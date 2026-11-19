import torch

from .._model import Model
from ._guide import BaseGuide


def BBVI(model: Model, guide, n_samples=1, epochs=100, lr=1e-2, betas=(0.9, 0.999), clip_norm=10.0, lrd=1.0, callback_frequency=1, verbose=True, batch_size=None, subsample_plate=None):
    """Black-box variational inference with a hand-written, clipped Adam optimiser.

    The ELBO gradient uses the reparameterisation trick for guide
    distributions with ``has_rsample`` and the score-function estimator with
    a baseline otherwise (see :func:`BayesianDLL.Variational.elbo`).

    Args:
        model: the model.
        guide: a :class:`Guide` bound to ``model`` (Gaussian groups and/or
            hand-written entries).
        n_samples: Monte Carlo samples per ELBO estimate.
        epochs: number of optimisation steps.
        lr: learning rate.
        betas: Adam parameters (beta1, beta2).
        clip_norm: maximum L2 norm of each parameter's gradient.
        lrd: learning-rate decay factor per epoch (applied before every
            step, so step ``t`` uses ``lr * lrd**t``).
        callback_frequency: record (and print) the ELBO every this many epochs.
        verbose: print the ELBO.
        batch_size: when given, every epoch uses a random minibatch of
            ``batch_size`` observations along a plate whose size comes from
            ``Data`` (stochastic variational inference). The likelihood of
            the minibatch is scaled by ``N / batch_size``, and the ``Data``
            inputs whose batch dimension is the plate dimension are sliced
            consistently.
        subsample_plate: name of the plate to subsample; needed only when
            several plates qualify.

    Variational parameters may be tensors of any shape (e.g. matrices). Each
    one takes its Adam steps in its ``unconstrained_value`` coordinates
    (``log`` scale, or a log-diagonal lower-triangular parameterisation for
    ``scale_tril``).

    Returns the list of recorded ELBO values.
    """
    if not isinstance(guide, BaseGuide):
        raise TypeError(f"guide must be a Guide, got {type(guide).__name__}.")
    guide.validate(model)
    table = guide.variational_parameters()
    m = {key: torch.zeros_like(parameter.unconstrained_value) for key, parameter in table.items()}
    v = {key: torch.zeros_like(parameter.unconstrained_value) for key, parameter in table.items()}
    current_lr = lr
    history = []

    subsampler = None
    if batch_size is not None:
        from ._minibatch import Subsampler
        subsampler = Subsampler(model, batch_size, subsample_plate)

    for epoch in range(1, epochs + 1):
        current_lr *= lrd
        if subsampler is None:
            current_elbo, grads = guide._elbo(model, n_samples)
        else:
            with subsampler.minibatch() as scales:
                current_elbo, grads = guide._elbo(model, n_samples, scales=scales)

        for key, grad in grads.items():
            if key not in table:
                raise ValueError(f"Parameter {key} not found in guide.")
            parameter = table[key]
            grad = parameter.unconstrained_grad(grad)
            grad_norm = torch.linalg.norm(grad.reshape(-1))
            clipped_grad = grad * torch.clamp(clip_norm / (grad_norm + 1e-8), max=1.0)

            m[key] = betas[0] * m[key] + (1 - betas[0]) * clipped_grad
            v[key] = betas[1] * v[key] + (1 - betas[1]) * clipped_grad ** 2
            m_hat = m[key] / (1 - betas[0] ** epoch)
            v_hat = v[key] / (1 - betas[1] ** epoch)

            new_value = parameter.unconstrained_value + current_lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            parameter.set_unconstrained_value(new_value)

        if epoch % callback_frequency == 0:
            history.append(float(current_elbo))
            if verbose:
                print(f"Epoch {epoch} | ELBO: {float(current_elbo):.4f}")

    return history
