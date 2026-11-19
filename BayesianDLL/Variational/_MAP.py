import torch

from .. import Model


def find_MAP(model: Model, lr=1e-2, epochs=100, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
    """Maximise the model's log joint density with Adam, in place.

    Every latent takes ``epochs`` Adam ascent steps (learning rate ``lr``,
    moment decay rates ``betas``) on its unconstrained value, using the
    hand-written gradient ``model.joint_grad_log_prob``. The objective is
    the log joint in the unconstrained space, which includes the
    ``log|det J|`` of every transform: the result is the mode of the
    posterior of the unconstrained values, which in general differs from
    the mode of the constrained parameters (e.g. for a ``HalfNormal``
    scale). For a GP likelihood (:class:`~BayesianDLL.GP.ExactGP`, ...)
    with priors on the hyperparameters this is a type-II MAP estimate; it
    is a type-II maximum-likelihood estimate only with flat priors in the
    unconstrained space.

    The model's latents are left at the final values. Every
    ``callback_frequency`` epochs the log joint is recorded (and printed if
    ``verbose``). Returns the list of recorded log joint values.
    """
    history = []

    m = {}
    v = {}
    t = 0

    for name, param in model.params.items():
        m[name] = torch.zeros_like(param.unconstrained_value)
        v[name] = torch.zeros_like(param.unconstrained_value)

    for epoch in range(epochs):
        t += 1

        grads = model.joint_grad_log_prob()
        for name, param in model.params.items():
            grad = grads[name]
            m[name] = betas[0] * m[name] + (1 - betas[0]) * grad
            v[name] = betas[1] * v[name] + (1 - betas[1]) * grad.pow(2)
            m_hat = m[name] / (1 - betas[0] ** t)
            v_hat = v[name] / (1 - betas[1] ** t)
            new_unconstrained_value = param.unconstrained_value + lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            param.set_unconstrained_value(new_unconstrained_value)

        if epoch % callback_frequency == 0:
            logp = model.model_log_prob()
            history.append(logp.item())
            if verbose:
                print(f"Epoch: {epoch + 1} - Log_prob: {logp.item():.2f}")

    return history

