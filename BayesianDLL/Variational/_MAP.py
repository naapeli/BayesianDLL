import torch

from .. import Model


class MAPResult:
    """The result of maximum a posteriori (MAP) optimization.

    Attributes
    ----------
    params : dict[str, torch.Tensor]
        Dictionary mapping parameter names to their constrained values at the mode.
    history : list[float]
        Log joint probability recorded at each callback frequency epoch.
    log_prob : float
        Final log joint probability at the mode.
    """

    def __init__(self, mode: dict[str, torch.Tensor], history: list[float], log_prob: float):
        self._mode = mode
        self.history = history
        self.log_prob = log_prob

    def mode(self) -> dict[str, torch.Tensor]:
        """Return the dictionary of parameter values at the posterior mode."""
        return {k: v.clone() for k, v in self._mode.items()}

    @property
    def params(self) -> dict[str, torch.Tensor]:
        return self._mode

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._mode:
                return self._mode[key]
            raise KeyError(f"Parameter '{key}' not found in MAP result.")
        elif isinstance(key, (int, slice)):
            return self.history[key]
        raise TypeError(f"Key must be str (for parameter) or int/slice (for history), got {type(key).__name__}")

    def __contains__(self, key):
        return key in self._mode

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)

    def keys(self):
        return self._mode.keys()

    def values(self):
        return self._mode.values()

    def items(self):
        return self._mode.items()

    def __repr__(self):
        params_str = ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self._mode.items())
        return f"MAPResult(log_prob={self.log_prob:.3f}, params={{{params_str}}})"


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
    ``verbose``).

    Parameters
    ----------
    model : Model
        The probabilistic model to optimize.
    lr : float, default 1e-2
        Learning rate (step size) for Adam optimization.
    epochs : int, default 100
        Number of optimization steps.
    betas : tuple[float, float], default (0.9, 0.999)
        Exponential decay rates for the first and second moment estimates in Adam.
    callback_frequency : int, default 1
        Frequency (in epochs) at which log joint probability is recorded and printed.
    verbose : bool, default True
        Whether to print log joint probability during optimization.

    Returns
    -------
    MAPResult
        Object containing the mode parameters, history of log joint probabilities,
        and final log probability.
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

    final_logp = history[-1] if history else model.model_log_prob().item()
    mode = {
        name: param.constrained_value.detach().clone()
        for name, param in model.params.items()
    }
    for name, param in model.deterministic_params.items():
        val = getattr(param, "constrained_value", getattr(param, "value", None))
        if val is not None:
            mode[name] = torch.as_tensor(val).detach().clone()

    return MAPResult(mode, history, final_logp)

