"""Automatic Gaussian guides over the unconstrained latent space."""

import math

import torch

from .._parameters import VariationalParameter
from ._ELBO import scaled_log_joint, scaled_log_joint_grad
from ._guide import BaseGuide, trace_from_unconstrained


_LOG_2PI = math.log(2 * math.pi)


class LatentLayout:
    """Concatenation of a model's unconstrained latent values into one vector."""

    def __init__(self, model, purpose="Automatic guides"):
        if not model._compiled:
            model.compile()
        self.names = list(model.params)
        if not self.names:
            raise ValueError("The model has no latent (random) parameters.")
        for name, parameter in model.params.items():
            if not parameter.distribution.transformed_state_space.is_continuous():
                raise ValueError(
                    f"'{name}' is a discrete latent. {purpose} need continuous latents; sum it out with discrete "
                    f"enumeration instead, e.g. RandomParameter('{name}', ..., enumerate=True) or "
                    f"BayesianDLL.marginalize(model, ['{name}'])."
                )
        self.shapes = {}
        self.slices = {}
        offset = 0
        for name in self.names:
            value = model.params[name].unconstrained_value
            self.shapes[name] = value.shape
            self.slices[name] = (offset, offset + value.numel())
            offset += value.numel()
        self.dim = offset
        first = model.params[self.names[0]].unconstrained_value
        dtype = first.dtype if first.is_floating_point() else torch.get_default_dtype()
        self.dtype = dtype
        self.device = first.device

    def check(self, model):
        for name in self.names:
            if name not in model.params or model.params[name].unconstrained_value.shape != self.shapes[name]:
                raise ValueError(f"The model does not match the latent layout of this guide ('{name}').")
        extra = [name for name in model.params if name not in self.shapes]
        if extra:
            raise ValueError(f"The guide has no distribution for the model latent(s) {extra}.")

    def pack(self, values):
        return torch.cat([values[name].reshape(-1).to(self.dtype) for name in self.names])

    def current(self, model):
        return self.pack({name: model.params[name].unconstrained_value for name in self.names})

    def unpack(self, flat):
        return {name: flat[start:end].reshape(self.shapes[name]) for name, (start, end) in self.slices.items()}

    def to_constrained(self, model, flat):
        return {name: model.params[name].distribution.transform.inverse(value) for name, value in self.unpack(flat).items()}


def _lower(matrix, diagonal=0):
    """``torch.tril`` by masking (elementwise, cheap for small matrices)."""
    n, m = matrix.shape[-2:]
    rows = torch.arange(n, device=matrix.device).unsqueeze(-1)
    cols = torch.arange(m, device=matrix.device).unsqueeze(0)
    return matrix * (cols - rows <= diagonal).to(matrix.dtype)


class ScaleTrilParameter(VariationalParameter):
    """A lower-triangular matrix with a positive diagonal, optimised in log-diagonal coordinates.

    The optimiser works on ``tril(L, -1) + diag(log diag L)``: the strictly
    upper triangle is never touched (its gradient is masked out) and the
    diagonal stays positive.
    """

    def __init__(self, name, value):
        value = torch.as_tensor(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("scale_tril must be a square matrix.")
        if not bool((torch.diagonal(value) > 0).all()):
            raise ValueError("The diagonal of scale_tril must be positive.")
        super().__init__(name, _lower(value))

    @property
    def unconstrained_value(self):
        return _lower(self.value, -1) + torch.diag_embed(torch.log(torch.diagonal(self.value)))

    def unconstrained_grad(self, grad):
        diagonal = torch.diagonal(grad) * torch.diagonal(self.value)
        return _lower(grad, -1) + torch.diag_embed(diagonal)

    def set_unconstrained_value(self, unconstrained_value):
        unconstrained_value = unconstrained_value.reshape(self.value.shape)
        value = _lower(unconstrained_value, -1) + torch.diag_embed(torch.exp(torch.diagonal(unconstrained_value)))
        self.set_new_value(value)


class AutoGuide(BaseGuide):
    """Base class of the automatic Gaussian guides over the concatenated unconstrained latents.

    The guide is built from ``model``: every latent gets a Normal family on
    its unconstrained value. For other variational families write a
    :class:`Guide` by hand.

    ``guide.sample(n)`` draws directly from the guide (no MCMC) and returns a
    :class:`SamplingResult` in the constrained space, including the
    model's deterministic nodes.
    """

    def __init__(self, model):
        self.model = model
        self.layout = LatentLayout(model, purpose=type(self).__name__)
        self._initial = self.layout.current(model)

    # --- to be implemented by subclasses ---------------------------------
    def variational_parameters(self):
        raise NotImplementedError

    def _transform(self, eps):
        """Map standard normal noise ``(n, D)`` to unconstrained draws ``(n, D)``."""
        raise NotImplementedError

    def _log_q(self, eps):
        raise NotImplementedError

    def _accumulate(self, grads, model_grad, eps):
        """Add the reparameterised gradient of ``log p(z(eps)) - log q(z(eps))`` for one draw."""
        raise NotImplementedError

    # --- shared -----------------------------------------------------------
    def _noise(self, n):
        return torch.randn(n, self.layout.dim, dtype=self.layout.dtype, device=self.layout.device)

    def _elbo(self, model, n_samples, scales=None):
        self.layout.check(model)
        table = self.variational_parameters()
        grads = {key: torch.zeros_like(parameter.value) for key, parameter in table.items()}
        eps = self._noise(n_samples)
        draws = self._transform(eps)
        total = 0.0
        for i in range(n_samples):
            with model.temporarily_set_many(self.layout.unpack(draws[i])):
                log_p = scaled_log_joint(model, scales)
                model_grads = scaled_log_joint_grad(model, self.layout.names, scales)
            total = total + log_p - self._log_q(eps[i])
            self._accumulate(grads, self.layout.pack(model_grads), eps[i])
        for key in grads:
            grads[key] = grads[key] / n_samples
        return total / n_samples, grads

    def sample_unconstrained(self, n):
        return self._transform(self._noise(n))

    def sample(self, n_samples=1000, n_chains=1):
        """Draw ``n_chains * n_samples`` independent draws from the guide."""
        draws = self.sample_unconstrained(n_chains * n_samples).reshape(n_chains, n_samples, self.layout.dim)
        return trace_from_unconstrained(self.model, self.layout, draws)

    @property
    def loc(self):
        """Mean of the guide in the unconstrained space, as a flat vector."""
        return self._transform(torch.zeros(1, self.layout.dim, dtype=self.layout.dtype, device=self.layout.device))[0]

    def median(self):
        """Constrained values at the guide's mean (the median of every latent)."""
        return self.layout.to_constrained(self.model, self.loc)


class AutoNormal(AutoGuide):
    """A Normal on every unconstrained latent, built automatically (mean-field, diagonal covariance).

    For each latent ``name`` the guide has the VariationalParameters
    ``f"{name}_loc"`` and ``f"{name}_scale"`` (optimised on the log scale).
    The guide starts at the model's current values with scale ``init_scale``.
    """

    def __init__(self, model, init_scale=0.1):
        super().__init__(model)
        self.locs = {}
        self.scales = {}
        for name, value in self.layout.unpack(self._initial).items():
            self.locs[name] = VariationalParameter(f"{name}_loc", value.clone())
            self.scales[name] = VariationalParameter(f"{name}_scale", torch.full_like(value, float(init_scale)), log_scale=True)

    def variational_parameters(self):
        table = {}
        for name in self.layout.names:
            table[f"{name}_loc"] = self.locs[name]
            table[f"{name}_scale"] = self.scales[name]
        return table

    def _flat(self, parameters):
        return torch.cat([parameters[name].value.reshape(-1) for name in self.layout.names])

    def _transform(self, eps):
        return self._flat(self.locs) + eps * self._flat(self.scales)

    def _log_q(self, eps):
        return (-0.5 * eps ** 2 - torch.log(self._flat(self.scales)) - 0.5 * _LOG_2PI).sum()

    def _accumulate(self, grads, model_grad, eps):
        scale = self._flat(self.scales)
        grad_scale = model_grad * eps + 1 / scale
        for name, (start, end) in self.layout.slices.items():
            shape = self.locs[name].value.shape
            grads[f"{name}_loc"] += model_grad[start:end].reshape(shape)
            grads[f"{name}_scale"] += grad_scale[start:end].reshape(shape)

    @property
    def scale(self):
        """Standard deviations in the unconstrained space, as a flat vector."""
        return self._flat(self.scales)


class AutoMultivariateNormal(AutoGuide):
    """A full-rank multivariate Normal over the concatenated unconstrained latents, built automatically.

    ``z = loc + L eps`` with a lower-triangular ``L = scale_tril`` whose
    diagonal is kept positive (log-diagonal parameterisation). The hand-written
    reparameterisation gradients are ``d/dloc = g`` and
    ``d/dL = tril(g eps^T) + diag(1 / L_ii)``, where the last term is the
    gradient of the entropy ``sum_i log L_ii``.
    """

    def __init__(self, model, init_scale=0.1):
        super().__init__(model)
        dim, dtype, device = self.layout.dim, self.layout.dtype, self.layout.device
        self.loc_parameter = VariationalParameter("loc", self._initial.clone())
        self.scale_tril_parameter = ScaleTrilParameter(
            "scale_tril", float(init_scale) * torch.eye(dim, dtype=dtype, device=device)
        )

    def variational_parameters(self):
        return {"loc": self.loc_parameter, "scale_tril": self.scale_tril_parameter}

    @property
    def scale_tril(self):
        return self.scale_tril_parameter.value

    @property
    def covariance(self):
        L = self.scale_tril
        return L @ L.transpose(-2, -1)

    def _transform(self, eps):
        return self.loc_parameter.value + eps @ self.scale_tril.transpose(-2, -1)

    def _log_q(self, eps):
        diagonal = torch.diagonal(self.scale_tril)
        return -0.5 * (eps ** 2).sum() - torch.log(diagonal).sum() - 0.5 * self.layout.dim * _LOG_2PI

    def _accumulate(self, grads, model_grad, eps):
        grads["loc"] += model_grad
        grads["scale_tril"] += _lower(torch.outer(model_grad, eps)) + torch.diag_embed(1 / torch.diagonal(self.scale_tril))
