import torch
import torch.nn.functional as F
import math
from functools import reduce
from abc import ABC, abstractmethod

from ._transforms import (
    IdentityTransform, LogitTransform, LogTransform, InverseSoftPlusTransform,
    StickBreakingTransform, CorrCholeskyTransform, ChainTransform, _apply_jacobian_transpose, _log1m_tanh_sq,
    PositiveDefiniteTransform, OrderedTransform, log_sigmoid,
)
from ._state_space import (
    ContinuousReal, ContinuousPositive, ContinuousRange, ContinuousSimplex, DiscretePositive, DiscreteRange, Union,
    DiscreteNonNegative, ContinuousCorrCholesky, ContinuousPositiveDefinite, ContinuousOrdered,
)
from .._parameters import RandomParameter, DeterministicParameter, ObservedParameter, VariationalParameter
from .._data import Data
from ._resolve import resolve
from .._batching import bsum as _bsum


_LOG_2PI = math.log(2 * math.pi)
_LOG_PI = math.log(math.pi)


# ========================= helpers =========================
def _float_dtype(*values):
    """Promoted floating dtype of the given tensors (default dtype if none is floating)."""
    floats = [v.dtype for v in values if isinstance(v, torch.Tensor) and v.is_floating_point()]
    if not floats:
        return torch.get_default_dtype()
    return reduce(torch.promote_types, floats)


def _device(*values):
    for v in values:
        if isinstance(v, torch.Tensor):
            return v.device
    return None


def _as_float(*values):
    """Resolve parameters to tensors of one common floating dtype and device."""
    tensors = [resolve(v) for v in values]
    # Python numbers adopt the dtype of the tensor-valued parameters.
    typed = [r for v, r in zip(values, tensors) if not isinstance(v, int | float)]
    dtype = _float_dtype(*typed)
    device = _device(*typed)
    out = [t.to(dtype=dtype, device=device) for t in tensors]
    return out if len(out) > 1 else out[0]


def _sample_shape(n_samples, batch_shape, event_shape=()):
    return (n_samples, *tuple(batch_shape), *tuple(event_shape))


def _standard_gamma(alpha):
    """Exact Gamma(alpha, 1) draws with one draw per element of ``alpha``.

    Marsaglia & Tsang (2000) squeeze/rejection method; ``alpha < 1`` uses the
    boost ``G(alpha) = G(alpha + 1) * U^(1/alpha)``.
    """
    boost = alpha < 1
    a = torch.where(boost, alpha + 1, alpha)
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = torch.zeros_like(a)
    undone = torch.ones_like(a, dtype=torch.bool)
    while bool(undone.any()):
        x = torch.randn_like(a)
        v = (1 + c * x) ** 3
        u = torch.rand_like(a)
        positive = v > 0
        log_v = torch.log(torch.where(positive, v, torch.ones_like(v)))
        accept = undone & positive & (torch.log(u) < 0.5 * x ** 2 + d - d * v + d * log_v)
        out = torch.where(accept, d * v, out)
        undone = undone & ~accept
    if bool(boost.any()):
        u = torch.rand_like(a)
        log_out = torch.log(out) + torch.log(u) / torch.where(boost, alpha, torch.ones_like(alpha))
        out = torch.where(boost, torch.exp(log_out), out)
    return out


def _sample_gamma(alpha, rate, shape):
    alpha = alpha.expand(shape)
    return _standard_gamma(alpha) / rate


def _sample_beta(a, b, shape):
    x = _standard_gamma(a.expand(shape))
    y = _standard_gamma(b.expand(shape))
    return x / (x + y)


def _uniform(shape, dtype, device):
    """Uniform draws on the open interval (0, 1)."""
    u = torch.rand(shape, dtype=dtype, device=device)
    tiny = torch.finfo(dtype).tiny
    return u.clamp(min=tiny)


def _discrete_grad_error():
    return NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")


def _is_integer(x):
    if not x.is_floating_point():
        return torch.ones_like(x, dtype=torch.bool)
    return x == torch.floor(x)


def _in_support(x, inside, fill=0):
    """``(inside, x_safe)``: the support mask and ``x`` with ``fill`` outside the support.

    The density is evaluated at ``x_safe`` and masked afterwards, so values
    outside the support (e.g. ``lgamma`` of a negative count) never produce
    NaNs in the selected branch or in its gradients.
    """
    if x.shape != inside.shape:
        x = x.expand(inside.shape)
    return inside, torch.where(inside, x, torch.full_like(x, fill))


def _restrict(log_p, inside):
    """``log_p`` inside the support and ``-inf`` outside."""
    return torch.where(inside, log_p, torch.full_like(log_p, -math.inf))


def _zero_outside(grad, inside):
    """A parameter gradient of ``log_p``: zero where ``log_p`` is a constant ``-inf``."""
    return torch.where(inside, grad, torch.zeros_like(grad))


def _check_exactly_one(names, values):
    given = [n for n, v in zip(names, values) if v is not None]
    if len(given) != 1:
        raise ValueError(f"Specify exactly one of {' / '.join(names)} (got {given or 'none'}).")


def _mvlgamma(a, p):
    j = torch.arange(1, p + 1, dtype=a.dtype, device=a.device)
    return p * (p - 1) / 4 * _LOG_PI + torch.lgamma(a.unsqueeze(-1) + (1 - j) / 2).sum(-1)


def _mvdigamma(a, p):
    j = torch.arange(1, p + 1, dtype=a.dtype, device=a.device)
    return torch.digamma(a.unsqueeze(-1) + (1 - j) / 2).sum(-1)


class Distribution(ABC):
    #: ``True`` when ``sample(..., _reparametrization_trick_grad=True)`` is supported.
    has_rsample = False

    def __init__(self, transform=IdentityTransform(), state_space=ContinuousReal(), transformed_state_space=ContinuousReal()):
        self.transform = transform
        self.state_space = state_space
        self.transformed_state_space = transformed_state_space
        self.parameters = set()
        self.variational_parameters = dict()

    @property
    def event_shape(self):
        """Shape of a single draw from this distribution."""
        return ()

    @property
    def batch_shape(self):
        """Shape of independent (batch) dimensions from parameter broadcasting."""
        return ()

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))

    @abstractmethod
    def log_pdf(self, x):
        """
        Log probability density.  Returns a tensor whose shape equals the
        broadcast batch dimensions (event dimensions are summed out).
        """
        pass

    @abstractmethod
    def log_pdf_grad(self, x):
        pass

    @abstractmethod
    def log_pdf_param_grads(self, x):
        pass

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Draw ``n_samples`` samples of shape ``(n_samples, *batch_shape, *event_shape)``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement sampling.")

    def cdf(self, x):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form cdf.")

    def icdf(self, u):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form inverse cdf.")

    def _no_reparam(self):
        return NotImplementedError(f"{type(self).__name__} has no reparameterised sampler; call sample() without _reparametrization_trick_grad.")

    def _log_prob_unconstrained(self, x_unconstrained):
        """
        Log probability in unconstrained space, including the Jacobian
        correction.  *x_unconstrained* has the same shape as the parameter's
        unconstrained value.  Returns a **scalar**.
        """
        x_constrained = self.transform.inverse(x_unconstrained)
        log_det = self.transform.log_abs_det_jacobian(x_unconstrained)
        return _bsum(self.log_pdf(x_constrained)) + log_det

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        """
        Gradient of ``_log_prob_unconstrained`` w.r.t. *x_unconstrained*.
        Returns a tensor with the same shape as *x_unconstrained*.
        """
        log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
        term1 = _apply_jacobian_transpose(self.transform, x_unconstrained, log_pdf_grad_x)
        d_log_det = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
        return term1 + d_log_det

    def resolve_name(self, name, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter | ObservedParameter | VariationalParameter | Data):
            return parameter.name
        return name

    def add_dependency(self, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter | ObservedParameter | Data):
            self.parameters.add(parameter.name)
        if isinstance(parameter, VariationalParameter):
            self.variational_parameters[parameter.name] = parameter

    def _add_dependencies_of(self, distribution):
        self.parameters |= set(distribution.parameters)
        self.variational_parameters.update(distribution.variational_parameters)


# ========================= CONTINUOUS =========================
class Normal(Distribution):
    """Normal distribution with mean ``mu`` and **standard deviation** ``sigma``."""

    has_rsample = True

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.add_dependency(mu)
        self.add_dependency(sigma)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.mu).shape, resolve(self.sigma).shape)

    def log_pdf(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        z = (x - mu) / sigma
        return -0.5 * z ** 2 - torch.log(sigma) - 0.5 * _LOG_2PI

    def log_pdf_grad(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        return -(x - mu) / sigma ** 2

    def log_pdf_param_grads(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        diff = x - mu
        grad_mu = diff / sigma ** 2
        grad_sigma = -1 / sigma + diff ** 2 / sigma ** 3
        return {self.resolve_name("mean", self.mu): grad_mu, self.resolve_name("sigma", self.sigma): grad_sigma}

    def cdf(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        return torch.special.ndtr((x - mu) / sigma)

    def icdf(self, u):
        mu, sigma = _as_float(self.mu, self.sigma)
        return mu + sigma * torch.special.ndtri(u)

    def cdf_param_grads(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        t = (x - mu) / sigma
        phi = torch.exp(-0.5 * t ** 2 - 0.5 * _LOG_2PI)
        return {self.resolve_name("mean", self.mu): -phi / sigma, self.resolve_name("sigma", self.sigma): -phi * t / sigma}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        mu, sigma = _as_float(self.mu, self.sigma)
        shape = _sample_shape(n_samples, self.batch_shape)
        eps = torch.randn(shape, dtype=mu.dtype, device=mu.device)
        samples = mu + eps * sigma
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): torch.ones_like(eps), self.resolve_name("sigma", self.sigma): eps}
        return samples


class HalfNormal(Distribution):
    """Half-normal distribution ``|N(0, sigma^2)|`` with scale ``sigma``."""

    has_rsample = True

    def __init__(self, sigma):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.sigma = sigma
        self.add_dependency(sigma)

    @property
    def batch_shape(self):
        return resolve(self.sigma).shape

    def log_pdf(self, x):
        sigma = _as_float(self.sigma)
        return 0.5 * math.log(2 / math.pi) - torch.log(sigma) - 0.5 * (x / sigma) ** 2

    def log_pdf_grad(self, x):
        sigma = _as_float(self.sigma)
        return -x / sigma ** 2

    def log_pdf_param_grads(self, x):
        sigma = _as_float(self.sigma)
        return {self.resolve_name("sigma", self.sigma): -1 / sigma + x ** 2 / sigma ** 3}

    def _log_prob_unconstrained(self, x_unconstrained):
        sigma = _as_float(self.sigma)
        z = x_unconstrained
        return _bsum(0.5 * math.log(2 / math.pi) - torch.log(sigma) - 0.5 * torch.exp(2 * z) / sigma ** 2 + z)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        sigma = _as_float(self.sigma)
        return 1 - torch.exp(2 * x_unconstrained) / sigma ** 2

    def cdf(self, x):
        sigma = _as_float(self.sigma)
        return torch.erf(x / (sigma * math.sqrt(2)))

    def icdf(self, u):
        sigma = _as_float(self.sigma)
        return sigma * torch.special.ndtri((1 + u) / 2)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        sigma = _as_float(self.sigma)
        eps = torch.randn(_sample_shape(n_samples, self.batch_shape), dtype=sigma.dtype, device=sigma.device).abs()
        samples = sigma * eps
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("sigma", self.sigma): eps}
        return samples


class LogNormal(Distribution):
    """``exp(N(mu, sigma^2))`` with ``sigma`` the standard deviation of the log."""

    has_rsample = True

    def __init__(self, mu, sigma):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.mu = mu
        self.sigma = sigma
        self.add_dependency(mu)
        self.add_dependency(sigma)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.mu).shape, resolve(self.sigma).shape)

    def log_pdf(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        log_x = torch.log(x)
        return -log_x - torch.log(sigma) - 0.5 * _LOG_2PI - 0.5 * ((log_x - mu) / sigma) ** 2

    def log_pdf_grad(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        return -(1 + (torch.log(x) - mu) / sigma ** 2) / x

    def log_pdf_param_grads(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        diff = torch.log(x) - mu
        return {self.resolve_name("mean", self.mu): diff / sigma ** 2,
                self.resolve_name("sigma", self.sigma): -1 / sigma + diff ** 2 / sigma ** 3}

    def _log_prob_unconstrained(self, x_unconstrained):
        mu, sigma = _as_float(self.mu, self.sigma)
        z = x_unconstrained
        return _bsum(-torch.log(sigma) - 0.5 * _LOG_2PI - 0.5 * ((z - mu) / sigma) ** 2)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        mu, sigma = _as_float(self.mu, self.sigma)
        return -(x_unconstrained - mu) / sigma ** 2

    def cdf(self, x):
        mu, sigma = _as_float(self.mu, self.sigma)
        return torch.special.ndtr((torch.log(x) - mu) / sigma)

    def icdf(self, u):
        mu, sigma = _as_float(self.mu, self.sigma)
        return torch.exp(mu + sigma * torch.special.ndtri(u))

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        mu, sigma = _as_float(self.mu, self.sigma)
        eps = torch.randn(_sample_shape(n_samples, self.batch_shape), dtype=mu.dtype, device=mu.device)
        samples = torch.exp(mu + sigma * eps)
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): samples, self.resolve_name("sigma", self.sigma): samples * eps}
        return samples


class MultivariateNormal(Distribution):
    """Multivariate normal parameterised by ``covariance`` **or** ``scale_tril``.

    ``MultivariateNormal(mu, covariance)`` or
    ``MultivariateNormal(mu, scale_tril=L)`` with ``L`` lower triangular and
    ``covariance = L @ L.T``. The covariance gradient returned by
    :meth:`log_pdf_param_grads` is the full symmetric matrix
    ``0.5 * (S^-1 d d^T S^-1 - S^-1)``; the ``scale_tril`` gradient is lower
    triangular.
    """

    _param_event_ndims = {"mu": 1, "covariance": 2, "scale_tril": 2}

    def __init__(self, mu, covariance=None, scale_tril=None):
        super().__init__()
        _check_exactly_one(["covariance", "scale_tril"], [covariance, scale_tril])
        self.mu = mu
        self.covariance = covariance
        self.scale_tril = scale_tril
        self.add_dependency(mu)
        self.add_dependency(covariance if covariance is not None else scale_tril)

    @property
    def has_rsample(self):
        return not isinstance(self.covariance if self.covariance is not None else self.scale_tril, VariationalParameter)

    def _params(self):
        if self.covariance is not None:
            mu, cov = _as_float(self.mu, self.covariance)
            L = torch.linalg.cholesky(cov)
        else:
            mu, L = _as_float(self.mu, self.scale_tril)
        if mu.ndim == 0:
            mu = mu.expand(L.shape[-1])
        return mu, L

    def _matrix(self):
        return resolve(self.covariance if self.covariance is not None else self.scale_tril)

    @property
    def event_shape(self):
        return self._matrix().shape[-1:]

    @property
    def batch_shape(self):
        mu = resolve(self.mu)
        return torch.broadcast_shapes(mu.shape[:-1], self._matrix().shape[:-2])

    def _whiten(self, x):
        """Return (y, a, L) with y = L^-1 (x - mu), a = Sigma^-1 (x - mu)."""
        mu, L = self._params()
        diff = x - mu
        batch = torch.broadcast_shapes(diff.shape[:-1], L.shape[:-2])
        D = L.shape[-1]
        L = L.expand(*batch, D, D)
        diff = diff.expand(*batch, D)
        y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
        a = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        return y.squeeze(-1), a.squeeze(-1), L

    def log_pdf(self, x):
        y, _, L = self._whiten(x)
        D = L.shape[-1]
        log_det = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return -0.5 * (y ** 2).sum(-1) - log_det - 0.5 * D * _LOG_2PI

    def log_pdf_grad(self, x):
        _, a, _ = self._whiten(x)
        return -a

    def log_pdf_param_grads(self, x):
        y, a, L = self._whiten(x)
        grads = {self.resolve_name("mean", self.mu): a}
        if self.covariance is not None:
            precision = torch.cholesky_inverse(L)
            grads[self.resolve_name("covariance", self.covariance)] = 0.5 * (a.unsqueeze(-1) * a.unsqueeze(-2) - precision)
        else:
            diag = torch.diagonal(L, dim1=-2, dim2=-1)
            grads[self.resolve_name("scale_tril", self.scale_tril)] = torch.tril(a.unsqueeze(-1) * y.unsqueeze(-2)) - torch.diag_embed(1 / diag)
        return grads

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad and not self.has_rsample:
            raise NotImplementedError("Reparameterised gradients w.r.t. a variational covariance / scale_tril are not implemented.")
        mu, L = self._params()
        shape = _sample_shape(n_samples, self.batch_shape, self.event_shape)
        eps = torch.randn(shape, dtype=mu.dtype, device=mu.device)
        samples = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): torch.ones_like(samples)}
        return samples


class Beta(Distribution):
    def __init__(self, a, b):
        super().__init__(LogitTransform(low=0, high=1), ContinuousRange(0, 1), ContinuousReal())
        self.a = a
        self.b = b
        self.add_dependency(a)
        self.add_dependency(b)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.a).shape, resolve(self.b).shape)

    def log_pdf(self, x):
        a, b = _as_float(self.a, self.b)
        return torch.xlogy(a - 1, x) + torch.special.xlog1py(b - 1, -x) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    def log_pdf_grad(self, x):
        a, b = _as_float(self.a, self.b)
        return (a - 1) / x - (b - 1) / (1 - x)

    def log_pdf_param_grads(self, x):
        a, b = _as_float(self.a, self.b)
        grad_a = torch.log(x) - torch.digamma(a) + torch.digamma(a + b)
        grad_b = torch.log1p(-x) - torch.digamma(b) + torch.digamma(a + b)
        return {self.resolve_name("a", self.a): grad_a, self.resolve_name("b", self.b): grad_b}

    def _log_prob_unconstrained(self, x_unconstrained):
        a, b = _as_float(self.a, self.b)
        log_x = log_sigmoid(x_unconstrained)
        log_1_minus_x = log_sigmoid(-x_unconstrained)
        log_prob = a * log_x + b * log_1_minus_x - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        return _bsum(log_prob)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        a, b = _as_float(self.a, self.b)
        return a * torch.sigmoid(-x_unconstrained) - b * torch.sigmoid(x_unconstrained)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        a, b = _as_float(self.a, self.b)
        return _sample_beta(a, b, _sample_shape(n_samples, self.batch_shape))


class Exponential(Distribution):
    has_rsample = True

    def __init__(self, rate):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.rate = rate
        self.add_dependency(rate)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return resolve(self.rate).shape

    def log_pdf(self, x):
        rate = _as_float(self.rate)
        return torch.log(rate) - rate * x

    def log_pdf_grad(self, x):
        rate = _as_float(self.rate)
        return -rate * torch.ones_like(x)

    def log_pdf_param_grads(self, x):
        rate = _as_float(self.rate)
        grad_rate = 1 / rate - x
        return {self.resolve_name("rate", self.rate): grad_rate}

    def _log_prob_unconstrained(self, x_unconstrained):
        if not isinstance(self.transform, LogTransform):
            raise RuntimeError("Exponential._log_prob_unconstrained can only be used if the transform is a log transform")
        rate = _as_float(self.rate)
        z = x_unconstrained
        return _bsum(torch.log(rate) - rate * torch.exp(z) + z)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        if not isinstance(self.transform, LogTransform):
            raise RuntimeError("Exponential._log_prob_unconstrained can only be used if the transform is a log transform")
        rate = _as_float(self.rate)
        return 1 - rate * torch.exp(x_unconstrained)

    def cdf(self, x):
        rate = _as_float(self.rate)
        return -torch.expm1(-rate * x)

    def icdf(self, u):
        rate = _as_float(self.rate)
        return -torch.log1p(-u) / rate

    def cdf_param_grads(self, x):
        rate = _as_float(self.rate)
        return {self.resolve_name("rate", self.rate): x * torch.exp(-rate * x)}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        rate = _as_float(self.rate)
        eps = _uniform(_sample_shape(n_samples, self.batch_shape), rate.dtype, rate.device)
        log_eps = torch.log(eps)
        samples = -log_eps / rate
        if _reparametrization_trick_grad:
            grad_rate = log_eps / rate ** 2
            return samples, {self.resolve_name("rate", self.rate): grad_rate}
        return samples


class Gamma(Distribution):
    """Gamma distribution with shape ``alpha`` and **rate** ``beta``."""

    def __init__(self, alpha, beta):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.alpha = alpha
        self.beta = beta
        self.add_dependency(alpha)
        self.add_dependency(beta)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.alpha).shape, resolve(self.beta).shape)

    def log_pdf(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return alpha * torch.log(beta) - torch.lgamma(alpha) + torch.xlogy(alpha - 1, x) - beta * x

    def log_pdf_grad(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return (alpha - 1) / x - beta

    def log_pdf_param_grads(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return {self.resolve_name("alpha", self.alpha): torch.log(beta) - torch.digamma(alpha) + torch.log(x),
                self.resolve_name("beta", self.beta): alpha / beta - x}

    def _log_prob_unconstrained(self, x_unconstrained):
        alpha, beta = _as_float(self.alpha, self.beta)
        z = x_unconstrained
        return _bsum(alpha * torch.log(beta) - torch.lgamma(alpha) + alpha * z - beta * torch.exp(z))

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        alpha, beta = _as_float(self.alpha, self.beta)
        return alpha - beta * torch.exp(x_unconstrained)

    def cdf(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return torch.special.gammainc(alpha, beta * x)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        alpha, beta = _as_float(self.alpha, self.beta)
        return _sample_gamma(alpha, beta, _sample_shape(n_samples, self.batch_shape))


class Uniform(Distribution):
    """Continuous uniform distribution on ``[low, high]``.

    ``log_pdf_param_grads`` returns the gradient of ``log p(x | low, high)``
    at a **fixed** value ``x`` inside the support (``d/dlow = 1/(high-low)``,
    ``d/dhigh = -1/(high-low)``, zero outside the support). This is exact for
    observed data. For a *latent* Uniform the sampling transform
    ``x = low + (high-low) sigmoid(z)`` itself depends on the bounds, and the
    unconstrained density ``log p(x(z)) + log|dx/dz| = log sigmoid(z) +
    log sigmoid(-z)`` does not depend on them at all; the model gradient does
    not account for ``x`` moving with the bounds, so for random bounds prefer
    the Metropolis sampler or write ``x = low + (high - low) * u`` with
    ``u ~ Uniform(0, 1)``.
    """

    has_rsample = True

    def __init__(self, low, high):
        super().__init__(LogitTransform(low=low, high=high), ContinuousRange(low=low, high=high), ContinuousReal())
        self.low = low
        self.high = high
        self.add_dependency(low)
        self.add_dependency(high)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.low).shape, resolve(self.high).shape)

    def _inside(self, x, low, high):
        return (x >= low) & (x <= high)

    def pdf(self, x):
        low, high = _as_float(self.low, self.high)
        return torch.where(self._inside(x, low, high), 1.0 / (high - low), torch.zeros_like(x))

    def log_pdf(self, x):
        low, high = _as_float(self.low, self.high)
        return torch.where(self._inside(x, low, high), -torch.log(high - low), torch.full_like(x, -torch.inf))

    def log_pdf_grad(self, x):
        return torch.zeros_like(x)

    def log_pdf_param_grads(self, x):
        low, high = _as_float(self.low, self.high)
        inside = self._inside(x, low, high)
        inv = torch.where(inside, 1.0 / (high - low), torch.zeros_like(x))
        return {self.resolve_name("low", self.low): inv, self.resolve_name("high", self.high): -inv}

    def _log_prob_unconstrained(self, x_unconstrained):
        # log(1/(high-low)) + log|dx/dz| = log sigmoid(z) + log sigmoid(-z): independent of the bounds.
        z = x_unconstrained
        return _bsum(log_sigmoid(z) + log_sigmoid(-z))

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        z = x_unconstrained
        return torch.sigmoid(-z) - torch.sigmoid(z)

    def cdf(self, x):
        low, high = _as_float(self.low, self.high)
        return ((x - low) / (high - low)).clamp(0, 1)

    def icdf(self, u):
        low, high = _as_float(self.low, self.high)
        return low + (high - low) * u

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        low, high = _as_float(self.low, self.high)
        u = torch.rand(_sample_shape(n_samples, self.batch_shape), dtype=low.dtype, device=low.device)
        samples = low + (high - low) * u
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("low", self.low): 1 - u, self.resolve_name("high", self.high): u}
        return samples


class InvGamma(Distribution):
    def __init__(self, alpha, beta):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.alpha = alpha
        self.beta = beta
        self.add_dependency(alpha)
        self.add_dependency(beta)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.alpha).shape, resolve(self.beta).shape)

    def log_pdf(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return alpha * torch.log(beta) - torch.lgamma(alpha) - (alpha + 1) * torch.log(x) - beta / x

    def log_pdf_grad(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return -(alpha + 1) / x + beta / x ** 2

    def log_pdf_param_grads(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        grad_alpha = torch.log(beta) - torch.digamma(alpha) - torch.log(x)
        grad_beta = alpha / beta - 1 / x
        return {self.resolve_name("alpha", self.alpha): grad_alpha, self.resolve_name("beta", self.beta): grad_beta}

    def _log_prob_unconstrained(self, x_unconstrained):
        alpha, beta = _as_float(self.alpha, self.beta)
        z = x_unconstrained
        return _bsum(alpha * torch.log(beta) - torch.lgamma(alpha) - alpha * z - beta * torch.exp(-z))

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        alpha, beta = _as_float(self.alpha, self.beta)
        return -alpha + beta * torch.exp(-x_unconstrained)

    def cdf(self, x):
        alpha, beta = _as_float(self.alpha, self.beta)
        return torch.special.gammaincc(alpha, beta / x)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        alpha, beta = _as_float(self.alpha, self.beta)
        shape = _sample_shape(n_samples, self.batch_shape)
        return beta / _standard_gamma(alpha.expand(shape))


class HalfCauchy(Distribution):
    has_rsample = True

    def __init__(self, scale):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.scale = scale
        self.add_dependency(scale)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return resolve(self.scale).shape

    def log_pdf(self, x):
        scale = _as_float(self.scale)
        return math.log(2.0) - _LOG_PI - torch.log(scale) - torch.log1p((x / scale) ** 2)

    def log_pdf_grad(self, x):
        scale = _as_float(self.scale)
        return -2 * x / (scale ** 2 + x ** 2)

    def log_pdf_param_grads(self, x):
        scale = _as_float(self.scale)
        grad_scale = -1 / scale + 2 * x ** 2 / (scale * (x ** 2 + scale ** 2))
        return {self.resolve_name("scale", self.scale): grad_scale}

    def _log_prob_unconstrained(self, x_unconstrained):
        scale = _as_float(self.scale)
        z = x_unconstrained
        x = torch.exp(z)
        return _bsum(math.log(2.0) - _LOG_PI - torch.log(scale) + z - torch.log1p((x / scale) ** 2))

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        scale = _as_float(self.scale)
        x = torch.exp(x_unconstrained)
        return (scale ** 2 - x ** 2) / (scale ** 2 + x ** 2)

    def cdf(self, x):
        scale = _as_float(self.scale)
        return 2 / math.pi * torch.atan(x / scale)

    def icdf(self, u):
        scale = _as_float(self.scale)
        return scale * torch.tan(math.pi / 2 * u)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        scale = _as_float(self.scale)
        u = torch.rand(_sample_shape(n_samples, self.batch_shape), dtype=scale.dtype, device=scale.device)
        unit = torch.tan(math.pi / 2 * u)
        samples = scale * unit
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("scale", self.scale): unit}
        return samples


class Cauchy(Distribution):
    has_rsample = True

    def __init__(self, mu, scale):
        super().__init__()
        self.mu = mu
        self.scale = scale
        self.add_dependency(mu)
        self.add_dependency(scale)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.mu).shape, resolve(self.scale).shape)

    def log_pdf(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        return -_LOG_PI - torch.log(scale) - torch.log1p(((x - mu) / scale) ** 2)

    def log_pdf_grad(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        diff = x - mu
        return -2 * diff / (scale ** 2 + diff ** 2)

    def log_pdf_param_grads(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        diff = x - mu
        denom = scale ** 2 + diff ** 2
        return {self.resolve_name("mean", self.mu): 2 * diff / denom,
                self.resolve_name("scale", self.scale): -1 / scale + 2 * diff ** 2 / (scale * denom)}

    def cdf(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        return 0.5 + torch.atan((x - mu) / scale) / math.pi

    def icdf(self, u):
        mu, scale = _as_float(self.mu, self.scale)
        return mu + scale * torch.tan(math.pi * (u - 0.5))

    def cdf_param_grads(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        t = (x - mu) / scale
        dens = 1 / (math.pi * (1 + t ** 2))
        return {self.resolve_name("mean", self.mu): -dens / scale, self.resolve_name("scale", self.scale): -dens * t / scale}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        mu, scale = _as_float(self.mu, self.scale)
        u = _uniform(_sample_shape(n_samples, self.batch_shape), mu.dtype, mu.device)
        unit = torch.tan(math.pi * (u - 0.5))
        samples = mu + scale * unit
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): torch.ones_like(unit), self.resolve_name("scale", self.scale): unit}
        return samples


class Laplace(Distribution):
    has_rsample = True

    def __init__(self, mu, scale):
        super().__init__()
        self.mu = mu
        self.scale = scale
        self.add_dependency(mu)
        self.add_dependency(scale)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.mu).shape, resolve(self.scale).shape)

    def log_pdf(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        return -math.log(2.0) - torch.log(scale) - (x - mu).abs() / scale

    def log_pdf_grad(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        return -torch.sign(x - mu) / scale

    def log_pdf_param_grads(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        diff = x - mu
        return {self.resolve_name("mean", self.mu): torch.sign(diff) / scale,
                self.resolve_name("scale", self.scale): -1 / scale + diff.abs() / scale ** 2}

    def cdf(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        t = (x - mu) / scale
        return 0.5 - 0.5 * torch.sign(t) * torch.expm1(-t.abs())

    def icdf(self, u):
        mu, scale = _as_float(self.mu, self.scale)
        v = u - 0.5
        return mu - scale * torch.sign(v) * torch.log1p(-2 * v.abs())

    def cdf_param_grads(self, x):
        mu, scale = _as_float(self.mu, self.scale)
        t = (x - mu) / scale
        dens = 0.5 * torch.exp(-t.abs())
        return {self.resolve_name("mean", self.mu): -dens / scale, self.resolve_name("scale", self.scale): -dens * t / scale}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        mu, scale = _as_float(self.mu, self.scale)
        u = _uniform(_sample_shape(n_samples, self.batch_shape), mu.dtype, mu.device)
        v = u - 0.5
        unit = -torch.sign(v) * torch.log1p(-2 * v.abs())
        samples = mu + scale * unit
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): torch.ones_like(unit), self.resolve_name("scale", self.scale): unit}
        return samples


class StudentT(Distribution):
    """Student's t with ``nu`` degrees of freedom, location ``mu`` and scale ``sigma``.

    Reparameterised sampling gives gradients w.r.t. ``mu`` and ``sigma`` only,
    so ``has_rsample`` is ``False`` when ``nu`` is a VariationalParameter.
    """

    def __init__(self, nu, mu=0.0, sigma=1.0):
        super().__init__()
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.add_dependency(nu)
        self.add_dependency(mu)
        self.add_dependency(sigma)

    @property
    def has_rsample(self):
        return not isinstance(self.nu, VariationalParameter)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.nu).shape, resolve(self.mu).shape, resolve(self.sigma).shape)

    def log_pdf(self, x):
        nu, mu, sigma = _as_float(self.nu, self.mu, self.sigma)
        t = (x - mu) / sigma
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                - torch.log(sigma) - (nu + 1) / 2 * torch.log1p(t ** 2 / nu))

    def log_pdf_grad(self, x):
        nu, mu, sigma = _as_float(self.nu, self.mu, self.sigma)
        diff = x - mu
        return -(nu + 1) * diff / (nu * sigma ** 2 + diff ** 2)

    def log_pdf_param_grads(self, x):
        nu, mu, sigma = _as_float(self.nu, self.mu, self.sigma)
        diff = x - mu
        t2 = (diff / sigma) ** 2
        grad_nu = (0.5 * torch.digamma((nu + 1) / 2) - 0.5 * torch.digamma(nu / 2) - 0.5 / nu
                   - 0.5 * torch.log1p(t2 / nu) + (nu + 1) * t2 / (2 * nu * (nu + t2)))
        grad_mu = (nu + 1) * diff / (nu * sigma ** 2 + diff ** 2)
        grad_sigma = -1 / sigma + (nu + 1) * t2 / (sigma * (nu + t2))
        return {self.resolve_name("nu", self.nu): grad_nu, self.resolve_name("mean", self.mu): grad_mu,
                self.resolve_name("sigma", self.sigma): grad_sigma}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad and not self.has_rsample:
            raise self._no_reparam()
        nu, mu, sigma = _as_float(self.nu, self.mu, self.sigma)
        shape = _sample_shape(n_samples, self.batch_shape)
        z = torch.randn(shape, dtype=mu.dtype, device=mu.device)
        chi2 = 2 * _standard_gamma((nu / 2).expand(shape))
        unit = z / torch.sqrt(chi2 / nu)
        samples = mu + sigma * unit
        if _reparametrization_trick_grad:
            return samples, {self.resolve_name("mean", self.mu): torch.ones_like(unit), self.resolve_name("sigma", self.sigma): unit}
        return samples


class Dirichlet(Distribution):
    """Dirichlet distribution on the simplex.

    Sampled in the ``K-1`` dimensional unconstrained space of a
    :class:`StickBreakingTransform` (the unconstrained value has shape
    ``(..., K-1)`` while the constrained value has shape ``(..., K)``).
    """

    _param_event_ndims = {"alpha": 1}

    def __init__(self, alpha):
        super().__init__(transform=StickBreakingTransform(), state_space=ContinuousSimplex(), transformed_state_space=ContinuousReal())
        self.alpha = alpha
        self.add_dependency(alpha)

    @property
    def event_shape(self):
        return resolve(self.alpha).shape[-1:]

    @property
    def batch_shape(self):
        return resolve(self.alpha).shape[:-1]

    def _log_norm(self, alpha):
        return torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))

    def log_pdf(self, x):
        alpha = _as_float(self.alpha)
        return torch.xlogy(alpha - 1, x).sum(-1) - self._log_norm(alpha)

    def log_pdf_grad(self, x):
        alpha = _as_float(self.alpha)
        return (alpha - 1) / x

    def log_pdf_param_grads(self, x):
        alpha = _as_float(self.alpha)
        digamma_alpha = torch.digamma(alpha)
        digamma_alpha_sum = torch.digamma(alpha.sum(-1, keepdim=True))
        grad_alpha = x.log() - digamma_alpha + digamma_alpha_sum
        return {self.resolve_name("alpha", self.alpha): grad_alpha}

    def _log_prob_unconstrained(self, x_unconstrained):
        alpha = _as_float(self.alpha)
        log_x = self.transform.log_simplex(x_unconstrained)
        log_prob = ((alpha - 1) * log_x).sum(-1) - self._log_norm(alpha)
        return _bsum(log_prob) + self.transform.log_abs_det_jacobian(x_unconstrained)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        alpha = _as_float(self.alpha)
        z = x_unconstrained
        a1 = alpha - 1
        batch = torch.broadcast_shapes(a1.shape[:-1], z.shape[:-1])
        a1 = a1.expand(*batch, a1.shape[-1])
        y = torch.sigmoid(z - StickBreakingTransform._offset(z))
        # sum_{i>k} (alpha_i - 1)
        tail = torch.flip(torch.cumsum(torch.flip(a1, dims=[-1]), dim=-1), dims=[-1])[..., 1:]
        grad = a1[..., :-1] * (1 - y) - y * tail
        return grad + self.transform.grad_log_abs_det_jacobian(z)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        alpha = _as_float(self.alpha)
        g = _standard_gamma(alpha.expand(_sample_shape(n_samples, self.batch_shape, self.event_shape)))
        return g / g.sum(-1, keepdim=True)


class LKJCholesky(Distribution):
    """LKJ distribution over Cholesky factors ``L`` of ``dim x dim`` correlation matrices.

    ``p(L) ∝ prod_{i>=2} L_ii^{dim - i + 2 eta - 2}`` (1-indexed), i.e. the
    correlation matrix ``L L^T`` has density ``∝ det(L L^T)^(eta - 1)``.
    The unconstrained space is ``R^{dim(dim-1)/2}`` via
    :class:`CorrCholeskyTransform` (tanh of canonical partial correlations).
    """

    def __init__(self, dim, eta=1.0):
        dim = int(dim)
        if dim < 2:
            raise ValueError("LKJCholesky needs dim >= 2.")
        super().__init__(CorrCholeskyTransform(dim), ContinuousCorrCholesky(), ContinuousReal())
        self.dim = dim
        self.eta = eta
        self.add_dependency(eta)

    @property
    def event_shape(self):
        return (self.dim, self.dim)

    @property
    def batch_shape(self):
        return resolve(self.eta).shape

    def _coefficients(self, eta):
        i = torch.arange(1, self.dim, dtype=eta.dtype, device=eta.device)
        return 2 * (eta.unsqueeze(-1) - 1) + (self.dim - 1 - i)

    def _log_norm(self, eta):
        dm1 = self.dim - 1
        alpha = eta + 0.5 * dm1
        return 0.5 * dm1 * _LOG_PI + _mvlgamma(alpha - 0.5, dm1) - dm1 * torch.lgamma(alpha)

    def log_pdf(self, x):
        eta = _as_float(self.eta)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)[..., 1:]
        return (self._coefficients(eta) * torch.log(diag)).sum(-1) - self._log_norm(eta)

    def log_pdf_grad(self, x):
        eta = _as_float(self.eta)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        coef = torch.cat([torch.zeros_like(eta).unsqueeze(-1), self._coefficients(eta)], dim=-1)
        return torch.diag_embed(coef / diag)

    def log_pdf_param_grads(self, x):
        eta = _as_float(self.eta)
        dm1 = self.dim - 1
        alpha = eta + 0.5 * dm1
        diag = torch.diagonal(x, dim1=-2, dim2=-1)[..., 1:]
        d_norm = _mvdigamma(alpha - 0.5, dm1) - dm1 * torch.digamma(alpha)
        return {self.resolve_name("eta", self.eta): 2 * torch.log(diag).sum(-1) - d_norm}

    def _log_prob_unconstrained(self, x_unconstrained):
        eta = _as_float(self.eta)
        z = x_unconstrained
        T = self.transform
        ell = T._to_matrix(_log1m_tanh_sq(z))
        log_diag = 0.5 * ell.sum(-1)[..., 1:]  # log L_ii, i >= 1
        log_prob = (self._coefficients(eta) * log_diag).sum(-1) - self._log_norm(eta)
        return _bsum(log_prob) + T.log_abs_det_jacobian(z)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        eta = _as_float(self.eta)
        z = x_unconstrained
        rows, cols = self.transform._indices(z.device)
        coef = self._coefficients(eta)[..., rows - 1]
        return -torch.tanh(z) * (coef + (rows - cols + 1).to(z.dtype))

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Onion method (Lewandowski, Kurowicka & Joe, 2009)."""
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        eta = _as_float(self.eta)
        D = self.dim
        batch = (n_samples, *self.batch_shape)
        # Row k >= 1: squared norm of its off-diagonal part ~ Beta(k / 2, eta + (D - 2) / 2 - (k - 1) / 2).
        # (Row 0 has no off-diagonal part; its draw is discarded.)
        marginal = eta + 0.5 * (D - 2)
        k = torch.arange(D, dtype=eta.dtype, device=eta.device).clamp(min=1)
        conc1 = (0.5 * k).expand(*batch, D)
        conc0 = (marginal.unsqueeze(-1) - 0.5 * (k - 1)).expand(*batch, D)
        y = _sample_beta(conc1, conc0, (*batch, D)).unsqueeze(-1)
        u = torch.tril(torch.randn(*batch, D, D, dtype=eta.dtype, device=eta.device), diagonal=-1)
        norm = u.norm(dim=-1, keepdim=True)
        u = u / torch.where(norm > 0, norm, torch.ones_like(norm))
        w = torch.sqrt(y) * u
        diag = (1 - (w ** 2).sum(-1)).clamp(min=torch.finfo(eta.dtype).tiny).sqrt()
        return w + torch.diag_embed(diag)


def _spd_parts(X):
    """Cholesky factor, inverse and log-determinant of SPD matrices."""
    L = torch.linalg.cholesky(X)
    inv = torch.cholesky_inverse(L)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return L, inv, logdet


def _bartlett(df, shape, D, dtype, device):
    """Lower-triangular Bartlett factor ``A`` with ``A A^T ~ Wishart(df, I)``."""
    i = torch.arange(D, dtype=dtype, device=device)
    chi2 = 2 * _standard_gamma((0.5 * (df.unsqueeze(-1) - i)).expand(*shape, D))
    A = torch.tril(torch.randn(*shape, D, D, dtype=dtype, device=device), diagonal=-1)
    return A + torch.diag_embed(torch.sqrt(chi2))


class _WishartFamily(Distribution):
    """Shared parameter handling of :class:`Wishart` and :class:`InverseWishart`."""

    _param_event_ndims = {"scale": 2, "scale_tril": 2}

    def __init__(self, df, scale=None, scale_tril=None):
        _check_exactly_one(["scale", "scale_tril"], [scale, scale_tril])
        D = resolve(scale if scale is not None else scale_tril).shape[-1]
        super().__init__(PositiveDefiniteTransform(D), ContinuousPositiveDefinite(), ContinuousReal())
        self.dim = D
        self.df = df
        self.scale = scale
        self.scale_tril = scale_tril
        self.add_dependency(df)
        self.add_dependency(scale if scale is not None else scale_tril)

    @property
    def event_shape(self):
        return (self.dim, self.dim)

    @property
    def batch_shape(self):
        matrix = resolve(self.scale if self.scale is not None else self.scale_tril)
        return torch.broadcast_shapes(resolve(self.df).shape, matrix.shape[:-2])

    def _params(self):
        """(df, L_S, S^-1, log det S)."""
        if self.scale is not None:
            df, S = _as_float(self.df, self.scale)
            L, S_inv, logdet = _spd_parts(S)
        else:
            df, L = _as_float(self.df, self.scale_tril)
            S_inv = torch.cholesky_inverse(L)
            logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return df, L, S_inv, logdet

    def _log_mvgamma(self, df):
        return _mvlgamma(0.5 * df, self.dim)

    def _matrix_grads(self, grad_scale, L):
        """Return the param-grad entry for ``scale`` (full symmetric) or ``scale_tril``."""
        if self.scale is not None:
            return {self.resolve_name("scale", self.scale): grad_scale}
        return {self.resolve_name("scale_tril", self.scale_tril): torch.tril(2 * grad_scale @ L)}

    def _no_rsample(self, flag):
        if flag:
            raise self._no_reparam()


class Wishart(_WishartFamily):
    """Wishart distribution over ``D x D`` SPD matrices with ``df`` degrees of freedom.

    ``Wishart(df, scale)`` or ``Wishart(df, scale_tril=L)`` with
    ``scale = L L^T``; ``E[X] = df * scale``. Sampled in the
    ``D(D+1)/2``-dimensional unconstrained space of a
    :class:`PositiveDefiniteTransform`. Matrix gradients are full symmetric
    matrices (entries treated as independent), the ``scale_tril`` gradient is
    lower triangular.
    """

    def log_pdf(self, x):
        df, _, S_inv, logdet_S = self._params()
        _, _, logdet_X = _spd_parts(x)
        D = self.dim
        trace = (S_inv * x).sum((-2, -1))
        return (0.5 * (df - D - 1) * logdet_X - 0.5 * trace - 0.5 * df * D * math.log(2.0)
                - 0.5 * df * logdet_S - self._log_mvgamma(df))

    def log_pdf_grad(self, x):
        df, _, S_inv, _ = self._params()
        _, X_inv, _ = _spd_parts(x)
        return 0.5 * (df - self.dim - 1)[..., None, None] * X_inv - 0.5 * S_inv

    def log_pdf_param_grads(self, x):
        df, L, S_inv, logdet_S = self._params()
        _, _, logdet_X = _spd_parts(x)
        D = self.dim
        grad_df = 0.5 * logdet_X - 0.5 * D * math.log(2.0) - 0.5 * logdet_S - 0.5 * _mvdigamma(0.5 * df, D)
        grad_scale = 0.5 * S_inv @ x @ S_inv - 0.5 * df[..., None, None] * S_inv
        grads = {self.resolve_name("df", self.df): grad_df}
        grads.update(self._matrix_grads(grad_scale, L))
        return grads

    def _log_prob_unconstrained(self, x_unconstrained):
        df, L_S, _, logdet_S = self._params()
        T = self.transform
        L = T.cholesky(x_unconstrained)
        D = self.dim
        diag_z = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
        batch = torch.broadcast_shapes(L.shape[:-2], L_S.shape[:-2])
        A = torch.linalg.solve_triangular(L_S.expand(*batch, D, D), L.expand(*batch, D, D), upper=False)
        log_prob = ((df - D - 1)[..., None] * diag_z).sum(-1) - 0.5 * (A ** 2).sum((-2, -1)) \
            - 0.5 * df * D * math.log(2.0) - 0.5 * df * logdet_S - self._log_mvgamma(df)
        return _bsum(log_prob) + T.log_abs_det_jacobian(x_unconstrained)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        df, _, S_inv, _ = self._params()
        T = self.transform
        z = x_unconstrained
        L = T.cholesky(z)
        rows, cols = T._indices(z.device)
        bar_L = -(S_inv @ L)[..., rows, cols]
        grad = torch.where(rows == cols, bar_L * L[..., rows, cols], bar_L)
        diag = torch.where(rows == cols, torch.ones((), dtype=z.dtype, device=z.device), torch.zeros((), dtype=z.dtype, device=z.device))
        return grad + (df - self.dim - 1)[..., None] * diag + T.grad_log_abs_det_jacobian(z)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Bartlett decomposition: ``X = L_S A A^T L_S^T``."""
        self._no_rsample(_reparametrization_trick_grad)
        df, L, _, _ = self._params()
        shape = (n_samples, *self.batch_shape)
        A = _bartlett(df.expand(shape), shape, self.dim, df.dtype, df.device)
        B = L @ A
        return B @ B.transpose(-2, -1)


class InverseWishart(_WishartFamily):
    """Inverse-Wishart distribution: ``X^-1 ~ Wishart(df, scale^-1)``.

    ``InverseWishart(df, scale)`` or ``InverseWishart(df, scale_tril=L)``;
    ``E[X] = scale / (df - D - 1)``. Same transform and gradient conventions
    as :class:`Wishart`.
    """

    def log_pdf(self, x):
        df, _, S, logdet_S = self._scale_params()
        _, X_inv, logdet_X = _spd_parts(x)
        D = self.dim
        trace = (S * X_inv).sum((-2, -1))
        return (0.5 * df * logdet_S - 0.5 * df * D * math.log(2.0) - self._log_mvgamma(df)
                - 0.5 * (df + D + 1) * logdet_X - 0.5 * trace)

    def _scale_params(self):
        """(df, L_S, S, log det S)."""
        df, L, _, logdet = self._params()
        return df, L, L @ L.transpose(-2, -1), logdet

    def log_pdf_grad(self, x):
        df, _, S, _ = self._scale_params()
        _, X_inv, _ = _spd_parts(x)
        return -0.5 * (df + self.dim + 1)[..., None, None] * X_inv + 0.5 * X_inv @ S @ X_inv

    def log_pdf_param_grads(self, x):
        df, L, S_inv, logdet_S = self._params()
        _, X_inv, logdet_X = _spd_parts(x)
        D = self.dim
        grad_df = 0.5 * logdet_S - 0.5 * D * math.log(2.0) - 0.5 * _mvdigamma(0.5 * df, D) - 0.5 * logdet_X
        grad_scale = 0.5 * df[..., None, None] * S_inv - 0.5 * X_inv
        grads = {self.resolve_name("df", self.df): grad_df}
        grads.update(self._matrix_grads(grad_scale, L))
        return grads

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """``X = L_S A^-T A^-1 L_S^T`` with a Bartlett factor ``A``."""
        self._no_rsample(_reparametrization_trick_grad)
        df, L, _, _ = self._params()
        shape = (n_samples, *self.batch_shape)
        A = _bartlett(df.expand(shape), shape, self.dim, df.dtype, df.device)
        # B = L A^-T  <=>  B^T = A^-1 L^T
        Bt = torch.linalg.solve_triangular(A, L.transpose(-2, -1).expand(*shape, self.dim, self.dim), upper=False)
        return Bt.transpose(-2, -1) @ Bt


# ========================= DISCRETE =========================
class _ProbsOrLogits:
    """Mixin for distributions parameterised by a probability ``probs`` or by ``logits``."""

    def _init_probs(self, probs, logits):
        _check_exactly_one(["probs", "logits"], [probs, logits])
        self.probs = probs
        self.logits = logits
        self.add_dependency(probs if probs is not None else logits)

    @property
    def _param(self):
        return self.probs if self.probs is not None else self.logits

    def _log_p_log_q(self):
        """(log p, log(1-p)) computed stably."""
        if self.logits is not None:
            logits = _as_float(self.logits)
            return log_sigmoid(logits), log_sigmoid(-logits)
        p = _as_float(self.probs)
        return torch.log(p), torch.log1p(-p)

    def _probs_float(self):
        if self.probs is not None:
            return _as_float(self.probs)
        return torch.sigmoid(_as_float(self.logits))


class Geometric(_ProbsOrLogits, Distribution):
    """Number of trials up to and including the first success (support 1, 2, ...)."""

    def __init__(self, probs=None, logits=None):
        Distribution.__init__(self, IdentityTransform(), DiscretePositive(), DiscretePositive())
        self._init_probs(probs, logits)

    @property
    def batch_shape(self):
        return resolve(self._param).shape

    @staticmethod
    def _support(x):
        x = torch.as_tensor(x)
        return _in_support(x, (x >= 1) & _is_integer(x), fill=1)

    def log_pdf(self, x):
        inside, x = self._support(x)
        if self.logits is not None:
            logits = _as_float(self.logits)
            return _restrict(log_sigmoid(logits) + (x - 1) * log_sigmoid(-logits), inside)
        p = _as_float(self.probs)
        return _restrict(torch.log(p) + torch.special.xlog1py(x - 1, -p), inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        if self.logits is not None:
            s = torch.sigmoid(_as_float(self.logits))
            return {self.resolve_name("logits", self.logits): _zero_outside(1 - x * s, inside)}
        p = _as_float(self.probs)
        return {self.resolve_name("probs", self.probs): _zero_outside(1 / p - (x - 1) / (1 - p), inside)}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        _, log_q = self._log_p_log_q()
        u = _uniform(_sample_shape(n_samples, self.batch_shape), log_q.dtype, log_q.device)
        return torch.ceil(torch.log(u) / log_q).clamp(min=1)


class Bernoulli(_ProbsOrLogits, Distribution):
    def __init__(self, probs=None, logits=None):
        Distribution.__init__(self, IdentityTransform(), DiscreteRange(0, 1), DiscreteRange(0, 1))
        self._init_probs(probs, logits)

    @property
    def batch_shape(self):
        return resolve(self._param).shape

    @staticmethod
    def _support(x):
        x = torch.as_tensor(x)
        return _in_support(x, (x == 0) | (x == 1))

    def log_pdf(self, x):
        inside, x = self._support(x)
        if self.logits is not None:
            logits = _as_float(self.logits)
            return _restrict(x * logits - F.softplus(logits), inside)
        p = _as_float(self.probs)
        return _restrict(torch.xlogy(x, p) + torch.special.xlog1py(1 - x, -p), inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        if self.logits is not None:
            grad = x - torch.sigmoid(_as_float(self.logits))
            return {self.resolve_name("logits", self.logits): _zero_outside(grad, inside)}
        p = _as_float(self.probs)
        return {self.resolve_name("probs", self.probs): _zero_outside(x / p - (1 - x) / (1 - p), inside)}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        p = self._probs_float()
        u = torch.rand(_sample_shape(n_samples, self.batch_shape), dtype=p.dtype, device=p.device)
        return (u < p).to(p.dtype)


class Binomial(_ProbsOrLogits, Distribution):
    def __init__(self, n, probs=None, logits=None):
        Distribution.__init__(self, IdentityTransform(), DiscreteRange(0, n), DiscreteRange(0, n))
        self.n = n
        self.add_dependency(n)
        self._init_probs(probs, logits)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.n).shape, resolve(self._param).shape)

    def _count(self, *like):
        n = resolve(self.n)
        dtype = _float_dtype(*like, resolve(self._param))
        return n.to(dtype=dtype, device=_device(*like, n))

    def _log_binom_coeff(self, x):
        n = self._count(x)
        return torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1)

    def _support(self, x):
        x = torch.as_tensor(x)
        return _in_support(x, (x >= 0) & (x <= self._count(x)) & _is_integer(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        n = self._count(x)
        if self.logits is not None:
            logits = _as_float(self.logits)
            log_p = self._log_binom_coeff(x) + x * logits - n * F.softplus(logits)
        else:
            p = _as_float(self.probs)
            log_p = self._log_binom_coeff(x) + torch.xlogy(x, p) + torch.special.xlog1py(n - x, -p)
        return _restrict(log_p, inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        n = self._count(x)
        if self.logits is not None:
            grad = x - n * torch.sigmoid(_as_float(self.logits))
            return {self.resolve_name("logits", self.logits): _zero_outside(grad, inside)}
        p = _as_float(self.probs)
        return {self.resolve_name("probs", self.probs): _zero_outside(x / p - (n - x) / (1 - p), inside)}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        p = self._probs_float()
        shape = _sample_shape(n_samples, self.batch_shape)
        n = self._count(p).expand(shape).contiguous()
        return torch.binomial(n, p.expand(shape).contiguous())


class DiscreteUniform(Distribution):
    """Uniform distribution on the integers ``low, ..., high``.

    ``log_pdf_param_grads`` treats the bounds as continuous-valued: inside the
    support ``d/dlow log p = 1/(high-low+1)`` and ``d/dhigh = -1/(high-low+1)``
    (zero outside the support).
    """

    def __init__(self, low, high):
        super().__init__(IdentityTransform(), DiscreteRange(low=low, high=high), DiscreteRange(low=low, high=high))
        self.low = low
        self.high = high
        self.add_dependency(low)
        self.add_dependency(high)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.low).shape, resolve(self.high).shape)

    def _bounds(self, x):
        low, high = resolve(self.low), resolve(self.high)
        dtype = _float_dtype(x, low, high)
        return low.to(dtype), high.to(dtype)

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        low, high = self._bounds(x)
        inside = (x >= low) & (x <= high) & _is_integer(x)
        return torch.where(inside, -torch.log(high - low + 1), torch.full_like(x, -torch.inf, dtype=low.dtype))

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        low, high = self._bounds(x)
        inside = (x >= low) & (x <= high) & _is_integer(x)
        inv = torch.where(inside, 1 / (high - low + 1), torch.zeros_like(x))
        return {self.resolve_name("low", self.low): inv, self.resolve_name("high", self.high): -inv}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        low, high = self._bounds(torch.empty(0))
        u = torch.rand(_sample_shape(n_samples, self.batch_shape), dtype=low.dtype, device=low.device)
        return low + torch.floor(u * (high - low + 1))


class Poisson(Distribution):
    def __init__(self, rate):
        super().__init__(IdentityTransform(), DiscreteNonNegative(), DiscreteNonNegative())
        self.rate = rate
        self.add_dependency(rate)

    @property
    def batch_shape(self):
        return resolve(self.rate).shape

    @staticmethod
    def _support(x):
        x = torch.as_tensor(x)
        return _in_support(x, (x >= 0) & _is_integer(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        rate = _as_float(self.rate)
        return _restrict(torch.xlogy(x, rate) - rate - torch.lgamma(x + 1), inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        rate = _as_float(self.rate)
        return {self.resolve_name("rate", self.rate): _zero_outside(x / rate - 1, inside)}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        rate = _as_float(self.rate)
        return torch.poisson(rate.expand(_sample_shape(n_samples, self.batch_shape)).contiguous())


class NegativeBinomial(_ProbsOrLogits, Distribution):
    """Number of successes before ``total_count`` failures, success probability ``probs``.

    ``log p(k) = log C(k + r - 1, k) + r log(1 - p) + k log p`` (the PyTorch /
    Pyro convention), mean ``r p / (1 - p)``.
    """

    def __init__(self, total_count, probs=None, logits=None):
        Distribution.__init__(self, IdentityTransform(), DiscreteNonNegative(), DiscreteNonNegative())
        self.total_count = total_count
        self.add_dependency(total_count)
        self._init_probs(probs, logits)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.total_count).shape, resolve(self._param).shape)

    @staticmethod
    def _support(x):
        x = torch.as_tensor(x)
        return _in_support(x, (x >= 0) & _is_integer(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        r = _as_float(self.total_count).to(_float_dtype(x, resolve(self._param)))
        log_norm = torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
        if self.logits is not None:
            logits = _as_float(self.logits)
            return _restrict(log_norm + x * log_sigmoid(logits) + r * log_sigmoid(-logits), inside)
        p = _as_float(self.probs)
        return _restrict(log_norm + torch.xlogy(x, p) + r * torch.log1p(-p), inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        r = _as_float(self.total_count).to(_float_dtype(x, resolve(self._param)))
        _, log_q = self._log_p_log_q()
        grads = {self.resolve_name("total_count", self.total_count): torch.digamma(x + r) - torch.digamma(r) + log_q}
        if self.logits is not None:
            grads[self.resolve_name("logits", self.logits)] = x - (x + r) * torch.sigmoid(_as_float(self.logits))
        else:
            p = _as_float(self.probs)
            grads[self.resolve_name("probs", self.probs)] = x / p - r / (1 - p)
        return {name: _zero_outside(grad, inside) for name, grad in grads.items()}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        if self.logits is not None:
            odds = torch.exp(_as_float(self.logits))
        else:
            p = _as_float(self.probs)
            odds = p / (1 - p)
        r = _as_float(self.total_count).to(odds.dtype)
        shape = _sample_shape(n_samples, self.batch_shape)
        rate = _standard_gamma(r.expand(shape)) * odds
        return torch.poisson(rate.contiguous())


class Categorical(Distribution):
    """Categorical distribution over ``{0, ..., K-1}`` given ``probs`` or ``logits`` (last dim = K).

    ``probs`` need not be normalised; the gradients are those of
    ``log(probs_k / sum(probs))``.
    """

    _param_event_ndims = {"probs": 1, "logits": 1}

    def __init__(self, probs=None, logits=None):
        _check_exactly_one(["probs", "logits"], [probs, logits])
        self.probs = probs
        self.logits = logits
        K = resolve(probs if probs is not None else logits).shape[-1]
        self.n_categories = K
        super().__init__(IdentityTransform(), DiscreteRange(0, K - 1), DiscreteRange(0, K - 1))
        self.add_dependency(probs if probs is not None else logits)

    @property
    def _param(self):
        return self.probs if self.probs is not None else self.logits

    @property
    def batch_shape(self):
        return resolve(self._param).shape[:-1]

    def _log_probs(self):
        if self.logits is not None:
            return torch.log_softmax(_as_float(self.logits), dim=-1)
        p = _as_float(self.probs)
        return torch.log(p) - torch.log(p.sum(-1, keepdim=True))

    def _support(self, x):
        """The support mask and the category indices (0 outside the support)."""
        x = torch.as_tensor(x)
        inside = (x >= 0) & (x <= self.n_categories - 1) & _is_integer(x)
        return inside, torch.where(inside, x, torch.zeros_like(x)).long()

    def _one_hot(self, idx, like):
        shape = torch.broadcast_shapes(idx.shape, like.shape[:-1])
        return F.one_hot(idx.expand(shape), self.n_categories).to(like.dtype)

    def log_pdf(self, x):
        inside, idx = self._support(x)
        logp = self._log_probs()
        shape = torch.broadcast_shapes(idx.shape, logp.shape[:-1])
        log_p = logp.expand(*shape, self.n_categories).gather(-1, idx.expand(shape).unsqueeze(-1)).squeeze(-1)
        return _restrict(log_p, inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, idx = self._support(x)
        if self.logits is not None:
            logits = _as_float(self.logits)
            grad = self._one_hot(idx, logits) - torch.softmax(logits, dim=-1)
            return {self.resolve_name("logits", self.logits): _zero_outside(grad, inside.unsqueeze(-1))}
        p = _as_float(self.probs)
        onehot = self._one_hot(idx, p)
        grad = torch.where(onehot > 0, 1 / p, torch.zeros_like(p)) - 1 / p.sum(-1, keepdim=True)
        return {self.resolve_name("probs", self.probs): _zero_outside(grad, inside.unsqueeze(-1))}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        p = torch.exp(self._log_probs())
        cdf = torch.cumsum(p, dim=-1)
        u = torch.rand((n_samples, *self.batch_shape, 1), dtype=p.dtype, device=p.device)
        idx = (cdf < u).sum(-1).clamp(max=self.n_categories - 1)
        return idx.to(p.dtype)


class Multinomial(Distribution):
    """Multinomial counts over K categories with ``total_count`` trials.

    Supports ``log_pdf``, parameter gradients and ``sample``; it cannot be
    used as a latent variable for MCMC (its support, vectors summing to
    ``total_count``, is not represented by the state space).
    """

    _param_event_ndims = {"probs": 1, "logits": 1}

    def __init__(self, total_count, probs=None, logits=None):
        _check_exactly_one(["probs", "logits"], [probs, logits])
        super().__init__(IdentityTransform(), DiscreteRange(0, total_count), DiscreteRange(0, total_count))
        self.total_count = total_count
        self.probs = probs
        self.logits = logits
        self.add_dependency(probs if probs is not None else logits)

    @property
    def _param(self):
        return self.probs if self.probs is not None else self.logits

    @property
    def event_shape(self):
        return resolve(self._param).shape[-1:]

    @property
    def batch_shape(self):
        return resolve(self._param).shape[:-1]

    def _log_probs(self):
        if self.logits is not None:
            return torch.log_softmax(_as_float(self.logits), dim=-1)
        p = _as_float(self.probs)
        return torch.log(p) - torch.log(p.sum(-1, keepdim=True))

    def _support(self, x):
        """Counts are non-negative integers that sum to ``total_count`` (zeros outside the support)."""
        x = torch.as_tensor(x)
        n = torch.as_tensor(resolve(self.total_count), device=x.device)
        inside = ((x >= 0) & _is_integer(x)).all(-1) & (x.sum(-1) == n)
        return inside, torch.where(inside.unsqueeze(-1), x, torch.zeros_like(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        logp = self._log_probs()
        n = resolve(self.total_count).to(logp.dtype)
        log_terms = torch.where(x > 0, x * logp, torch.zeros_like(x * logp))
        return _restrict(torch.lgamma(n + 1) - torch.lgamma(x + 1).sum(-1) + log_terms.sum(-1), inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        n = x.sum(-1, keepdim=True)
        if self.logits is not None:
            logits = _as_float(self.logits)
            grad = x - n * torch.softmax(logits, dim=-1)
            return {self.resolve_name("logits", self.logits): _zero_outside(grad, inside.unsqueeze(-1))}
        p = _as_float(self.probs)
        grad = torch.where(x > 0, x / p, torch.zeros_like(x * p)) - n / p.sum(-1, keepdim=True)
        return {self.resolve_name("probs", self.probs): _zero_outside(grad, inside.unsqueeze(-1))}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Sequential conditional binomial draws."""
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        p = torch.exp(self._log_probs())
        K = p.shape[-1]
        shape = (n_samples, *self.batch_shape)
        p = p.expand(*shape, K)
        tail = torch.flip(torch.cumsum(torch.flip(p, dims=[-1]), dim=-1), dims=[-1])
        remaining = resolve(self.total_count).to(p.dtype).expand(shape).clone()
        counts = []
        for k in range(K - 1):
            q = (p[..., k] / tail[..., k]).clamp(0, 1).nan_to_num(0.0)
            c = torch.binomial(remaining.contiguous(), q.contiguous())
            counts.append(c)
            remaining = remaining - c
        counts.append(remaining)
        return torch.stack(counts, dim=-1)


class ZeroInflated(Distribution):
    """Zero-inflated discrete distribution: ``0`` with probability ``gate``, else ``base``.

    E.g. ``ZeroInflated(Poisson(rate), gate)`` is the zero-inflated Poisson.
    """

    def __init__(self, base, gate):
        super().__init__(base.transform, base.state_space, base.transformed_state_space)
        self.base = base
        self.gate = gate
        self._add_dependencies_of(base)
        self.add_dependency(gate)

    @property
    def event_shape(self):
        return self.base.event_shape

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(self.base.batch_shape, resolve(self.gate).shape)

    def _parts(self, x):
        gate = _as_float(self.gate)
        base_lp = self.base.log_pdf(x)
        log_keep = torch.log1p(-gate) + base_lp
        log_zero = torch.logaddexp(torch.log(gate) + torch.zeros_like(log_keep), log_keep)
        is_zero = x == 0
        return gate, base_lp, log_keep, log_zero, is_zero

    def log_pdf(self, x):
        _, _, log_keep, log_zero, is_zero = self._parts(x)
        return torch.where(is_zero, log_zero, log_keep)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        gate, base_lp, log_keep, log_zero, is_zero = self._parts(x)
        b0 = torch.exp(base_lp)
        grad_gate = torch.where(is_zero, (1 - b0) * torch.exp(-log_zero), -1 / (1 - gate))
        grad_gate = _zero_outside(grad_gate, is_zero | torch.isfinite(base_lp))
        weight = torch.where(is_zero, torch.exp(log_keep - log_zero), torch.ones_like(log_keep))
        grads = {}
        for name, g in self.base.log_pdf_param_grads(x).items():
            w = weight
            if isinstance(g, torch.Tensor) and g.ndim > w.ndim:
                w = w.reshape(*w.shape, *([1] * (g.ndim - w.ndim)))
            grads[name] = w * g
        gate_name = self.resolve_name("gate", self.gate)
        grads[gate_name] = grads.get(gate_name, 0) + grad_gate
        return grads

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        gate = _as_float(self.gate)
        base = self.base.sample(n_samples)
        shape = _sample_shape(n_samples, self.batch_shape)
        base = base.expand(shape)
        u = torch.rand(shape, dtype=base.dtype, device=base.device)
        return torch.where(u < gate, torch.zeros_like(base), base)


class Truncated(Distribution):
    """A continuous ``base`` distribution truncated to ``[low, high]``.

    Either bound may be ``None`` (no truncation on that side). The base must
    provide ``cdf`` (and ``icdf`` for sampling). Gradients w.r.t. the base
    parameters additionally need ``base.cdf_param_grads`` (Normal,
    Exponential, Cauchy and Laplace provide it).
    """

    def __init__(self, base, low=None, high=None):
        if low is None and high is None:
            transform = base.transform
        elif low is None:
            transform = LogTransform(border=high, side="smaller")
        elif high is None:
            transform = LogTransform(border=low, side="larger")
        else:
            transform = LogitTransform(low=low, high=high)
        state_space = ContinuousRange(low=-math.inf if low is None else low, high=math.inf if high is None else high)
        super().__init__(transform, state_space, ContinuousReal())
        self.base = base
        self.low = low
        self.high = high
        self._add_dependencies_of(base)
        if low is not None:
            self.add_dependency(low)
        if high is not None:
            self.add_dependency(high)

    @property
    def has_rsample(self):
        return False

    @property
    def event_shape(self):
        return self.base.event_shape

    @property
    def batch_shape(self):
        shapes = [self.base.batch_shape]
        shapes += [resolve(b).shape for b in (self.low, self.high) if b is not None]
        return torch.broadcast_shapes(*shapes)

    def _cdf_bounds(self, like):
        F_low = self.base.cdf(_as_float(self.low).to(like.dtype)) if self.low is not None else torch.zeros((), dtype=like.dtype, device=like.device)
        F_high = self.base.cdf(_as_float(self.high).to(like.dtype)) if self.high is not None else torch.ones((), dtype=like.dtype, device=like.device)
        return F_low, F_high

    def _inside(self, x):
        inside = torch.ones_like(x, dtype=torch.bool)
        if self.low is not None:
            inside = inside & (x >= _as_float(self.low))
        if self.high is not None:
            inside = inside & (x <= _as_float(self.high))
        return inside

    def log_pdf(self, x):
        F_low, F_high = self._cdf_bounds(x)
        log_z = torch.log(F_high - F_low)
        return torch.where(self._inside(x), self.base.log_pdf(x) - log_z, torch.full_like(x, -torch.inf))

    def log_pdf_grad(self, x):
        return torch.where(self._inside(x), self.base.log_pdf_grad(x), torch.zeros_like(x))

    def log_pdf_param_grads(self, x):
        F_low, F_high = self._cdf_bounds(x)
        Z = F_high - F_low
        grads = dict(self.base.log_pdf_param_grads(x))
        if hasattr(self.base, "cdf_param_grads"):
            dF_high = self.base.cdf_param_grads(_as_float(self.high).to(x.dtype)) if self.high is not None else {}
            dF_low = self.base.cdf_param_grads(_as_float(self.low).to(x.dtype)) if self.low is not None else {}
            for name in set(dF_high) | set(dF_low):
                dZ = dF_high.get(name, 0) - dF_low.get(name, 0)
                grads[name] = grads.get(name, 0) - dZ / Z
        elif self.base.parameters or self.base.variational_parameters:
            raise NotImplementedError(f"Truncated({type(self.base).__name__}) parameter gradients need {type(self.base).__name__}.cdf_param_grads.")
        else:
            grads = {}  # constant base parameters: only the bound gradients are meaningful
        if self.low is not None:
            name = self.resolve_name("low", self.low)
            grads[name] = grads.get(name, 0) + self.base.pdf(_as_float(self.low).to(x.dtype)) / Z * torch.ones_like(x)
        if self.high is not None:
            name = self.resolve_name("high", self.high)
            grads[name] = grads.get(name, 0) - self.base.pdf(_as_float(self.high).to(x.dtype)) / Z * torch.ones_like(x)
        inside = self._inside(x)
        return {k: torch.where(inside, v, torch.zeros_like(v)) if isinstance(v, torch.Tensor) and v.shape == inside.shape else v
                for k, v in grads.items()}

    def cdf(self, x):
        F_low, F_high = self._cdf_bounds(x)
        return ((self.base.cdf(x) - F_low) / (F_high - F_low)).clamp(0, 1)

    def icdf(self, u):
        F_low, F_high = self._cdf_bounds(u)
        return self._clamp(self.base.icdf(F_low + u * (F_high - F_low)))

    def _clamp(self, x):
        if self.low is not None:
            x = torch.maximum(x, _as_float(self.low).to(x.dtype))
        if self.high is not None:
            x = torch.minimum(x, _as_float(self.high).to(x.dtype))
        return x

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        ref = self.base.sample(1)
        u = _uniform(_sample_shape(n_samples, self.batch_shape, self.event_shape), ref.dtype, ref.device)
        return self.icdf(u)


class TransformedDistribution(Distribution):
    """Distribution of ``y = transform.inverse(x)`` with ``x ~ base``.

    The library's transforms map unconstrained -> constrained with
    ``inverse``, so e.g. ``TransformedDistribution(Normal(mu, sigma),
    LogTransform())`` is a log-normal. Densities use the change of variables
    ``log p(y) = log p_base(x) - log|det d inverse/dx|`` with
    ``x = transform.forward(y)``. MCMC runs in the base distribution's
    unconstrained space. ``state_space`` describes the support of ``y``
    (default: real line).
    """

    def __init__(self, base, transform, state_space=None):
        super().__init__(ChainTransform(base.transform, transform), state_space or ContinuousReal(), base.transformed_state_space)
        self.base = base
        self.outer = transform
        self._add_dependencies_of(base)

    @property
    def event_shape(self):
        return self.outer.constrained_shape(self.base.event_shape) if self.base.event_shape else ()

    @property
    def batch_shape(self):
        return self.base.batch_shape

    def _log_det(self, x):
        d = self.outer.derivative(x)
        n_event = len(self.base.event_shape)
        if d.shape == x.shape:
            log_d = torch.log(d.abs())
            return log_d.sum(tuple(range(-n_event, 0))) if n_event else log_d
        return torch.linalg.slogdet(d)[1]

    def log_pdf(self, x_y):
        x = self.outer.forward(x_y)
        return self.base.log_pdf(x) - self._log_det(x)

    def log_pdf_grad(self, x_y):
        x = self.outer.forward(x_y)
        g = self.base.log_pdf_grad(x) - self.outer.grad_log_abs_det_jacobian(x)
        d = self.outer.derivative(x)
        if d.shape == x.shape:
            return g / d
        return torch.linalg.solve(d.transpose(-2, -1), g.unsqueeze(-1)).squeeze(-1)

    def log_pdf_param_grads(self, x_y):
        return self.base.log_pdf_param_grads(self.outer.forward(x_y))

    def _log_prob_unconstrained(self, x_unconstrained):
        return self.base._log_prob_unconstrained(x_unconstrained)

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        return self.base._log_prob_grad_unconstrained(x_unconstrained)

    @property
    def has_rsample(self):
        return self.base.has_rsample

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            x, grads = self.base.sample(n_samples, _reparametrization_trick_grad=True)
            d = self.outer.derivative(x)
            if d.shape != x.shape:
                raise NotImplementedError("Reparameterised gradients need an element-wise transform.")
            return self.outer.inverse(x), {k: v * d for k, v in grads.items()}
        return self.outer.inverse(self.base.sample(n_samples))

    def cdf(self, x_y):
        # valid for increasing element-wise transforms
        return self.base.cdf(self.outer.forward(x_y))

    def icdf(self, u):
        return self.outer.inverse(self.base.icdf(u))


class Ordered(Distribution):
    """Strictly increasing vector of length ``K`` with elementwise density ``base``.

    ``Ordered(base, K)`` is the distribution of a vector ``x`` of shape
    ``(*batch, K)`` with ``x_0 < x_1 < ... < x_{K-1}`` and (unnormalised)
    log density ``sum_k base.log_pdf(x_k)``: the density of ``K`` draws of
    ``base`` restricted to the ordered region. For a base with parameters
    shared by all elements the normalising constant is ``K!``, which is left
    out (it does not depend on ``x`` or on the parameters). The parameters of
    ``base`` broadcast against ``x``, so they may be scalars or have a
    trailing dimension of size ``K`` (e.g. one prior mean per element).

    The sampler works in an unconstrained space through
    :class:`OrderedTransform` (``x_0 = y_0``, ``x_k = x_{k-1} + exp(y_k)``).
    A base with a bounded support (e.g. ``Exponential``) composes its own
    increasing elementwise transform after the ordering, so the ordered
    values stay in the base support.

    The typical use is to break the label-switching symmetry of a mixture::

        means = RandomParameter("means", Ordered(Normal(0.0, 5.0), 2))
    """

    _param_event_ndims = {"base": 1}

    def __init__(self, base, K):
        if base.event_shape != ():
            raise ValueError("Ordered needs a univariate base distribution.")
        if not base.state_space.is_continuous():
            raise ValueError("Ordered needs a continuous base distribution.")
        base_transform = base.transform
        if isinstance(base_transform, IdentityTransform):
            transform = OrderedTransform()
        elif isinstance(base_transform, (LogTransform, InverseSoftPlusTransform)) and base_transform.side == "larger" or isinstance(base_transform, LogitTransform):
            # Increasing elementwise maps keep the order of the ordered vector.
            transform = ChainTransform(OrderedTransform(), base_transform)
        else:
            raise ValueError(f"Ordered does not support a base with the transform {type(base_transform).__name__}.")
        super().__init__(transform, ContinuousOrdered(base.state_space), ContinuousReal())
        self.base = base
        self.K = int(K)
        if self.K < 1:
            raise ValueError("K must be positive.")
        self._add_dependencies_of(base)

    @property
    def event_shape(self):
        return (self.K,)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(tuple(self.base.batch_shape), (self.K,))[:-1]

    def log_pdf(self, x):
        log_p = self.base.log_pdf(x)
        return log_p.expand(torch.broadcast_shapes(log_p.shape, x.shape)).sum(-1)

    def log_pdf_grad(self, x):
        grad = self.base.log_pdf_grad(x)
        return grad.expand(torch.broadcast_shapes(grad.shape, x.shape))

    def log_pdf_param_grads(self, x):
        # Elementwise in x; the model sums them onto the shapes of the parameters.
        return self.base.log_pdf_param_grads(x)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        base_batch = tuple(self.base.batch_shape)
        if base_batch[-1:] == (self.K,):
            draws = self.base.sample(n_samples)
        else:
            draws = self.base.sample(n_samples * self.K)
            draws = draws.reshape(n_samples, self.K, *base_batch).movedim(1, -1)
            draws = draws.expand(n_samples, *self.batch_shape, self.K)
        return torch.sort(draws, dim=-1).values


# ========================= Miscellaneous distributions =========================
class Mixture(Distribution):
    _param_event_ndims = {"weights": 1}

    def __init__(self, components, weights):
        super().__init__(IdentityTransform(), Union(*[component.state_space for component in components]), Union(*[component.transformed_state_space for component in components]))
        self.components = components
        self.weights = weights
        self.add_dependency(weights)
        for component in components:
            for param in component.parameters:
                self.parameters.add(param)

    @property
    def event_shape(self):
        return self.components[0].event_shape if self.components else ()

    @property
    def batch_shape(self):
        shapes = [c.batch_shape for c in self.components]
        if shapes:
            return torch.broadcast_shapes(*shapes)
        return ()

    def _weights(self):
        weights = resolve(self.weights)
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        return weights

    def pdf(self, x):
        weights = self._weights()
        return sum(weights[..., k] * component.pdf(x) for k, component in enumerate(self.components))

    def log_pdf(self, x):
        weights = self._weights()
        log_pdfs = [(weights[..., k] + 1e-12).log() + component.log_pdf(x) for k, component in enumerate(self.components)]
        shape = torch.broadcast_shapes(*(t.shape for t in log_pdfs))
        log_pdfs = torch.stack([t.expand(shape) for t in log_pdfs])
        return _logsumexp(log_pdfs, dim=0)

    def log_pdf_grad(self, x):
        weights = self._weights()
        log_weighted = [torch.log(weights[..., k] + 1e-12) + component.log_pdf(x) for k, component in enumerate(self.components)]
        grads = [component.log_pdf_grad(x) for component in self.components]
        shape = torch.broadcast_shapes(*(t.shape for t in log_weighted))
        log_weighted = torch.stack([t.expand(shape) for t in log_weighted], dim=0)  # (K, *batch)
        grad_shape = torch.broadcast_shapes(*(g.shape for g in grads))
        grads = torch.stack([g.expand(grad_shape) for g in grads], dim=0)  # (K, *batch, ...)
        log_mixture_pdf = _logsumexp(log_weighted, dim=0)
        posterior_weights = torch.exp(log_weighted - log_mixture_pdf)
        if grads.ndim > posterior_weights.ndim:
            posterior_weights = posterior_weights.reshape(*posterior_weights.shape, *([1] * (grads.ndim - posterior_weights.ndim)))
        return (posterior_weights * grads).sum(dim=0)

    def log_pdf_param_grads(self, x):
        weights = self._weights()
        grads = {}
        log_pdfs = torch.stack([component.log_pdf(x) for component in self.components], dim=-1)
        log_w = torch.log(weights + 1e-12)
        log_weighted = log_w + log_pdfs
        log_mixture_pdf = torch.logsumexp(log_weighted, dim=-1, keepdim=True)

        grads[self.resolve_name("weights", self.weights)] = torch.exp(log_pdfs - log_mixture_pdf)
        posterior_weights = torch.exp(log_weighted - log_mixture_pdf)

        for i, component in enumerate(self.components):
            param_grads = component.log_pdf_param_grads(x)
            coefficient = posterior_weights[..., i]
            for name, gradient in param_grads.items():
                if not isinstance(gradient, torch.Tensor):
                    gradient = torch.as_tensor(gradient, dtype=coefficient.dtype)
                coeff = coefficient
                if gradient.ndim > coefficient.ndim:
                    coeff = coeff.reshape(*coeff.shape, *([1] * (gradient.ndim - coefficient.ndim)))
                grads[name] = grads.get(name, 0) + coeff * gradient
        return grads

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        weights = self._weights()
        shape = _sample_shape(n_samples, self.batch_shape, self.event_shape)
        def _draw(component):
            s = component.sample(n_samples)
            missing = len(shape) - s.ndim
            return s.reshape(n_samples, *([1] * missing), *s.shape[1:]).expand(shape)
        draws = torch.stack([_draw(c) for c in self.components], dim=0)  # (K, *shape)
        index_shape = (n_samples, *self.batch_shape)
        probs = weights.to(draws.dtype) / weights.sum()
        cdf = torch.cumsum(probs, dim=0)
        u = torch.rand((*index_shape, 1), dtype=draws.dtype, device=draws.device)
        idx = (cdf < u).sum(-1).clamp(max=len(self.components) - 1)
        idx = idx.reshape(1, *index_shape, *([1] * len(self.event_shape))).expand(1, *shape)
        return draws.gather(0, idx).squeeze(0)


def _logsumexp(log_values, dim=0):
    max_val = torch.amax(log_values, dim=dim, keepdim=True)
    sum_exp = torch.sum(torch.exp(log_values - max_val), dim=dim, keepdim=True)
    return (max_val + torch.log(sum_exp)).squeeze(dim)
