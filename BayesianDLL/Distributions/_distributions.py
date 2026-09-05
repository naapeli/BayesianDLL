import torch
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod

from ._transforms import IdentityTransform, LogitTransform, LogTransform, SoftMaxTransform, InverseSoftPlusTransform
from ._state_space import ContinuousReal, ContinuousPositive, ContinuousRange, ContinuousSimplex, DiscretePositive, DiscreteRange, Union
from .._parameters import RandomParameter, DeterministicParameter, VariationalParameter
from .._data import Data
from ._resolve import resolve


class Distribution(ABC):
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

    @abstractmethod
    def pdf(self, x):
        pass

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

    def _log_prob_unconstrained(self, x_unconstrained):
        """
        Log probability in unconstrained space, including the Jacobian
        correction.  *x_unconstrained* has the same shape as the parameter's
        unconstrained value (``event_shape`` for element-wise transforms).
        Returns a **scalar**.
        """
        x_constrained = self.transform.inverse(x_unconstrained)
        log_det = self.transform.log_abs_det_jacobian(x_unconstrained)
        return self.log_pdf(x_constrained).sum() + log_det

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        """
        Gradient of ``_log_prob_unconstrained`` w.r.t. *x_unconstrained*.
        Returns a tensor with the same shape as *x_unconstrained*.
        """
        log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
        dx_dz = self.transform.derivative(x_unconstrained)
        # For element-wise transforms dx_dz has the same shape as x
        if dx_dz.shape == x_unconstrained.shape:
            term1 = log_pdf_grad_x * dx_dz
        else:
            # SoftMax: dx_dz is a full Jacobian matrix
            term1 = (dx_dz.transpose(-2, -1) @ log_pdf_grad_x.unsqueeze(-1)).squeeze(-1)
        d_log_det = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
        return term1 + d_log_det

    def resolve_name(self, name, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter | VariationalParameter | Data):
            return parameter.name
        return name

    def add_dependency(self, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter | Data):
            self.parameters.add(parameter.name)
        if isinstance(parameter, VariationalParameter):
            self.variational_parameters[parameter.name] = parameter


# ========================= CONTINUOUS =========================
class Normal(Distribution):
    def __init__(self, mu, variance):
        super().__init__()
        self.mu = mu
        self.variance = variance
        self.add_dependency(mu)
        self.add_dependency(variance)

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        mu = resolve(self.mu)
        var = resolve(self.variance)
        return torch.broadcast_shapes(mu.shape, var.shape)

    def pdf(self, x):
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return torch.exp(-0.5 * (x - mu) ** 2 / variance) / torch.sqrt(2 * math.pi * variance)

    def log_pdf(self, x):
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -0.5 * (x - mu) ** 2 / variance - 0.5 * torch.log(2 * math.pi * variance)

    def log_pdf_grad(self, x):
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -(x - mu) / variance

    def log_pdf_param_grads(self, x):
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        diff = x - mu
        grad_mu = diff / variance
        grad_var = 0.5 * (diff ** 2 / variance ** 2 - 1 / variance)
        return {self.resolve_name("mean", self.mu): grad_mu, self.resolve_name("variance", self.variance): grad_var}

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        eps = torch.randn((n_samples, *mu.shape), dtype=mu.dtype)
        samples = mu + eps * torch.sqrt(variance)
        if _reparametrization_trick_grad:
            grad_mu = torch.ones_like(eps)
            grad_var = eps / (2 * torch.sqrt(variance))
            return samples, {self.resolve_name("mean", self.mu): grad_mu, self.resolve_name("variance", self.variance): grad_var}
        return samples


class MultivariateNormal(Distribution):
    def __init__(self, mu, covariance):
        super().__init__()
        self.mu = mu
        self.covariance = covariance
        self.add_dependency(mu)
        self.add_dependency(covariance)

    @property
    def event_shape(self):
        return resolve(self.mu).shape[-1:]

    @property
    def batch_shape(self):
        return resolve(self.mu).shape[:-1]

    def pdf(self, x):
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        return torch.exp(self.log_pdf(x))

    def log_pdf(self, x):
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        n_features = x.shape[-1]
        diff = x - mu
        L = torch.linalg.cholesky(covariance)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), L)
        quad_form = (diff.unsqueeze(-1) * sol).sum(dim=-2)
        log_det = 2 * torch.log(torch.diag(L)).sum()
        log_norm_const = 0.5 * (n_features * math.log(2 * math.pi) + log_det)
        return (-0.5 * quad_form - log_norm_const).squeeze(-1)

    def log_pdf_grad(self, x):
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        grad = -torch.cholesky_solve(diff.unsqueeze(-1), torch.linalg.cholesky(covariance)).squeeze(-1)
        return grad

    def log_pdf_param_grads(self, x):
        raise NotImplementedError("Make sure MultivariateNormal.log_pdf_param_grads works as expected!")


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

    def pdf(self, x):
        a = resolve(self.a)
        b = resolve(self.b)
        beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        return (x ** (a - 1) * (1 - x) ** (b - 1)) / beta

    def log_pdf(self, x):
        a = resolve(self.a)
        b = resolve(self.b)
        return (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    def log_pdf_grad(self, x):
        a = resolve(self.a)
        b = resolve(self.b)
        return (a - 1) / x - (b - 1) / (1 - x)

    def log_pdf_param_grads(self, x):
        a = resolve(self.a)
        b = resolve(self.b)
        grad_a = torch.log(x) - torch.digamma(a) + torch.digamma(a + b)
        grad_b = torch.log(1 - x) - torch.digamma(b) + torch.digamma(a + b)
        return {self.resolve_name("a", self.a): grad_a, self.resolve_name("b", self.b): grad_b}

    def _log_prob_unconstrained(self, x_unconstrained):
        a = resolve(self.a)
        b = resolve(self.b)
        log_x = F.logsigmoid(x_unconstrained)
        log_1_minus_x = F.logsigmoid(-x_unconstrained)
        log_prob = a * log_x + b * log_1_minus_x - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        return log_prob.sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        a = resolve(self.a)
        b = resolve(self.b)
        return a * torch.sigmoid(-x_unconstrained) - b * torch.sigmoid(x_unconstrained)


class Exponential(Distribution):
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

    def pdf(self, x):
        rate = resolve(self.rate)
        return rate * torch.exp(-rate * x)

    def log_pdf(self, x):
        rate = resolve(self.rate).clamp(min=1e-8)
        return torch.log(rate) - rate * x

    def log_pdf_grad(self, x):
        rate = resolve(self.rate).clamp(min=1e-8)
        return -rate * torch.ones_like(x)

    def log_pdf_param_grads(self, x):
        rate = resolve(self.rate).clamp(min=1e-8)
        grad_rate = 1 / rate - x
        return {self.resolve_name("rate", self.rate): grad_rate}

    def _log_prob_unconstrained(self, x_unconstrained):
        if not isinstance(self.transform, LogTransform):
            raise RuntimeError("Exponential._log_prob_unconstrained can only be used if the transform is a log transform")
        
        rate = resolve(self.rate)
        z = x_unconstrained
        x = torch.exp(z)
        return (torch.log(rate) - rate * x + z).sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        if not isinstance(self.transform, LogTransform):
            raise RuntimeError("Exponential._log_prob_unconstrained can only be used if the transform is a log transform")
        
        rate = resolve(self.rate)
        x = torch.exp(x_unconstrained)
        return 1 - rate * x

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        rate = resolve(self.rate)
        eps = torch.rand(size=(n_samples, *rate.shape), dtype=rate.dtype)
        log_eps = torch.log(eps)
        samples = -1 / rate * log_eps
        if _reparametrization_trick_grad:
            grad_rate = log_eps / rate ** 2
            return samples, {self.resolve_name("rate", self.rate): grad_rate}
        return samples


class Uniform(Distribution):
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

    def pdf(self, x):
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.where((x >= low) & (x <= high), 1.0 / (high - low), torch.zeros_like(x))

    def log_pdf(self, x):
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.where((x >= low) & (x <= high), -torch.log(high - low), torch.full_like(x, -torch.inf))

    def log_pdf_grad(self, x):
        return torch.zeros_like(x)

    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")

    def _log_prob_unconstrained(self, x_unconstrained):
        z = x_unconstrained
        return (F.logsigmoid(z) + F.logsigmoid(-z)).sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        z = x_unconstrained
        return torch.sigmoid(-z) - torch.sigmoid(z)


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

    def pdf(self, x):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        return torch.exp(self.log_pdf(x))

    def log_pdf(self, x):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        return alpha * torch.log(beta) - torch.lgamma(alpha) - (alpha + 1) * torch.log(x) - beta / x

    def log_pdf_grad(self, x):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        return -(alpha + 1) / x + beta / x ** 2

    def log_pdf_param_grads(self, x):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        grad_alpha = torch.log(beta) - torch.digamma(alpha) - torch.log(x)
        grad_beta = alpha / beta - 1 / x
        return {self.resolve_name("alpha", self.alpha): grad_alpha, self.resolve_name("beta", self.beta): grad_beta}

    def _log_prob_unconstrained(self, x_unconstrained):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        z = x_unconstrained
        return (alpha * torch.log(beta) - torch.lgamma(alpha) - alpha * z - beta * torch.exp(-z)).sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        x = torch.exp(x_unconstrained)
        return -alpha + beta / x


class HalfCauchy(Distribution):
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

    def pdf(self, x):
        scale = resolve(self.scale)
        return 2.0 / (math.pi * scale * (1 + (x / scale) ** 2))

    def log_pdf(self, x):
        scale = resolve(self.scale)
        return math.log(2.0) - math.log(math.pi) - torch.log(scale) - torch.log(1 + (x / scale) ** 2)

    def log_pdf_grad(self, x):
        scale = resolve(self.scale)
        return -2 * x / (scale ** 2 + x ** 2)

    def log_pdf_param_grads(self, x):
        scale = resolve(self.scale)
        grad_scale = -1 / scale + 2 * x ** 2 / (scale * (x ** 2 + scale ** 2))
        return {self.resolve_name("scale", self.scale): grad_scale}

    def _log_prob_unconstrained(self, x_unconstrained):
        scale = resolve(self.scale)
        z = x_unconstrained
        x = torch.exp(z)
        return (math.log(2.0) - math.log(math.pi) - torch.log(scale) + z - torch.log(1 + (x / scale) ** 2)).sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        scale = resolve(self.scale)
        x = torch.exp(x_unconstrained)
        return (scale ** 2 - x ** 2) / (scale ** 2 + x ** 2)


class Dirichlet(Distribution):
    def __init__(self, alpha):
        super().__init__(transform=SoftMaxTransform(dim=-1), state_space=ContinuousSimplex(), transformed_state_space=ContinuousReal())
        self.alpha = alpha
        self.add_dependency(alpha)

    @property
    def event_shape(self):
        return resolve(self.alpha).shape[-1:]

    @property
    def batch_shape(self):
        return resolve(self.alpha).shape[:-1]

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))

    def log_pdf(self, x):
        alpha = resolve(self.alpha)
        alpha_sum = alpha.sum(-1, keepdim=True)
        log_norm_const = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha_sum).squeeze(-1)
        return ((alpha - 1) * x.log()).sum(-1) - log_norm_const

    def log_pdf_grad(self, x):
        alpha = resolve(self.alpha)
        return (alpha - 1) / x

    def log_pdf_param_grads(self, x):
        alpha = resolve(self.alpha)
        digamma_alpha = torch.digamma(alpha)
        digamma_alpha_sum = torch.digamma(alpha.sum(-1, keepdim=True))
        grad_alpha = x.log() - digamma_alpha + digamma_alpha_sum
        return {self.resolve_name("alpha", self.alpha): grad_alpha}

    def _log_prob_unconstrained(self, x_unconstrained):
        alpha = resolve(self.alpha)
        alpha_sum = alpha.sum(-1, keepdim=True)
        log_norm_const = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(alpha_sum)
        z = x_unconstrained
        log_x = z - torch.logsumexp(z, dim=-1, keepdim=True)
        log_prob = ((alpha - 1) * log_x).sum(-1, keepdim=True) - log_norm_const
        return log_prob.sum()

    def _log_prob_grad_unconstrained(self, x_unconstrained):
        alpha = resolve(self.alpha)
        x = torch.softmax(x_unconstrained, dim=-1)
        return (alpha - 1) - x * (alpha - 1).sum(-1, keepdim=True)


# ========================= DISCRETE =========================
class Geometric(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscretePositive(), DiscretePositive())
        self.p = p
        self.add_dependency(p)

    def pdf(self, x):
        p = resolve(self.p)
        return (1 - p) ** (x - 1) * p

    def log_pdf(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        return (x - 1) * torch.log(1 - p) + torch.log(p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        grad = (1 / p) - (x - 1) / (1 - p)
        return {self.resolve_name("p", self.p): grad}


class Bernoulli(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscreteRange(0, 1), DiscreteRange(0, 1))
        self.p = p
        self.add_dependency(p)

    def pdf(self, x):
        p = resolve(self.p)
        return p ** x * (1 - p) ** (1 - x)

    def log_pdf(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        return x * torch.log(p) + (1 - x) * torch.log(1 - p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        grad = (x / p - (1 - x) / (1 - p))
        return {self.resolve_name("p", self.p): grad}


class Binomial(Distribution):
    def __init__(self, n, p):
        super().__init__(IdentityTransform(), DiscreteRange(0, n), DiscreteRange(0, n))
        self.n = n
        self.p = p
        self.add_dependency(n)
        self.add_dependency(p)

    def _log_binom_coeff(self, x):
        n = resolve(self.n).to(dtype=torch.float32)
        return torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1)

    def pdf(self, x):
        return self.log_pdf(x).exp()

    def log_pdf(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        n = resolve(self.n)
        return self._log_binom_coeff(x) + x * torch.log(p) + (n - x) * torch.log(1 - p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        p = resolve(self.p).clamp(1e-8, 1 - 1e-8)
        n = resolve(self.n)
        grad = x / p - (n - x) / (1 - p)
        return {self.resolve_name("p", self.p): grad}


class DiscreteUniform(Distribution):
    def __init__(self, low, high):
        super().__init__(IdentityTransform(), DiscreteRange(low=low, high=high), DiscreteRange(low=low, high=high))
        self.low = low
        self.high = high
        self.add_dependency(low)
        self.add_dependency(high)

    def pdf(self, x):
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.full_like(x, 1.0 / (high - low + 1))

    def log_pdf(self, x):
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.full_like(x, -torch.log(torch.as_tensor(high - low + 1, dtype=x.dtype)))

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the discrete uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")


# ========================= Miscellaneous distributions =========================
class Mixture(Distribution):
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

    def pdf(self, x):
        weights = resolve(self.weights)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        return sum(w * component.pdf(x) for w, component in zip(weights, self.components))

    def log_pdf(self, x):
        weights = resolve(self.weights)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        log_pdfs = torch.stack([(w + 1e-12).log() + component.log_pdf(x) for w, component in zip(weights, self.components)])
        return _logsumexp(log_pdfs, dim=0)

    def log_pdf_grad(self, x):
        weights = resolve(self.weights)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        log_pdfs = torch.stack([component.log_pdf(x) for component in self.components], dim=0)  # (K, *batch)
        grads = torch.stack([component.log_pdf_grad(x) for component in self.components], dim=0)  # (K, *batch, ...)
        # Reshape weights (K,) to (K, 1, 1, ...) to broadcast with batch dims
        n_batch_dims = log_pdfs.ndim - 1
        w = weights.reshape(-1, *([1] * n_batch_dims)) if n_batch_dims > 0 else weights
        log_weighted = torch.log(w + 1e-12) + log_pdfs
        log_mixture_pdf = _logsumexp(log_weighted, dim=0)
        log_posterior_weights = log_weighted - log_mixture_pdf
        posterior_weights = torch.exp(log_posterior_weights)
        if grads.ndim > posterior_weights.ndim:
            posterior_weights = posterior_weights.reshape(*posterior_weights.shape, *([1] * (grads.ndim - posterior_weights.ndim)))
        return (posterior_weights * grads).sum(dim=0)

    def log_pdf_param_grads(self, x):
        weights = resolve(self.weights)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        grads = {}
        log_pdfs = torch.stack([component.log_pdf(x) for component in self.components], dim=-1)
        log_w = torch.log(weights + 1e-12)
        log_weighted = log_w + log_pdfs
        log_mixture_pdf = torch.logsumexp(log_weighted, dim=-1, keepdim=True)

        grad_weights = torch.exp(log_pdfs - log_mixture_pdf)
        grads[self.resolve_name("weights", self.weights)] = grad_weights

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


def _logsumexp(log_values, dim=0):
    max_val, _ = torch.max(log_values, dim=dim, keepdim=True)
    sum_exp = torch.sum(torch.exp(log_values - max_val), dim=dim, keepdim=True)
    return (max_val + torch.log(sum_exp)).squeeze(dim)
