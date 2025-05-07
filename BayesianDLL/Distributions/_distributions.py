import torch
import math
from abc import ABC, abstractmethod

from ._transforms import IdentityTransform, LogitTransform, LogTransform
from ._state_space import ContinuousReal, ContinuousPositive, ContinuousRange, DiscretePositive, DiscreteRange
from .._parameters import RandomParameter, DeterministicParameter
from ._resolve import resolve


class Distribution(ABC):
    def __init__(self, transform=IdentityTransform(), state_space=ContinuousReal(), transformed_state_space=ContinuousReal()):
        self.transform = transform
        self.state_space = state_space
        self.transformed_state_space = transformed_state_space
        self.random_parameters = set()  # used to store the random variables the distribution depends on
        self.deterministic_parameters = set()

    @abstractmethod
    def pdf(self, x_constrained):
        pass

    @abstractmethod
    def log_pdf(self, x_constrained):
        pass

    @abstractmethod
    def log_pdf_grad(self, x_constrained):
        pass

    @abstractmethod
    def log_pdf_param_grads(self, x_constrained):
        pass

    def _depends_on_random_variable(self, name):
        return name in self.random_parameters

    def _log_prob_unconstrained(self, x_unconstrained):
        # x_unconstrained.shape = (n_features,) or (n_samples, n_features)
        x_constrained = self.transform.inverse(x_unconstrained)
        log_det_abs_jacobian = self.transform.derivative(x_unconstrained).abs().log()
        log_det_abs_jacobian = log_det_abs_jacobian.sum(dim=1) if x_unconstrained.ndim == 2 else log_det_abs_jacobian.sum()
        return self.log_pdf(x_constrained).squeeze() + log_det_abs_jacobian
    
    def _log_prob_grad_unconstrained(self, x_unconstrained):
        # x_unconstrained.shape = (n_features,) or (n_samples, n_features)
        log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
        dx_dz = self.transform.derivative(x_unconstrained)
        d_log_det_jacobian = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
        return log_pdf_grad_x * dx_dz + d_log_det_jacobian
    
    def resolve_name(self, name, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter):
            return parameter.name
        return name
    
    def add_dependecy(self, parameter):
        if isinstance(parameter, RandomParameter): self.random_parameters.add(parameter.name)
        if isinstance(parameter, DeterministicParameter): self.deterministic_parameters.add(parameter.name)


# ========================= CONTINUOUS =========================
class Normal(Distribution):
    def __init__(self, mu, variance):
        super().__init__()
        self.mu = mu
        self.variance = variance
        self.add_dependecy(mu)
        self.add_dependecy(variance)

    def pdf(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return torch.exp(-0.5 * (x - mu) ** 2 / variance) / torch.sqrt(2 * math.pi * variance)

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -0.5 * (x - mu) ** 2 / variance - 0.5 * torch.log(2 * math.pi * variance)

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -(x - mu) / variance

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        diff = x - mu
        grad_mu = diff / variance
        grad_var = 0.5 * (diff ** 2 / variance ** 2 - 1 / variance)
        return {self.resolve_name("mean", self.mu): grad_mu, self.resolve_name("variance", self.variance): grad_var}

class MultivariateNormal(Distribution):
    def __init__(self, mu, covariance):
        super().__init__()
        self.mu = mu
        self.covariance = covariance
        self.add_dependecy(mu)
        self.add_dependecy(covariance)

    def pdf(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        exponent = -0.5 * (diff @ torch.cholesky_solve(diff.unsqueeze(-1), torch.linalg.cholesky(covariance)).squeeze(-1))
        denominator = (2 * math.pi) ** (-len(diff) / 2) * torch.det(self.covariance) ** -0.5
        return torch.exp(exponent) / denominator

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        exponent = -0.5 * (diff @ torch.cholesky_solve(diff.unsqueeze(-1), torch.linalg.cholesky(covariance)).squeeze(-1))
        denominator = (2 * math.pi) ** (-len(diff) / 2) * torch.det(self.covariance) ** -0.5
        return exponent + denominator

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        grad = -torch.cholesky_solve(diff.unsqueeze(-1), torch.linalg.cholesky(covariance)).squeeze(-1)
        return grad

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        L = torch.linalg.cholesky(covariance)
        inv_diff = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
        grad_mu = inv_diff

        outer = inv_diff.unsqueeze(1) @ inv_diff.unsqueeze(0)
        inv_cov = torch.cholesky_inverse(L)
        grad_cov = 0.5 * (outer - inv_cov)
        return {self.resolve_name("mean", self.mu): grad_mu, self.resolve_name("covariance", self.covariance): grad_cov}

class Beta(Distribution):
    def __init__(self, a, b):
        super().__init__(LogitTransform(low=0, high=1), ContinuousRange(0, 1), ContinuousReal())
        self.a = a
        self.b = b
        self.add_dependecy(a)
        self.add_dependecy(b)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        a = resolve(self.a)
        b = resolve(self.b)
        beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        return (x ** (a - 1) * (1 - x) ** (b - 1)) / beta

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        a = resolve(self.a)
        b = resolve(self.b)
        return (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        a = resolve(self.a)
        b = resolve(self.b)
        return (a - 1) / x - (b - 1) / (1 - x)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        a = resolve(self.a)
        b = resolve(self.b)
        grad_a = torch.log(x) - torch.digamma(a) + torch.digamma(a + b)
        grad_b = torch.log(1 - x) - torch.digamma(b) + torch.digamma(a + b)
        return {self.resolve_name("a", self.a): grad_a, self.resolve_name("b", self.b): grad_b}

class Exponential(Distribution):
    def __init__(self, rate):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.rate = rate
        self.add_dependecy(rate)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        rate = resolve(self.rate)
        return self.rate * torch.exp(-rate * x)
    
    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        rate = resolve(self.rate)
        return torch.log(rate) - rate * x

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        rate = resolve(self.rate)
        return -rate * torch.ones_like(x)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        rate = resolve(self.rate)
        grad_rate = 1 / rate - x
        return {self.resolve_name("rate", self.rate): grad_rate}

class Uniform(Distribution):
    def __init__(self, low, high):
        super().__init__(LogitTransform(low=low, high=high), ContinuousRange(low=low, high=high), ContinuousReal())
        self.low = low
        self.high = high
        self.add_dependecy(low)
        self.add_dependecy(high)
    
    def pdf(self, x):
        x = torch.as_tensor(x)
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.where((low <= x) & (x <= high), torch.tensor([1 / (high - low)], dtype=x.dtype), torch.tensor([0], dtype=x.dtype))
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.where((low <= x) & (x <= high), -torch.tensor([high - low], dtype=x.dtype).log(), torch.tensor([-float("inf")], dtype=x.dtype).log())
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return torch.tensor([0], dtype=x.dtype)
    
    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")

class InvGamma(Distribution):
    def __init__(self, alpha, beta):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.alpha = alpha
        self.beta = beta
        self.add_dependecy(alpha)
        self.add_dependecy(beta)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        log_gamma_alpha = torch.lgamma(alpha)
        return (beta ** alpha / torch.exp(log_gamma_alpha)) * x ** (-alpha - 1) * torch.exp(-beta / x)

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        return alpha * torch.log(beta) - torch.lgamma(alpha) - (alpha + 1) * torch.log(x) - beta / x

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        return -(alpha + 1) / x + beta / x ** 2

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)

        grad_alpha = torch.log(beta) - torch.digamma(alpha) - torch.log(x)
        grad_beta = alpha / beta - 1 / x

        return {self.resolve_name("alpha", self.alpha): grad_alpha, self.resolve_name("beta", self.beta): grad_beta}

class HalfCauchy(Distribution):
    def __init__(self, scale):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.scale = scale
        self.add_dependecy(scale)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        scale = resolve(self.scale)
        denom = math.pi * scale * (1 + (x / scale) ** 2)
        return 2.0 / denom

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        scale = resolve(self.scale)
        return math.log(2.0) - math.log(math.pi) - torch.log(scale) - torch.log(1 + (x / scale) ** 2)

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        scale = resolve(self.scale)
        return -2 * x / (scale ** 2 + x ** 2)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        scale = resolve(self.scale)
        grad_scale = -1 / scale + 2 * x ** 2 / (scale * (x ** 2 + scale ** 2))
        return {self.resolve_name("scale", self.scale): grad_scale}


# ========================= DISCRETE =========================
class Geometric(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscretePositive(), DiscretePositive())
        self.p = p
        self.add_dependecy(p)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1)
        p = resolve(self.p)
        return (1 - p) ** (x - 1) * p

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1)
        p = resolve(self.p)
        return (x - 1) * torch.log(1 - p) + torch.log(p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x).clamp(min=1)
        p = resolve(self.p)
        grad = (1 / p) - (x - 1) / (1 - p)
        return {self.resolve_name("p", self.p): grad}


class Bernoulli(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscreteRange(0, 1), DiscreteRange(0, 1))
        self.p = p
        self.add_dependecy(p)

    def pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        p = resolve(self.p)
        return p ** x * (1 - p) ** (1 - x)

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        p = resolve(self.p)
        return x * torch.log(p) + (1 - x) * torch.log(1 - p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        p = resolve(self.p)
        grad = (x / p - (1 - x) / (1 - p))
        return {self.resolve_name("p", self.p): grad}

class Binomial(Distribution):
    def __init__(self, n, p):
        super().__init__(IdentityTransform(), DiscreteRange(0, n), DiscreteRange(0, n))
        self.n = n
        self.p = p
        self.add_dependecy(n)
        self.add_dependecy(p)

    def _log_binom_coeff(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        n = resolve(self.n).to(dtype=torch.float32)
        return torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1)

    def pdf(self, x):
        return self.log_pdf(x).exp()

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        p = resolve(self.p)
        n = resolve(self.n)
        return self._log_binom_coeff(x) + x * torch.log(p) + (n - x) * torch.log(1 - p)

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")
    
    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        p = resolve(self.p)
        n = resolve(self.n)
        return {self.resolve_name("p", self.p): x / p - (n - x) / (1 - p)}

class DiscreteUniform(Distribution):
    def __init__(self, low, high):
        super().__init__(IdentityTransform(), DiscreteRange(low=low, high=high), DiscreteRange(low=low, high=high))
        self.low = low
        self.high = high
        self.add_dependecy(low)
        self.add_dependecy(high)
    
    def pdf(self, x):
        x = torch.as_tensor(x)
        low = resolve(self.low)
        high = resolve(self.high)
        prob = 1.0 / (high - low + 1)
        return torch.where((x >= low) & (x <= high), torch.full_like(x, prob, dtype=x.dtype), torch.zeros_like(x, dtype=x.dtype))
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        low = resolve(self.low)
        high = resolve(self.high)
        log_prob = -torch.log(high - low + 1)
        return torch.where((x >= low) & (x <= high), torch.full_like(x, log_prob, dtype=x.dtype), torch.full_like(x, -float('inf'), dtype=x.dtype))
    
    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")
    
    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the discrete uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")
