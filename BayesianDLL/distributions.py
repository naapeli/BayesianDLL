import torch
import math
from abc import ABC, abstractmethod

from .parameters import Parameter
from ._transforms import IdentityTransform, LogitTransform, LogTransform


class Distribution(ABC):
    def __init__(self, transform=IdentityTransform()):
        self.transform = transform

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

class Normal(Distribution):
    def __init__(self, mu, variance):
        super().__init__(IdentityTransform())
        self.mu = torch.as_tensor(mu)
        self.variance = torch.as_tensor(variance)

    def pdf(self, x):
        x = torch.as_tensor(x)
        return torch.exp(-0.5 * (x - self.mu) ** 2 / self.variance) / torch.sqrt(2 * math.pi * self.variance)

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        return -0.5 * (x - self.mu) ** 2 / self.variance - 0.5 * torch.log(2 * math.pi * self.variance)

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return -(x - self.mu) / self.variance

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        diff = x - self.mu
        grad_mu = diff / self.variance
        grad_var = 0.5 * (diff ** 2 / self.variance ** 2 - 1 / self.variance)
        return {"mean": grad_mu, "variance": grad_var}

class MultivariateNormal(Distribution):
    def __init__(self, mu, covariance):
        super().__init__(IdentityTransform())
        self.mu = mu
        self.covariance = covariance
        self._dim = len(mu)
        self._inv_cov = torch.linalg.inv(self.covariance)
        self._norm_const = -0.5 * (self._dim * math.log(2 * math.pi) + torch.logdet(self.covariance))

    def pdf(self, x):
        x = torch.as_tensor(x)
        diff = x - self.mu
        exponent = -0.5 * (diff @ self._inv_cov @ diff)
        return torch.exp(exponent) / torch.exp(self._norm_const)

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        diff = x - self.mu
        exponent = -0.5 * (diff @ self._inv_cov @ diff)
        return self._norm_const + exponent

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        diff = x - self.mu
        return -self._inv_cov @ diff

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        diff = x - self.mu
        grad_mu = self._inv_cov @ diff
        outer = torch.ger(diff, diff)
        grad_cov = 0.5 * (self._inv_cov @ outer @ self._inv_cov - self._inv_cov)
        return {"mean": grad_mu, "covariance": grad_cov}

class Beta(Distribution):
    def __init__(self, a, b):
        super().__init__(LogitTransform(low=0, high=1))
        self.a = torch.as_tensor(a)
        self.b = torch.as_tensor(b)
        self._beta = torch.as_tensor(torch.math.gamma(a) * torch.math.gamma(b) / torch.math.gamma(a + b))
        self._betaln = self._beta.log()

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        return (x ** (self.a - 1) * (1 - x) ** (self.b - 1)) / self._beta
    
    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x) - self._betaln
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        return (self.a - 1)/x - (self.b - 1)/(1 - x)
    
    def log_pdf_param_grads(self, x):
        raise NotImplementedError()
        x = torch.as_tensor(x).clamp(1e-8, 1 - 1e-8)
        grad_a = torch.log(x) - torch.digamma(self.a) + torch.digamma(self.a + self.b)
        grad_b = torch.log(1 - x) - torch.digamma(self.b) + torch.digamma(self.a + self.b)
        return {'a': grad_a.sum(), 'b': grad_b.sum()}

class Exponential(Distribution):
    def __init__(self, rate):
        super().__init__(LogTransform(border=0, side="larger"))
        self.rate = torch.as_tensor(rate)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return self.rate * torch.exp(-self.rate * x)
    
    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return torch.log(self.rate) - self.rate * x

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return -self.rate * torch.ones_like(x)

    def log_pdf_param_grads(self, x):
        raise NotImplementedError()
        x = torch.as_tensor(x).clamp(min=1e-8)
        n = x.numel()
        grad_rate = n / self.rate - x.sum()
        return {'rate': grad_rate}

class Uniform(Distribution):
    def __init__(self, low, high):
        super().__init__(LogitTransform(low=low, high=high))
        self.low = low
        self.high = high
    
    def pdf(self, x):
        x = torch.as_tensor(x)
        return torch.where((self.low <= x) & (x <= self.high), torch.tensor([1 / (self.high - self.low)], dtype=x.dtype), torch.tensor([0], dtype=x.dtype))
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        return torch.where((self.low <= x) & (x <= self.high), -torch.tensor([self.high - self.low], dtype=x.dtype).log(), torch.tensor([-float("inf")], dtype=x.dtype).log())
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return torch.tensor([0], dtype=x.dtype)
    
    def log_pdf_param_grads(self, x):
        raise NotImplementedError()

class Bernoulli(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform())
        self.p = torch.as_tensor(p).clamp(1e-8, 1 - 1e-8)

    def pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.p ** x * (1 - self.p) ** (1 - x)

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x * torch.log(self.p) + (1 - x) * torch.log(1 - self.p)

    def log_pdf_grad(self, x):
        raise NotImplementedError()
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x / self.p - (1 - x) / (1 - self.p)) * self.p * (1 - self.p)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return {"p": x / self.p - (1 - x) / (1 - self.p)}

class Binomial(Distribution):
    def __init__(self, n, p):
        super().__init__(LogitTransform(low=0.0, high=1.0))
        self.n = n
        self.p = torch.as_tensor(p).clamp(1e-8, 1 - 1e-8)
        self._log_p = torch.log(self.p)
        self._log_1mp = torch.log(1 - self.p)

    def _log_binom_coeff(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        n = torch.tensor(self.n, dtype=torch.float32, device=x.device)
        return torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1)

    def pdf(self, x):
        return self.log_pdf(x).exp()

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self._log_binom_coeff(x) + x * self._log_p + (self.n - x) * self._log_1mp

    def log_pdf_grad(self, x):
        raise NotImplementedError()
        x = torch.as_tensor(x, dtype=torch.float32)
        return self._log_p - self._log_1mp
    
    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return {"p": x / self.p - (self.n - x) / (1 - self.p)}

class InvGamma(Distribution):
    def __init__(self, alpha, beta):
        super().__init__(LogTransform(border=0, side="larger"))
        self.alpha = alpha
        self.beta = beta
        self._log_gamma_alpha = torch.lgamma(torch.as_tensor(self.alpha))

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return (self.beta ** self.alpha / torch.exp(self._log_gamma_alpha)) * x ** (-self.alpha - 1) * torch.exp(-self.beta / x)

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return self.alpha * math.log(self.beta) - self._log_gamma_alpha - (self.alpha + 1) * torch.log(x) - self.beta / x

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return -(self.alpha + 1) / x + self.beta / x ** 2
    
    def log_pdf_param_grads(self, x):
        raise NotImplementedError()

class HalfCauchy(Distribution):
    def __init__(self, scale):
        super().__init__(LogTransform(border=0, side="larger"))
        self.scale = torch.as_tensor(scale).clamp(min=1e-8)

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        denom = math.pi * self.scale * (1 + (x / self.scale) ** 2)
        return 2.0 / denom

    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return math.log(2.0) - math.log(math.pi) - torch.log(self.scale) - torch.log(1 + (x / self.scale) ** 2)

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-8)
        return -2 * x / (self.scale ** 2 + x ** 2)

    def log_pdf_param_grads(self, x):
        raise NotImplementedError()
        x = torch.as_tensor(x).clamp(min=1e-8)
        d_log_pdf_dscale = -1 / self.scale + 2 * (x ** 2) / (self.scale * (x ** 2 + self.scale ** 2))
        return {'scale': d_log_pdf_dscale}
