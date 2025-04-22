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

    # @abstractmethod
    # def log_pdf_param_grads(self, x_constrained):
    #     pass

    def _log_prob_unconstrained(self, x_unconstrained):
        x_constrained = self.transform.inverse(x_unconstrained)
        return self.log_pdf(x_constrained) + self.transform.derivative(x_unconstrained).abs().log()
    
    def _log_prob_grad_unconstrained(self, x_unconstrained):
        log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
        dx_dz = self.transform.derivative(x_unconstrained)
        d_log_det_jacobian = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
        return log_pdf_grad_x * dx_dz + d_log_det_jacobian

class Normal(Distribution):
    def __init__(self, mu, sigma):
        super().__init__(IdentityTransform())
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x):
        x = torch.as_tensor(x)
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * math.sqrt(2 * math.pi))
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return -(x - self.mu) / (self.sigma ** 2)
    
    # def log_pdf_param_grads(self, x):
    #     x = torch.as_tensor(x)
    #     grad_mu = ((x - self.mu).sum()) / (self.sigma ** 2)
    #     grad_sigma = (-len(x) / self.sigma) + (((x - self.mu) ** 2).sum()) / (self.sigma ** 3)
    #     return {'mu': grad_mu, 'sigma': grad_sigma}

class Beta(Distribution):
    def __init__(self, a, b):
        super().__init__(LogitTransform(low=0, high=1))
        self.a = a
        self.b = b
        self._beta = torch.lgamma(torch.as_tensor(a)).exp() * torch.lgamma(torch.as_tensor(b)).exp() / torch.lgamma(torch.as_tensor(a + b)).exp()
        self._betaln = self._beta.log()

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-7, 1-1e-7)
        return (x ** (self.a - 1) * (1 - x) ** (self.b - 1)) / self._beta
    
    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(1e-7, 1-1e-7)
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x) - self._betaln
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(1e-7, 1-1e-7)
        return (self.a - 1)/x - (self.b - 1)/(1 - x)
    
    # def log_pdf_param_grads(self, x):
    #     x = torch.as_tensor(x).clamp(1e-7, 1-1e-7)
    #     grad_a = torch.log(x) - torch.digamma(self.a) + torch.digamma(self.a + self.b)
    #     grad_b = torch.log(1 - x) - torch.digamma(self.b) + torch.digamma(self.a + self.b)
    #     return {'a': grad_a.sum(), 'b': grad_b.sum()}

class Exponential(Distribution):
    def __init__(self, rate):
        super().__init__(LogTransform(border=0, side="larger"))
        self.rate = rate

    def pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-10)
        return self.rate * torch.exp(-self.rate * x)
    
    def log_pdf(self, x):
        x = torch.as_tensor(x).clamp(min=1e-10)
        return torch.log(torch.tensor(self.rate)) - self.rate * x

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x).clamp(min=1e-10)
        return -self.rate * torch.ones_like(x)

    # def log_pdf_param_grads(self, x):
    #     x = torch.as_tensor(x).clamp(min=1e-10)
    #     n = x.numel()
    #     grad_rate = n / self.rate - x.sum()
    #     return {'rate': grad_rate}

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
        return torch.where((self.low <= x) & (x <= self.high), -torch.tensor([self.high - self.low], dtype=x.dtype), torch.tensor([-float("inf")], dtype=x.dtype))
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return torch.tensor([0], dtype=x.dtype)

class Bernoulli(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform())
        self.p = torch.as_tensor(p).clamp(1e-7, 1 - 1e-7)

    def pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.p ** x * (1 - self.p) ** (1 - x)

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x * torch.log(self.p) + (1 - x) * torch.log(1 - self.p)

    def log_pdf_grad(self, x):
        raise NotImplementedError()
        # x = torch.as_tensor(x, dtype=torch.float32)
        # return (x / self.p - (1 - x) / (1 - self.p)) * self.p * (1 - self.p)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x / self.p - (1 - x) / (1 - self.p)

class Binomial(Distribution):
    def __init__(self, n, p):
        super().__init__(LogitTransform(low=0.0, high=1.0))
        self.n = n
        self.p = torch.as_tensor(p).clamp(1e-7, 1 - 1e-7)
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
        # x = torch.as_tensor(x, dtype=torch.float32)
        # return self._log_p - self._log_1mp
    
    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x / self.p - (self.n - x) / (1 - self.p)
