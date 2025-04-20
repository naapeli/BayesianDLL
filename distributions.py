import torch
import math
from abc import ABC, abstractmethod

from model import Parameter


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def log_pdf(self, x):
        pass

    @abstractmethod
    def log_pdf_grad(self, x):
        pass

    @abstractmethod
    def log_pdf_param_grads(self, x):
        pass


class Beta(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._betaln = torch.lgamma(torch.as_tensor(self.a)) + torch.lgamma(torch.as_tensor(self.b)) - torch.lgamma(torch.as_tensor(self.a) + torch.as_tensor(self.b))

    def pdf(self, x):
        x = torch.as_tensor(x)
        return (x ** (self.a - 1) * (1 - x) ** (self.b - 1)) / torch.special.beta(self.a, self.b)
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        return (self.a - 1) * torch.log(x) + (self.b - 1) * torch.log(1 - x) - self._betaln
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return (self.a - 1)/x - (self.b - 1)/(1 - x)
    
    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        grad_a = torch.log(x) - torch.digamma(self.a) + torch.digamma(self.a + self.b)
        grad_b = torch.log(1 - x) - torch.digamma(self.b) + torch.digamma(self.a + self.b)
        return {'a': grad_a.sum(), 'b': grad_b.sum()}


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def _resolve(self, x):
        return x.init_value if isinstance(x, Parameter) else x
    
    def pdf(self, x):
        x = torch.as_tensor(x)
        mu = self._resolve(self.mu)
        sigma = self._resolve(self.sigma)
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    
    def log_pdf(self, x):
        x = torch.as_tensor(x)
        mu = self._resolve(self.mu)
        sigma = self._resolve(self.sigma)
        return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    
    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        mu = self._resolve(self.mu)
        sigma = self._resolve(self.sigma)
        return -(x - mu) / (sigma ** 2)
    
    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        mu = self._resolve(self.mu)
        sigma = self._resolve(self.sigma)
        grad_mu = ((x - mu).sum()) / (sigma ** 2)
        grad_sigma = (-len(x) / sigma) + (((x - mu) ** 2).sum()) / (sigma ** 3)
        return {'mu': grad_mu, 'sigma': grad_sigma}
