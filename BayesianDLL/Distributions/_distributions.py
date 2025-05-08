import torch
import math
from abc import ABC, abstractmethod

from ._transforms import IdentityTransform, LogitTransform, LogTransform, SoftMaxTransform
from ._state_space import ContinuousReal, ContinuousPositive, ContinuousRange, DiscretePositive, DiscreteRange, Union
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

    # def _log_prob_unconstrained(self, x_unconstrained):
    #     if not isinstance(x_unconstrained, torch.Tensor):
    #         raise TypeError("x_unconstrained should be a torch.Tensor.")
    #     if x_unconstrained.ndim != 2:
    #         raise ValueError("x_unconstrained.shape should be (n_samples, n_features).")

    #     x_constrained = self.transform.inverse(x_unconstrained)
    #     log_det_abs_jacobian = self.transform.derivative(x_unconstrained).abs().log()
    #     log_det_abs_jacobian = log_det_abs_jacobian.sum(dim=1) if x_unconstrained.ndim == 2 else log_det_abs_jacobian.sum()
    #     return self.log_pdf(x_constrained).squeeze() + log_det_abs_jacobian
    
    # def _log_prob_grad_unconstrained(self, x_unconstrained):
    #     if not isinstance(x_unconstrained, torch.Tensor):
    #         raise TypeError("x_unconstrained should be a torch.Tensor.")
    #     if x_unconstrained.ndim != 2:
    #         raise ValueError("x_unconstrained.shape should be (n_samples, n_features).")
        
    #     log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
    #     dx_dz = self.transform.derivative(x_unconstrained)
    #     d_log_det_jacobian = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
    #     return log_pdf_grad_x * dx_dz + d_log_det_jacobian

    def _log_prob_unconstrained(self, x_unconstrained):
        if not isinstance(x_unconstrained, torch.Tensor):
            raise TypeError("x_unconstrained should be a torch.Tensor.")
        if x_unconstrained.ndim != 2:
            raise ValueError("x_unconstrained.shape should be (n_samples, n_features).")

        x_constrained = self.transform.inverse(x_unconstrained)
        jacobian = self.transform.derivative(x_unconstrained)
        log_det_abs_jacobian = torch.linalg.slogdet(jacobian).logabsdet
        return self.log_pdf(x_constrained).squeeze() + log_det_abs_jacobian
    
    def _log_prob_grad_unconstrained(self, x_unconstrained):
        if not isinstance(x_unconstrained, torch.Tensor):
            raise TypeError("x_unconstrained should be a torch.Tensor.")
        if x_unconstrained.ndim != 2:
            raise ValueError("x_unconstrained.shape should be (n_samples, n_features).")
        
        log_pdf_grad_x = self.log_pdf_grad(self.transform.inverse(x_unconstrained))
        dx_dz = self.transform.derivative(x_unconstrained)  # (n, d, d)
        log_pdf_grad_x = log_pdf_grad_x.unsqueeze(2)        # (n, d, 1)
        term1 = (dx_dz.transpose(1, 2) @ log_pdf_grad_x).squeeze(2)  # (n, d)
        d_log_det_jacobian = self.transform.grad_log_abs_det_jacobian(x_unconstrained)
        return term1 + d_log_det_jacobian
    
    def resolve_name(self, name, parameter):
        if isinstance(parameter, RandomParameter | DeterministicParameter):
            return parameter.name
        return name
    
    def add_dependency(self, parameter):
        if isinstance(parameter, RandomParameter): self.random_parameters.add(parameter.name)
        if isinstance(parameter, DeterministicParameter): self.deterministic_parameters.add(parameter.name)


# ========================= CONTINUOUS =========================
class Normal(Distribution):
    def __init__(self, mu, variance):
        super().__init__()
        self.mu = mu
        self.variance = variance
        self.add_dependency(mu)
        self.add_dependency(variance)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return torch.exp(-0.5 * (x - mu) ** 2 / variance) / torch.sqrt(2 * math.pi * variance)

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -0.5 * (x - mu) ** 2 / variance - 0.5 * torch.log(2 * math.pi * variance)

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        mu = resolve(self.mu)
        variance = resolve(self.variance)
        return -(x - mu) / variance

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
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
        self.add_dependency(mu)
        self.add_dependency(covariance)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        exponent = -0.5 * (diff @ torch.cholesky_solve(diff.T, torch.linalg.cholesky(covariance)))
        denominator = (2 * math.pi) ** (-len(diff) / 2) * torch.linalg.det(self.covariance) ** -0.5
        prob = torch.exp(exponent) / denominator
        return prob

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        exponent = -0.5 * (diff @ torch.cholesky_solve(diff.T, torch.linalg.cholesky(covariance)))
        denominator = 0.5 * torch.log(torch.det(self.covariance)) + 0.5 * len(diff) * math.log(2 * math.pi)
        log_prob = exponent + denominator
        return log_prob

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        grad = -torch.cholesky_solve(diff.T, torch.linalg.cholesky(covariance)).T
        return grad

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        raise NotImplementedError("Make sure MultivariateNormal.log_pdf_param_grads works as expected!")
        
        mu = resolve(self.mu)
        covariance = resolve(self.covariance)
        diff = x - mu
        L = torch.linalg.cholesky(covariance)
        inv_diff = torch.cholesky_solve(diff.T, L)
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
        self.add_dependency(a)
        self.add_dependency(b)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        a = resolve(self.a)
        b = resolve(self.b)
        beta = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        prob = (x ** (a - 1) * (1 - x) ** (b - 1)) / beta
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(prob))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        a = resolve(self.a)
        b = resolve(self.b)
        log_prob = (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        a = resolve(self.a)
        b = resolve(self.b)
        grad = (a - 1) / x - (b - 1) / (1 - x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, grad, torch.full_like(grad, torch.nan))

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        a = resolve(self.a)
        b = resolve(self.b)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad_a = torch.log(x) - torch.digamma(a) + torch.digamma(a + b)
        grad_a = torch.where(mask, grad_a, torch.full_like(grad_a, torch.nan))
        grad_b = torch.log(1 - x) - torch.digamma(b) + torch.digamma(a + b)
        grad_b = torch.where(mask, grad_b, torch.full_like(grad_b, torch.nan))
        return {self.resolve_name("a", self.a): grad_a, self.resolve_name("b", self.b): grad_b}

class Exponential(Distribution):
    def __init__(self, rate):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.rate = rate
        self.add_dependency(rate)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        rate = resolve(self.rate)
        prob = self.rate * torch.exp(-rate * x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(prob))
    
    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        rate = resolve(self.rate)
        log_prob = torch.log(rate) - rate * x
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        rate = resolve(self.rate)
        grad = -rate * torch.ones_like(x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, grad, torch.full_like(grad, torch.nan))

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        rate = resolve(self.rate)
        grad_rate = 1 / rate - x
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad_rate = torch.where(mask, grad_rate, torch.full_like(grad_rate, torch.nan))
        return {self.resolve_name("rate", self.rate): grad_rate}

class Uniform(Distribution):
    def __init__(self, low, high):
        super().__init__(LogitTransform(low=low, high=high), ContinuousRange(low=low, high=high), ContinuousReal())
        self.low = low
        self.high = high
        self.add_dependency(low)
        self.add_dependency(high)
    
    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        low = resolve(self.low)
        high = resolve(self.high)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, torch.full_like(x, 1 / (high - low)), torch.zeros_like(x))
    
    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        low = resolve(self.low)
        high = resolve(self.high)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, -torch.full_like(x, high - low).log(), torch.full_like(x, -torch.inf))
    
    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, torch.zeros_like(x), torch.full_like(x, torch.nan))
    
    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")

class InvGamma(Distribution):
    def __init__(self, alpha, beta):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.alpha = alpha
        self.beta = beta
        self.add_dependency(alpha)
        self.add_dependency(beta)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        log_gamma_alpha = torch.lgamma(alpha)
        prob = (beta ** alpha / torch.exp(log_gamma_alpha)) * x ** (-alpha - 1) * torch.exp(-beta / x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(prob))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        log_prob = alpha * torch.log(beta) - torch.lgamma(alpha) - (alpha + 1) * torch.log(x) - beta / x
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)
        grad = -(alpha + 1) / x + beta / x ** 2
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, grad, torch.full_like(grad, torch.nan))

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        alpha = resolve(self.alpha)
        beta = resolve(self.beta)

        grad_alpha = torch.log(beta) - torch.digamma(alpha) - torch.log(x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad_alpha = torch.where(mask, grad_alpha, torch.full_like(grad_alpha, torch.nan))
        grad_beta = alpha / beta - 1 / x
        grad_beta = torch.where(mask, grad_beta, torch.full_like(grad_beta, torch.nan))
        return {self.resolve_name("alpha", self.alpha): grad_alpha, self.resolve_name("beta", self.beta): grad_beta}

class HalfCauchy(Distribution):
    def __init__(self, scale):
        super().__init__(LogTransform(border=0, side="larger"), ContinuousPositive(), ContinuousReal())
        self.scale = scale
        self.add_dependency(scale)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        scale = resolve(self.scale)
        denom = math.pi * scale * (1 + (x / scale) ** 2)
        prob = 2.0 / denom
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(x))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        scale = resolve(self.scale)
        log_prob = math.log(2.0) - math.log(math.pi) - torch.log(scale) - torch.log(1 + (x / scale) ** 2)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        scale = resolve(self.scale)
        grad = -2 * x / (scale ** 2 + x ** 2)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, grad, torch.full_like(grad, torch.nan))

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        scale = resolve(self.scale)
        grad_scale = -1 / scale + 2 * x ** 2 / (scale * (x ** 2 + scale ** 2))
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad_scale = torch.where(mask, grad_scale, torch.full_like(grad_scale, torch.nan))
        return {self.resolve_name("scale", self.scale): grad_scale}

class Dirichlet(Distribution):  # TODO: make new class for the dirichlet state space before transformation
    def __init__(self, alpha):
        super().__init__(transform=SoftMaxTransform(dim=-1), state_space=ContinuousRange(0, 1), transformed_state_space=ContinuousReal())
        self.alpha = alpha
        self.add_dependency(alpha)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        alpha = resolve(self.alpha)
        alpha_sum = alpha.sum(-1, keepdim=True)
        norm_const = torch.exp(torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha_sum).squeeze(-1))
        prod = torch.prod(x ** (alpha - 1), dim=-1, keepdim=True)
        prob = prod / norm_const
        mask = torch.stack([torch.all(point >= 0) & torch.all(point <= 1) & torch.allclose(point.sum(), torch.ones_like(point.sum())) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(prob))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        alpha = resolve(self.alpha)
        alpha_sum = alpha.sum(-1, keepdim=True)
        log_norm_const = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha_sum).squeeze(-1)
        log_prob = ((alpha - 1) * x.log()).sum(-1, keepdim=True) - log_norm_const
        mask = torch.stack([torch.all(point >= 0) & torch.all(point <= 1) & torch.allclose(point.sum(), torch.ones_like(point.sum())) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        alpha = resolve(self.alpha)
        grad = (alpha - 1) / x
        mask = torch.stack([torch.all(point >= 0) & torch.all(point <= 1) & torch.allclose(point.sum(), torch.ones_like(point.sum())) for point in x]).unsqueeze(1)
        return torch.where(mask, grad, torch.full_like(grad, torch.nan))

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        alpha = resolve(self.alpha)
        digamma_alpha = torch.digamma(alpha)
        digamma_alpha_sum = torch.digamma(alpha.sum(-1, keepdim=True))
        grad_alpha = x.log() - digamma_alpha + digamma_alpha_sum
        mask = torch.stack([torch.all(point >= 0) & torch.all(point <= 1) & torch.allclose(point.sum(), torch.ones_like(point.sum())) for point in x]).unsqueeze(1)
        grad_alpha = torch.where(mask, grad_alpha, torch.full_like(grad_alpha, torch.nan))
        return {self.resolve_name("alpha", self.alpha): grad_alpha}



# ========================= DISCRETE =========================
class Geometric(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscretePositive(), DiscretePositive())
        self.p = p
        self.add_dependency(p)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        prob = (1 - p) ** (x - 1) * p
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(x))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        log_prob = (x - 1) * torch.log(1 - p) + torch.log(p)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        grad = (1 / p) - (x - 1) / (1 - p)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad = torch.where(mask, grad, torch.full_like(grad, torch.nan))
        return {self.resolve_name("p", self.p): grad}


class Bernoulli(Distribution):
    def __init__(self, p):
        super().__init__(IdentityTransform(), DiscreteRange(0, 1), DiscreteRange(0, 1))
        self.p = p
        self.add_dependency(p)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        prob = p ** x * (1 - p) ** (1 - x)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, prob, torch.zeros_like(prob))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        log_prob = x * torch.log(p) + (1 - x) * torch.log(1 - p)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        grad = (x / p - (1 - x) / (1 - p))
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad = torch.where(mask, grad, torch.full_like(grad, torch.nan))
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
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, self.log_pdf(x).exp(), torch.zeros_like(x))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        n = resolve(self.n)
        log_prob = self._log_binom_coeff(x) + x * torch.log(p) + (n - x) * torch.log(1 - p)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, log_prob, torch.full_like(log_prob, -torch.inf))

    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")
    
    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        p = resolve(self.p)
        n = resolve(self.n)
        grad = x / p - (n - x) / (1 - p)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        grad = torch.where(mask, grad, torch.full_like(grad, torch.nan))
        return {self.resolve_name("p", self.p): grad}

class DiscreteUniform(Distribution):
    def __init__(self, low, high):
        super().__init__(IdentityTransform(), DiscreteRange(low=low, high=high), DiscreteRange(low=low, high=high))
        self.low = low
        self.high = high
        self.add_dependency(low)
        self.add_dependency(high)
    
    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        low = resolve(self.low)
        high = resolve(self.high)
        prob = 1.0 / (high - low + 1)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, torch.full_like(x, prob), torch.zeros_like(x))
    
    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, 1).")
        
        low = resolve(self.low)
        high = resolve(self.high)
        log_prob = -torch.log(high - low + 1)
        mask = torch.tensor([self.state_space.contains(point) for point in x]).unsqueeze(1)
        return torch.where(mask, torch.full_like(x, log_prob), torch.full_like(x, -torch.inf))
    
    def log_pdf_grad(self, x):
        raise NotImplementedError("Gradient w.r.t. x is undefined for discrete distributions.")
    
    def log_pdf_param_grads(self, x):
        raise RuntimeError("The parameters of the discrete uniform distribution are not differentiable. Consider using the metropolis sampler instead of NUTS if the likelihood is uniform.")


# ========================= Mixture of distributions =========================
class Mixture(Distribution):
    def __init__(self, components, weights):
        super().__init__(IdentityTransform(), Union(*[component.state_space for component in components]), Union(*[component.transformed_state_space for component in components]))

        self.components = components
        self.weights = weights

        # Add the dependencies of this class
        self.add_dependency(weights)
        for component in components:
            for param in component.random_parameters:
                self.random_parameters.add(param)
            for param in component.deterministic_parameters:
                self.deterministic_parameters.add(param)

    def pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        weights = resolve(self.weights)
        return sum(w * component.pdf(x) for w, component in zip(weights, self.components))

    def log_pdf(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        weights = resolve(self.weights)
        log_pdfs = torch.stack([(w + 1e-12).log() + component.log_pdf(x) for w, component in zip(weights.T, self.components)])  # TODO: make sure weights.T is correct
        return _logsumexp(log_pdfs, dim=0)

    def log_pdf_grad(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        weights = resolve(self.weights)
        
        # TODO: Change the distributions to always return consistant shapes
        log_weights = torch.log(weights + 1e-12)
        log_pdfs = torch.stack([component.log_pdf(x) for component in self.components], dim=0).squeeze()
        if log_pdfs.ndim == 0: log_pdfs = log_pdfs.unsqueeze(0)  # fix the inconsistent shapes in the log_pdf functions of distributions
        # log_pdfs.shape should be (n_components,)
        grads = torch.stack([component.log_pdf_grad(x) for component in self.components], dim=0).squeeze()
        if grads.ndim == 0: grads = grads.unsqueeze(0)  # fix the inconsistent shapes in the log_pdf functions of distributions
        if grads.ndim == 1: grads = grads.unsqueeze(1)
        # grads.shape should be (n_components, 1)

        log_weighted = log_weights + log_pdfs
        log_mixture_pdf = _logsumexp(log_weighted, dim=0)
        log_posterior_weights = log_weighted - log_mixture_pdf
        posterior_weights = torch.exp(log_posterior_weights)
        return (posterior_weights.unsqueeze(-1) * grads).sum(dim=0)

    def log_pdf_param_grads(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x.shape should be (n_samples, n_features).")
        
        weights = resolve(self.weights)
        grads = {}
        
        # gradient of weights
        pdfs = torch.stack([component.pdf(x) for component in self.components]).squeeze()
        if pdfs.ndim == 0: pdfs = pdfs.unsqueeze(0)
        grad_weights = pdfs / self.pdf(x)
        grads[self.resolve_name("weights", self.weights)] = grad_weights
    
        # TODO: implement and test gradients of the components' parameters
        # for w, comp in zip(self.weights, self.components):
        #     param_grads = comp.log_pdf_param_grads(x)
        #     for k, v in param_grads.items():
        #         if k in grads:
        #             grads[k] += w * comp.pdf(x) * v
        #         else:
        #             grads[k] = w * comp.pdf(x) * v
        # # Normalize by mixture PDF for gradient of log
        # mixture_pdf = self.pdf(x)
        # for k in grads:
        #     grads[k] /= mixture_pdf
        return grads

def _logsumexp(log_values, dim=0):
    max_val, _ = torch.max(log_values, dim=dim, keepdim=True)
    sum_exp = torch.sum(torch.exp(log_values - max_val), dim=dim, keepdim=True)
    return (max_val + torch.log(sum_exp)).squeeze(dim)
