"""Curvature of log densities w.r.t. their parameters (used by the Gauss-Newton Laplace approximation).

``log_pdf_param_curvature(x)`` of a distribution returns the Fisher information
of the parameters of ``log_pdf`` (the expected negative Hessian, positive
semi-definite). Every unordered pair of parameters appears once, keyed by the
names used in :meth:`~BayesianDLL.Distributions.Distribution.log_pdf_param_grads`.
A block has the broadcast shape of ``log_pdf(x)`` followed by the event
dimensions of the first and of the second parameter. Pairs that are zero are
left out.

A distribution without an entry here raises ``NotImplementedError``; the
Laplace approximation then differentiates that term by finite differences.
"""

import torch

from ._distributions import (
    Bernoulli, Beta, Binomial, Categorical, Cauchy, Exponential, Gamma, Laplace, LogNormal,
    Multinomial, MultivariateNormal, Normal, Poisson, StudentT, _as_float,
)
from ._resolve import resolve

_REGISTRY = {}


def _register(*classes):
    def decorator(function):
        for cls in classes:
            _REGISTRY[cls] = function
        return function
    return decorator


def parameter_curvature(distribution, x):
    """Fisher information of ``distribution`` w.r.t. its parameters, ``{(name_a, name_b): block}``."""
    for cls in type(distribution).__mro__:
        if cls in _REGISTRY:
            return _REGISTRY[cls](distribution, x)
    raise NotImplementedError(f"{type(distribution).__name__} has no parameter curvature.")


@_register(Normal, LogNormal)
def _normal(d, x):
    _, sigma = _as_float(d.mu, d.sigma)
    return {(d.resolve_name("mean", d.mu),) * 2: 1 / sigma ** 2,
            (d.resolve_name("sigma", d.sigma),) * 2: 2 / sigma ** 2}


@_register(Laplace)
def _laplace(d, x):
    _, scale = _as_float(d.mu, d.scale)
    return {(d.resolve_name("mean", d.mu),) * 2: 1 / scale ** 2,
            (d.resolve_name("scale", d.scale),) * 2: 1 / scale ** 2}


@_register(Cauchy)
def _cauchy(d, x):
    _, scale = _as_float(d.mu, d.scale)
    return {(d.resolve_name("mean", d.mu),) * 2: 0.5 / scale ** 2,
            (d.resolve_name("scale", d.scale),) * 2: 0.5 / scale ** 2}


@_register(StudentT)
def _student_t(d, x):
    nu, _, sigma = _as_float(d.nu, d.mu, d.sigma)
    nu_key, mu_key, sigma_key = d.resolve_name("nu", d.nu), d.resolve_name("mean", d.mu), d.resolve_name("sigma", d.sigma)
    trigamma = lambda z: torch.polygamma(1, z)
    return {
        (mu_key, mu_key): (nu + 1) / ((nu + 3) * sigma ** 2),
        (sigma_key, sigma_key): 2 * nu / ((nu + 3) * sigma ** 2),
        (nu_key, sigma_key): -2 / (sigma * (nu + 1) * (nu + 3)),
        (nu_key, nu_key): 0.25 * (trigamma(nu / 2) - trigamma((nu + 1) / 2)) - (nu + 5) / (2 * nu * (nu + 1) * (nu + 3)),
    }


@_register(Exponential)
def _exponential(d, x):
    return {(d.resolve_name("rate", d.rate),) * 2: 1 / _as_float(d.rate) ** 2}


@_register(Gamma)
def _gamma(d, x):
    alpha, beta = _as_float(d.alpha, d.beta)
    a, b = d.resolve_name("alpha", d.alpha), d.resolve_name("beta", d.beta)
    return {(a, a): torch.polygamma(1, alpha), (a, b): -1 / beta, (b, b): alpha / beta ** 2}


@_register(Beta)
def _beta(d, x):
    a_value, b_value = _as_float(d.a, d.b)
    a, b = d.resolve_name("a", d.a), d.resolve_name("b", d.b)
    joint = torch.polygamma(1, a_value + b_value)
    return {(a, a): torch.polygamma(1, a_value) - joint, (a, b): -joint, (b, b): torch.polygamma(1, b_value) - joint}


@_register(Poisson)
def _poisson(d, x):
    return {(d.resolve_name("rate", d.rate),) * 2: 1 / _as_float(d.rate)}


@_register(Bernoulli, Binomial)
def _bernoulli(d, x):
    count = d._count(torch.as_tensor(x)) if isinstance(d, Binomial) else 1
    if d.logits is not None:
        p = torch.sigmoid(_as_float(d.logits))
        return {(d.resolve_name("logits", d.logits),) * 2: count * p * (1 - p)}
    p = _as_float(d.probs)
    return {(d.resolve_name("probs", d.probs),) * 2: count / (p * (1 - p))}


@_register(Categorical, Multinomial)
def _categorical(d, x):
    probs = d.logits is None
    p = _as_float(d.probs if probs else d.logits)
    count = 1
    if isinstance(d, Multinomial):
        count = torch.as_tensor(resolve(d.total_count)).to(dtype=p.dtype, device=p.device)
        count = count[..., None, None] if count.ndim else count
    if not probs:
        p = torch.softmax(p, dim=-1)
        return {(d.resolve_name("logits", d.logits),) * 2: count * (torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2))}
    total = p.sum(-1, keepdim=True)
    return {(d.resolve_name("probs", d.probs),) * 2:
            count * (torch.diag_embed(1 / (total * p)) - (1 / total ** 2).unsqueeze(-1))}


@_register(MultivariateNormal)
def _multivariate_normal(d, x):
    # Only the mean block; a latent covariance makes the caller fall back to finite differences.
    _, L = d._params()
    return {(d.resolve_name("mean", d.mu),) * 2: torch.cholesky_inverse(L)}
