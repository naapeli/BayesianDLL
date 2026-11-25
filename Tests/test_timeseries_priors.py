"""Gaussian random walk and autoregressive distributions over a series."""

import math
import warnings

import numpy as np
import pytest
import torch
from scipy.linalg import toeplitz
from scipy.signal import lfilter
from scipy.stats import multivariate_normal

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import HalfNormal, Normal, Uniform
from BayesianDLL.TimeSeries import AutoRegressive, GaussianRandomWalk, is_stationary
from test_vectorized import chain_by_chain, random_values

t = torch.tensor


def ar_autocovariance(phi, sigma, n):
    """Toeplitz covariance of a stationary AR process from its MA(infinity) weights (independent of the Lyapunov solve)."""
    psi = lfilter([1.0], np.r_[1.0, -np.asarray(phi)], np.r_[1.0, np.zeros(4000)])
    gamma = np.array([sigma ** 2 * psi[: len(psi) - h] @ psi[h:] for h in range(n)])
    return toeplitz(gamma)


def autograd_grads(make, x, params):
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    xg = x.clone().requires_grad_(True)
    make(**leaves).log_pdf(xg).sum().backward()
    return xg.grad, {k: v.grad for k, v in leaves.items()}


# ============================================================ Gaussian random walk
def grw_covariance(T, sigma, s0):
    idx = np.arange(T)
    return s0 ** 2 + sigma ** 2 * np.minimum.outer(idx, idx)


@pytest.mark.parametrize("init_sigma", [None, 2.0])
def test_random_walk_matches_dense_normal(init_sigma):
    T, sigma, mu, m0 = 7, 0.8, 0.3, -0.5
    s0 = sigma if init_sigma is None else init_sigma
    x = torch.randn(T)
    d = GaussianRandomWalk(sigma, mu=mu, init_mu=m0, init_sigma=init_sigma, T=T)
    mean = m0 + mu * np.arange(T)
    ref = multivariate_normal(mean=mean, cov=grw_covariance(T, sigma, s0)).logpdf(x.numpy())
    assert float(d.log_pdf(x)) == pytest.approx(ref, abs=1e-10)
    torch.testing.assert_close(d.mean(), t(mean))
    torch.testing.assert_close(d.variance(), t(np.diag(grw_covariance(T, sigma, s0))))


@pytest.mark.parametrize("init_sigma", [None, 2.0])
def test_random_walk_gradients_match_autograd(init_sigma):
    x = torch.randn(3, 6)
    params = {"sigma": t(0.8), "mu": t(0.3), "init_mu": t(-0.5)}
    if init_sigma:
        params["init_sigma"] = t(init_sigma)
    make = lambda **p: GaussianRandomWalk(**p, T=6)
    gx, gp = autograd_grads(make, x, params)
    d = make(**params)
    torch.testing.assert_close(d.log_pdf_grad(x), gx)
    grads = d.log_pdf_param_grads(x)
    for k in params:
        torch.testing.assert_close(grads[k].sum(), gp[k])


def test_random_walk_batched_parameters_and_sampling():
    d = GaussianRandomWalk(t([0.5, 2.0]), mu=t([0.0, 0.1]), T=5)
    assert d.batch_shape == (2,) and d.event_shape == (5,)
    s = d.sample(40000)
    assert s.shape == (40000, 2, 5)
    torch.testing.assert_close(s.mean(0), d.mean(), atol=0.04, rtol=0)
    torch.testing.assert_close(s.var(0), d.variance(), rtol=0.05, atol=0)
    lp = d.log_pdf(s[:3])
    assert lp.shape == (3, 2)
    one = GaussianRandomWalk(0.5, mu=0.0, T=5)
    torch.testing.assert_close(d.log_pdf(s[:3])[:, 0], one.log_pdf(s[:3, 0]))


# ============================================================ autoregressive
PHIS = [[0.6], [0.5, -0.3], [0.4, 0.2, -0.1], [-0.3, 0.1, 0.2, 0.1]]


@pytest.mark.parametrize("phi", PHIS)
def test_ar_stationary_matches_dense_normal(phi):
    n, sigma, mu = 10, 1.3, 0.7
    x = torch.randn(n)
    d = AutoRegressive(t(phi), sigma, mu, T=n)
    ref = multivariate_normal(mean=np.full(n, mu), cov=ar_autocovariance(phi, sigma, n)).logpdf(x.numpy())
    assert float(d.log_pdf(x)) == pytest.approx(ref, abs=1e-10)
    assert float(d.variance()[0]) == pytest.approx(ar_autocovariance(phi, sigma, 1)[0, 0], rel=1e-10)


@pytest.mark.parametrize("phi", PHIS)
@pytest.mark.parametrize("init", ["stationary", "conditional"])
def test_ar_gradients_match_autograd(phi, init):
    x = torch.randn(2, 12)
    params = {"coefficients": t(phi), "sigma": t(1.3), "mu": t(0.4)}
    make = lambda coefficients, sigma, mu: AutoRegressive(coefficients, sigma, mu, init=init)
    gx, gp = autograd_grads(make, x, params)
    d = make(**params)
    torch.testing.assert_close(d.log_pdf_grad(x), gx, rtol=1e-9, atol=1e-10)
    grads = d.log_pdf_param_grads(x)
    torch.testing.assert_close(grads["coefficients"].sum(0), gp["coefficients"], rtol=1e-9, atol=1e-10)
    for k in ("sigma", "mu"):
        torch.testing.assert_close(grads[k].sum(), gp[k], rtol=1e-9, atol=1e-10)


def test_ar_conditional_equals_least_squares_gaussian_likelihood():
    phi = [0.5, -0.3]
    x = torch.randn(30)
    d = AutoRegressive(t(phi), 1.1, 0.2, init="conditional")
    z = x.numpy() - 0.2
    e = z[2:] - phi[0] * z[1:-1] - phi[1] * z[:-2]
    ref = (-0.5 * np.log(2 * np.pi * 1.1 ** 2) - 0.5 * e ** 2 / 1.1 ** 2).sum()
    assert float(d.log_pdf(x)) == pytest.approx(ref, abs=1e-10)


def test_ar_outside_stationary_region_is_minus_infinity_with_zero_gradients():
    x = torch.randn(8)
    for phi in ([1.2], [0.5, 0.7], [-1.0]):
        d = AutoRegressive(t(phi), 1.0)
        assert float(d.log_pdf(x)) == -math.inf
        assert all(torch.all(g == 0) for g in d.log_pdf_param_grads(x).values())
        assert torch.all(d.log_pdf_grad(x) == 0)
        assert not bool(is_stationary(t(phi)))
    assert bool(is_stationary(t([0.5, -0.3])))
    # a conditional likelihood needs no stationarity
    assert math.isfinite(float(AutoRegressive(t([1.2]), 1.0, init="conditional").log_pdf(x)))


def test_ar_batched_coefficients_agree_with_a_loop():
    phi = t([[0.6, 0.0], [0.5, -0.3], [-0.2, 0.3]])
    sigma = t([1.0, 0.5, 2.0])
    x = torch.randn(4, 3, 9)
    d = AutoRegressive(phi, sigma, 0.1, T=9)
    assert d.batch_shape == (3,)
    lp = d.log_pdf(x)
    assert lp.shape == (4, 3)
    for b in range(3):
        one = AutoRegressive(phi[b], sigma[b], 0.1)
        torch.testing.assert_close(lp[:, b], one.log_pdf(x[:, b]))
        grads = d.log_pdf_param_grads(x)
        torch.testing.assert_close(grads["coefficients"][:, b], one.log_pdf_param_grads(x[:, b])["coefficients"])


def test_ar_sampling_reproduces_stationary_moments():
    phi, sigma, mu = [0.6, -0.25], 1.2, 0.5
    d = AutoRegressive(t(phi), sigma, mu, T=6)
    s = d.sample(60000)
    assert s.shape == (60000, 6)
    cov = ar_autocovariance(phi, sigma, 6)
    emp = np.cov(s.numpy().T)
    np.testing.assert_allclose(emp, cov, atol=0.06)
    np.testing.assert_allclose(s.mean(0).numpy(), np.full(6, mu), atol=0.03)
    # the sample is distributed according to log_pdf: its mean log density is minus the entropy
    entropy = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * cov)[1]
    assert float(d.log_pdf(s).mean()) == pytest.approx(-entropy, abs=0.05)
    with pytest.raises(ValueError):
        AutoRegressive(t([1.1]), 1.0, T=4).sample(2)


def test_series_length_is_needed_for_sampling_only():
    d = AutoRegressive(t([0.5]), 1.0)
    assert d.event_shape == ()
    assert math.isfinite(float(d.log_pdf(torch.randn(5))))
    with pytest.raises(ValueError):
        d.sample(2)
    with pytest.raises(ValueError):
        AutoRegressive(t([0.5, 0.2]), 1.0, init="conditional").log_pdf(torch.randn(2))


# ============================================================ models
def ar_model():
    x = torch.sin(torch.arange(15.0) * 0.7) + 0.3 * torch.randn(15, generator=torch.Generator().manual_seed(2))
    with Model() as model:
        phi = RandomParameter("phi", Uniform(-0.8, 0.8), shape=2)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("x", AutoRegressive(phi, sigma, mu, init="conditional"), x)
    return model


def ar_stationary_model():
    x = torch.sin(torch.arange(15.0) * 0.7) + 0.3 * torch.randn(15, generator=torch.Generator().manual_seed(2))
    with Model() as model:
        phi = RandomParameter("phi", Uniform(-0.4, 0.4), shape=2)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        ObservedParameter("x", AutoRegressive(phi, sigma, 0.2), x)
    return model


def random_walk_model():
    x = torch.cumsum(0.4 * torch.randn(12, generator=torch.Generator().manual_seed(3)), 0)
    with Model() as model:
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("x", GaussianRandomWalk(sigma, mu=mu), x)
    return model


def latent_series_model():
    y = torch.randn(10, generator=torch.Generator().manual_seed(4))
    with Model() as model:
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        level = RandomParameter("level", GaussianRandomWalk(sigma, T=10))
        ObservedParameter("y", Normal(level, 0.5), y)
    return model


ZOO = {"ar_conditional": ar_model, "ar_stationary": ar_stationary_model, "random_walk": random_walk_model,
       "latent_series": latent_series_model}


@pytest.mark.parametrize("name", list(ZOO))
def test_batched_models_equal_chain_by_chain_and_finite_differences(name):
    model = ZOO[name]()
    n_chains = 4
    values = random_values(model, n_chains, torch.Generator().manual_seed(5))
    names = list(model.params)
    expected_lp, expected_grads = chain_by_chain(model, values, names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(n_chains, values=values):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            lp = model.model_log_prob()
            grads = model.joint_grad_log_prob(names)
    torch.testing.assert_close(lp, expected_lp, rtol=1e-12, atol=1e-12)
    for n in names:
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=1e-9, atol=1e-10)
    # finite differences of the unbatched model
    for n in names:
        u = model.params[n].unconstrained_value.clone()
        flat = u.reshape(-1)
        fd = torch.zeros_like(flat)
        for i in range(flat.numel()):
            for sign in (1, -1):
                v = flat.clone()
                v[i] += sign * 1e-6
                model.params[n].set_unconstrained_value(v.reshape(u.shape))
                fd[i] += sign * float(model.model_log_prob()) / 2e-6
        model.params[n].set_unconstrained_value(u)
        torch.testing.assert_close(model.joint_grad_log_prob(names)[n].reshape(-1), fd, rtol=1e-5, atol=1e-6)


def _posterior(model, n=500, warmup=400, chains=2, seed=5):
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample(n, warmup, n_chains=chains, model=model, progress_bar=False, seed=seed)


def _covers(draws, truth, k=3.5):
    d = draws.reshape(-1, *draws.shape[2:])
    return bool(torch.all((d.mean(0) - truth).abs() < k * d.std(0)))


def test_nuts_recovers_ar2_parameters_with_the_stationary_likelihood():
    truth = t([0.5, -0.3])
    x = AutoRegressive(truth, 0.8, 1.0, T=400).sample(1)[0]
    with Model() as model:
        phi = RandomParameter("phi", Uniform(-0.9, 0.9), shape=2)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 3.0))
        ObservedParameter("x", AutoRegressive(phi, sigma, mu), x)
    result = _posterior(model)
    assert _covers(result["phi"], truth) and _covers(result["sigma"], t(0.8)) and _covers(result["mu"], t(1.0))


def test_nuts_recovers_random_walk_drift_and_scale():
    x = GaussianRandomWalk(0.5, mu=0.2, T=300).sample(1)[0]
    with Model() as model:
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("x", GaussianRandomWalk(sigma, mu=mu), x)
    result = _posterior(model)
    assert _covers(result["sigma"], t(0.5)) and _covers(result["mu"], t(0.2))


def test_latent_random_walk_prior_gives_the_conjugate_posterior_mean():
    """A random-walk prior and Gaussian noise: the posterior mean of the latent series is the exact smoother."""
    T, sigma, tau = 6, 0.7, 0.5
    y = torch.randn(T, generator=torch.Generator().manual_seed(9))
    cov = grw_covariance(T, sigma, sigma)
    prior_precision = np.linalg.inv(cov)
    precision = prior_precision + np.eye(T) / tau ** 2
    mean = np.linalg.solve(precision, y.numpy() / tau ** 2)
    with Model() as model:
        level = RandomParameter("level", GaussianRandomWalk(sigma, T=T))
        ObservedParameter("y", Normal(level, tau), y)
    result = _posterior(model, n=1500, warmup=500)
    d = result["level"].reshape(-1, T)
    sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    assert np.all(np.abs(d.mean(0).numpy() - mean) < 0.15 * sd + 0.02)
    assert np.all(np.abs(d.std(0).numpy() / sd - 1) < 0.15)


def test_time_series_priors_support_prior_and_posterior_predictive():
    from BayesianDLL import prior_predicative
    with Model() as model:
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        phi = RandomParameter("phi", Uniform(-0.8, 0.8), shape=1)
        ObservedParameter("x", AutoRegressive(phi, sigma, T=20), torch.zeros(20))
    result = prior_predicative(300, samples_per_step=1, model=model, progress_bar=False)
    assert result["x"].shape[-1] == 20 and torch.isfinite(result["x"]).all()
