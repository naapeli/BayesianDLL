"""Ordered logistic, censored, hurdle, beta-binomial and Dirichlet-multinomial distributions, and the new CDFs.

``scipy`` and ``torch.autograd`` are reference oracles only.
"""

import math
import warnings

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import (
    Beta, BetaBinomial, Censored, DirichletMultinomial, Exponential, Gamma, HalfNormal, Hurdle, LogNormal,
    NegativeBinomial, Normal, Ordered, OrderedLogistic, Poisson, StudentT, ZeroInflated,
)
from test_vectorized import chain_by_chain, random_values

t = torch.tensor


def autograd_param_grads(make, x, params):
    """``{name: d sum log_pdf / d param}`` by autograd, for ``make(**params)``."""
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    make(**leaves).log_pdf(x).sum().backward()
    return {k: v.grad for k, v in leaves.items()}


def hand_param_grads(make, x, params):
    grads = make(**params).log_pdf_param_grads(x)
    out = {}
    for k, v in params.items():
        g = grads[k]
        while g.ndim > v.ndim:
            g = g.sum(0)
        out[k] = g
    return out


# ============================================================ new CDFs
@pytest.mark.parametrize("a, b", [(2.5, 0.7), (0.4, 3.0), (30.0, 45.0), (300.0, 200.0)])
def test_beta_cdf_matches_scipy(a, b):
    x = torch.linspace(0.01, 0.99, 23)
    torch.testing.assert_close(Beta(a, b).cdf(x), t(st.beta(a, b).cdf(x.numpy())), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("nu", [0.8, 3.5, 40.0])
def test_student_t_cdf_matches_scipy(nu):
    x = torch.linspace(-6, 6, 25)
    torch.testing.assert_close(StudentT(nu, 0.2, 1.3).cdf(x), t(st.t(nu, 0.2, 1.3).cdf(x.numpy())), rtol=1e-9, atol=1e-12)


CDF_GRAD_CASES = {
    "normal": (lambda mean, sigma: Normal(mean, sigma), {"mean": 0.3, "sigma": 1.4}, [-1.0, 0.5, 2.0]),
    "lognormal": (lambda mean, sigma: LogNormal(mean, sigma), {"mean": 0.3, "sigma": 0.8}, [0.4, 1.5, 3.0]),
    "halfnormal": (lambda sigma: HalfNormal(sigma), {"sigma": 1.4}, [0.4, 1.5, 3.0]),
    "exponential": (lambda rate: Exponential(rate), {"rate": 1.7}, [0.4, 1.5, 3.0]),
    "beta": (lambda a, b: Beta(a, b), {"a": 2.2, "b": 1.4}, [0.1, 0.5, 0.9]),
    "student": (lambda nu, mean, sigma: StudentT(nu, mean, sigma), {"nu": 3.5, "mean": 0.2, "sigma": 1.3}, [-2.0, 0.4, 3.0]),
}


@pytest.mark.parametrize("case", list(CDF_GRAD_CASES))
def test_cdf_param_grads_match_autograd(case):
    make, params, xs = CDF_GRAD_CASES[case]
    params = {k: t(v) for k, v in params.items()}
    x = t(xs)
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    make(**leaves).cdf(x).sum().backward()
    # constant parameters are keyed by the library's fixed labels
    grads = make(**params).cdf_param_grads(x)
    for k in params:
        label = {"rate": "rate", "mean": "mean"}.get(k, k)
        torch.testing.assert_close(grads[label].sum(), leaves[k].grad, rtol=1e-6, atol=1e-8)


def test_gamma_cdf_param_grads_against_finite_differences():
    x = t([0.4, 1.5, 3.0])
    alpha, beta = 2.3, 1.7
    grads = Gamma(alpha, beta).cdf_param_grads(x)
    h = 1e-6
    fd_a = (st.gamma(alpha + h, scale=1 / beta).cdf(x.numpy()) - st.gamma(alpha - h, scale=1 / beta).cdf(x.numpy())) / (2 * h)
    fd_b = (st.gamma(alpha, scale=1 / (beta + h)).cdf(x.numpy()) - st.gamma(alpha, scale=1 / (beta - h)).cdf(x.numpy())) / (2 * h)
    np.testing.assert_allclose(grads["alpha"].numpy(), fd_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(grads["beta"].numpy(), fd_b, rtol=1e-5, atol=1e-8)


def test_survival_function_is_complement_of_cdf_and_accurate_in_the_tail():
    d = Normal(0.0, 1.0)
    torch.testing.assert_close(d.sf(t([40.0])), t(st.norm.sf([40.0])), rtol=1e-10, atol=0)
    torch.testing.assert_close(Gamma(2.0, 1.0).sf(t([1.0])), 1 - Gamma(2.0, 1.0).cdf(t([1.0])))


# ============================================================ ordered logistic
CUTS = [-1.2, 0.1, 1.5]


def ordered_logistic_reference(y, eta, cuts):
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    return np.log(sp.expit(edges[y + 1] - eta) - sp.expit(edges[y] - eta))


def test_ordered_logistic_log_pdf_matches_reference():
    eta = t([-0.7, 0.3, 2.1, 0.0])
    y = t([0, 2, 3, 1])
    d = OrderedLogistic(eta, t(CUTS))
    expected = ordered_logistic_reference(y.numpy(), eta.numpy(), np.array(CUTS))
    torch.testing.assert_close(d.log_pdf(y), t(expected))
    torch.testing.assert_close(d.probs().sum(-1), torch.ones(4))
    assert d.log_pdf(t([4.0, -1.0, 0.5, 1.0])).tolist()[:3] == [-math.inf] * 3


def test_ordered_logistic_is_stable_for_extreme_predictors():
    d = OrderedLogistic(t([-60.0, 60.0, 0.0]), t(CUTS))
    lp = d.log_pdf(t([3.0, 0.0, 2.0]))
    assert torch.isfinite(lp).all()
    g = d.log_pdf_param_grads(t([3.0, 0.0, 2.0]))
    assert all(torch.isfinite(v).all() for v in g.values())
    # a middle category far from the predictor: log P ~ -|distance|
    d = OrderedLogistic(t([200.0]), t([-1.0, 1.0]))
    assert torch.isfinite(d.log_pdf(t([1.0])))


@pytest.mark.parametrize("y", [0.0, 1.0, 2.0, 3.0])
def test_ordered_logistic_param_grads_match_autograd(y):
    x = t([y, y])
    params = {"eta": t([0.4, -0.9]), "cutpoints": t(CUTS)}
    make = lambda eta, cutpoints: OrderedLogistic(eta, cutpoints)
    expected = autograd_param_grads(make, x, params)
    got = hand_param_grads(make, x, params)
    for k in params:
        torch.testing.assert_close(got[k], expected[k])


def test_ordered_logistic_batched_cutpoints_and_sampling_frequencies():
    eta = t([0.0, 1.0])
    cuts = t([[-1.0, 0.5, 2.0], [-0.5, 0.0, 1.0]])
    d = OrderedLogistic(eta, cuts)
    assert d.batch_shape == (2,)
    draws = d.sample(60000)
    assert draws.shape == (60000, 2)
    freq = torch.stack([(draws == k).double().mean(0) for k in range(4)], -1)
    torch.testing.assert_close(freq, d.probs(), atol=0.01, rtol=0)
    torch.testing.assert_close(draws.mean(0), d.mean(), atol=0.02, rtol=0)
    torch.testing.assert_close(draws.var(0), d.variance(), atol=0.03, rtol=0)


# ============================================================ censored
@pytest.mark.parametrize("make", [
    lambda mean, sigma: Normal(mean, sigma),
    lambda mean, sigma: LogNormal(mean, sigma),
    lambda mean, sigma: StudentT(4.0, mean, sigma),
])
def test_censored_values_gradients_and_limits(make):
    low, high = t(0.6), t(2.5)
    x = t([0.6, 0.9, 1.7, 2.5, 0.4, 3.0])
    params = {"mean": t(0.3), "sigma": t(0.9), "low": low, "high": high}
    build = lambda mean, sigma, low, high: Censored(make(mean, sigma), low=low, high=high)
    dist = build(**params)
    base = make(params["mean"], params["sigma"])
    expected = torch.where(x <= low, torch.log(base.cdf(low)),
                           torch.where(x >= high, torch.log(1 - base.cdf(high)), base.log_pdf(x)))
    torch.testing.assert_close(dist.log_pdf(x), expected)
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    build(**leaves).log_pdf(x).sum().backward()
    grads = dist.log_pdf_param_grads(x)
    for k in params:
        torch.testing.assert_close(grads[k].sum(), leaves[k].grad, rtol=1e-8, atol=1e-10)
    xg = x.clone().requires_grad_(True)
    build(**params).log_pdf(xg).sum().backward()
    torch.testing.assert_close(dist.log_pdf_grad(x), xg.grad)


def test_censored_one_sided_matches_scipy_and_normalises():
    d = Censored(Normal(0.5, 1.5), high=2.0)
    x = t([-1.0, 1.0, 2.0])
    expected = [st.norm.logpdf(-1.0, 0.5, 1.5), st.norm.logpdf(1.0, 0.5, 1.5), st.norm.logsf(2.0, 0.5, 1.5)]
    torch.testing.assert_close(d.log_pdf(x), t(expected))
    # the density integrates, with the point mass at the limit, to one
    grid = torch.linspace(-12, 2.0, 200001)[:-1]
    mass = torch.trapezoid(torch.exp(d.log_pdf(grid)), grid) + math.exp(float(d.log_pdf(t(2.0))))
    assert abs(float(mass) - 1) < 1e-4


def test_censored_sampling_clips_at_the_limits():
    d = Censored(Normal(0.0, 1.0), low=-0.5, high=1.0)
    s = d.sample(40000)
    assert float(s.min()) == -0.5 and float(s.max()) == 1.0
    assert abs(float((s == 1.0).double().mean()) - st.norm.sf(1.0)) < 0.01
    assert abs(float((s == -0.5).double().mean()) - st.norm.cdf(-0.5)) < 0.01


def test_censored_requires_cdf_param_grads_for_random_base_parameters():
    from BayesianDLL.Distributions import Cauchy, Uniform  # noqa: F401
    with Model():
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        d = Censored(Uniform(mu, 5.0), high=4.0)
        with pytest.raises(NotImplementedError):
            d.log_pdf_param_grads(t([1.0, 4.0]))


# ============================================================ hurdle
def hurdle_poisson_reference(x, rate, pi):
    x = np.asarray(x)
    pos = np.log1p(-pi) + st.poisson.logpmf(x, rate) - np.log1p(-np.exp(-rate))
    return np.where(x == 0, np.log(pi), pos)


def test_hurdle_poisson_matches_reference_and_normalises():
    d = Hurdle(Poisson(2.3), 0.35)
    x = t([0.0, 1.0, 2.0, 5.0, 9.0])
    torch.testing.assert_close(d.log_pdf(x), t(hurdle_poisson_reference(x.numpy(), 2.3, 0.35)))
    k = torch.arange(0, 80.0)
    assert abs(float(d.pdf(k).sum()) - 1) < 1e-12
    assert d.log_pdf(t([-1.0, 1.5])).tolist() == [-math.inf, -math.inf]


def test_hurdle_differs_from_zero_inflated():
    base = Poisson(2.3)
    h, z = Hurdle(base, 0.35), ZeroInflated(base, 0.35)
    assert float(h.pdf(t(0.0))) == pytest.approx(0.35)
    assert float(z.pdf(t(0.0))) == pytest.approx(0.35 + 0.65 * math.exp(-2.3))


@pytest.mark.parametrize("base_case", ["poisson", "nbinom"])
def test_hurdle_param_grads_match_autograd(base_case):
    x = t([0.0, 1.0, 3.0, 0.0, 6.0])
    if base_case == "poisson":
        params = {"rate": t(2.1), "zero_prob": t(0.3)}
        make = lambda rate, zero_prob: Hurdle(Poisson(rate), zero_prob)
    else:
        params = {"total_count": t(2.1), "probs": t(0.4), "zero_prob": t(0.3)}
        make = lambda total_count, probs, zero_prob: Hurdle(NegativeBinomial(total_count, probs=probs), zero_prob)
    expected = autograd_param_grads(make, x, params)
    grads = make(**params).log_pdf_param_grads(x)
    for k in params:
        torch.testing.assert_close(grads[k].sum(), expected[k], rtol=1e-8, atol=1e-10)


def test_hurdle_moments_and_sampling():
    d = Hurdle(Poisson(1.2), 0.4)
    s = d.sample(80000)
    assert abs(float((s == 0).double().mean()) - 0.4) < 0.01
    assert abs(float(s.mean()) - float(d.mean())) < 0.02
    assert abs(float(s.var()) - float(d.variance())) < 0.05
    k = torch.arange(0, 60.0)
    p = d.pdf(k)
    assert float((p * k).sum()) == pytest.approx(float(d.mean()), abs=1e-10)


# ============================================================ beta-binomial
def test_beta_binomial_matches_scipy_and_normalises():
    d = BetaBinomial(12, 2.5, 4.0)
    k = torch.arange(0, 13.0)
    torch.testing.assert_close(d.log_pdf(k), t(st.betabinom(12, 2.5, 4.0).logpmf(k.numpy())))
    assert float(d.pdf(k).sum()) == pytest.approx(1.0)
    assert d.log_pdf(t([-1.0, 13.0, 2.5])).tolist() == [-math.inf] * 3
    assert float(d.mean()) == pytest.approx(st.betabinom(12, 2.5, 4.0).mean())
    assert float(d.variance()) == pytest.approx(st.betabinom(12, 2.5, 4.0).var())


def test_beta_binomial_param_grads_match_autograd():
    x = t([0.0, 3.0, 9.0, 12.0])
    params = {"alpha": t(2.5), "beta": t(4.0)}
    make = lambda alpha, beta: BetaBinomial(12, alpha, beta)
    expected = autograd_param_grads(make, x, params)
    got = make(**params).log_pdf_param_grads(x)
    for k in params:
        torch.testing.assert_close(got[k].sum(), expected[k])


def test_beta_binomial_sampling_moments():
    d = BetaBinomial(10, 1.5, 2.5)
    s = d.sample(100000)
    assert abs(float(s.mean()) - float(d.mean())) < 0.03
    assert abs(float(s.var()) - float(d.variance())) < 0.1


# ============================================================ Dirichlet-multinomial
def test_dirichlet_multinomial_matches_scipy_and_normalises():
    alpha = t([0.8, 1.7, 2.6])
    d = DirichletMultinomial(6, alpha)
    counts = [(i, j, 6 - i - j) for i in range(7) for j in range(7 - i)]
    x = t(counts, dtype=torch.float64)
    expected = st.dirichlet_multinomial(alpha.numpy(), 6).logpmf(np.array(counts))
    torch.testing.assert_close(d.log_pdf(x), t(expected))
    assert float(d.pdf(x).sum()) == pytest.approx(1.0)
    assert d.log_pdf(t([[1.0, 1.0, 1.0]])).tolist() == [-math.inf]


def test_dirichlet_multinomial_param_grads_match_autograd():
    x = t([[2.0, 1.0, 3.0], [0.0, 0.0, 6.0]])
    params = {"concentration": t([0.8, 1.7, 2.6])}
    make = lambda concentration: DirichletMultinomial(6, concentration)
    expected = autograd_param_grads(make, x, params)["concentration"]
    got = make(**params).log_pdf_param_grads(x)["concentration"].sum(0)
    torch.testing.assert_close(got, expected)


def test_dirichlet_multinomial_sampling_moments_and_batch():
    alpha = t([[0.8, 1.7, 2.6], [3.0, 1.0, 1.0]])
    d = DirichletMultinomial(8, alpha)
    assert d.batch_shape == (2,) and d.event_shape == (3,)
    s = d.sample(60000)
    assert s.shape == (60000, 2, 3) and torch.all(s.sum(-1) == 8)
    torch.testing.assert_close(s.mean(0), d.mean(), atol=0.04, rtol=0)
    torch.testing.assert_close(s.var(0), d.variance(), atol=0.15, rtol=0)
    cov = d.covariance()
    emp = torch.einsum("sbi,sbj->bij", s - s.mean(0), s - s.mean(0)) / (s.shape[0] - 1)
    torch.testing.assert_close(emp, cov, atol=0.15, rtol=0)


# ============================================================ models: vectorized chains and inference
def ordinal_model():
    y = t([0.0, 2.0, 1.0, 3.0, 3.0, 1.0, 0.0, 2.0])
    x = t([-1.0, 0.3, -0.2, 1.5, 0.9, 0.1, -1.4, 0.5])
    with Model() as model:
        beta = RandomParameter("beta", Normal(0.0, 1.5))
        cuts = RandomParameter("cuts", Ordered(Normal(0.0, 2.0), 3), shape=3)
        from BayesianDLL.Deterministic import Multiply
        eta = Multiply("eta", x, beta)
        ObservedParameter("y", OrderedLogistic(eta, cuts), y)
    return model


def censored_model():
    y = t([0.5, 1.2, 2.0, 2.0, 0.5, 1.7, 0.9])
    with Model() as model:
        mu = RandomParameter("mu", Normal(1.0, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        ObservedParameter("y", Censored(Normal(mu, sigma), low=0.5, high=2.0), y)
    return model


def hurdle_model():
    y = t([0.0, 3.0, 1.0, 0.0, 0.0, 2.0, 5.0, 1.0])
    with Model() as model:
        rate = RandomParameter("rate", Gamma(2.0, 1.0))
        pi = RandomParameter("pi", Beta(2.0, 2.0))
        ObservedParameter("y", Hurdle(Poisson(rate), pi), y)
    return model


def beta_binomial_model():
    y = t([3.0, 7.0, 5.0, 9.0, 1.0])
    with Model() as model:
        a = RandomParameter("a", Gamma(2.0, 1.0))
        b = RandomParameter("b", Gamma(2.0, 1.0))
        ObservedParameter("y", BetaBinomial(10, a, b), y)
    return model


def dirichlet_multinomial_model():
    y = t([[3.0, 2.0, 5.0], [0.0, 4.0, 6.0], [5.0, 5.0, 0.0]])
    with Model() as model:
        conc = RandomParameter("conc", Gamma(2.0, 1.0), shape=3)
        ObservedParameter("y", DirichletMultinomial(10, conc), y)
    return model


EXTRA_ZOO = {
    "ordinal": ordinal_model, "censored": censored_model, "hurdle": hurdle_model,
    "beta_binomial": beta_binomial_model, "dirichlet_multinomial": dirichlet_multinomial_model,
}


@pytest.mark.parametrize("name", list(EXTRA_ZOO))
def test_batched_models_equal_chain_by_chain(name):
    model = EXTRA_ZOO[name]()
    n_chains = 4
    values = random_values(model, n_chains, torch.Generator().manual_seed(11))
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
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", list(EXTRA_ZOO))
def test_model_gradients_match_finite_differences(name):
    model = EXTRA_ZOO[name]()
    names = list(model.params)
    g = model.joint_grad_log_prob(names)
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
        torch.testing.assert_close(g[n].reshape(-1), fd, rtol=1e-5, atol=1e-6)


# ---- enumeration of a finite discrete latent with the new distributions
@pytest.mark.parametrize("make", [lambda: BetaBinomial(3, 2.0, 3.0), lambda: OrderedLogistic(0.3, t([-0.5, 0.4, 1.1]))])
def test_marginalised_latent_equals_brute_force_sum(make):
    prior = make()
    y = 1.3
    support = prior.state_space.values.to(torch.float64)
    with Model() as model:
        z = RandomParameter("z", prior, shape=(), marginalize=True)
        ObservedParameter("y", Normal(z, 1.0), t(y))
    expected = torch.logsumexp(prior.log_pdf(support) + Normal(support, 1.0).log_pdf(t(y)), 0)
    assert float(model.model_log_prob()) == pytest.approx(float(expected), abs=1e-10)


def test_gibbs_posterior_of_a_beta_binomial_latent_matches_enumeration():
    from BayesianDLL import sample
    prior = BetaBinomial(4, 1.5, 2.5)
    y = 2.2
    support = torch.arange(0, 5.0)
    post = torch.softmax(prior.log_pdf(support) + Normal(support, 1.0).log_pdf(t(y)), 0)
    with Model() as model:
        z = RandomParameter("z", prior, shape=())
        ObservedParameter("y", Normal(z, 1.0), t(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sample(3000, 200, n_chains=2, model=model, progress_bar=False, seed=3)
    z = result["z"].reshape(-1)
    freq = torch.stack([(z == k).double().mean() for k in range(5)])
    torch.testing.assert_close(freq, post, atol=0.03, rtol=0)


# ---- parameter recovery with NUTS on simulated data
def _posterior(model, n=500, warmup=400, chains=2, seed=5):
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample(n, warmup, n_chains=chains, model=model, progress_bar=False, seed=seed)


def _covers(draws, truth, k=3.5):
    d = draws.reshape(-1, *draws.shape[2:])
    return bool(torch.all((d.mean(0) - truth).abs() < k * d.std(0)))


def test_ordinal_regression_recovers_slope_and_cutpoints():
    from BayesianDLL.Deterministic import Multiply
    g = torch.Generator().manual_seed(1)
    x = torch.randn(600, generator=g)
    cuts_true = t([-1.0, 0.3, 1.4])
    y = OrderedLogistic(1.2 * x, cuts_true).sample(1)[0]
    with Model() as model:
        beta = RandomParameter("beta", Normal(0.0, 2.0))
        cuts = RandomParameter("cuts", Ordered(Normal(0.0, 3.0), 3), shape=3)
        ObservedParameter("y", OrderedLogistic(Multiply("eta", x, beta), cuts), y)
    result = _posterior(model)
    assert _covers(result["beta"], t(1.2)) and _covers(result["cuts"], cuts_true)


def test_censored_regression_recovers_parameters_where_naive_fit_is_biased():
    n, lo, hi = 800, 0.0, 2.5
    latent = Normal(1.0, 1.5).sample(n)
    y = latent.clamp(lo, hi)
    assert (y == lo).any() and (y == hi).any()
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        sigma = RandomParameter("sigma", HalfNormal(5.0))
        ObservedParameter("y", Censored(Normal(mu, sigma), low=lo, high=hi), y)
    result = _posterior(model)
    assert _covers(result["mu"], t(1.0)) and _covers(result["sigma"], t(1.5))
    with Model() as naive:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        sigma = RandomParameter("sigma", HalfNormal(5.0))
        ObservedParameter("y", Normal(mu, sigma), y)
    assert not _covers(_posterior(naive)["sigma"], t(1.5))


def test_hurdle_poisson_recovers_zero_probability_and_rate():
    d = Hurdle(Poisson(2.5), 0.4)
    y = d.sample(800)
    with Model() as model:
        rate = RandomParameter("rate", Gamma(2.0, 0.5))
        pi = RandomParameter("pi", Beta(1.0, 1.0))
        ObservedParameter("y", Hurdle(Poisson(rate), pi), y)
    result = _posterior(model)
    assert _covers(result["rate"], t(2.5)) and _covers(result["pi"], t(0.4))


def test_beta_binomial_recovers_mean_and_concentration():
    y = BetaBinomial(20, 3.0, 5.0).sample(500)
    with Model() as model:
        a = RandomParameter("a", Gamma(2.0, 0.5))
        b = RandomParameter("b", Gamma(2.0, 0.5))
        ObservedParameter("y", BetaBinomial(20, a, b), y)
    result = _posterior(model, n=600)
    ratio = result["a"] / (result["a"] + result["b"])
    assert _covers(ratio, t(3 / 8)) and _covers(result["a"] + result["b"], t(8.0))


def test_dirichlet_multinomial_recovers_concentration():
    truth = t([1.0, 2.0, 3.0])
    y = DirichletMultinomial(30, truth).sample(500)
    with Model() as model:
        conc = RandomParameter("conc", Gamma(2.0, 0.5), shape=3)
        ObservedParameter("y", DirichletMultinomial(30, conc), y)
    assert _covers(_posterior(model, n=600)["conc"], truth)


def test_new_distributions_support_prior_and_posterior_predictive():
    from BayesianDLL import prior_predicative
    with Model() as model:
        eta = RandomParameter("eta", Normal(0.0, 1.0))
        cuts = RandomParameter("cuts", Ordered(Normal(0.0, 1.5), 2), shape=2)
        ObservedParameter("y", OrderedLogistic(eta, cuts), t([0.0, 1.0, 2.0]))
    result = prior_predicative(200, samples_per_step=1, model=model, progress_bar=False)
    assert set(result["y"].unique().tolist()) <= {0.0, 1.0, 2.0}
