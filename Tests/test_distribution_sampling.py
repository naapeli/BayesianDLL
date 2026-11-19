"""Exact samplers, reparameterisation gradients and NUTS on the new transforms."""

import math

import pytest
import torch

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet, DiscreteUniform, Exponential, Gamma, Geometric,
    HalfCauchy, HalfNormal, InvGamma, Laplace, LKJCholesky, LogNormal, LogTransform, Mixture, Multinomial,
    MultivariateNormal, NegativeBinomial, Normal, Poisson, StudentT, TransformedDistribution, Truncated, Uniform,
    ZeroInflated, Wishart, InverseWishart,
)


t = torch.tensor
N = 100_000


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _truncated_normal_moments(a, b):
    Z = 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
    mean = (_phi(a) - _phi(b)) / Z
    var = 1 + (a * _phi(a) - b * _phi(b)) / Z - mean ** 2
    return mean, var


MOMENT_CASES = {
    # name: (distribution factory, mean, variance) -- None skips a moment
    "normal": (lambda: Normal(1.0, 2.0), 1.0, 4.0),
    "half_normal": (lambda: HalfNormal(1.5), 1.5 * math.sqrt(2 / math.pi), 1.5 ** 2 * (1 - 2 / math.pi)),
    "log_normal": (lambda: LogNormal(0.2, 0.5), math.exp(0.2 + 0.125), (math.exp(0.25) - 1) * math.exp(0.4 + 0.25)),
    "beta": (lambda: Beta(2.0, 3.0), 0.4, 0.04),
    "exponential": (lambda: Exponential(2.0), 0.5, 0.25),
    "gamma": (lambda: Gamma(2.5, 1.5), 2.5 / 1.5, 2.5 / 1.5 ** 2),
    "gamma_small_shape": (lambda: Gamma(0.4, 1.0), 0.4, 0.4),
    "uniform": (lambda: Uniform(-1.0, 3.0), 1.0, 16 / 12),
    "inverse_gamma": (lambda: InvGamma(4.0, 3.0), 1.0, 0.5),
    "laplace": (lambda: Laplace(0.5, 1.5), 0.5, 4.5),
    "student_t": (lambda: StudentT(5.0, 1.0, 2.0), 1.0, 4 * 5 / 3),
    "bernoulli": (lambda: Bernoulli(0.3), 0.3, 0.21),
    "bernoulli_logits": (lambda: Bernoulli(logits=math.log(0.3 / 0.7)), 0.3, 0.21),
    "binomial": (lambda: Binomial(10, 0.3), 3.0, 2.1),
    "geometric": (lambda: Geometric(0.3), 1 / 0.3, 0.7 / 0.09),
    "geometric_logits": (lambda: Geometric(logits=math.log(0.3 / 0.7)), 1 / 0.3, 0.7 / 0.09),
    "poisson": (lambda: Poisson(3.5), 3.5, 3.5),
    "negative_binomial": (lambda: NegativeBinomial(3.0, 0.4), 2.0, 2.0 / 0.6),
    "negative_binomial_logits": (lambda: NegativeBinomial(3.0, logits=math.log(0.4 / 0.6)), 2.0, 2.0 / 0.6),
    "categorical": (lambda: Categorical(t([0.2, 0.5, 0.3])), 1.1, 0.5 + 4 * 0.3 - 1.21),
    "discrete_uniform": (lambda: DiscreteUniform(2, 5), 3.5, 15 / 12),
    "zero_inflated_poisson": (lambda: ZeroInflated(Poisson(3.0), 0.3), 2.1, 0.7 * 3.0 * (1 + 0.3 * 3.0)),
    "mixture": (lambda: Mixture([Normal(-1.0, 0.5), Normal(2.0, 1.0)], t([0.3, 0.7])), 1.1, 2.665),
    "truncated_normal": (lambda: Truncated(Normal(0.0, 1.0), -1.0, 2.0), *_truncated_normal_moments(-1.0, 2.0)),
    "transformed": (lambda: TransformedDistribution(Normal(0.2, 0.5), LogTransform()), math.exp(0.325), (math.exp(0.25) - 1) * math.exp(0.65)),
}


@pytest.mark.parametrize("name", list(MOMENT_CASES))
def test_sample_moments(name):
    factory, mean, var = MOMENT_CASES[name]
    dist = factory()
    samples = dist.sample(N)
    assert samples.shape == (N,)
    assert samples.dtype == torch.float64
    sd = math.sqrt(var)
    # five standard errors of the mean, ~5% on the variance
    assert samples.mean().item() == pytest.approx(mean, abs=5 * sd / math.sqrt(N) + 1e-12)
    assert samples.var().item() == pytest.approx(var, rel=0.05)
    if name not in ("normal", "laplace", "student_t", "mixture", "uniform", "truncated_normal"):
        state_space = dist.state_space
        assert bool(torch.as_tensor(dist.log_pdf(samples[:200]) > -math.inf).all()), state_space


@pytest.mark.parametrize("dist,median,q75", [
    (Cauchy(0.5, 2.0), 0.5, 2.5),
    (HalfCauchy(1.5), 1.5, 1.5 * math.tan(3 * math.pi / 8)),
])
def test_heavy_tailed_quantiles(dist, median, q75):
    samples = dist.sample(N)
    assert samples.median().item() == pytest.approx(median, abs=0.05)
    assert samples.quantile(0.75).item() == pytest.approx(q75, rel=0.04)
    assert dist.icdf(dist.cdf(t([0.7]))).item() == pytest.approx(0.7)


def test_multivariate_samples():
    mu = t([1.0, -2.0, 0.5])
    A = t([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.3, 0.2, 0.5]])
    cov = A @ A.T
    for dist in (MultivariateNormal(mu, cov), MultivariateNormal(mu, scale_tril=A)):
        s = dist.sample(N)
        assert s.shape == (N, 3)
        torch.testing.assert_close(s.mean(0), mu, atol=0.02, rtol=0)
        torch.testing.assert_close(torch.cov(s.T), cov, atol=0.03, rtol=0)

    alpha = t([2.0, 3.0, 5.0])
    s = Dirichlet(alpha).sample(N)
    torch.testing.assert_close(s.sum(-1), torch.ones(N))
    torch.testing.assert_close(s.mean(0), alpha / 10, atol=0.003, rtol=0)
    torch.testing.assert_close(s.var(0), alpha * (10 - alpha) / (100 * 11), atol=0.001, rtol=0)

    probs = t([0.2, 0.5, 0.3])
    s = Multinomial(10, probs).sample(20_000)
    assert (s.sum(-1) == 10).all()
    torch.testing.assert_close(s.mean(0), 10 * probs, atol=0.05, rtol=0)


def test_lkj_cholesky_samples():
    L = LKJCholesky(3, 2.0).sample(40_000)
    assert L.shape == (40_000, 3, 3)
    assert (torch.triu(L, diagonal=1) == 0).all()
    corr = L @ L.transpose(-2, -1)
    torch.testing.assert_close(torch.diagonal(corr, dim1=-2, dim2=-1), torch.ones(40_000, 3))
    off = corr[:, [1, 2, 2], [0, 0, 1]]
    # marginal of a correlation: (r + 1) / 2 ~ Beta(eta - 1 + D/2, same) -> var = 1 / (2 eta + D - 1)
    torch.testing.assert_close(off.mean(0), torch.zeros(3), atol=0.01, rtol=0)
    torch.testing.assert_close(off.var(0), torch.full((3,), 1 / 6), atol=0.01, rtol=0)


def test_batched_sample_shapes():
    assert Normal(torch.zeros(3, 1), torch.ones(4)).sample(5).shape == (5, 3, 4)
    assert MultivariateNormal(torch.zeros(2, 3), torch.eye(3)).sample(5).shape == (5, 2, 3)
    assert Dirichlet(torch.ones(2, 3)).sample(5).shape == (5, 2, 3)
    assert Categorical(torch.ones(2, 3)).sample(5).shape == (5, 2)
    assert Multinomial(4, torch.ones(2, 3)).sample(5).shape == (5, 2, 3)
    assert LKJCholesky(3, torch.ones(2)).sample(5).shape == (5, 2, 3, 3)
    assert Gamma(torch.ones(4), 1.0).sample(5).shape == (5, 4)
    assert Mixture([Normal(torch.zeros(3), 1.0), Normal(2.0, 1.0)], [0.4, 0.6]).sample(5).shape == (5, 3)
    float32 = Normal(torch.zeros(2, dtype=torch.float32), 1.0).sample(3)
    assert float32.dtype == torch.float32


REPARAM_CASES = [
    (lambda m, s: Normal(m, s), {"mean": 0.3, "sigma": 1.4}),
    (lambda s: HalfNormal(s), {"sigma": 1.4}),
    (lambda m, s: LogNormal(m, s), {"mean": 0.3, "sigma": 0.6}),
    (lambda r: Exponential(r), {"rate": 1.5}),
    (lambda lo, hi: Uniform(lo, hi), {"low": -1.0, "high": 2.0}),
    (lambda s: HalfCauchy(s), {"scale": 1.4}),
    (lambda m, s: Cauchy(m, s), {"mean": 0.3, "scale": 1.4}),
    (lambda m, s: Laplace(m, s), {"mean": 0.3, "scale": 1.4}),
    (lambda m, s: StudentT(4.0, m, s), {"mean": 0.3, "sigma": 1.4}),
    (lambda m: MultivariateNormal(m, torch.eye(2) * 2.0), {"mean": [0.3, -0.2]}),
    (lambda m, s: TransformedDistribution(Normal(m, s), LogTransform()), {"mean": 0.3, "sigma": 0.6}),
]


@pytest.mark.parametrize("factory,params", REPARAM_CASES)
def test_reparameterisation_gradients_match_finite_differences(factory, params):
    values = {k: torch.tensor(v) for k, v in params.items()}
    torch.manual_seed(11)
    dist = factory(*values.values())
    assert dist.has_rsample
    samples, grads = dist.sample(50, _reparametrization_trick_grad=True)
    torch.manual_seed(11)
    torch.testing.assert_close(dist.sample(50), samples)
    eps = 1e-6
    for key in params:
        assert grads[key].shape == samples.shape
        directions = torch.eye(values[key].numel()).reshape(-1, *values[key].shape) if values[key].ndim else [torch.tensor(1.0)]
        numeric = 0
        for direction in directions:
            plus = {k: v + eps * direction if k == key else v for k, v in values.items()}
            minus = {k: v - eps * direction if k == key else v for k, v in values.items()}
            torch.manual_seed(11)
            upper = factory(*plus.values()).sample(50)
            torch.manual_seed(11)
            lower = factory(*minus.values()).sample(50)
            numeric = numeric + (upper - lower) / (2 * eps)
        torch.testing.assert_close(grads[key], numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dist", [Gamma(2.0, 1.0), Beta(2.0, 2.0), Dirichlet(t([1.0, 2.0])), Poisson(2.0), Bernoulli(0.3),
                                  LKJCholesky(2, 1.0), Truncated(Normal(0.0, 1.0), 0.0, 1.0)])
def test_reparameterisation_unavailable(dist):
    assert not dist.has_rsample
    assert dist.sample(3).shape[0] == 3
    with pytest.raises(NotImplementedError):
        dist.sample(3, _reparametrization_trick_grad=True)


def test_dirichlet_nuts_recovers_moments():
    with Model() as model:
        RandomParameter("p", Dirichlet(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)))
    assert model.params["p"].unconstrained_value.shape == (2,)
    torch.testing.assert_close(model.params["p"].constrained_value, torch.full((3,), 1 / 3))
    trace = model.sample(1000, 500, n_chains=2, progress_bar=False)
    p = trace["p"].reshape(-1, 3)
    torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[0]))
    torch.testing.assert_close(p.mean(0), torch.full((3,), 1 / 3), atol=0.03, rtol=0)
    # Var = a (a0 - a) / (a0^2 (a0 + 1)) = 2 * 4 / (36 * 7) = 0.0317
    torch.testing.assert_close(p.var(0), torch.full((3,), 8 / 252), atol=0.007, rtol=0)


def test_lkj_cholesky_nuts_recovers_marginal_variance():
    with Model() as model:
        RandomParameter("L", LKJCholesky(3, 2.0))
    trace = model.sample(1000, 500, n_chains=2, progress_bar=False)
    L = trace["L"].reshape(-1, 3, 3)
    corr = L @ L.transpose(-2, -1)
    off = corr[:, [1, 2, 2], [0, 0, 1]]
    torch.testing.assert_close(off.mean(0), torch.zeros(3), atol=0.06, rtol=0)
    torch.testing.assert_close(off.var(0), torch.full((3,), 1 / 6), atol=0.035, rtol=0)


def test_mvn_with_random_mean_and_deterministic_covariance_samples_with_nuts():
    torch.manual_seed(3)
    true_cov = torch.tensor([[1.0, 0.5], [0.5, 0.8]], dtype=torch.float64)
    data = MultivariateNormal(torch.tensor([1.0, -1.0], dtype=torch.float64), true_cov).sample(200)
    base = true_cov / 2
    with Model() as model:
        mean = RandomParameter("mean", Normal(torch.zeros(2, dtype=torch.float64), 3.0), shape=2)
        log_scale = RandomParameter("log_scale", Normal(0.0, 1.0))
        covariance = DeterministicParameter(
            "covariance", lambda s: torch.exp(s) * base, lambda s: {"log_scale": torch.exp(s) * base}, [log_scale])
        ObservedParameter("data", MultivariateNormal(mean, covariance), data)
    trace = model.sample(300, 300, n_chains=2, progress_bar=False)
    torch.testing.assert_close(trace["mean"].reshape(-1, 2).mean(0), data.mean(0), atol=0.1, rtol=0)
    assert trace["log_scale"].mean().item() == pytest.approx(math.log(2.0), abs=0.25)


def test_wishart_family_samples():
    S = torch.tensor([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 1.5]])
    L = torch.linalg.cholesky(S)
    for dist in (Wishart(5.0, S), Wishart(5.0, scale_tril=L)):
        X = dist.sample(N)
        assert X.shape == (N, 3, 3)
        torch.testing.assert_close(X.mean(0), 5.0 * S, atol=0.06, rtol=0)
        # Var(X_ij) = df (S_ij^2 + S_ii S_jj)
        torch.testing.assert_close(X.var(0), 5.0 * (S ** 2 + torch.outer(S.diagonal(), S.diagonal())), atol=0.0, rtol=0.05)
        assert bool(torch.isfinite(dist.log_pdf(X[:100])).all())
    X = InverseWishart(8.0, S).sample(N)
    torch.testing.assert_close(X.mean(0), S / 4.0, atol=0.01, rtol=0)
    assert Wishart(4.0 * torch.ones(2), S).sample(5).shape == (5, 2, 3, 3)
    with pytest.raises(NotImplementedError):
        Wishart(5.0, S).sample(2, _reparametrization_trick_grad=True)


def test_wishart_nuts_recovers_mean():
    S = torch.tensor([[1.0, 0.3], [0.3, 0.5]], dtype=torch.float64)
    with Model() as model:
        RandomParameter("X", Wishart(6.0, S))
    assert model.params["X"].unconstrained_value.shape == (3,)
    trace = model.sample(1000, 500, n_chains=2, progress_bar=False)
    X = trace["X"].reshape(-1, 2, 2)
    torch.testing.assert_close(X.mean(0), 6.0 * S, atol=0.4, rtol=0)


@pytest.mark.parametrize("batched_components", [False, True])
def test_mixture_sample_with_batched_weights(batched_components):
    """Component frequencies per batch row follow that row's weights (normalised over the last dim)."""
    torch.manual_seed(0)
    weights = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]], dtype=torch.float64)
    if batched_components:
        # Component 0 sits at -100 in rows 0 and 2 and at +100 in row 1.
        components = [Normal(torch.tensor([-100.0, 100.0, -100.0]), 1.0), Normal(torch.tensor([100.0, -100.0, 100.0]), 1.0)]
        is_first_negative = torch.tensor([True, False, True])
    else:
        components = [Normal(-100.0, 1.0), Normal(100.0, 1.0)]
        is_first_negative = torch.tensor([True, True, True])
    draws = Mixture(components, 2 * weights).sample(20000)  # unnormalised weights are normalised per row
    assert draws.shape == (20000, 3)
    first = torch.where(is_first_negative, draws < 0, draws > 0).double().mean(0)
    torch.testing.assert_close(first, weights[:, 0], atol=0.015, rtol=0)
