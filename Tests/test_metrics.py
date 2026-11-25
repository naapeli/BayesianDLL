"""Inverse-metric estimators and their use in NUTS."""

import pytest
import torch

from BayesianDLL.Samplers import NUTS
from BayesianDLL.Samplers._metrics import (AUTO_DENSE_MAX_DIM, LowRankMetric, check_metric, inverse_metric_matrix,
                                           low_rank_inverse_metric, low_rank_matrix, low_rank_momentum,
                                           low_rank_velocity, metric_kind, resolve_metric, shrinkage_inverse_metric,
                                           stan_regularise)
from BayesianDLL.Samplers._vectorized_samplers import VectorizedNUTS

DTYPE = torch.float64
METRICS = ("diag", "dense", "ledoit_wolf", "oas", "low_rank", "auto")


def correlated_gaussian(d=5, seed=0):
    """A covariance with scales spanning two orders of magnitude and strong correlations."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=generator, dtype=DTYPE)
    correlation = a @ a.T / d + 0.05 * torch.eye(d, dtype=DTYPE)
    scale = torch.logspace(-1, 1, d, dtype=DTYPE)
    return scale[:, None] * correlation * scale[None, :]


def gaussian_draws(covariance, n, seed=1):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n, covariance.shape[0], generator=generator, dtype=DTYPE)
    return z @ torch.linalg.cholesky(covariance).T


def reference_shrinkage(draws, method):
    """Straightforward (loop) version of the Ledoit-Wolf / OAS correlation shrinkage."""
    n, p = draws.shape
    x = draws - draws.mean(0)
    sd = x.square().mean(0).sqrt()
    z = x / sd
    r = z.T @ z / n
    mu = torch.trace(r) / p
    identity = torch.eye(p, dtype=DTYPE)
    if method == "ledoit_wolf":
        d2 = ((r - mu * identity) ** 2).sum() / p
        b2 = sum(((torch.outer(row, row) - r) ** 2).sum() for row in z) / (n * n * p)
        rho = min(b2, d2) / d2
    else:
        tr2 = (r ** 2).sum()
        rho = min(1.0, ((1 - 2 / p) * tr2 + torch.trace(r) ** 2) / ((n + 1 - 2 / p) * (tr2 - torch.trace(r) ** 2 / p)))
    shrunk = (1 - rho) * r + rho * mu * identity
    sd = sd * (n / (n - 1)) ** 0.5
    return stan_regularise(sd[:, None] * shrunk * sd[None, :], n)


# ------------------------------------------------------------------ estimators
def test_check_and_resolve_metric():
    for metric in METRICS:
        check_metric(metric)
    with pytest.raises(ValueError, match="metric"):
        check_metric("full")
    assert resolve_metric("auto", AUTO_DENSE_MAX_DIM) == "dense"
    assert resolve_metric("auto", AUTO_DENSE_MAX_DIM + 1) == "low_rank"
    assert resolve_metric("oas", 1000) == "oas"
    assert metric_kind(None) is None
    assert metric_kind(torch.ones(3)) == "diag"
    assert metric_kind(torch.eye(3)) == "dense"
    assert metric_kind(torch.ones(2, 3), batch_dims=1) == "diag"
    assert metric_kind(torch.ones(2, 3, 3), batch_dims=1) == "dense"
    assert metric_kind(LowRankMetric(torch.ones(3), torch.zeros(3, 0), torch.ones(0))) == "low_rank"


@pytest.mark.parametrize("method", ["ledoit_wolf", "oas"])
def test_shrinkage_matches_reference_and_batches(method):
    covariance = correlated_gaussian(4)
    draws = torch.stack([gaussian_draws(covariance, 30, seed=1), gaussian_draws(covariance, 30, seed=2)])
    batched = shrinkage_inverse_metric(draws, method)
    assert batched.shape == (2, 4, 4)
    for chain in range(2):
        torch.testing.assert_close(batched[chain], reference_shrinkage(draws[chain], method), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", ["ledoit_wolf", "oas"])
def test_shrinkage_tends_to_sample_covariance_and_shrinks_small_samples(method):
    covariance = correlated_gaussian(4)
    estimate = shrinkage_inverse_metric(gaussian_draws(covariance, 200000), method)
    torch.testing.assert_close(estimate, covariance, rtol=0.05, atol=0.05 * covariance.abs().max().item())
    # With fewer draws than dimensions the off-diagonals are pulled towards zero.
    small = gaussian_draws(correlated_gaussian(30), 10)
    sample = torch.cov(small.T)
    shrunk = shrinkage_inverse_metric(small, method)
    off = ~torch.eye(30, dtype=torch.bool)
    assert shrunk[off].abs().sum() < sample[off].abs().sum()
    assert torch.linalg.eigvalsh(shrunk).min() > 0


def low_rank_target(d=40, seed=0):
    """Diagonal scales times the identity plus a wide and a narrow direction."""
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.linalg.qr(torch.randn(d, 2, generator=generator, dtype=DTYPE))[0]
    scale = torch.logspace(-1, 1, d, dtype=DTYPE)
    inner = torch.eye(d, dtype=DTYPE) + vectors @ torch.diag(torch.tensor([29.0, -0.98], dtype=DTYPE)) @ vectors.T
    return scale[:, None] * inner * scale[None, :]


def test_low_rank_recovers_gaussian_covariance():
    covariance = low_rank_target()
    precision = torch.linalg.inv(covariance)
    draws = gaussian_draws(covariance, 20000)
    metric = low_rank_inverse_metric(draws, -draws @ precision)
    assert 2 <= metric.vectors.shape[-1] <= 6
    # In the metric's coordinates the target is close to a standard normal: all
    # eigenvalues of M^-1 Sigma are within (a bit more than) the cutoff.
    chol = torch.linalg.cholesky(low_rank_matrix(metric))
    whitened = torch.linalg.solve_triangular(chol, torch.linalg.solve_triangular(chol, covariance, upper=False).T, upper=False)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    assert eigenvalues.min() > 0.3 and eigenvalues.max() < 3.0
    # The plain diagonal metric would leave them spread over orders of magnitude.
    diagonal_chol = torch.diag(metric.scale)
    plain = torch.linalg.inv(diagonal_chol) @ covariance @ torch.linalg.inv(diagonal_chol)
    assert torch.linalg.eigvalsh(plain).max() > 10.0
    torch.testing.assert_close(metric.vectors.T @ metric.vectors, torch.eye(metric.vectors.shape[-1], dtype=DTYPE),
                               atol=1e-9, rtol=0)


def test_low_rank_with_few_draws_has_bounded_rank_and_is_valid():
    covariance = low_rank_target(d=60)
    precision = torch.linalg.inv(covariance)
    draws = gaussian_draws(covariance, 12)
    metric = low_rank_inverse_metric(draws, -draws @ precision)
    assert metric.vectors.shape[-1] <= 2 * 12
    assert torch.linalg.eigvalsh(low_rank_matrix(metric)).min() > 0


def test_low_rank_batches_match_single_estimates():
    covariance = low_rank_target()
    precision = torch.linalg.inv(covariance)
    first = gaussian_draws(covariance, 4000, seed=1)
    second = gaussian_draws(torch.diag(covariance.diagonal()), 4000, seed=2)
    draws = torch.stack([first, second])
    gradients = torch.stack([-first @ precision, -second / covariance.diagonal()])
    batched = low_rank_inverse_metric(draws, gradients)
    assert batched.scale.shape == (2, 40)
    for chain in range(2):
        single = low_rank_inverse_metric(draws[chain], gradients[chain])
        batch_chain = LowRankMetric(batched.scale[chain], batched.vectors[chain], batched.values[chain])
        torch.testing.assert_close(low_rank_matrix(batch_chain), low_rank_matrix(single), rtol=1e-6, atol=1e-9)


def random_low_rank(d=7, k=3, batch=(), seed=0):
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.linalg.qr(torch.randn(*batch, d, k, generator=generator, dtype=DTYPE))[0]
    values = torch.exp(torch.randn(*batch, k, generator=generator, dtype=DTYPE))
    scale = torch.exp(0.5 * torch.randn(*batch, d, generator=generator, dtype=DTYPE))
    return LowRankMetric(scale, vectors, values)


@pytest.mark.parametrize("batch", [(), (3,)])
def test_low_rank_velocity_and_momentum_match_dense_form(batch):
    metric = random_low_rank(batch=batch)
    matrix = low_rank_matrix(metric)
    r = torch.randn(*batch, 7, dtype=DTYPE)
    torch.testing.assert_close(low_rank_velocity(r, metric), (matrix @ r.unsqueeze(-1)).squeeze(-1))
    # The momentum covariance is the inverse of the inverse metric.
    z = torch.randn(200000, *batch, 7, dtype=DTYPE, generator=torch.Generator().manual_seed(3))
    momenta = low_rank_momentum(z, metric)
    expected = torch.linalg.inv(matrix)
    covariance = torch.einsum("n...i,n...j->...ij", momenta, momenta) / momenta.shape[0]
    torch.testing.assert_close(covariance, expected, rtol=0.05, atol=0.03 * expected.abs().max().item())
    torch.testing.assert_close(inverse_metric_matrix(metric, batch_dims=len(batch)), matrix)


def test_nuts_low_rank_momentum_kinetic_and_velocity_use_one_metric():
    metric = random_low_rank()
    matrix = low_rank_matrix(metric)
    sampler = NUTS(lambda x: x.sum(), lambda x: torch.ones_like(x), lambda x: x, metric="low_rank")
    sampler.inv_metric = metric
    theta = torch.zeros(7, dtype=DTYPE)
    momenta = torch.stack([sampler._sample_momentum(theta) for _ in range(20000)])
    expected = torch.linalg.inv(matrix)
    torch.testing.assert_close(torch.cov(momenta.T), expected, rtol=0.1, atol=0.05 * expected.abs().max().item())
    r = torch.randn(7, dtype=DTYPE)
    torch.testing.assert_close(sampler._kinetic(r), 0.5 * r @ matrix @ r)
    torch.testing.assert_close(sampler._velocity(r), matrix @ r)

    vectorized = VectorizedNUTS(lambda x: x.sum(-1), lambda x: torch.ones_like(x), 3, metric="low_rank")
    vectorized.inv_metric = random_low_rank(batch=(3,))
    matrices = low_rank_matrix(vectorized.inv_metric)
    r = torch.randn(3, 7, dtype=DTYPE)
    torch.testing.assert_close(vectorized._kinetic(r), 0.5 * torch.einsum("ci,cij,cj->c", r, matrices, r))


def test_defaults_and_invalid_metric():
    assert NUTS(lambda x: x, lambda x: x, lambda x: x).metric == "dense"
    assert VectorizedNUTS(lambda x: x.sum(-1), lambda x: x, 2).metric == "dense"
    with pytest.raises(ValueError, match="metric"):
        VectorizedNUTS(lambda x: x.sum(-1), lambda x: x, 2, metric="full")


# ------------------------------------------------------------------ invariance
def gaussian_target(covariance):
    precision = torch.linalg.inv(covariance)
    return (lambda x: -0.5 * ((x @ precision) * x).sum(-1)), (lambda x: -(x @ precision))


def check_moments(x, covariance):
    scale = covariance.diagonal().sqrt()
    outer = scale[:, None] * scale[None]
    assert (x.mean(0) / scale).abs().max() < 0.2
    assert (torch.cov(x.T) / outer - covariance / outer).abs().max() < 0.25


@pytest.mark.integration
@pytest.mark.parametrize("metric", METRICS)
def test_vectorized_nuts_leaves_correlated_gaussian_invariant(metric):
    covariance = correlated_gaussian(4)
    log_target, gradient = gaussian_target(covariance)
    chains, warmup, draws = 4, 300, 400
    sampler = VectorizedNUTS(log_target, gradient, chains, metric=metric)
    sampler.set_warmup_length(warmup)
    theta = torch.randn(chains, 4, dtype=DTYPE)
    stored = []
    for i in range(warmup + draws):
        theta = sampler.step(theta, warmup=i < warmup)[0]
        if i >= warmup:
            stored.append(theta)
    check_moments(torch.stack(stored).reshape(-1, 4), covariance)


@pytest.mark.integration
@pytest.mark.parametrize("metric", ["ledoit_wolf", "oas", "low_rank", "auto"])
def test_nuts_leaves_correlated_gaussian_invariant(metric):
    covariance = correlated_gaussian(3)
    log_target, gradient = gaussian_target(covariance)
    sampler = NUTS(log_target, gradient, lambda x: x, metric=metric)
    sampler.set_warmup_length(300)
    theta = torch.zeros(3, dtype=DTYPE)
    stored = []
    for i in range(300 + 1200):
        theta = sampler.step(theta, warmup=i < 300)[0]
        if i >= 300:
            stored.append(theta)
    assert sampler.inv_metric is not None
    check_moments(torch.stack(stored), covariance)


@pytest.mark.integration
def test_low_rank_nuts_adapts_to_strong_correlation():
    covariance = low_rank_target(d=20)
    log_target, gradient = gaussian_target(covariance)
    chains, warmup = 2, 300
    steps = {}
    for metric in ("diag", "low_rank"):
        torch.manual_seed(0)
        sampler = VectorizedNUTS(log_target, gradient, chains, metric=metric)
        sampler.set_warmup_length(warmup)
        theta = torch.randn(chains, 20, dtype=DTYPE)
        for i in range(warmup + 40):
            theta = sampler.step(theta, warmup=i < warmup)[0]
        steps[metric] = sampler.last_stats["n_leapfrog"].mean().item()
    assert isinstance(sampler.inv_metric, LowRankMetric)
    assert sampler.inv_metric.vectors.shape[-1] >= 1
    # Following the dominant direction needs fewer leapfrog steps.
    assert steps["low_rank"] < steps["diag"]
