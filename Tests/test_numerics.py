"""The torch helpers that replace scipy in the diagnostics, checked against scipy as an oracle."""

import numpy as np
import pytest
import torch

scipy_stats = pytest.importorskip("scipy.stats")
scipy_special = pytest.importorskip("scipy.special")
scipy_fftpack = pytest.importorskip("scipy.fftpack")

from BayesianDLL.Evaluation._information_criteria import _logsumexp
from BayesianDLL.Evaluation._numerics import autocov, gaussian_kde, next_fast_len, rankdata, z_scale


@pytest.mark.parametrize("values", [
    np.random.default_rng(0).integers(0, 4, size=1000).astype(float),  # many ties
    np.random.default_rng(1).integers(-2, 3, size=(7, 33)).astype(float),
    np.round(np.random.default_rng(2).normal(size=500), 1),
    np.random.default_rng(3).normal(size=257),  # no ties
    np.array([3.0, 3.0, 3.0, 3.0]),
    np.array([5.0]),
    np.array([2.0, -np.inf, np.inf, 2.0, -np.inf]),
])
def test_rankdata_matches_scipy_average_method(values):
    ranks = rankdata(torch.from_numpy(values))
    assert ranks.dtype == torch.float64
    np.testing.assert_array_equal(ranks.numpy(), scipy_stats.rankdata(values.reshape(-1), method="average"))


def test_rankdata_handles_integer_and_float32_input():
    values = torch.randint(0, 3, (200,))
    expected = scipy_stats.rankdata(values.numpy(), method="average")
    np.testing.assert_array_equal(rankdata(values).numpy(), expected)
    np.testing.assert_array_equal(rankdata(values.float()).numpy(), expected)


def test_rankdata_propagates_nan_and_empty_input():
    assert torch.isnan(rankdata(torch.tensor([1.0, float("nan"), 2.0]))).all()
    assert rankdata(torch.tensor([])).numel() == 0


def test_z_scale_matches_scipy_rank_normalisation():
    values = np.random.default_rng(4).integers(0, 6, size=(4, 300)).astype(float)
    u = (scipy_stats.rankdata(values.reshape(-1), method="average") - 0.375) / (values.size + 0.25)
    expected = scipy_stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10)).reshape(values.shape)
    np.testing.assert_allclose(z_scale(torch.from_numpy(values)).numpy(), expected, rtol=1e-14, atol=0)


def test_ndtr_and_ndtri_match_scipy_normal():
    u = torch.linspace(1e-10, 1 - 1e-10, 1001, dtype=torch.float64)
    np.testing.assert_allclose(torch.special.ndtri(u).numpy(), scipy_stats.norm.ppf(u.numpy()), rtol=1e-14, atol=1e-15)
    # torch's ndtr loses relative accuracy in the far lower tail (x < -2), unlike scipy's.
    x = torch.linspace(-2, 8, 1001, dtype=torch.float64)
    np.testing.assert_allclose(torch.special.ndtr(x).numpy(), scipy_stats.norm.cdf(x.numpy()), rtol=1e-14, atol=0)


def test_next_fast_len_matches_scipy_fftpack():
    for n in range(1, 5000):
        assert next_fast_len(n) == scipy_fftpack.next_fast_len(n), n


@pytest.mark.parametrize("shape, dim", [((3, 200), 1), ((1, 7), 1), ((50, 4), 0), ((2, 3, 101), -1)])
def test_fft_autocovariance_matches_direct_sum(shape, dim):
    values = torch.randn(*shape, dtype=torch.float64)
    x = np.moveaxis(values.numpy(), dim, -1)
    n = x.shape[-1]
    centered = x - x.mean(axis=-1, keepdims=True)
    direct = np.stack([(centered[..., : n - k] * centered[..., k:]).sum(axis=-1) / n for k in range(n)], axis=-1)
    result = np.moveaxis(autocov(values, dim=dim).numpy(), dim, -1)
    np.testing.assert_allclose(result, direct, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed, n", [(0, 2), (1, 50), (2, 3000)])
def test_gaussian_kde_matches_scipy(seed, n):
    rng = np.random.default_rng(seed)
    samples = rng.standard_t(4, size=n) * 2.0 + 1.0
    grid = np.linspace(samples.min() - 3, samples.max() + 3, 400)
    expected = scipy_stats.gaussian_kde(samples)(grid)
    result = gaussian_kde(torch.from_numpy(samples), torch.from_numpy(grid))
    assert result.dtype == torch.float64 and result.shape == grid.shape
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10, atol=0)


def test_gaussian_kde_chunking_does_not_change_the_result():
    samples = torch.randn(1000, dtype=torch.float64)
    grid = torch.linspace(-4, 4, 333, dtype=torch.float64).reshape(9, 37)
    full = gaussian_kde(samples, grid)
    chunked = gaussian_kde(samples, grid, chunk_size=1500)
    assert full.shape == (9, 37)
    torch.testing.assert_close(chunked, full, rtol=1e-15, atol=0)


@pytest.mark.parametrize("samples", [torch.ones(10), torch.tensor([1.0])])
def test_gaussian_kde_rejects_degenerate_samples(samples):
    with pytest.raises(ValueError):
        gaussian_kde(samples, torch.linspace(0, 2, 5))


def test_logsumexp_matches_scipy():
    values = np.random.default_rng(5).normal(scale=50, size=(40, 6))
    np.testing.assert_allclose(_logsumexp(values), scipy_special.logsumexp(values), rtol=1e-14)
    np.testing.assert_allclose(_logsumexp(values, axis=0), scipy_special.logsumexp(values, axis=0), rtol=1e-14)
    assert _logsumexp(np.full(3, -np.inf)) == -np.inf
