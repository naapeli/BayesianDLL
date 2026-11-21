"""Ancestral (exact) prior and posterior predictive sampling."""

import importlib
import math

import pytest
import torch

from BayesianDLL import (
    DeterministicParameter, Model, ObservedParameter, RandomParameter, plate,
    posterior_predicative, sample_prior_predicative,
)
from BayesianDLL.Distributions import Bernoulli, Beta, Exponential, HalfNormal, Normal, Poisson
from BayesianDLL.Samplers import SamplingResult


def _forbid_mcmc(monkeypatch):
    module = importlib.import_module("BayesianDLL.Samplers._sample")

    def fail(*args, **kwargs):
        raise AssertionError("MCMC must not be used")

    monkeypatch.setattr(module, "sample", fail)
    monkeypatch.setattr(module, "_decide_predicative_step", fail)


def test_posterior_predictive_matches_conjugate_normal(normal_model, monkeypatch):
    """Posterior N(1.6, 4/13) and unit noise: predictive N(1.6, 4/13 + 1)."""
    _forbid_mcmc(monkeypatch)
    n_draws = 4000
    draws = 1.6 + math.sqrt(4 / 13) * torch.randn(2, n_draws // 2, 1)
    result = posterior_predicative({"mean": draws}, n_samples=n_draws, samples_per_step=5, model=normal_model, progress_bar=False)
    samples = result["data"]
    assert samples.shape == (n_draws, 5, 3)
    assert samples.mean().item() == pytest.approx(1.6, abs=0.05)
    assert samples.var().item() == pytest.approx(4 / 13 + 1, rel=0.05)
    # Given a posterior draw the replicated data have unit variance.
    assert samples.var(dim=(1, 2)).mean().item() == pytest.approx(1.0, rel=0.05)


def test_prior_predictive_needs_no_mcmc_and_returns_prior_draws(monkeypatch):
    _forbid_mcmc(monkeypatch)
    with Model() as model:
        mean = RandomParameter("mean", Normal(1.0, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        DeterministicParameter("twice", lambda m: 2 * m, lambda m: {"mean": 2 * torch.ones_like(m)}, [mean])
        with plate("data", 4):
            ObservedParameter("y", Normal(mean, sigma), torch.zeros(4))
    original = model.params["mean"].unconstrained_value.clone()
    result = sample_prior_predicative(3000, samples_per_step=4, model=model, progress_bar=False)
    assert result["y"].shape == (3000, 4, 4)
    prior = result.prior
    assert isinstance(prior, SamplingResult)
    assert prior["mean"].shape == (1, 3000, 1)
    assert prior["mean"].mean().item() == pytest.approx(1.0, abs=0.1)
    assert prior["mean"].std().item() == pytest.approx(2.0, rel=0.05)
    assert prior["sigma"].mean().item() == pytest.approx(math.sqrt(2 / math.pi), rel=0.05)
    torch.testing.assert_close(prior["twice"], 2 * prior["mean"])
    # y = mean + sigma * eps: E = 1, Var = 4 + E[sigma^2] = 5.
    assert result["y"].mean().item() == pytest.approx(1.0, abs=0.1)
    assert result["y"].var().item() == pytest.approx(5.0, rel=0.05)
    torch.testing.assert_close(model.params["mean"].unconstrained_value, original)
    # The model method returns the same kind of result.
    assert model.sample_prior_predicative(5, samples_per_step=2, progress_bar=False).prior["sigma"].shape == (1, 5, 1)


def test_predictive_broadcasts_scalar_parameters_over_plates_independently(monkeypatch):
    _forbid_mcmc(monkeypatch)
    with Model() as model:
        rate = RandomParameter("rate", Exponential(1.0))
        with plate("data", 50):
            ObservedParameter("counts", Poisson(rate), torch.zeros(50, dtype=torch.int64))
    trace = {"rate": torch.full((1, 2, 1), 4.0)}
    result = posterior_predicative(trace, n_samples=2, samples_per_step=200, model=model, progress_bar=False)
    samples = result["counts"]
    assert samples.dtype == torch.int64
    assert samples.shape == (2, 200, 50)
    assert samples.double().mean().item() == pytest.approx(4.0, rel=0.05)
    # Elements along the plate are independent draws, not copies.
    assert samples.double().var(dim=-1).mean().item() == pytest.approx(4.0, rel=0.1)


def test_downstream_observed_nodes_condition_on_replicated_values(monkeypatch):
    _forbid_mcmc(monkeypatch)
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        with plate("data", 3):
            y = ObservedParameter("y", Normal(mean, 1.0), torch.full((3,), 10.0))
            shifted = DeterministicParameter("shifted", lambda y: y + 1, lambda y: {"y": torch.ones_like(y)}, [y])
            ObservedParameter("z", Normal(shifted, 0.1), torch.zeros(3))
            ObservedParameter("w", Normal(y, 0.1), torch.zeros(3))
    trace = {"mean": torch.zeros(1, 50, 1)}
    result = posterior_predicative(trace, n_samples=50, samples_per_step=20, model=model, progress_bar=False)
    y_rep, z_rep, w_rep = result["y"], result["z"], result["w"]
    assert (z_rep - y_rep - 1).std().item() == pytest.approx(0.1, rel=0.1)
    assert (w_rep - y_rep).std().item() == pytest.approx(0.1, rel=0.1)
    # The observed values are restored afterwards.
    torch.testing.assert_close(model.observed_params["y"].observed_values, torch.full((3,), 10.0))


def test_predictive_with_enumerated_parameter(monkeypatch):
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 2.0))
        with plate("n", 4):
            z = RandomParameter("z", Bernoulli(p), shape=(), marginalize=True)
            mean = DeterministicParameter(
                "mean", lambda z: 10 * z, lambda z: {}, [z],
            )
            ObservedParameter("y", Normal(mean, 0.1), torch.zeros(4))
    _forbid_mcmc(monkeypatch)
    prior = sample_prior_predicative(2000, samples_per_step=1, model=model, progress_bar=False)
    assert prior.prior["z"].shape == (1, 2000, 4)
    assert prior.prior["z"].mean().item() == pytest.approx(0.5, abs=0.05)
    torch.testing.assert_close(prior["y"][:, 0] / 10, prior.prior["z"][0], atol=0.1, rtol=0)

    trace = {"p": torch.full((1, 2, 1), 0.5), "z": torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])}
    result = posterior_predicative(trace, n_samples=2, samples_per_step=10, model=model, progress_bar=False)
    torch.testing.assert_close(result["y"].mean(dim=1) / 10, trace["z"][0], atol=0.02, rtol=0)
