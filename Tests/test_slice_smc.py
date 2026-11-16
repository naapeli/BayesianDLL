"""Slice sampler and Sequential Monte Carlo."""

import math

import pytest
import torch

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter, sample_smc
from BayesianDLL.Distributions import (
    Bernoulli, Beta, ContinuousReal, DiscreteRange, DiscreteUniform, Exponential, Gamma, Normal,
)
from BayesianDLL.Samplers import Slice, SamplingBlock
from BayesianDLL.Samplers._SMC import _next_beta, _ess, _systematic_resample
from BayesianDLL.Samplers._sample import _build_blocks, _select_sampler


# ------------------------------------------------------------------------ slice
def test_slice_recovers_standard_normal_and_records_stats():
    sampler = Slice(lambda x: -0.5 * x.square().sum(), ContinuousReal(), width=0.1)
    theta = torch.zeros(1)
    draws = []
    for i in range(3500):
        theta, step_size, acceptance, diverging = sampler.step(theta, warmup=i < 500)
        assert acceptance == 1.0 and not diverging
        if i >= 500:
            draws.append(theta.clone())
    samples = torch.stack(draws)
    assert samples.mean().item() == pytest.approx(0.0, abs=0.12)
    assert samples.var().item() == pytest.approx(1.0, abs=0.15)
    # The width adapts from 0.1 towards the scale of the target.
    assert 0.8 < sampler.width.item() < 4.0
    assert set(sampler.last_stats) == {"n_evals", "accept_stat", "step_size", "lp"}
    assert sampler.last_stats["n_evals"] >= 1
    assert sampler.last_stats["lp"] == pytest.approx(-0.5 * theta.square().sum().item())
    sampler.reset()
    assert sampler.m == 0 and sampler.width is None


def test_slice_updates_each_coordinate_and_keeps_widths_without_adaptation():
    scales = torch.tensor([0.1, 10.0])
    sampler = Slice(lambda x: -0.5 * (x / scales).square().sum(), ContinuousReal(), width=torch.tensor([0.2, 20.0]), adapt_width=False)
    theta = torch.zeros(2)
    draws = []
    for _ in range(3000):
        theta, *_ = sampler.step(theta, warmup=True)
        draws.append(theta.clone())
    torch.testing.assert_close(sampler.width, torch.tensor([0.2, 20.0]))
    std = torch.stack(draws).std(0)
    torch.testing.assert_close(std, scales, rtol=0.15, atol=0)


def test_slice_registration_and_errors():
    assert isinstance(_select_sampler(lambda x: -x.square().sum(), ContinuousReal(), sampler="slice"), Slice)
    with pytest.raises(RuntimeError, match="continuous"):
        _select_sampler(lambda x: x.sum(), DiscreteRange(0, 1), sampler="slice")
    with pytest.raises(ValueError, match="width"):
        Slice(lambda x: x.sum(), ContinuousReal(), width=0.0)
    with Model() as model:
        RandomParameter("a", Normal(0.0, 1.0), sampler="slice")
        RandomParameter("b", Normal(0.0, 1.0))
        RandomParameter("c", Bernoulli(0.5))
    blocks = _build_blocks(model, sampler_params={"width": 0.5})
    assert [(b.param_names, b.sampler_type) for b in blocks] == [(["b"], "nuts"), (["a"], "slice"), (["c"], "gibbs")]
    assert blocks[1].sampler.initial_width == 0.5
    assert SamplingBlock(model, ["a", "b"]).sampler_type == "slice"
    with Model() as discrete:
        RandomParameter("z", Bernoulli(0.5), sampler="slice")
    with pytest.raises(ValueError, match="continuous"):
        _build_blocks(discrete)


@pytest.mark.integration
def test_slice_samples_skewed_gamma_posterior():
    # Gamma(2, 1) prior on an exponential rate; the posterior is Gamma(2 + n, 1 + sum(y)).
    y = torch.tensor([0.5, 1.5, 0.3])
    with Model() as model:
        rate = RandomParameter("rate", Gamma(2.0, 1.0), sampler="slice")
        ObservedParameter("y", Exponential(rate), y)
    result = model.sample(1500, 300, n_chains=2, progress_bar=False, check_convergence=False, seed=1)
    alpha, beta = 2.0 + 3, 1.0 + y.sum().item()
    samples = result["rate"].flatten()
    assert samples.mean().item() == pytest.approx(alpha / beta, rel=0.06)
    assert samples.var().item() == pytest.approx(alpha / beta ** 2, rel=0.2)
    skewness = ((samples - samples.mean()) ** 3).mean() / samples.std() ** 3
    assert skewness.item() == pytest.approx(2 / math.sqrt(alpha), abs=0.35)
    assert result.sample_stats.samplers == ["slice"]
    assert set(result.sample_stats[0]) == {"n_evals", "accept_stat", "step_size", "lp"}
    assert result.sample_stats["n_evals"].shape == (2, 1500)


@pytest.mark.integration
def test_slice_samples_bounded_beta_through_transform():
    with Model() as model:
        RandomParameter("p", Beta(2.0, 5.0), sampler="slice")
    result = model.sample(2000, 300, n_chains=2, progress_bar=False, check_convergence=False, seed=2)
    samples = result["p"].flatten()
    assert bool(((samples > 0) & (samples < 1)).all())
    assert samples.mean().item() == pytest.approx(2 / 7, abs=0.02)
    assert samples.var().item() == pytest.approx(10 / (49 * 8), rel=0.15)


@pytest.mark.integration
def test_multi_block_model_with_nuts_and_slice():
    torch.manual_seed(0)
    y = 1.0 + 0.5 * torch.randn(40)
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 10.0))
        sigma = RandomParameter("sigma", Gamma(2.0, 2.0), sampler="slice")
        ObservedParameter("y", Normal(mu, sigma), y)
    result = model.sample(800, 300, n_chains=2, progress_bar=False, check_convergence=False, seed=3)
    assert result.sample_stats.samplers == ["nuts", "slice"]
    assert result["mu"].mean().item() == pytest.approx(y.mean().item(), abs=0.05)
    assert result["sigma"].mean().item() == pytest.approx(y.std().item(), abs=0.08)


# -------------------------------------------------------------------------- SMC
def test_smc_helpers():
    log_likelihood = torch.randn(200) * 5
    beta = _next_beta(0.0, log_likelihood, 100.0)
    assert 0 < beta < 1
    assert _ess(beta * log_likelihood) == pytest.approx(100.0, rel=1e-3)
    assert _next_beta(0.3, torch.zeros(10), 5.0) == 1.0
    weights = torch.tensor([0.1, 0.0, 0.6, 0.3])
    counts = torch.bincount(_systematic_resample(weights), minlength=4)
    assert counts.sum() == 4 and counts[1] == 0 and counts[2] >= 2


def _normal_normal_log_evidence(y, m0, s0, sigma):
    n = y.numel()
    covariance = sigma ** 2 * torch.eye(n) + s0 ** 2 * torch.ones(n, n)
    residual = y - m0
    return -0.5 * (n * math.log(2 * math.pi) + torch.logdet(covariance) + residual @ torch.linalg.solve(covariance, residual)).item()


@pytest.mark.integration
@pytest.mark.parametrize("kernel", ["metropolis", "nuts", "slice"])
def test_smc_normal_normal_moments_and_evidence(kernel):
    y = torch.tensor([1.2, 2.1, 0.4, 1.9, 1.5, 0.8, 2.6, 1.1])
    m0, s0, sigma = 0.0, 2.0, 1.0
    with Model() as model:
        mu = RandomParameter("mu", Normal(m0, s0))
        ObservedParameter("y", Normal(mu, sigma), y)
    result = sample_smc(model, n_particles=600, kernel=kernel, seed=0, progress_bar=False)
    post_var = 1 / (1 / s0 ** 2 + y.numel() / sigma ** 2)
    post_mean = post_var * (m0 / s0 ** 2 + y.sum().item() / sigma ** 2)
    samples = result["mu"]
    assert samples.shape == (1, 600, 1)
    assert samples.mean().item() == pytest.approx(post_mean, abs=0.06)
    assert samples.var().item() == pytest.approx(post_var, rel=0.2)
    assert result.log_marginal_likelihood == pytest.approx(_normal_normal_log_evidence(y, m0, s0, sigma), abs=0.25)
    assert result.beta_schedule[0] == 0.0 and result.beta_schedule[-1] == 1.0
    assert all(b < c for b, c in zip(result.beta_schedule, result.beta_schedule[1:]))
    assert len(result.stage_ess) == len(result.stage_acceptance) == len(result.beta_schedule) - 1
    assert result.sample_stats.samplers == ["smc"]
    assert result.sample_stats["lp"].shape == (1, 600)
    # The model's values are restored.
    assert model.params["mu"].constrained_value.item() == pytest.approx(0.0)


@pytest.mark.integration
def test_smc_beta_bernoulli_evidence():
    y = torch.tensor([1.0] * 14 + [0.0] * 6)
    a, b = 2.0, 2.0
    with Model() as model:
        p = RandomParameter("p", Beta(a, b))
        ObservedParameter("y", Bernoulli(p), y)
    result = sample_smc(model, n_particles=800, seed=1, progress_bar=False)
    k, n = 14, 20

    def log_beta(x, z):
        return math.lgamma(x) + math.lgamma(z) - math.lgamma(x + z)

    exact = log_beta(a + k, b + n - k) - log_beta(a, b)
    assert result.log_marginal_likelihood == pytest.approx(exact, abs=0.15)
    assert result["p"].mean().item() == pytest.approx((a + k) / (a + b + n), abs=0.02)


@pytest.mark.integration
def test_smc_finds_both_modes_where_single_chain_nuts_does_not():
    torch.manual_seed(0)
    y = 2.5 + torch.randn(20)
    with Model() as model:
        theta = RandomParameter("theta", Normal(0.0, 3.0))
        magnitude = DeterministicParameter("magnitude", lambda t: t.abs(), lambda t: {"theta": torch.sign(t)}, [theta])
        ObservedParameter("y", Normal(magnitude, 1.0), y)
    smc = sample_smc(model, n_particles=1000, seed=4, progress_bar=False)
    positive = (smc["theta"] > 0).double().mean().item()
    assert 0.35 < positive < 0.65
    torch.testing.assert_close(smc["magnitude"], smc["theta"].abs())
    assert smc["magnitude"].mean().item() == pytest.approx(y.mean().item(), abs=0.1)
    nuts = model.sample(400, 200, n_chains=1, progress_bar=False, check_convergence=False, seed=4)
    nuts_positive = (nuts["theta"] > 0).double().mean().item()
    assert nuts_positive < 0.02 or nuts_positive > 0.98


@pytest.mark.integration
@pytest.mark.parametrize("kernel", ["metropolis", "nuts"])
def test_smc_discrete_parameter_uses_gibbs(kernel):
    y = torch.tensor([3.2, 2.7, 3.9, 2.4])
    with Model() as model:
        k = RandomParameter("k", DiscreteUniform(1, 6))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        location = DeterministicParameter("location", lambda k, mu: k + 0.1 * mu, lambda k, mu: {"k": torch.ones_like(k), "mu": 0.1 * torch.ones_like(mu)}, [k, mu])
        ObservedParameter("y", Normal(location, 1.0), y)
    result = sample_smc(model, n_particles=400, kernel=kernel, n_mutation_steps=4 if kernel == "metropolis" else 2, seed=5, progress_bar=False)
    assert set(result["k"].unique().tolist()) <= set(range(1, 7))
    # Posterior of k with mu integrated out: y_i ~ Normal(k, 1 + 0.01) jointly.
    support = torch.arange(1, 7, dtype=torch.float64)
    n = y.numel()
    covariance = torch.eye(n) + 0.01 * torch.ones(n, n)
    log_marginals = torch.stack([
        -0.5 * (n * math.log(2 * math.pi) + torch.logdet(covariance) + (y - value) @ torch.linalg.solve(covariance, y - value))
        for value in support
    ])
    probabilities = torch.softmax(log_marginals, 0)
    frequencies = torch.stack([(result["k"] == value).double().mean() for value in support])
    torch.testing.assert_close(frequencies, probabilities, atol=0.06, rtol=0)
    exact = (torch.logsumexp(log_marginals, 0) - math.log(6)).item()
    assert result.log_marginal_likelihood == pytest.approx(exact, abs=0.2)


def test_smc_argument_validation():
    with Model() as model:
        RandomParameter("mu", Normal(0.0, 1.0))
    with pytest.raises(ValueError, match="kernel"):
        sample_smc(model, kernel="hmc", progress_bar=False)
    with pytest.raises(ValueError, match="target_ess"):
        sample_smc(model, target_ess=1.5, progress_bar=False)


def test_smc_seed_is_reproducible_and_leaves_global_rng():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.3, -0.2]))
    state = torch.random.get_rng_state()
    first = sample_smc(model, n_particles=50, n_mutation_steps=2, seed=7, progress_bar=False)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = sample_smc(model, n_particles=50, n_mutation_steps=2, seed=7, progress_bar=False)
    torch.testing.assert_close(first["mu"], second["mu"])
    assert first.log_marginal_likelihood == second.log_marginal_likelihood
