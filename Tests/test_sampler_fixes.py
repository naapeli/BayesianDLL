"""Regression tests for sampler accounting, diagnostics and plate fixes."""
import inspect
import math
import warnings

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate, sample, sample_smc
from BayesianDLL.Distributions import Bernoulli, ContinuousRange, DiscreteRange, Normal
from BayesianDLL.Evaluation import gelman_rubin, summary
from BayesianDLL.Samplers import NUTS, FiniteGibbs, Metropolis, SampleStats, SamplingResult
from BayesianDLL.Samplers import _sample as sample_module
from BayesianDLL.Samplers._vectorized import sample_vectorized
from BayesianDLL.Samplers._vectorized_samplers import VectorizedMetropolis, VectorizedNUTS


# ---------------------------------------------------------------- Metropolis out-of-support proposals
def _run_scalar(sampler, theta, n_steps, warmup):
    """Rates of moves, of accepted proposals and the mean accept_stat over ``n_steps`` steps."""
    moved, accepted, accept_stat = [], [], []
    for _ in range(n_steps):
        new, *_ = sampler.step(theta, warmup=warmup)
        moved.append(float(not torch.equal(new, theta)))
        accepted.append(sampler.last_stats["accepted"])
        accept_stat.append(sampler.last_stats["accept_stat"])
        theta = new
    mean = lambda values: sum(values) / len(values)
    return theta, mean(moved), mean(accepted), mean(accept_stat)


def test_metropolis_out_of_support_proposal_is_a_rejection():
    torch.manual_seed(0)
    # Flat target on [0, 1]: a proposal is accepted iff it stays in the support.
    sampler = Metropolis(lambda x: torch.zeros(()), ContinuousRange(0.0, 1.0), proposal_scale=5.0)
    theta, *_ = _run_scalar(sampler, torch.full((1,), 0.5), 500, warmup=True)
    # The adaptation reaches the target acceptance rate of in-support moves ...
    theta, moved, accepted, accept_stat = _run_scalar(sampler, theta, 3000, warmup=True)
    assert accepted == moved
    assert accept_stat == moved
    assert moved == pytest.approx(sampler.target_accept_rate, abs=0.04)
    # ... and the rolling rate is the fraction of moves of the last 20 steps.
    moves = []
    for _ in range(20):
        new, _, rolling, _ = sampler.step(theta)
        moves.append(float(not torch.equal(new, theta)))
        theta = new
    assert rolling == pytest.approx(sum(moves) / 20)


def test_discrete_metropolis_boundary_counts_rejections():
    torch.manual_seed(1)
    sampler = Metropolis(lambda x: torch.zeros(()), DiscreteRange(0, 2))
    _, moved, accepted, accept_stat = _run_scalar(sampler, torch.ones(1), 3000, warmup=False)
    assert accepted == moved
    assert accept_stat == moved
    # From 1 both moves stay inside, from 0 or 2 half of them leave: rate 2/3.
    assert moved == pytest.approx(2 / 3, abs=0.04)


def test_vectorized_metropolis_out_of_support_proposal_is_a_rejection():
    torch.manual_seed(2)
    n_chains = 4
    sampler = VectorizedMetropolis(lambda x: torch.zeros(x.shape[0]), ContinuousRange(0.0, 1.0), n_chains, proposal_scale=5.0)
    theta = torch.full((n_chains, 1), 0.5)
    for _ in range(500):
        theta, *_ = sampler.step(theta, warmup=True)
    moved = accepted = accept_stat = 0.0
    for _ in range(1000):
        new, *_ = sampler.step(theta, warmup=True)
        moved += float((new != theta).any(-1).float().mean())
        accepted += float(sampler.last_stats["accepted"].mean())
        accept_stat += float(sampler.last_stats["accept_stat"].mean())
        theta = new
    assert accepted == pytest.approx(moved)
    assert accept_stat == pytest.approx(moved)
    assert moved / 1000 == pytest.approx(sampler.target_accept_rate, abs=0.04)


# ---------------------------------------------------------------- acceptance accumulators
def _normal_model():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.3, -0.2, 0.5]))
    return model


@pytest.mark.parametrize("chain_method", ["parallel", "vectorized"])
def test_acceptance_probabilities_are_the_mean_accept_stat(chain_method):
    # Without warmup every iteration is recorded, so the mean over all
    # iterations equals the mean of the recorded accept_stat.
    result = sample(20, 0, n_chains=2, model=_normal_model(), progress_bar=False, seed=0,
                    check_convergence=False, chain_method=chain_method, log_likelihood=False)
    means = result.sample_stats["accept_stat"].mean(1)
    for chain in range(2):
        assert result.acceptance_probabilities[chain][0] == pytest.approx(float(means[chain]), abs=1e-6)


# ---------------------------------------------------------------- init_scale rename
def test_init_scale_replaces_start_point_variance():
    for function in (sample, Model.sample, sample_vectorized, sample_module._sample_single_chain):
        parameters = inspect.signature(function).parameters
        assert "init_scale" in parameters
        assert "start_point_variance" not in parameters
    with pytest.raises(TypeError):
        sample(5, 5, n_chains=1, model=_normal_model(), progress_bar=False, chain_method="vectorized",
               start_point_variance=0.1)


# ---------------------------------------------------------------- R-hat with zero within-chain variance
def _stuck(values, n=60):
    return torch.tensor(values, dtype=torch.float64)[:, None].expand(len(values), n).clone()


@pytest.mark.parametrize("method", ["classical", "split", "rank"])
def test_rhat_zero_within_chain_variance(method):
    different = gelman_rubin({"x": _stuck([0.0, 1.0, 2.0, 3.0])}, method=method)["x"]
    same = gelman_rubin({"x": _stuck([2.0, 2.0, 2.0, 2.0])}, method=method)["x"]
    assert math.isinf(float(different)) and float(different) > 0
    assert math.isnan(float(same))
    torch.manual_seed(0)
    normal = gelman_rubin({"x": torch.randn(4, 200)}, method=method)["x"]
    assert 0.98 < float(normal) < 1.03


def test_summary_and_convergence_warning_with_degenerate_rhat():
    traces = {"a": _stuck([0.0, 1.0]).unsqueeze(-1), "b": _stuck([1.0, 1.0])}
    table = summary(traces)
    assert math.isinf(table.loc["a[0]", "r_hat"])
    assert math.isnan(table.loc["b", "r_hat"])

    model = _normal_model()

    def finish(trace):
        n_samples = trace.shape[1]
        return sample_module._finish_sampling(
            model, n_samples, 2, True, {"mu": trace}, {}, {}, None, [0, 0], [[1.0], [1.0]], [[1.0], [1.0]],
            SampleStats(), [], {},
        )

    with pytest.warns(UserWarning, match="Gelman–Rubin R-hat of mu"):
        finish(_stuck([0.0, 1.0]).unsqueeze(-1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        finish(_stuck([1.0, 1.0]).unsqueeze(-1))


# ---------------------------------------------------------------- nested plates sized by 1-d Data / tensors
def _nested_plate_model(outer, inner):
    y = torch.arange(6.0).reshape(2, 3, 1) / 4
    with Model() as model:
        sizes = {"outer": outer, "inner": inner}
        for key, size in sizes.items():
            if size == "data":
                sizes[key] = Data(f"{key}_size", torch.zeros(2 if key == "outer" else 3))
            elif size == "tensor":
                sizes[key] = torch.zeros(2 if key == "outer" else 3)
        with plate("outer", sizes["outer"]):
            with plate("inner", sizes["inner"]):
                z = RandomParameter("z", Normal(0.0, 1.0))
                observed = ObservedParameter("y", Normal(z, 1.0), y)
    z.set_constrained_value(torch.linspace(-1.0, 1.0, 6).reshape(z.constrained_value.shape))
    return model, z, observed


@pytest.mark.parametrize("outer, inner", [(2, "data"), ("data", 3), ("tensor", "tensor")])
def test_nested_plates_sized_by_one_dimensional_data_or_tensors(outer, inner):
    reference_model, reference_z, reference_y = _nested_plate_model(2, 3)
    model, z, observed = _nested_plate_model(outer, inner)
    assert z.constrained_value.shape == reference_z.constrained_value.shape
    assert z.constrained_value.shape[:2] == (2, 3)
    assert observed.predictive_shape == reference_y.predictive_shape == (2, 3, 1)
    assert [p.size for p in observed.plates] == [2, 3]
    torch.testing.assert_close(model.model_log_prob(), reference_model.model_log_prob())


def test_one_dimensional_plate_size_rejects_other_dims():
    with Model():
        with plate("outer", 2):
            with pytest.raises(ValueError, match="dim"):
                with plate("inner", torch.zeros(3), dim=-2):
                    RandomParameter("z", Normal(0.0, 1.0))


# ---------------------------------------------------------------- thin() keeps the SMC attributes
def test_thin_keeps_smc_attributes():
    model = _normal_model()
    result = sample_smc(model, n_particles=40, n_mutation_steps=1, seed=3, progress_bar=False)
    thinned = result.thin(2)
    assert thinned["mu"].shape[1] == 20
    for name in ("log_marginal_likelihood", "beta_schedule", "stage_ess", "stage_acceptance", "kernel"):
        assert getattr(thinned, name) == getattr(result, name)
        assert getattr(result, name) is not None
    plain = SamplingResult({"x": torch.zeros(1, 4)}, [0], [[1.0]], [[1.0]])
    assert plain.log_marginal_likelihood is None and plain.thin(2).kernel is None


# ---------------------------------------------------------------- FiniteGibbs lp after the independent update
def test_finite_gibbs_independent_update_records_finite_lp():
    torch.manual_seed(0)
    distribution = Bernoulli(torch.tensor([0.2, 0.7, 0.5]))
    log_target = lambda x: distribution.log_pdf(x).sum()
    sampler = FiniteGibbs(log_target, distribution.state_space, elementwise_log_target=distribution.log_pdf)
    theta = torch.zeros(3)
    for _ in range(3):
        theta, *_ = sampler.step(theta)
        assert math.isfinite(sampler.last_stats["lp"])
        assert sampler.last_stats["lp"] == pytest.approx(float(log_target(theta)))


# ---------------------------------------------------------------- NUTS uses eps_bar from the first post-warmup draw
def test_nuts_first_post_warmup_step_uses_averaged_step_size():
    torch.manual_seed(0)
    sampler = NUTS(lambda x: -0.5 * x.square().sum(), lambda x: -x, lambda x: x, max_depth=4)
    theta = torch.zeros(2)
    for _ in range(30):
        theta, *_ = sampler.step(theta, warmup=True)
    step_size_bar = sampler.step_size_bar
    assert sampler.step_size != step_size_bar
    for _ in range(3):
        theta, next_step_size, *_ = sampler.step(theta, warmup=False)
        assert sampler.last_stats["step_size"] == step_size_bar
        assert next_step_size == step_size_bar


def test_vectorized_nuts_first_post_warmup_step_uses_averaged_step_size():
    torch.manual_seed(0)
    sampler = VectorizedNUTS(lambda x: -0.5 * x.square().sum(-1), lambda x: -x, 3, max_depth=4)
    theta = torch.zeros(3, 2)
    for _ in range(30):
        theta, *_ = sampler.step(theta, warmup=True)
    step_size_bar = sampler.step_size_bar.clone()
    assert not torch.equal(sampler.step_size, step_size_bar)
    for _ in range(3):
        theta, next_step_size, *_ = sampler.step(theta, warmup=False)
        torch.testing.assert_close(sampler.last_stats["step_size"], step_size_bar)
        torch.testing.assert_close(next_step_size, step_size_bar)


# ---------------------------------------------------------------- exact warmup / recording counts
def _metropolis_model():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0), sampler="metropolis")
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.3, -0.2]))
    return model


def _count_calls(monkeypatch, cls):
    flags = []
    original = cls.step

    def step(self, theta, warmup=False):
        flags.append(bool(warmup))
        new_theta, step_size, acceptance, diverging = original(self, theta, warmup=warmup)
        always = torch.ones_like(torch.as_tensor(diverging), dtype=torch.bool)
        return new_theta, step_size, acceptance, always if always.ndim else True

    monkeypatch.setattr(cls, "step", step)
    return flags


@pytest.mark.parametrize("warmup_length, n_samples", [(0, 3), (1, 2), (4, 3)])
def test_parallel_chain_warmup_and_recording_counts(monkeypatch, warmup_length, n_samples):
    flags = _count_calls(monkeypatch, Metropolis)
    model = _metropolis_model()
    model.prepare_enumeration()
    initial = {name: p.unconstrained_value for name, p in model.params.items()}
    trace, _, divergences, acceptance, *_ = sample_module._sample_single_chain(
        0, 1, model, initial, 1.0, n_samples, warmup_length, False, seed=0)
    assert flags == [True] * warmup_length + [False] * n_samples
    assert trace["mu"].shape[0] == n_samples
    assert divergences == n_samples


@pytest.mark.parametrize("warmup_length, n_samples", [(0, 3), (1, 2), (4, 3)])
def test_vectorized_chain_warmup_and_recording_counts(monkeypatch, warmup_length, n_samples):
    flags = _count_calls(monkeypatch, VectorizedMetropolis)
    with pytest.warns(UserWarning, match="divergences"):
        result = sample(n_samples, warmup_length, n_chains=2, model=_metropolis_model(), progress_bar=False, seed=0,
                        chain_method="vectorized", check_convergence=False, log_likelihood=False)
    assert flags == [True] * warmup_length + [False] * n_samples
    assert result["mu"].shape[:2] == (2, n_samples)
    assert result.sample_stats["accept_stat"].shape == (2, n_samples)
    assert result.divergences == [n_samples, n_samples]


def test_nuts_warmup_length_matches_the_adapting_iterations(monkeypatch):
    lengths = []
    original = NUTS.set_warmup_length
    monkeypatch.setattr(NUTS, "set_warmup_length", lambda self, n: (lengths.append(n), original(self, n))[1])
    model = _normal_model()
    model.prepare_enumeration()
    initial = {name: p.unconstrained_value for name, p in model.params.items()}
    sample_module._sample_single_chain(0, 1, model, initial, 1.0, 2, 25, False, seed=0)
    assert lengths == [25]
