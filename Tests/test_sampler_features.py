"""Discrete samplers, Gibbs block targets, dense NUTS metric, sampler statistics and seeding."""

import math
import threading
import warnings

import pytest
import torch

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter, condition, plate
from BayesianDLL.Distributions import Bernoulli, ContinuousReal, DiscretePositive, DiscreteRange, Normal
from BayesianDLL.Distributions._state_space import ContinuousPositive, ContinuousSimplex, JointStateSpace
from BayesianDLL.Evaluation import bfmi, ess_bulk
from BayesianDLL.Samplers import FiniteGibbs, Metropolis, NUTS, SampleStats, SamplingBlock, SamplingResult
from BayesianDLL.Samplers._sample import _build_blocks, _check_hamiltonian_diagnostics, _select_sampler
from BayesianDLL._active_model import _active_model
from BayesianDLL._active_model._active_model import get_active_model
from BayesianDLL._parameters import ObservedParameter as ObservedParameterClass


# ----------------------------------------------------------------- state spaces
def test_discrete_contains_is_elementwise_and_returns_bool():
    space = DiscreteRange(0, 2)
    assert space.contains(torch.tensor([0.0, 1.0, 2.0])) is True
    assert space.contains(torch.tensor([[0.0, 1.0], [2.0, 1.0]])) is True
    assert space.contains(torch.tensor([0.0, 3.0])) is False
    assert space.contains(torch.tensor([0.0, 0.5])) is False
    assert space.contains(torch.tensor([0, 1, 2])) is True
    assert space.contains(torch.tensor(1.0)) is True

    positive = DiscretePositive()
    assert positive.contains(torch.tensor([1.0, 7.0])) is True
    assert positive.contains(torch.tensor([1.0, 0.0])) is False
    assert positive.contains(torch.tensor([1.5])) is False

    assert ContinuousReal().contains(torch.tensor([1.0, -2.0])) is True
    assert ContinuousPositive().contains(torch.tensor([1.0, -2.0])) is False
    assert ContinuousSimplex().contains(torch.tensor([[0.2, 0.8], [0.5, 0.5]])) is True

    joint = JointStateSpace(
        {"a": DiscreteRange(0, 1), "b": DiscretePositive()},
        {"a": (0, 2, (2,)), "b": (2, 3, (1,))},
    )
    assert joint.contains(torch.tensor([0.0, 1.0, 4.0])) is True
    assert joint.contains(torch.tensor([0.0, 2.0, 4.0])) is False


@pytest.mark.integration
def test_vector_bernoulli_latents_sample():
    with Model() as model:
        with plate("n", 20):
            RandomParameter("z", Bernoulli(torch.tensor(0.5, dtype=torch.float64)))
    result = model.sample(150, 50, n_chains=2, progress_bar=False, check_convergence=False, seed=0)
    assert result["z"].shape == (2, 150, 20, 1)
    assert set(result["z"].unique().tolist()) <= {0.0, 1.0}
    assert result["z"].mean().item() == pytest.approx(0.5, abs=0.05)
    assert result.sample_stats.samplers == ["gibbs"]


# ------------------------------------------------------------------- Metropolis
def counting(fn):
    calls = []

    def wrapped(x):
        calls.append(1)
        return fn(x)

    return wrapped, calls


def test_metropolis_evaluates_target_once_per_continuous_step():
    target, calls = counting(lambda x: -0.5 * x.square().sum())
    sampler = Metropolis(target, ContinuousReal())
    theta = torch.zeros(2)
    theta, *_ = sampler.step(theta)
    assert len(calls) == 2  # current state (not cached yet) and proposal
    for _ in range(10):
        theta, *_ = sampler.step(theta)
    assert len(calls) == 12
    # A different tensor invalidates the cache.
    sampler.step(theta.clone())
    assert len(calls) == 14


def test_discrete_metropolis_updates_one_coordinate_at_a_time():
    space = DiscreteRange(0, 1)
    target, calls = counting(lambda x: torch.zeros(()))
    sampler = Metropolis(target, space)
    theta = torch.zeros(30)
    acceptance = []
    for _ in range(20):
        new_theta, _, _, _ = sampler.step(theta)
        acceptance.append(sampler.last_stats["accept_stat"])
        # Every in-support single-coordinate move is accepted for a flat target.
        theta = new_theta
    assert sum(acceptance) / len(acceptance) > 0.3
    assert set(theta.unique().tolist()) <= {0.0, 1.0}
    assert 0 < theta.sum() < 30


def test_proposal_variance_keyword_is_removed():
    with pytest.raises(TypeError):
        Metropolis(lambda x: -x.square().sum(), ContinuousReal(), proposal_variance=0.3)
    sampler = Metropolis(lambda x: -x.square().sum(), ContinuousReal(), proposal_scale=0.3)
    assert sampler.proposal_scale == 0.3
    assert not hasattr(sampler, "proposal_variance")
    assert not hasattr(sampler, "adapt_proposal_variance")


# ------------------------------------------------------------------ FiniteGibbs
def test_finite_gibbs_samples_exact_conditionals():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    log_probs = probs.log()

    def target(x):
        return log_probs[x.long()].sum()

    sampler = FiniteGibbs(target, DiscreteRange(0, 3))
    theta = torch.zeros(2)
    draws = []
    for _ in range(3000):
        theta, *_ = sampler.step(theta)
        draws.append(theta.clone())
    draws = torch.stack(draws)
    frequencies = torch.stack([(draws == k).double().mean() for k in range(4)])
    torch.testing.assert_close(frequencies, probs.double(), atol=0.03, rtol=0)


def test_finite_gibbs_elementwise_update_and_auto_selection():
    p = torch.tensor([0.1, 0.9])
    sampler = FiniteGibbs(
        lambda x: (x * p.log() + (1 - x) * (1 - p).log()).sum(), DiscreteRange(0, 1),
        elementwise_log_target=lambda x: x * p.log() + (1 - x) * (1 - p).log(),
    )
    draws = torch.stack([sampler.step(torch.zeros(2))[0] for _ in range(2000)])
    torch.testing.assert_close(draws.mean(0), p.double(), atol=0.03, rtol=0)

    assert isinstance(_select_sampler(lambda x: x.sum(), DiscreteRange(0, 1)), FiniteGibbs)
    assert isinstance(_select_sampler(lambda x: x.sum(), DiscretePositive()), Metropolis)
    assert isinstance(_select_sampler(lambda x: x.sum(), DiscreteRange(0, 1), sampler="metropolis"), Metropolis)
    with pytest.raises(RuntimeError, match="finite support"):
        _select_sampler(lambda x: x.sum(), DiscretePositive(), sampler="gibbs")


def test_sampler_settings_are_routed_to_the_samplers_that_accept_them():
    with Model() as model:
        RandomParameter("a", Normal(0.0, 1.0), shape=2)
        RandomParameter("c", Bernoulli(0.5))
        RandomParameter("d", Normal(0.0, 1.0), sampler="metropolis", proposal_scale=0.2)
    blocks = _build_blocks(model, sampler_params={"metric": "dense", "max_depth": 6})
    assert [block.sampler_type for block in blocks] == ["nuts", "metropolis", "gibbs"]
    assert blocks[0].sampler.metric == "dense" and blocks[0].sampler.max_depth == 6
    assert blocks[1].sampler.proposal_scale == 0.2
    with pytest.raises(TypeError, match="Unknown sampler setting"):
        _build_blocks(model, sampler_params={"not_a_setting": 1})


# ---------------------------------------------------------------- block targets
def chain_model():
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(a, 1.0))
        shifted = DeterministicParameter("shifted", lambda b: b + 1, lambda b: {"b": torch.ones_like(b)}, [b])
        c = RandomParameter("c", Normal(shifted, 1.0))
        ObservedParameter("obs", Normal(c, 1.0), torch.tensor([0.3, -0.2]))
    return model


def test_block_terms_are_the_markov_blanket():
    model = chain_model()
    random_terms, observed_terms, nodes = model.block_terms(["a"])
    assert [p.name for p in random_terms] == ["a", "b"]
    assert observed_terms == []
    random_terms, observed_terms, nodes = model.block_terms(["b"])
    assert [p.name for p in random_terms] == ["b", "c"]
    assert nodes == {"b", "shifted", "c"}
    random_terms, observed_terms, _ = model.block_terms(["c"])
    assert [p.name for p in random_terms] == ["c"]
    assert [o.name for o in observed_terms] == ["obs"]


@pytest.mark.parametrize("names", [["a"], ["b"], ["c"], ["a", "c"], ["a", "b", "c"]])
def test_block_log_target_matches_full_log_prob_differences(names):
    model = chain_model()
    block = SamplingBlock(model, names)
    theta0 = torch.randn(len(names), dtype=torch.float64)
    theta1 = torch.randn(len(names), dtype=torch.float64)

    def full(theta):
        block.unpack_and_set(theta)
        return model.model_log_prob()

    block_diff = block.sampler.log_target(theta1) - block.sampler.log_target(theta0)
    torch.testing.assert_close(block_diff, full(theta1) - full(theta0))

    # The restricted gradient equals the gradient of the full joint density.
    block.unpack_and_set(theta1)
    expected = model.joint_grad_log_prob(names)
    restricted = block.sampler.gradient(theta1)
    torch.testing.assert_close(restricted, torch.cat([expected[n].reshape(-1) for n in names]))


def test_single_block_log_target_equals_full_log_prob_up_to_a_constant():
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        ObservedParameter("obs", Normal(a, 1.0), torch.tensor([0.5]))
        ObservedParameter("const", Normal(0.0, 1.0), torch.tensor([2.0]))  # no parameter dependence
    block = SamplingBlock(model, ["a"])
    differences = []
    for value in (-1.0, 0.0, 2.5):
        theta = torch.tensor([value], dtype=torch.float64)
        block_value = block.sampler.log_target(theta)
        differences.append(model.model_log_prob() - block_value)
    torch.testing.assert_close(differences[0], differences[1])
    torch.testing.assert_close(differences[0], differences[2])


@pytest.mark.integration
@pytest.mark.parametrize("blocks", [[["a", "b"]], [["a"], ["b"]]])
def test_multi_block_gibbs_matches_joint_posterior(blocks):
    # a ~ N(0, 1), b ~ N(a, 1), y = 1 ~ N(b, 1): posterior mean (1/3, 2/3), covariance [[2, 1], [1, 2]] / 3.
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(a, 1.0))
        ObservedParameter("y", Normal(b, 1.0), torch.tensor([1.0]))
    result = model.sample(1000, 300, n_chains=2, progress_bar=False, blocks=blocks, seed=11)
    draws = torch.cat([result["a"], result["b"]], dim=-1).reshape(-1, 2)
    torch.testing.assert_close(draws.mean(0), torch.tensor([1 / 3, 2 / 3], dtype=torch.float64), atol=0.12, rtol=0)
    torch.testing.assert_close(torch.cov(draws.T), torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64) / 3, atol=0.12, rtol=0)


@pytest.mark.integration
def test_gibbs_with_discrete_and_continuous_blocks_has_no_stale_caches():
    # z ~ Bernoulli(0.5), mu ~ N(0, 1), y = 2 ~ N(mu + 3 z, 1).
    # Marginally y | z ~ N(3 z, 2), so P(z = 1 | y) = 1 / (1 + exp(-0.75)).
    with Model() as model:
        z = RandomParameter("z", Bernoulli(0.5))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        mean = DeterministicParameter(
            "mean", lambda mu, z: mu + 3 * z,
            lambda mu, z: {"mu": torch.ones_like(mu), "z": 3 * torch.ones_like(z)}, [mu, z],
        )
        ObservedParameter("y", Normal(mean, 1.0), torch.tensor([2.0]))
    result = model.sample(1500, 300, n_chains=2, progress_bar=False, seed=5)
    p_z = 1 / (1 + math.exp(-0.75))
    assert result["z"].mean().item() == pytest.approx(p_z, abs=0.06)
    expected_mu = p_z * (-0.5) + (1 - p_z) * 1.0
    assert result["mu"].mean().item() == pytest.approx(expected_mu, abs=0.12)


# --------------------------------------------------------------- dense metric
@pytest.mark.integration
def test_dense_metric_learns_correlation_and_beats_diagonal():
    covariance = torch.tensor([[1.0, 1.98], [1.98, 4.0]], dtype=torch.float64)
    precision = torch.linalg.inv(covariance)
    results = {}
    for metric in ("diag", "dense"):
        torch.manual_seed(0)
        sampler = NUTS(lambda x: -0.5 * x @ precision @ x, lambda x: -(precision @ x), lambda x: x, metric=metric)
        sampler.set_warmup_length(400)
        theta = torch.zeros(2, dtype=torch.float64)
        draws = []
        for i in range(1400):
            theta, _, _, _ = sampler.step(theta, warmup=i < 400)
            if i >= 400:
                draws.append(theta)
        draws = torch.stack(draws)
        results[metric] = (sampler, draws, ess_bulk({"x": draws.unsqueeze(0)})["x"].min().item())

    dense, dense_draws, dense_ess = results["dense"]
    diag, _, diag_ess = results["diag"]
    assert dense.inv_metric.shape == (2, 2)
    torch.testing.assert_close(dense.inv_metric, covariance, rtol=0.35, atol=0.1)
    torch.testing.assert_close(torch.cov(dense_draws.T), covariance, rtol=0.25, atol=0.1)
    assert dense.step_size > 2 * diag.step_size
    assert dense_ess > diag_ess


def test_dense_momentum_has_metric_covariance():
    sampler = NUTS(lambda x: x.sum(), lambda x: torch.ones_like(x), lambda x: x, metric="dense")
    inv_metric = torch.tensor([[2.0, 0.6], [0.6, 0.5]], dtype=torch.float64)
    sampler.inv_metric = inv_metric
    sampler._inv_metric_chol = torch.linalg.cholesky(inv_metric)
    momenta = torch.stack([sampler._sample_momentum(torch.zeros(2, dtype=torch.float64)) for _ in range(20000)])
    torch.testing.assert_close(torch.cov(momenta.T), torch.linalg.inv(inv_metric), rtol=0.1, atol=0.05)
    r = torch.tensor([0.3, -1.0], dtype=torch.float64)
    torch.testing.assert_close(sampler._kinetic(r), 0.5 * r @ inv_metric @ r)
    with pytest.raises(ValueError, match="metric"):
        NUTS(lambda x: x, lambda x: x, lambda x: x, metric="full")


# --------------------------------------------------------------- statistics
@pytest.mark.integration
def test_sample_stats_are_recorded_per_block():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        z = RandomParameter("z", Bernoulli(0.5))
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.5, 1.0]))
    result = model.sample(60, 40, n_chains=2, progress_bar=False, check_convergence=False, seed=3)
    stats = result.sample_stats
    assert isinstance(stats, SampleStats)
    assert stats.samplers == ["nuts", "gibbs"]
    assert stats.params == [["mu"], ["z"]]
    for name in ("step_size", "accept_stat", "tree_depth", "n_leapfrog", "diverging", "energy", "lp"):
        assert stats[0][name].shape == (2, 60)
    assert stats["energy"].shape == (2, 60)
    assert ((stats["tree_depth"] >= 1) & (stats["n_leapfrog"] >= 1)).all()
    with pytest.raises(KeyError, match="Several"):
        stats["lp"]
    # Existing attributes keep their structure.
    assert len(result.divergences) == 2 and len(result.step_sizes[0]) == 2
    values = bfmi(result)
    assert values.shape == (2,) and (values > 0).all()
    assert result.thin(3).sample_stats["energy"].shape == (2, 20)


def test_bfmi_formula_and_warnings():
    torch.manual_seed(0)
    iid = torch.randn(2, 4000, dtype=torch.float64)
    torch.testing.assert_close(bfmi(iid), torch.full((2,), 2.0, dtype=torch.float64), atol=0.15, rtol=0)
    random_walk = torch.cumsum(torch.randn(2, 500, dtype=torch.float64), dim=1)
    assert (bfmi(random_walk) < 0.3).all()

    stats = SampleStats([{
        "params": ["x"], "sampler": "nuts",
        "stats": {"energy": random_walk, "tree_depth": torch.full((2, 500), 4.0)},
    }])
    with pytest.warns(UserWarning) as record:
        _check_hamiltonian_diagnostics(stats, [4], 500)
    messages = " ".join(str(w.message) for w in record)
    assert "E-BFMI" in messages and "maximum tree depth" in messages

    with pytest.raises(ValueError, match="no energy"):
        bfmi(SamplingResult({}, [], [], []))


# ------------------------------------------------------------------- seeding
@pytest.mark.integration
def test_seed_makes_runs_reproducible_and_chains_distinct():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.5]))
    kwargs = dict(n_chains=2, progress_bar=False, check_convergence=False)
    first = model.sample(20, 10, seed=123, **kwargs)
    second = model.sample(20, 10, seed=123, **kwargs)
    other = model.sample(20, 10, seed=124, **kwargs)
    torch.testing.assert_close(first["mu"], second["mu"], rtol=0, atol=0)
    torch.testing.assert_close(first.sample_stats["energy"], second.sample_stats["energy"], rtol=0, atol=0)
    assert not torch.equal(first["mu"], other["mu"])
    assert not torch.equal(first["mu"][0], first["mu"][1])

    unseeded = model.sample(20, 10, **kwargs)
    assert not torch.equal(unseeded["mu"][0], unseeded["mu"][1])


# -------------------------------------------------------------- model internals
def test_active_model_is_a_thread_local_stack():
    assert _active_model._active_model is None
    with Model() as outer:
        assert get_active_model() is outer
        with Model() as inner:
            assert _active_model._active_model is inner
            seen = []
            thread = threading.Thread(target=lambda: seen.append(get_active_model()))
            thread.start()
            thread.join()
            assert seen == [None]
        assert _active_model._active_model is outer
        RandomParameter("x", Normal(0.0, 1.0))
    assert get_active_model() is None
    assert "x" in outer.params and "x" not in inner.params


def test_model_exit_does_not_hide_errors(monkeypatch):
    def broken_compile(self):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(Model, "compile", broken_compile)
    with pytest.raises(ValueError, match="original"):
        with Model():
            raise ValueError("original")
    assert get_active_model() is None


def test_condition_builds_a_real_observed_parameter():
    with Model() as model:
        with plate("n", 2):
            x = RandomParameter("x", Normal(0.0, 1.0), sampler="metropolis", proposal_scale=0.3)
        ObservedParameter("y", Normal(x, 1.0), torch.tensor([[1.0], [2.0]]))

    conditioned = condition(model, x=torch.tensor([[0.5], [1.5]]))
    observed = conditioned.observed_params["x"]
    assert type(observed) is ObservedParameterClass
    assert [p.name for p in observed.plates] == ["n"]
    assert observed.sampler == "metropolis" and observed.sampler_params == {"proposal_scale": 0.3}
    assert type(model.params["x"]).__name__ == "RandomParameter"
    # The child's distribution refers to the new observed parameter.
    child_distribution = conditioned.observed_params["y"].distribution
    assert any(value is observed for value in vars(child_distribution).values())

    def expected(value):
        value = torch.tensor(value, dtype=torch.float64)
        return (
            torch.distributions.Normal(0.0, 1.0).log_prob(value).sum()
            + torch.distributions.Normal(value, 1.0).log_prob(torch.tensor([[1.0], [2.0]])).sum()
        )

    torch.testing.assert_close(conditioned.model_log_prob(), expected([[0.5], [1.5]]))
    # Conditioning the conditioned model again updates every reference.
    again = condition(conditioned, x=torch.tensor([[2.0], [0.0]]))
    torch.testing.assert_close(again.model_log_prob(), expected([[2.0], [0.0]]))
    torch.testing.assert_close(conditioned.model_log_prob(), expected([[0.5], [1.5]]))
