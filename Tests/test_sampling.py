import importlib

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Bernoulli, ContinuousReal, DiscreteRange, Exponential, Normal
from BayesianDLL.Samplers import (
    Metropolis, NUTS, PredicativeResult, SamplingBlock, SamplingResult,
    posterior_predicative, sample_predicative,
)
from BayesianDLL.Samplers._sample import _build_blocks, _select_sampler


def normal_logp(x):
    return -0.5 * x.square().sum()


def normal_gradient(x):
    return -x


def make_nuts():
    return NUTS(normal_logp, normal_gradient, lambda x: x, max_depth=5, delta=0.8)


def test_nuts_leapfrog_is_reversible():
    sampler = make_nuts()
    theta, momentum = torch.tensor([0.4, -0.2]), torch.tensor([0.3, 0.7])
    new_theta, new_momentum, grad, logp = sampler.leapfrog(theta, momentum, -theta, 0.1)
    back_theta, back_momentum, _, _ = sampler.leapfrog(new_theta, new_momentum, grad, -0.1)
    torch.testing.assert_close(back_theta, theta)
    torch.testing.assert_close(back_momentum, momentum)
    torch.testing.assert_close(logp, normal_logp(new_theta))


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["nuts", "metropolis"])
def test_samplers_recover_standard_normal(kind):
    sampler = make_nuts() if kind == "nuts" else Metropolis(normal_logp, ContinuousReal())
    sampler.init_sampler()
    theta = torch.zeros(1)
    draws = []
    for i in range(2400):
        theta, step_size, acceptance, diverging = sampler.step(theta, warmup=i < 400)
        assert torch.isfinite(theta).all()
        assert step_size > 0
        assert 0 <= acceptance <= 1
        if i >= 400:
            draws.append(theta.clone())
    samples = torch.stack(draws)
    assert samples.mean().item() == pytest.approx(0, abs=0.18)
    assert samples.var().item() == pytest.approx(1, abs=0.25)
    sampler.reset()
    if kind == "nuts":
        assert not hasattr(sampler, "step_size")
    else:
        assert sampler.m == 0
        assert not sampler.accept_queue


def test_metropolis_rejects_impossible_proposals(monkeypatch):
    sampler = Metropolis(lambda x: torch.tensor(0.0) if x.item() == 0 else torch.tensor(-torch.inf), ContinuousReal())
    monkeypatch.setattr(sampler, "get_proposal", lambda theta: torch.ones_like(theta))
    theta, _, acceptance, diverging = sampler.step(torch.zeros(1))
    torch.testing.assert_close(theta, torch.zeros(1))
    assert acceptance == 0
    assert not diverging


@pytest.mark.integration
def test_discrete_metropolis_recovers_bernoulli():
    dist = Bernoulli(0.7)
    sampler = Metropolis(lambda x: dist.log_pdf(x).sum(), dist.state_space)
    theta = torch.zeros(1)
    draws = []
    for _ in range(2500):
        theta, _, _, _ = sampler.step(theta)
        assert theta.item() in (0, 1)
        draws.append(theta.item())
    assert sum(draws) / len(draws) == pytest.approx(0.7, abs=0.06)


@pytest.mark.parametrize("space,gradient,kind,error", [
    (DiscreteRange(0, 1), normal_gradient, "nuts", "continuous"),
    (ContinuousReal(), None, "nuts", "requires a gradient"),
    (ContinuousReal(), normal_gradient, "unknown", "chosen sampler"),
])
def test_sampler_selection_errors(space, gradient, kind, error):
    with pytest.raises(RuntimeError, match=error):
        _select_sampler(normal_logp, space, gradient, kind)


def test_blocks_pack_unpack_and_auto_selection():
    with Model() as model:
        RandomParameter("a", Normal(0.0, 1.0), shape=2)
        RandomParameter("b", Normal(0.0, 1.0))
        RandomParameter("c", Bernoulli(0.5))
    blocks = _build_blocks(model)
    assert [block.param_names for block in blocks] == [["a", "b"], ["c"]]
    assert isinstance(blocks[0].sampler, NUTS)
    assert isinstance(blocks[1].sampler, Metropolis)
    blocks[0].unpack_and_set(torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(model.params["a"].constrained_value, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(blocks[0].pack(), torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Continuous and discrete"):
        SamplingBlock(model, ["a", "c"])
    with pytest.raises(KeyError, match="not found"):
        SamplingBlock(model, ["missing"])
    with pytest.raises(ValueError, match="multiple blocks"):
        _build_blocks(model, ["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="not included"):
        _build_blocks(model, ["a"])
    with pytest.raises(TypeError, match="Invalid block"):
        _build_blocks(model, [123])


@pytest.mark.integration
def test_public_sampling_runs_multiple_chains_and_records_deterministics():
    # Exercise the real process executor; check structural invariants rather than
    # moments because worker processes have independent random states.
    with Model() as model:
        x = RandomParameter("x", Normal(0.0, 1.0), sampler="metropolis")
        DeterministicParameter("twice", lambda x: 2 * x, lambda x: {"x": 2 * torch.ones_like(x)}, [x])
    result = model.sample(8, 5, n_chains=2, progress_bar=False, check_convergence=False)
    assert isinstance(result, SamplingResult)
    assert result["x"].shape == (2, 8, 1)
    assert torch.isfinite(result["x"]).all()
    torch.testing.assert_close(result["twice"], 2 * result["x"])
    assert result.divergences == [0, 0]
    assert len(result.step_sizes) == len(result.acceptance_probabilities) == 2
    assert model.params["x"].constrained_value.item() == 0


@pytest.mark.parametrize("wrapper", [sample_predicative, posterior_predicative])
def test_predictive_sampling_shapes_and_restoration(normal_model, wrapper):
    normal_model.observed_params["data"].sampler = "metropolis"
    original = normal_model.params["mean"].constrained_value.clone()
    trace = {"mean": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])}
    result = wrapper(trace, n_samples=3, samples_per_step=6, warmup_per_sample=3, model=normal_model, progress_bar=False)
    assert isinstance(result, PredicativeResult)
    assert result["data"].shape == (3, 6, 3)
    assert torch.isfinite(result["data"]).all()
    torch.testing.assert_close(normal_model.params["mean"].constrained_value, original)


def test_predictive_default_uses_all_trace_draws(normal_model):
    normal_model.observed_params["data"].sampler = "metropolis"
    result = sample_predicative({"mean": torch.zeros(2, 3, 1)}, samples_per_step=2, warmup_per_sample=1, model=normal_model, progress_bar=False)
    assert result["data"].shape == (6, 2, 3)


def test_predictive_sampling_resolves_data_backed_plate_at_runtime():
    with Model() as model:
        x = Data("x", torch.arange(4.0))
        y = Data("y", torch.arange(4.0))
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        fitted_mean = DeterministicParameter(
            "fitted_mean",
            lambda mean, x: mean + x,
            lambda mean, x: {"mean": torch.ones_like(x)},
            [mean, x],
        )
        with plate("observations", x):
            likelihood = ObservedParameter(
                "likelihood", Normal(fitted_mean, 1.0), y,
                sampler="metropolis",
            )

    assert likelihood.predictive_shape == (4,)
    x.set_value(torch.arange(2.0))
    assert likelihood.plates[0].size == 2
    assert likelihood.predictive_shape == (2,)

    trace = {"mean": torch.zeros(1, 2, 1)}
    result = model.posterior_predicative(
        trace, n_samples=2, samples_per_step=3, warmup_per_sample=1,
        progress_bar=False,
    )
    assert result["likelihood"].shape == (2, 3, 2)


def test_predictive_rejects_more_draws_than_trace(normal_model):
    with pytest.raises(RuntimeError, match="total trace length"):
        sample_predicative({"mean": torch.zeros(1, 2, 1)}, n_samples=3, model=normal_model, progress_bar=False)


@pytest.mark.parametrize("prior", [False, True])
def test_predictive_wrappers_condition_on_correct_model(normal_model, monkeypatch, prior):
    module = importlib.import_module("BayesianDLL.Samplers._sample")
    original = normal_model.observed_params
    calls = []
    def fake_sample(*args, **kwargs):
        calls.append(set(normal_model.observed_params))
        return {"mean": torch.zeros(1, 2, 1)}
    monkeypatch.setattr(module, "sample", fake_sample)
    normal_model.observed_params["data"].sampler = "metropolis"
    wrapper = normal_model.sample_prior_predicative if prior else normal_model.sample_posterior_predicative
    result = wrapper(n_samples=2, warmup_length=1, samples_per_step=2, warmup_per_sample=1, progress_bar=False)
    assert calls == [set() if prior else {"data"}]
    assert normal_model.observed_params is original
    assert result["data"].shape == (2, 2, 3)


def test_prior_predictive_restores_observations_on_failure(normal_model, monkeypatch):
    module = importlib.import_module("BayesianDLL.Samplers._sample")
    original = normal_model.observed_params
    def fail(*args, **kwargs):
        raise ValueError("sampling failed")
    monkeypatch.setattr(module, "sample", fail)
    with pytest.raises(ValueError, match="sampling failed"):
        normal_model.sample_prior_predicative(progress_bar=False)
    assert normal_model.observed_params is original


def test_predictive_restores_parameters_on_failure(normal_model, monkeypatch):
    module = importlib.import_module("BayesianDLL.Samplers._sample")
    original = normal_model.params["mean"].unconstrained_value.clone()
    class FailingSampler:
        def reset(self):
            pass

        def step(self, *args):
            assert normal_model.params["mean"].constrained_value.item() == 2
            raise ValueError("step failed")
    monkeypatch.setattr(module, "_decide_predicative_step", lambda parameter: FailingSampler())
    with pytest.raises(ValueError, match="step failed"):
        sample_predicative({"mean": torch.full((1, 2, 1), 2.0)}, n_samples=1, model=normal_model, progress_bar=False)
    torch.testing.assert_close(normal_model.params["mean"].unconstrained_value, original)


def test_predictive_transforms_constrained_priors_and_observations():
    with Model() as model:
        rate = RandomParameter("rate", Exponential(1.0))
        ObservedParameter("data", Exponential(rate), torch.ones(1), sampler="metropolis")
    trace = SamplingResult({"rate": torch.tensor([[[2.0], [3.0]]])}, [], [], [], {"twice": torch.tensor([[[4.0], [6.0]]])})
    result = sample_predicative(trace, n_samples=2, samples_per_step=8, warmup_per_sample=5, model=model, progress_bar=False)
    assert (result["data"] > 0).all()
    assert rate.constrained_value.item() == 1
