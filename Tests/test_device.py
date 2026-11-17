"""Device and dtype cleanliness of the library.

The same scenarios (vectorized NUTS, Gibbs, slice, SMC, BBVI, Laplace,
predictive, GP, enumeration, LOO) run

* on ``torch.device("cuda")`` when a GPU is available (on AMD GPUs with the
  ROCm build of PyTorch, which also exposes the GPU as ``"cuda"``); skipped
  otherwise. Run with ``uv run pytest -q Tests/test_device.py``;
* on the CPU with every torch factory function patched to fail when library
  code creates a floating-point tensor without an explicit ``device`` --
  such a tensor would land on the CPU while the model lives on a GPU;
* on the CPU with a float64 model while torch's default dtype is float32: a
  tensor created with the default dtype (a dtype leak, usually also a device
  leak) changes the dtype of the results.
"""

import sys
import warnings

import pytest
import torch

from BayesianDLL import (
    Data, Model, ObservedParameter, RandomParameter, laplace_approximation, plate, sample, sample_smc,
)
from BayesianDLL.Deterministic import Affine, Index
from BayesianDLL.Distributions import Bernoulli, HalfCauchy, HalfNormal, Normal, Uniform
from BayesianDLL.Evaluation import loo
from BayesianDLL.GP import ExactGP, RBF
from BayesianDLL.Variational import BBVI
from BayesianDLL import AutoNormal


CUDA = torch.cuda.is_available()


# ============================================================ models
def eight_schools():
    y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    with Model() as model:
        school = Data("school", torch.arange(8))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        with plate("schools", 8):
            eta = RandomParameter("eta", Normal(0.0, 1.0), shape=())
        theta = Affine("theta", eta, loc=mu, scale=tau)
        ObservedParameter("y", Normal(Index("theta_obs", theta, school), sigma), y)
    return model


def gibbs_model():
    y = torch.tensor([-1.0, 2.2, 1.8])
    with Model() as model:
        means = RandomParameter("means", Normal(0.0, 3.0), shape=2)
        with plate("data", 3):
            z = RandomParameter("z", Bernoulli(0.5), shape=())
        ObservedParameter("y", Normal(Index("mz", means, z), 1.0), y)
    return model


def slice_model():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 2.0), sampler="slice")
        sigma = RandomParameter("sigma", HalfNormal(1.0), sampler="slice")
        ObservedParameter("y", Normal(mu, sigma), torch.tensor([0.3, 1.2, -0.4, 2.0]))
    return model


def enumerated_mixture():
    y = torch.tensor([-2.1, -1.7, -2.5, 1.9, 2.4, 2.2])
    with Model() as model:
        means = RandomParameter("means", Normal(torch.tensor([-2.0, 2.0]), 1.0), shape=2)
        with plate("data", len(y)):
            z = RandomParameter("z", Bernoulli(0.4), shape=(), enumerate=True)
        ObservedParameter("y", Normal(Index("mean", means, z), 0.7), y)
    return model


def gp_model():
    x = torch.linspace(0.0, 1.0, 6)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Uniform(0.2, 1.0))
        variance = RandomParameter("variance", Uniform(0.25, 2.0))
        ObservedParameter("y", ExactGP("f", inputs, RBF(lengthscale, variance), noise_variance=0.05), torch.sin(6 * x))
    return model


def regression_model():
    x = torch.linspace(-1.0, 1.0, 8)
    with Model() as model:
        X = Data("x", x)
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(0.0, 1.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        with plate("obs", X):
            ObservedParameter("y", Normal(Affine("mean", X, loc=a, scale=b), sigma), 0.5 + 2 * x)
    return model


def _sample(model, **kwargs):
    return sample(30, 30, n_chains=4, model=model, progress_bar=False, chain_method="vectorized", seed=1, **kwargs)


# ============================================================ scenarios
def scenario_nuts(device, dtype):
    result = _sample(eight_schools().to(device, dtype), trace_device=None)
    assert result["eta"].device.type == torch.device(device).type
    return [result["eta"], result.sample_stats[0]["energy"].to(dtype)]


def scenario_dense_nuts(device, dtype):
    return [_sample(regression_model().to(device, dtype), metric="dense")["a"]]


def scenario_gibbs(device, dtype):
    return [_sample(gibbs_model().to(device, dtype))["means"]]


def scenario_slice(device, dtype):
    return [_sample(slice_model().to(device, dtype))["sigma"]]


def scenario_smc(device, dtype):
    out = []
    for kernel in ("metropolis", "nuts", "slice"):
        result = sample_smc(regression_model().to(device, dtype), n_particles=32, kernel=kernel, seed=2, progress_bar=False)
        out.append(result["b"])
    result = sample_smc(gibbs_model().to(device, dtype), n_particles=32, seed=2, progress_bar=False)
    out.append(result["means"])
    return out


def scenario_bbvi(device, dtype):
    model = regression_model().to(device, dtype)
    guide = AutoNormal(model)
    BBVI(model, guide, n_samples=2, epochs=20, lr=0.03, verbose=False)
    return [guide.sample(20)["b"]]


def scenario_laplace(device, dtype):
    model = regression_model().to(device, dtype)
    approximation = laplace_approximation(model, map_epochs=30, lr=0.05)
    return [approximation.sample(20)["b"]]


def scenario_predictive(device, dtype):
    model = regression_model().to(device, dtype)
    result = _sample(model)
    predictive = model.posterior_predicative(result, n_samples=5, samples_per_step=2, progress_bar=False)
    prior = model.sample_prior_predicative(n_samples=5, samples_per_step=2, progress_bar=False)
    return [predictive["y"], prior["y"]]


def scenario_gp(device, dtype):
    return [_sample(gp_model().to(device, dtype))["lengthscale"]]


def scenario_enumeration_and_loo(device, dtype):
    result = _sample(enumerated_mixture().to(device, dtype))
    loo(result)
    return [result["means"], result["z"].to(dtype), result.enumerated_probs["z"], result.log_likelihood["y"]]


SCENARIOS = {
    name[len("scenario_"):]: function
    for name, function in dict(globals()).items() if name.startswith("scenario_")
}


def run(scenario, device, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = SCENARIOS[scenario](device, dtype)
    for output in outputs:
        assert torch.isfinite(output).all()
        if output.is_floating_point():
            assert output.dtype == dtype, (scenario, output.dtype)


# ============================================================ GPU
@pytest.mark.skipif(not CUDA, reason="needs a CUDA or ROCm GPU")
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenarios_on_gpu(scenario):
    run(scenario, torch.device("cuda"), torch.float64)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA or ROCm GPU")
def test_gpu_vectorized_is_the_default_and_parallel_is_rejected():
    model = eight_schools().to("cuda")
    result = sample(10, 10, n_chains=2, model=model, progress_bar=False, seed=0)
    assert result["eta"].shape == (2, 10, 8)
    with pytest.raises(ValueError):
        sample(10, 10, n_chains=2, model=model, progress_bar=False, chain_method="parallel")


@pytest.mark.skipif(not CUDA, reason="needs a CUDA or ROCm GPU")
def test_gpu_float32_model():
    run("nuts", torch.device("cuda"), torch.float32)


# ============================================================ CPU leak checks
_FACTORIES = ("zeros", "ones", "empty", "full", "eye", "arange", "linspace", "randn", "rand", "tensor", "normal")


@pytest.fixture
def strict_device_factories(monkeypatch):
    """Fail when library code creates a floating tensor without ``device=``.

    Only calls made directly from the library (not from the Evaluation
    package, whose diagnostics run on the CPU traces) are checked.
    """
    violations = []

    def wrap(name, original):
        def factory(*args, **kwargs):
            result = original(*args, **kwargs)
            caller = sys._getframe(1)
            module = caller.f_globals.get("__name__", "")
            if (
                module.startswith("BayesianDLL")
                and not module.startswith("BayesianDLL.Evaluation")
                and "device" not in kwargs
                and isinstance(result, torch.Tensor)
                and result.is_floating_point()
            ):
                violations.append(f"torch.{name} in {module}:{caller.f_lineno}")
            return result
        return factory

    for name in _FACTORIES:
        monkeypatch.setattr(torch, name, wrap(name, getattr(torch, name)))
    yield violations


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_library_creates_tensors_on_the_model_device(scenario, strict_device_factories):
    run(scenario, torch.device("cpu"), torch.float64)
    assert not strict_device_factories, sorted(set(strict_device_factories))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_float64_model_with_float32_default_dtype(scenario):
    torch.set_default_dtype(torch.float32)
    run(scenario, torch.device("cpu"), torch.float64)


@pytest.mark.parametrize("scenario", ["nuts", "gibbs", "smc", "enumeration_and_loo"])
def test_float32_model(scenario):
    run(scenario, torch.device("cpu"), torch.float32)


def test_model_to_moves_every_tensor():
    x = torch.linspace(0.0, 1.0, 4)
    offset = torch.tensor([0.5, 0.25, 0.0, 1.0])
    with Model() as model:
        X = Data("x", x)
        a = RandomParameter("a", Normal(torch.zeros(1), 1.0))
        from BayesianDLL import DeterministicParameter
        mean = DeterministicParameter("mean", lambda a, x: a * x + offset, lambda a, x: {"a": x}, inputs=[a, X])
        z = RandomParameter("z", Bernoulli(0.3), enumerate=True)
        ObservedParameter("y", Normal(Affine("m", mean, loc=z), 1.0), torch.ones(4))
    model.to(dtype=torch.float32)
    assert model.data["x"].value.dtype == torch.float32
    assert model.params["a"].unconstrained_value.dtype == torch.float32
    assert model.params["a"].distribution.mu.dtype == torch.float32
    assert model.observed_params["y"].observed_values.dtype == torch.float32
    assert model.deterministic_params["mean"].constrained_value.dtype == torch.float32  # closure tensor moved
    assert model.model_log_prob().dtype == torch.float32
    assert torch.as_tensor(model.data["x"].value).device == model.device
