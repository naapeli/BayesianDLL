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


def test_gp_posterior_predictive_with_exact_and_latent_gp():
    from BayesianDLL import Data
    from BayesianDLL.GP import ExactGP, LatentGP, RBF
    from BayesianDLL.Distributions import Uniform

    x_train = torch.linspace(0, 3, 10, dtype=torch.float64)
    y_train = torch.sin(x_train)

    with Model() as model:
        inputs = Data("inputs", x_train)
        ls = RandomParameter("ls", Uniform(0.5, 2.0))
        noise = RandomParameter("noise", Uniform(0.001, 0.05))
        f = ExactGP("f", inputs, RBF(ls, 1.0), noise_variance=noise)
        ObservedParameter("y", f, y_train)

    trace = {"ls": torch.full((1, 4, 1), 1.0), "noise": torch.full((1, 4, 1), 0.01)}
    x_new = torch.linspace(-0.5, 3.5, 18, dtype=torch.float64)
    inputs.set_value(x_new)
    pred = model.posterior_predicative(trace, samples_per_step=5, progress_bar=False)
    assert pred["y"].shape == (4, 5, 18)

    # Latent GP
    with Model() as model_lat:
        inputs_lat = Data("inputs_lat", x_train)
        ls_l = RandomParameter("ls_l", Uniform(0.5, 2.0))
        f_lat = LatentGP("f_lat", inputs_lat, RBF(ls_l, 1.0))
        ObservedParameter("y_lat", Bernoulli(logits=f_lat), (y_train > 0).double())

    trace_lat = {"ls_l": torch.full((1, 3, 1), 1.0), "f_lat_white": torch.zeros(1, 3, 10)}
    inputs_lat.set_value(x_new)
    pred_lat = model_lat.posterior_predicative(trace_lat, samples_per_step=4, progress_bar=False)
    assert pred_lat["y_lat"].shape == (3, 4, 18)
    assert "f_lat" in pred_lat
    assert pred_lat["f_lat"].shape == (3, 18)


def test_gp_posterior_predictive_toeplitz_and_sparse():
    from BayesianDLL import Data
    from BayesianDLL.GP import ExactGP, ToeplitzGP, RBF
    from BayesianDLL.Distributions import Uniform

    t_train = torch.linspace(0, 5, 20, dtype=torch.float64)
    y_train = torch.sin(t_train)

    with Model() as model_toep:
        t_data = Data("t", t_train)
        ls = RandomParameter("ls", Uniform(0.5, 2.0))
        noise = RandomParameter("noise", Uniform(0.001, 0.05))
        f_toep = ToeplitzGP("f_toep", t_data, RBF(ls, 1.0), noise_variance=noise)
        ObservedParameter("y", f_toep, y_train)

    t_new = torch.linspace(4.0, 7.0, 15, dtype=torch.float64)
    t_data.set_value(t_new)
    # Using point_trace (uses current model values)
    pred_toep = model_toep.posterior_predicative(model_toep.point_trace(), samples_per_step=10, progress_bar=False)
    assert pred_toep["y"].shape == (1, 10, 15)

    # Sparse GP with inducing points
    z = torch.linspace(0.5, 4.5, 5, dtype=torch.float64)
    with Model() as model_sparse:
        x_data = Data("x", t_train)
        ls_s = RandomParameter("ls_s", Uniform(0.5, 2.0))
        noise_s = RandomParameter("noise_s", Uniform(0.001, 0.05))
        f_sparse = ExactGP("f_sparse", x_data, RBF(ls_s, 1.0), inducing_points=z, noise_variance=noise_s, method="vfe")
        ObservedParameter("y_sparse", f_sparse, y_train)

    x_data.set_value(t_new)
    pred_sparse = model_sparse.posterior_predicative(model_sparse.point_trace(), samples_per_step=8, progress_bar=False)
    assert pred_sparse["y_sparse"].shape == (1, 8, 15)


def test_posterior_predicative_requires_trace():
    from BayesianDLL import Data
    with Model() as model:
        x_data = Data("x", torch.ones(5))
        p = RandomParameter("p", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(p, 1.0), torch.zeros(5))

    with pytest.raises(ValueError, match="trace must be provided"):
        model.posterior_predicative(None)

    pt = model.point_trace()
    assert "p" in pt
    assert pt["p"].shape == (1, 1, *p.constrained_value.shape)
    pred = model.posterior_predicative(pt, samples_per_step=3, progress_bar=False)
    assert pred["y"].shape == (1, 3, 5)


def test_map_result_and_predict():
    from BayesianDLL import Data, MAPResult, Linear, predict
    x_val = torch.linspace(-1.0, 1.0, 10)
    with Model() as model:
        x_data = Data("x", x_val)
        slope = RandomParameter("slope", Normal(0.0, 2.0))
        intercept = RandomParameter("intercept", Normal(0.0, 2.0))
        line = Linear("line", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Normal(line, 0.1), 2.0 * x_val + 1.0)

    res = model.find_MAP(lr=0.05, epochs=150, verbose=False)
    assert isinstance(res, MAPResult)
    assert "slope" in res.mode()
    assert "intercept" in res.mode()
    assert res["slope"].numel() == 1
    assert len(res) == len(res.history)

    # Predict with no arguments (current model params)
    p1 = model.predict(samples_per_step=5)
    assert p1["y"].shape == (1, 5, 10)

    # Predict with MAPResult
    p2 = model.predict(res, samples_per_step=5)
    assert p2["y"].shape == (1, 5, 10)

    # Predict with map_result.mode()
    p3 = model.predict(res.mode(), samples_per_step=5)
    assert p3["y"].shape == (1, 5, 10)

    # Top-level predict
    p4 = predict(res.mode(), samples_per_step=5, model=model)
    assert p4["y"].shape == (1, 5, 10)


def test_sampling_result_mode_median_mean_quantile():
    from BayesianDLL import Data
    with Model() as model:
        x_data = Data("x", torch.linspace(-1.0, 1.0, 5))
        w = RandomParameter("w", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(w, 0.1), torch.zeros(5))

    trace = model.sample(20, 10, n_chains=2, progress_bar=False)
    
    # Check trace statistics
    med = trace.median()
    assert "w" in med
    mn = trace.mean()
    assert "w" in mn
    q = trace.quantile(0.9)
    assert "w" in q
    mode_val = trace.mode()
    assert "w" in mode_val

    # Predict with trace point estimates
    p_med = model.predict(med, samples_per_step=4)
    assert p_med["y"].shape == (1, 4, 5)

    p_mode = model.predict(mode_val, samples_per_step=4)
    assert p_mode["y"].shape == (1, 4, 5)

    # Set params on model
    model.set_params(med)
    torch.testing.assert_close(model.params["w"].constrained_value, med["w"])


