import math
import numpy as np
import pytest
import torch

from BayesianDLL.Evaluation import (
    effective_sample_size, gelman_rubin, mcse, summary,
)
from BayesianDLL.Samplers import PredicativeResult, SamplingResult


@pytest.fixture
def trace():
    return {"x": torch.randn(4, 500, 2)}


@pytest.mark.parametrize("method", ["classical", "split", "rank"])
def test_rhat_distinguishes_converged_and_shifted_chains(trace, method):
    rhat = gelman_rubin(trace, method=method)["x"]
    assert rhat.shape == (2,)
    torch.testing.assert_close(rhat, torch.ones(2), atol=0.03, rtol=0)
    shifted = trace["x"].clone()
    shifted[0] += 5
    assert (gelman_rubin({"x": shifted}, method=method)["x"] > 1.1).all()


def test_classical_rhat_matches_variance_formula():
    chains = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    # Within-chain variance = 5/3, between-chain variance = 2.
    expected = ((3 / 4 * (5 / 3) + 2 / 4) / (5 / 3)) ** 0.5
    assert gelman_rubin({"x": chains}, method="classical")["x"].item() == pytest.approx(expected)


@pytest.mark.parametrize("method", ["mean", "bulk", "tail"])
def test_ess_detects_autocorrelation(trace, method):
    independent = effective_sample_size(trace, method=method)["x"]
    assert independent.shape == (2,)
    assert (independent > 1000).all()
    correlated = trace["x"].clone()
    for i in range(1, correlated.shape[1]):
        correlated[:, i] += 0.9 * correlated[:, i - 1]
    actual = effective_sample_size({"x": correlated}, method=method)["x"]
    assert (actual > 0).all()
    assert (actual < independent / 2).all()


def test_tail_probability_pair_matches_scalar(trace):
    torch.testing.assert_close(
        effective_sample_size(trace, method="tail", prob=0.9)["x"],
        effective_sample_size(trace, method="tail", prob=(0.1, 0.9))["x"],
    )


def test_mcse_matches_moment_formulas(trace):
    x = trace["x"]
    ess = effective_sample_size(trace, method="mean")["x"]
    flat = x.flatten(0, 1)
    torch.testing.assert_close(mcse(trace, method="mean")["x"], flat.std(0) / ess.sqrt())
    squared = (flat - flat.mean(0)).square()
    variance = squared.mean(0)
    # The ESS of the delta-method formula is that of the squared deviations, whose mean is the variance.
    ess_squared = effective_sample_size({"c": (x - flat.mean(0)).square()}, method="mean")["c"]
    expected_sd = ((squared.square().mean(0) - variance.square()) / ess_squared / variance / 4).sqrt()
    torch.testing.assert_close(mcse(trace, method="sd")["x"], expected_sd)


@pytest.mark.parametrize("function", [gelman_rubin, effective_sample_size, mcse])
def test_diagnostics_validate_inputs(function, trace):
    with pytest.raises(ValueError, match="Unknown method"):
        function(trace, method="invalid")
    with pytest.raises(NotImplementedError, match="at least 2 dimensions"):
        function({"x": torch.ones(5)})


def _ar1_chains(n_chains, n_draws, phi, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_chains, n_draws))
    draws = np.zeros_like(noise)
    draws[:, 0] = noise[:, 0] / np.sqrt(1 - phi**2)
    for t in range(1, n_draws):
        draws[:, t] = phi * draws[:, t - 1] + noise[:, t]
    return draws


@pytest.mark.parametrize("phi, seed", [(0.0, 0), (0.9, 1)])
@pytest.mark.parametrize("prob", [0.025, 0.3, 0.77, 0.975])
def test_mcse_quantile_matches_arviz(phi, seed, prob):
    az = pytest.importorskip("arviz")
    draws = _ar1_chains(4, 400, phi, seed)
    actual = mcse({"x": torch.from_numpy(draws)}, method="quantile", prob=prob)["x"].item()
    expected = float(az.mcse(draws, method="quantile", prob=prob, chain_axis=0, draw_axis=1))
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("phi, seed", [(0.0, 3), (0.8, 4)])
def test_mcse_median_matches_arviz(phi, seed):
    az = pytest.importorskip("arviz")
    draws = _ar1_chains(3, 401, phi, seed)
    tensor = {"x": torch.from_numpy(draws)}
    expected = float(az.mcse(draws, method="median", chain_axis=0, draw_axis=1))
    assert mcse(tensor, method="median")["x"].item() == pytest.approx(expected, rel=1e-8)
    assert mcse(tensor, method="median")["x"].item() == mcse(tensor, method="quantile", prob=0.5)["x"].item()


@pytest.mark.parametrize("phi, seed", [(0.0, 5), (0.9, 6)])
def test_mcse_mean_and_sd_match_arviz(phi, seed):
    az = pytest.importorskip("arviz")
    draws = _ar1_chains(4, 400, phi, seed)
    tensor = {"x": torch.from_numpy(draws)}
    kwargs = dict(chain_axis=0, draw_axis=1)
    assert mcse(tensor, method="mean")["x"].item() == pytest.approx(float(az.mcse(draws, method="mean", **kwargs)), rel=1e-8)
    # ArviZ 1.x divides by the ESS of x in the sd formula; Vehtari et al. (2021), followed here, by the
    # ESS of the squared deviations (arviz's ``ess(method="sd")``). The formulas coincide up to that factor.
    ess_x = float(az.ess(draws, method="mean", **kwargs))
    ess_sq = float(az.ess(draws, method="sd", **kwargs))
    expected = float(az.mcse(draws, method="sd", **kwargs)) * np.sqrt(ess_x / ess_sq)
    assert mcse(tensor, method="sd")["x"].item() == pytest.approx(expected, rel=1e-8)


def test_mcse_validates_prob(trace):
    with pytest.raises(ValueError, match="requires prob"):
        mcse(trace, method="quantile")
    for prob in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between"):
            mcse(trace, method="quantile", prob=prob)
    for method in ("mean", "sd", "median"):
        with pytest.raises(ValueError, match="only used by"):
            mcse(trace, method=method, prob=0.3)
    assert mcse(trace, method="quantile", prob=0.3)["x"].shape == (2,)


@pytest.mark.parametrize("function", [gelman_rubin, effective_sample_size, mcse])
def test_diagnostics_preserve_multidimensional_feature_shape(function):
    assert function({"x": torch.randn(3, 60, 2, 3)})["x"].shape == (2, 3)


def test_constant_chain_diagnostics():
    trace = {"x": torch.ones(4, 100, 1)}
    # All chains are the same constant: the R-hat is undefined.
    assert math.isnan(gelman_rubin(trace)["x"].item())
    assert effective_sample_size(trace, method="mean")["x"].item() == 400
    assert mcse(trace, method="mean")["x"].item() == 0
    assert mcse(trace, method="sd")["x"].item() == 0
    assert mcse(trace, method="median")["x"].item() == 0


def test_sampling_result_mapping(trace):
    result = SamplingResult(trace, [0] * 4, [[0.8]] * 4, [[0.2]] * 4, {"twice": 2 * trace["x"]})
    assert set(result.keys()) == {"x", "twice"}
    assert "x" in result and "twice" in result and "missing" not in result
    assert dict(result.items())["x"] is trace["x"]
    assert len(list(result.values())) == 2
    torch.testing.assert_close(result["twice"], 2 * result["x"])
    with pytest.raises(KeyError):
        result["missing"]
    assert "divergences" in repr(result)


def test_predictive_result_mapping():
    samples = {"data": torch.randn(2, 10, 3)}
    result = PredicativeResult(samples)
    assert len(result) == 1
    assert list(result) == list(result.keys()) == ["data"]
    assert "data" in result and "missing" not in result
    assert list(result.values())[0] is samples["data"]
    assert dict(result.items())["data"] is result["data"]
    with pytest.raises(KeyError):
        result["missing"]
    assert "data" in repr(result)


@pytest.mark.parametrize("ci_kind", ["eti", "hdi"])
def test_summary_statistics_and_intervals(trace, ci_kind):
    table = summary(trace, hdi_prob=0.8, ci_kind=ci_kind, round_to=None)
    assert list(table.index) == ["x[0]", "x[1]"]
    assert list(table.columns) == ["mean", "sd", f"{ci_kind}_10%", f"{ci_kind}_90%", "ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd"]
    flat = trace["x"].numpy().reshape(-1, 2)
    np.testing.assert_allclose(table["mean"], flat.mean(0))
    np.testing.assert_allclose(table["sd"], flat.std(0, ddof=1))
    if ci_kind == "eti":
        np.testing.assert_allclose(table["eti_10%"], np.quantile(flat, 0.1, axis=0))
        np.testing.assert_allclose(table["eti_90%"], np.quantile(flat, 0.9, axis=0))
    else:
        for feature in range(2):
            low, high = table.iloc[feature][["hdi_10%", "hdi_90%"]]
            coverage = ((flat[:, feature] >= low) & (flat[:, feature] <= high)).mean()
            assert coverage == pytest.approx(0.8, abs=0.002)


def test_result_summary_selects_deterministics_and_rounds(trace):
    result = SamplingResult(trace, [], [], [], {"twice": 2 * trace["x"]})
    table = result.summary(round_to=2)
    assert list(table.index) == ["x[0]", "x[1]"]
    assert table.loc["x[0]", "mean"] == round(trace["x"][:, :, 0].mean().item(), 2)

    table_with_deterministics = result.summary(include_deterministic=True)
    assert list(table_with_deterministics.index) == ["x[0]", "x[1]", "twice[0]", "twice[1]"]

    with pytest.raises(TypeError, match="result must be"):
        summary([1, 2, 3])
