import numpy as np
import pytest
import torch

from BayesianDLL.Evaluation import (
    effective_sample_size, ess_bulk, ess_mean, ess_tail, gelman_rubin,
    mcse, mcse_mean, mcse_sd, summary,
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


@pytest.mark.parametrize("wrapper,method", [(ess_mean, "mean"), (ess_bulk, "bulk"), (ess_tail, "tail")])
def test_ess_convenience_functions(trace, wrapper, method):
    torch.testing.assert_close(wrapper(trace)["x"], effective_sample_size(trace, method=method)["x"])


def test_tail_probability_pair_matches_scalar(trace):
    torch.testing.assert_close(ess_tail(trace, prob=0.9)["x"], ess_tail(trace, prob=(0.1, 0.9))["x"])


def test_mcse_matches_moment_formulas(trace):
    x = trace["x"]
    ess = ess_mean(trace)["x"]
    flat = x.flatten(0, 1)
    torch.testing.assert_close(mcse_mean(trace)["x"], flat.std(0) / ess.sqrt())
    squared = (flat - flat.mean(0)).square()
    variance = squared.mean(0)
    expected_sd = ((squared.square().mean(0) - variance.square()) / ess / variance / 4).sqrt()
    torch.testing.assert_close(mcse_sd(trace)["x"], expected_sd)
    torch.testing.assert_close(mcse(trace, method="sd")["x"], expected_sd)


@pytest.mark.parametrize("function", [gelman_rubin, effective_sample_size, mcse])
def test_diagnostics_validate_inputs(function, trace):
    with pytest.raises(ValueError, match="Unknown method"):
        function(trace, method="invalid")
    with pytest.raises(NotImplementedError, match="at least 2 dimensions"):
        function({"x": torch.ones(5)})


@pytest.mark.parametrize("function", [gelman_rubin, effective_sample_size, mcse])
def test_diagnostics_preserve_multidimensional_feature_shape(function):
    assert function({"x": torch.randn(3, 60, 2, 3)})["x"].shape == (2, 3)


def test_constant_chain_diagnostics():
    trace = {"x": torch.ones(4, 100, 1)}
    assert gelman_rubin(trace)["x"].item() == 1
    assert ess_mean(trace)["x"].item() == 400
    assert mcse_mean(trace)["x"].item() == 0
    assert mcse_sd(trace)["x"].item() == 0


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
    assert len(result.summary()) == 4
    table = result.summary(include_deterministic=False, round_to=2)
    assert list(table.index) == ["x[0]", "x[1]"]
    assert table.loc["x[0]", "mean"] == round(trace["x"][:, :, 0].mean().item(), 2)
    with pytest.raises(TypeError, match="result must be"):
        summary([1, 2, 3])
