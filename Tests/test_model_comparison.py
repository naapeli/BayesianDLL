"""Pointwise log-likelihood, PSIS-LOO, WAIC, model comparison, PPC statistics and ArviZ export."""

import math
import warnings

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Bernoulli, Beta, HalfCauchy, Normal
from BayesianDLL.Evaluation import (
    Graphics, compare, effective_sample_size, loo, ppc_pvalue, psis, to_arviz, waic,
)
from BayesianDLL.Evaluation._information_criteria import psis_tail_length, relative_efficiency, stacking_weights
from BayesianDLL.Samplers import PredicativeResult

az = pytest.importorskip("arviz")


def _normal_log_lik(y, mu, sigma=1.0):
    log_sigma = torch.log(sigma) if isinstance(sigma, torch.Tensor) else math.log(sigma)
    return -0.5 * ((y - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)


def _synthetic_log_lik(seed, shift=0.0, outlier=None, chains=4, draws=225, n=30):
    """Log-likelihood of y_i ~ N(mu, 1) with draws of mu from (an approximation to) its posterior."""
    rng = np.random.default_rng(seed)
    y = np.random.default_rng(0).normal(size=n)
    if outlier is not None:
        y[0] = outlier
    mu = rng.normal(y.mean() + shift, 1 / math.sqrt(n), size=(chains, draws))
    return _normal_log_lik(y[None, None, :], mu[..., None])


def _arviz_tree(log_lik):
    chains, draws = log_lik.shape[:2]
    return az.from_dict({
        "posterior": {"mu": np.zeros((chains, draws))},
        "log_likelihood": {"y": log_lik},
    })


# ---------------------------------------------------------------- PSIS-LOO
@pytest.mark.parametrize("outlier", [None, 4.0, 7.0])
def test_loo_matches_arviz(outlier):
    # 900 draws with r_eff = 1 give a tail of exactly 90 draws in both
    # implementations (arviz uses floor, the R package loo ceil).
    log_lik = _synthetic_log_lik(1, outlier=outlier)
    assert psis_tail_length(900, 1.0) == 90
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = az.loo(_arviz_tree(log_lik), pointwise=True, reff=1.0)
        result = loo(torch.from_numpy(log_lik), pointwise=True, reff=1.0)
    assert result.kind == "loo"
    assert result.elpd == pytest.approx(float(expected.elpd), abs=1e-6)
    assert result.p == pytest.approx(float(expected.p), abs=1e-6)
    assert result.se == pytest.approx(float(expected.se), abs=1e-6)
    np.testing.assert_allclose(result.elpd_i.numpy(), expected.elpd_i.values, atol=1e-6)
    np.testing.assert_allclose(result.pareto_k.numpy(), expected.pareto_k.values, rtol=0, atol=1e-6)
    assert result.good_k == pytest.approx(expected.good_k)


def test_loo_with_estimated_reff_is_close_to_arviz():
    log_lik = _synthetic_log_lik(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = az.loo(_arviz_tree(log_lik), pointwise=True)
        result = loo(torch.from_numpy(log_lik), pointwise=True)
    # r_eff is estimated per observation here and from the posterior in arviz,
    # which changes the tail length slightly.
    assert result.elpd == pytest.approx(float(expected.elpd), abs=1e-2)
    np.testing.assert_allclose(result.pareto_k.numpy(), expected.pareto_k.values, atol=0.1)


def test_relative_efficiency_uses_ess_of_the_likelihood():
    log_lik = _synthetic_log_lik(3)
    expected = effective_sample_size({"x": torch.from_numpy(np.exp(log_lik))}, method="mean")["x"].numpy() / 900
    np.testing.assert_allclose(relative_efficiency(log_lik), expected, rtol=1e-10)


def test_loo_warns_about_large_pareto_k():
    log_lik = _synthetic_log_lik(4, outlier=12.0, draws=250)
    with pytest.warns(UserWarning, match="Pareto shape"):
        result = loo(torch.from_numpy(log_lik))
    assert result.warning
    assert result.pareto_k.max() > result.good_k
    assert result.good_k == pytest.approx(min(1 - 1 / math.log10(1000), 0.7))


def test_psis_truncation_bounds_the_largest_weight():
    rng = np.random.default_rng(5)
    log_ratios = rng.exponential(1.5, 2000)  # Pareto ratios with shape 1.5
    untruncated, k = psis(log_ratios, truncate=False)
    truncated, k_truncated = psis(log_ratios, truncate=True)
    assert k == k_truncated and k > 1
    assert np.exp(logsumexp(truncated)) == pytest.approx(1.0)
    assert truncated.max() < untruncated.max()
    # The truncation level is S^(3/4) times the mean (smoothed) weight.
    level = 0.75 * math.log(2000) + logsumexp(untruncated) - math.log(2000)
    assert truncated.max() - truncated.min() == pytest.approx(level - untruncated.min(), rel=1e-12)


# -------------------------------------------------------------------- WAIC
def test_waic_matches_formula_and_warns():
    log_lik = _synthetic_log_lik(6, outlier=5.0)
    flat = log_lik.reshape(-1, log_lik.shape[-1])
    lppd = logsumexp(flat, axis=0) - math.log(flat.shape[0])
    p_waic = flat.var(axis=0, ddof=1)
    with pytest.warns(UserWarning, match="0.4"):
        result = waic({"y": torch.from_numpy(log_lik)}, pointwise=True)
    assert result.elpd == pytest.approx(np.sum(lppd - p_waic), abs=1e-10)
    assert result.p == pytest.approx(np.sum(p_waic), abs=1e-10)
    assert result.se == pytest.approx(math.sqrt(30) * np.std(lppd - p_waic), abs=1e-10)
    np.testing.assert_allclose(result.elpd_i.numpy(), lppd - p_waic)


def test_waic_and_loo_sum_several_observed_nodes_or_select_one():
    a = torch.from_numpy(_synthetic_log_lik(7))
    b = torch.from_numpy(_synthetic_log_lik(8))[..., :5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        both = waic({"a": a, "b": b})
        only_a = waic({"a": a, "b": b}, var_name="a")
        only_b = waic({"a": a, "b": b}, var_name="b")
        assert both.n_data_points == 35
        assert both.elpd == pytest.approx(only_a.elpd + only_b.elpd)
        assert loo({"a": a, "b": b}, reff=1.0).elpd == pytest.approx(
            loo({"a": a}, reff=1.0).elpd + loo({"b": b}, reff=1.0).elpd
        )


# ----------------------------------------------------------------- compare
def _compare_inputs():
    logs = {
        "good": _synthetic_log_lik(10),
        "shifted": _synthetic_log_lik(11, shift=0.3),
        "bad": _synthetic_log_lik(12, shift=-0.6),
    }
    # Make the models complementary on some observations so the stacking
    # weights are not all on one model.
    logs["shifted"][..., :8] += 0.4
    return logs


def test_compare_matches_arviz():
    logs = _compare_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = az.compare({n: _arviz_tree(v) for n, v in logs.items()}, round_to="None")
        estimates = {n: loo(torch.from_numpy(v), pointwise=True, reff=1.0) for n, v in logs.items()}
        table = compare(estimates)
    assert list(table.index) == list(expected.index)
    for column in ["elpd", "p", "se"]:
        np.testing.assert_allclose(table[column].to_numpy(float), expected[column].to_numpy(float), atol=1e-6)
    np.testing.assert_allclose(table["elpd_diff"].to_numpy(float), expected["elpd_diff"].to_numpy(float), atol=1e-6)
    np.testing.assert_allclose(table["dse"].to_numpy(float), expected["dse"].to_numpy(float), atol=1e-6)
    np.testing.assert_allclose(table["weight"].to_numpy(float), expected["weight"].to_numpy(float), atol=2e-3)
    assert table["weight"].sum() == pytest.approx(1.0)
    assert (table["weight"] > 0.05).sum() == 2  # a genuine mixture of two models

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_bma = az.compare({n: _arviz_tree(v) for n, v in logs.items()}, method="pseudo-BMA", round_to="None")
    table_bma = compare(estimates, method="pseudo-BMA")
    np.testing.assert_allclose(table_bma["weight"].to_numpy(float), expected_bma["weight"].to_numpy(float), atol=1e-8)


def test_stacking_weights_maximise_the_log_score():
    logs = _compare_inputs()
    pointwise = np.stack([loo(torch.from_numpy(v), pointwise=True, reff=1.0).elpd_i.numpy() for v in logs.values()], axis=1)
    weights = stacking_weights(pointwise)
    score = lambda w: np.sum(logsumexp(pointwise, b=w, axis=1))
    rng = np.random.default_rng(0)
    for candidate in rng.dirichlet(np.ones(3), size=200):
        assert score(candidate) <= score(weights) + 1e-9


def test_compare_with_waic_and_validation():
    logs = _compare_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = compare({n: {"y": torch.from_numpy(v)} for n, v in logs.items()}, ic="waic")
    assert table["elpd"].is_monotonic_decreasing
    assert table["elpd_diff"].iloc[0] == 0 and (table["elpd_diff"].iloc[1:] < 0).all()
    with pytest.raises(ValueError, match="same"):
        compare({"a": torch.from_numpy(logs["good"]), "b": torch.from_numpy(logs["bad"][..., :10])}, ic="waic")
    with pytest.raises(ValueError):
        compare({"a": torch.from_numpy(logs["good"])})


# ------------------------------------------------- recorded log-likelihood
@pytest.fixture(scope="module")
def regression_results():
    torch.manual_seed(0)
    n = 30
    x = torch.linspace(-2, 2, n, dtype=torch.float64)
    y = 1 + 2 * x - 0.8 * x ** 2 + 0.5 * torch.randn(n, dtype=torch.float64)

    def fit(quadratic):
        with Model() as model:
            a = RandomParameter("a", Normal(0.0, 5.0))
            b = RandomParameter("b", Normal(0.0, 5.0))
            c = RandomParameter("c", Normal(0.0, 5.0)) if quadratic else None
            sigma = RandomParameter("sigma", HalfCauchy(1.0))
            if quadratic:
                mu = DeterministicParameter(
                    "mu", lambda a, b, c: a + b * x + c * x ** 2,
                    lambda a, b, c: {"a": torch.ones_like(x), "b": x, "c": x ** 2}, [a, b, c],
                )
            else:
                mu = DeterministicParameter("mu", lambda a, b: a + b * x, lambda a, b: {"a": torch.ones_like(x), "b": x}, [a, b])
            ObservedParameter("y", Normal(mu, sigma), y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.sample(450, 300, n_chains=2, progress_bar=False, seed=1)

    return x, y, fit(False), fit(True)


def test_sampling_records_pointwise_log_likelihood(regression_results):
    x, y, linear, _ = regression_results
    log_lik = linear.log_likelihood["y"]
    assert log_lik.shape == (2, 450, 30)
    mu = linear["a"] + linear["b"] * x
    expected = _normal_log_lik(y, mu, linear["sigma"])
    torch.testing.assert_close(log_lik, expected)

    thinned = linear.thin(3)
    assert thinned.log_likelihood["y"].shape == (2, 150, 30)
    torch.testing.assert_close(thinned.log_likelihood["y"], log_lik[:, ::3])


def test_log_likelihood_is_opt_out(normal_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = normal_model.sample(20, 20, n_chains=1, progress_bar=False, seed=0, log_likelihood=False)
    assert result.log_likelihood == {}
    with pytest.raises(ValueError, match="log_likelihood=True"):
        loo(result)


def test_compare_prefers_the_quadratic_model(regression_results):
    _, _, linear, quadratic = regression_results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = compare({"linear": linear, "quadratic": quadratic})
    assert table.index[0] == "quadratic"
    assert table.loc["quadratic", "weight"] > 0.9
    assert table.loc["linear", "elpd_diff"] < -2 * table.loc["linear", "dse"]


def test_to_arviz_supports_summary_and_loo(regression_results):
    x, y, linear, _ = regression_results
    predictive = PredicativeResult({"y": torch.randn(5, 4, 30, dtype=torch.float64)})
    tree = to_arviz(linear, predictive=predictive, observed=y)
    assert set(tree.children) >= {"posterior", "sample_stats", "log_likelihood", "posterior_predictive", "observed_data"}
    assert {"energy", "diverging", "tree_depth", "step_size", "acceptance_rate", "lp"} <= set(tree["sample_stats"].data_vars)
    assert tree["posterior_predictive"]["y"].shape == (1, 20, 30)
    assert "mu" in tree["posterior"].data_vars
    summary = az.summary(tree, var_names=["a", "b", "sigma"])
    assert len(summary) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = az.loo(tree, reff=1.0, pointwise=True)
        result = loo(linear, reff=1.0, pointwise=True)
    # 900 draws: 3 * sqrt(900) = 90 is an integer, so arviz (which floors it)
    # and loo (which ceils it) smooth the same tail length and agree exactly.
    assert result.elpd == pytest.approx(float(expected.elpd), abs=1e-2)
    np.testing.assert_allclose(result.pareto_k.numpy(), expected.pareto_k.values, rtol=0, atol=1e-6)


def _mixture_model(n, data, shared_assignment=False):
    with Model() as model:
        weight = RandomParameter("weight", Beta(2.0, 2.0))
        mean0 = RandomParameter("mean0", Normal(-1.0, 1.0))
        mean1 = RandomParameter("mean1", Normal(1.0, 1.0))
        if shared_assignment:
            z = RandomParameter("z", Bernoulli(weight), marginalize=True)
            mean = DeterministicParameter(
                "mean", lambda m0, m1, z: m0 + (m1 - m0) * z,
                lambda m0, m1, z: {"mean0": 1 - z, "mean1": z}, [mean0, mean1, z],
            )
            ObservedParameter("y", Normal(mean, 1.0), data)
        else:
            with plate("data", n):
                z = RandomParameter("z", Bernoulli(weight), shape=(), marginalize=True)
                mean = DeterministicParameter(
                    "mean", lambda m0, m1, z: m0 + (m1 - m0) * z,
                    lambda m0, m1, z: {"mean0": 1 - z, "mean1": z}, [mean0, mean1, z],
                )
                ObservedParameter("y", Normal(mean, 1.0), data)
    return model


def test_enumerated_latents_give_marginal_pointwise_terms():
    n = 40
    assignments = (torch.rand(n) < 0.4).double()
    data = torch.where(assignments == 1, 2.5, -2.0) + torch.randn(n, dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _mixture_model(n, data).sample(30, 30, n_chains=1, progress_bar=False, seed=0)
    log_lik = result.log_likelihood["y"]
    assert log_lik.shape == (1, 30, n)
    w = result["weight"][..., None]
    expected = torch.logaddexp(
        torch.log1p(-w) + _normal_log_lik(data, result["mean0"][..., None]),
        torch.log(w) + _normal_log_lik(data, result["mean1"][..., None]),
    ).reshape(log_lik.shape)
    torch.testing.assert_close(log_lik, expected)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isfinite(loo(result).elpd)


def test_non_separable_enumeration_raises_a_clear_error():
    data = torch.tensor([-2.0, -1.5, -2.5, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _mixture_model(4, data, shared_assignment=True).sample(10, 10, n_chains=1, progress_bar=False, seed=0)
    assert "y" not in result.log_likelihood
    assert "y" in result.log_likelihood_unavailable
    with pytest.raises(ValueError, match="pointwise log-likelihood of 'y' is unavailable"):
        loo(result)
    with pytest.raises(ValueError, match="unavailable"):
        waic(result, var_name="y")


# ------------------------------------------------------------- PPC p-value
def test_ppc_pvalue_builtin_and_callable_statistics():
    torch.manual_seed(0)
    replicated = torch.randn(10, 20, 15, dtype=torch.float64)
    observed = torch.randn(15, dtype=torch.float64) + 0.3
    predictive = PredicativeResult({"y": replicated})
    flat = replicated.reshape(200, 15)

    result = ppc_pvalue(predictive, observed, "mean")
    assert result.t_rep.shape == (200,)
    assert result.t_observed == pytest.approx(observed.mean().item())
    assert result.p_value == pytest.approx((flat.mean(dim=1) >= observed.mean()).double().mean().item())

    for name, function in {
        "std": lambda y: y.std(), "min": lambda y: y.min(), "max": lambda y: y.max(),
        "median": lambda y: y.quantile(0.5), "q05": lambda y: y.quantile(0.05), "q95": lambda y: y.quantile(0.95),
    }.items():
        builtin = ppc_pvalue(predictive, observed, name)
        custom = ppc_pvalue(predictive, observed, function)
        torch.testing.assert_close(builtin.t_rep, custom.t_rep)
        assert builtin.p_value == custom.p_value

    from scipy.stats import skew
    skewness = ppc_pvalue(predictive, observed, "skew")
    np.testing.assert_allclose(skewness.t_rep.numpy(), skew(flat.numpy(), axis=1))

    counts = PredicativeResult({"y": torch.poisson(torch.full((4, 5, 12), 0.7))})
    zeros = ppc_pvalue(counts, torch.zeros(12), "n_zeros")
    assert zeros.t_observed == 12 and zeros.p_value == pytest.approx((zeros.t_rep >= 12).double().mean().item())

    with pytest.raises(ValueError, match="Unknown statistic"):
        ppc_pvalue(predictive, observed, "kurtosis")
    with pytest.raises(ValueError, match="var_name"):
        ppc_pvalue(PredicativeResult({"a": replicated, "b": replicated}), observed)


def test_plot_ppc_statistic():
    replicated = torch.randn(10, 20, 15, dtype=torch.float64)
    observed = torch.randn(15, dtype=torch.float64)
    ax = Graphics.plot_ppc_statistic(PredicativeResult({"y": replicated}), observed, statistic="max")
    result = ppc_pvalue(PredicativeResult({"y": replicated}), observed, "max")
    assert f"p = {result.p_value:.3f}" in ax.get_title()
    assert any(line.get_xdata()[0] == pytest.approx(result.t_observed) for line in ax.get_lines())
    ax = Graphics.plot_ppc_statistic(result)
    assert ax.get_title().startswith("max")
