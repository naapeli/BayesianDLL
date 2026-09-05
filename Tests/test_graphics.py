import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from BayesianDLL.Evaluation.Graphics import plot_model, plot_posterior, plot_predicative_distribution
from BayesianDLL.Samplers import PredicativeResult, SamplingResult


def test_plot_model_draws_dependency_graph(normal_model):
    plot_model(normal_model)
    assert {text.get_text() for text in plt.gca().texts} == {"mean", "data"}
    plt.gcf().canvas.draw()


@pytest.mark.parametrize("method", ["hist", "kde"])
def test_plot_posterior_draws_density_and_trace(method):
    result = SamplingResult({"x": torch.randn(2, 40, 1)}, [], [], [])
    plot_posterior(result, method=method, bins={"x": 8}, parameters=["x"])
    axes = plt.gcf().axes
    assert len(axes) == 2
    assert all(len(ax.lines) == 2 for ax in axes)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), result["x"][0, :, 0])
    plt.gcf().canvas.draw()


@pytest.mark.parametrize("method", ["hist", "kde"])
@pytest.mark.parametrize("kind", ["pdf", "cdf"])
def test_predictive_plots(method, kind):
    result = PredicativeResult({"data": torch.randn(3, 30, 1)})
    plot_predicative_distribution(result, data=torch.randn(50), method=method, kind=kind, bins=10)
    lines = plt.gca().lines
    assert len(lines) == 5  # three conditional curves, mean, and observations
    assert {line.get_label() for line in lines} >= {"Predicative mean", "Observed"}
    if kind == "cdf":
        for line in lines:
            values = line.get_ydata()
            assert (np.diff(values) >= -1e-12).all()
            assert values[-1] == pytest.approx(1.0)
    plt.gcf().canvas.draw()


@pytest.mark.parametrize("kwargs", [{"kind": "invalid"}, {"method": "invalid"}])
def test_predictive_plot_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError, match="should be in"):
        plot_predicative_distribution(PredicativeResult({}), **kwargs)


def test_posterior_plot_rejects_invalid_method():
    with pytest.raises(ValueError, match="method should be"):
        plot_posterior(SamplingResult({}, [], [], []), method="invalid")
