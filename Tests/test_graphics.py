import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from BayesianDLL.Evaluation.Graphics import plot_model, plot_posterior, plot_predicative_distribution
from BayesianDLL.Samplers import PredicativeResult, SamplingResult


def test_plot_model_draws_dependency_graph(normal_model):
    ax = plot_model(normal_model)
    assert ax is plt.gca()
    labels = {text.get_text() for text in plt.gca().texts}
    assert any(label.startswith("mean\n~\n") for label in labels)
    assert any(label.startswith("data\n~\n") for label in labels)
    plt.gcf().canvas.draw()


def test_plot_model_can_hide_data_and_metadata():
    from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
    from BayesianDLL.Distributions import Normal

    with Model() as model:
        x = Data("x", torch.arange(5.0))
        y = Data("y", torch.arange(5.0))
        slope = RandomParameter("slope", Normal(0.0, 1.0))
        mu = DeterministicParameter(
            "mu", lambda m, values: m * values,
            lambda m, values: {"slope": values}, [slope, x],
        )
        with plate("observations", x):
            ObservedParameter("likelihood", Normal(mu, 1.0), y)

    ax = plot_model(model, include_data=False, show_distributions=False, legend=False)
    labels = {text.get_text() for text in ax.texts}
    assert labels == {"slope", "mu", "likelihood", "observations  [5]"}
    assert len(ax.patches) == 6  # three nodes, two arrows, and one plate
    plt.gcf().canvas.draw()


def test_plot_model_uses_length_for_raw_tensor_plate_size():
    from BayesianDLL import Model, ObservedParameter, plate
    from BayesianDLL.Distributions import Normal

    values = torch.arange(5.0)
    with Model() as model:
        with plate("observations", values):
            ObservedParameter("likelihood", Normal(0.0, 1.0), values)

    ax = plot_model(model, show_distributions=False)
    labels = {text.get_text() for text in ax.texts}
    assert "observations  [5]" in labels
    assert not any("tensor(" in label for label in labels)
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


def test_plot_posterior_aggregates_parameter_components():
    samples = torch.arange(2 * 40 * 3, dtype=torch.float32).reshape(2, 40, 3)
    result = SamplingResult({"x": samples}, [], [], [])

    axes = plot_posterior(result, method="hist", bins={"x": 8}, parameters=["x"], aggregate=True)

    assert axes.shape == (1, 2)
    assert all(len(ax.lines) == 6 for ax in axes[0])  # three components across two chains
    assert axes[0, 0].get_title() == "x"
    assert axes[0, 1].get_title() == "x"
    assert {text.get_text() for text in axes[0, 1].get_legend().get_texts()} == {"x[0]", "x[1]", "x[2]"}
    plt.gcf().canvas.draw()


@pytest.mark.parametrize(
    "vars, expected_titles",
    [
        ("random", {"x"}),
        ("deterministic", {"f"}),
        ("all", {"x", "f"}),
    ],
)
def test_plot_posterior_selects_random_and_deterministic_variables(vars, expected_titles):
    result = SamplingResult(
        {"x": torch.randn(2, 40)},
        [],
        [],
        [],
        deterministic_trace={"f": torch.randn(2, 40)},
    )

    axes = plot_posterior(result, method="hist", vars=vars, aggregate=True)

    assert axes.shape == (len(expected_titles), 2)
    assert {axis.get_title() for axis in axes[:, 0]} == expected_titles


def test_plot_posterior_rejects_invalid_variable_selection():
    result = SamplingResult({"x": torch.randn(2, 40, 1)}, [], [], [])
    with pytest.raises(ValueError, match="vars should be in"):
        plot_posterior(result, vars="invalid")


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


def test_posterior_plot_uses_supplied_axes():
    result = SamplingResult({"x": torch.randn(2, 40, 1)}, [], [], [])
    _, supplied_axes = plt.subplots(1, 2)

    returned_axes = plot_posterior(result, axes=supplied_axes)

    assert returned_axes.shape == (1, 2)
    assert list(returned_axes[0]) == list(supplied_axes)
    assert all(ax.lines for ax in supplied_axes)


def test_posterior_plot_rejects_wrong_number_of_axes():
    result = SamplingResult({"x": torch.randn(2, 40, 1)}, [], [], [])
    _, supplied_ax = plt.subplots()

    with pytest.raises(ValueError, match="exactly 2 axes"):
        plot_posterior(result, axes=supplied_ax)


def test_predictive_plot_uses_supplied_axis():
    result = PredicativeResult({"data": torch.randn(3, 30, 1)})
    _, supplied_ax = plt.subplots()

    returned_ax = plot_predicative_distribution(result, ax=supplied_ax)

    assert returned_ax is supplied_ax
    assert supplied_ax.lines
