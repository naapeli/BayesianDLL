import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp, Linear, Log, Sigmoid
from BayesianDLL.Distributions import Bernoulli, Normal


def test_linear_regression_is_a_deterministic_parameter():
    x = torch.linspace(0.0, 1.0, 4)
    with Model() as model:
        inputs = Data("inputs", x)
        slope = RandomParameter("slope", Normal(0.0, 1.0))
        intercept = RandomParameter("intercept", Normal(0.0, 1.0))
        mean = Linear("mean", inputs, slope=slope, intercept=intercept)
        ObservedParameter("observations", Normal(mean, 0.1), torch.zeros(4))

    assert isinstance(mean, DeterministicParameter)
    assert model.deterministic_params["mean"] is mean
    torch.testing.assert_close(mean.constrained_value, slope.constrained_value * x + intercept.constrained_value)
    assert all(torch.isfinite(value).all() for value in model.joint_grad_log_prob().values())


def test_linear_regression_supports_design_matrices_and_weights():
    design = torch.tensor([[1.0, 0.0], [1.0, 2.0], [1.0, 4.0]])
    weights = torch.tensor([0.5, 2.0])
    with Model() as model:
        inputs = Data("inputs", design)
        mean = Linear("mean", inputs, weights, intercept=1.0)

    torch.testing.assert_close(mean.constrained_value, design @ weights + 1.0)
    assert mean.derivatives() == {}


def test_exponential_transform_has_matching_derivative():
    with Model() as model:
        log_scale = RandomParameter("log_scale", Normal(0.0, 1.0))
        scale = Exp("scale", log_scale)

    assert isinstance(scale, DeterministicParameter)
    torch.testing.assert_close(scale.constrained_value, torch.exp(log_scale.constrained_value))
    torch.testing.assert_close(scale.derivative("log_scale"), scale.constrained_value)


def test_log_transform_has_matching_derivative():
    with Model() as model:
        positive = RandomParameter("positive", Normal(0.0, 1.0))
        # Set a positive value explicitly because Normal is unconstrained.
        positive.set_constrained_value(torch.tensor([2.0]))
        log_positive = Log("log_positive", positive)

    torch.testing.assert_close(log_positive.constrained_value, torch.log(positive.constrained_value))
    torch.testing.assert_close(
        log_positive.derivative("positive"), positive.constrained_value.reciprocal()
    )


def test_sigmoid_transform_has_matching_derivative_and_finite_logistic_gradient():
    x = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    with Model() as model:
        inputs = Data("inputs", x)
        slope = RandomParameter("slope", Normal(0.0, 1.0))
        intercept = RandomParameter("intercept", Normal(0.0, 1.0))
        logits = Linear("logits", inputs, slope=slope, intercept=intercept)
        probability = Sigmoid("probability", logits)
        ObservedParameter("observations", Bernoulli(probability), y)

    expected = torch.sigmoid(logits.constrained_value)
    torch.testing.assert_close(probability.constrained_value, expected)
    torch.testing.assert_close(
        probability.derivative("logits"), expected * (1 - expected)
    )
    assert all(torch.isfinite(value).all() for value in model.joint_grad_log_prob().values())


def test_linear_regression_rejects_ambiguous_coefficient_arguments():
    with Model() as model:
        inputs = Data("inputs", torch.ones(2))
        with pytest.raises(TypeError, match="exactly one"):
            Linear("mean", inputs, coefficients=1.0, slope=1.0)
