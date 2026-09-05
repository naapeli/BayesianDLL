import pytest
import torch

from BayesianDLL import (
    Data, DeterministicParameter, Model, ObservedParameter, RandomParameter,
    VariationalParameter, plate, condition,
)
from BayesianDLL.Distributions import Bernoulli, Dirichlet, Exponential, Normal
from BayesianDLL.Distributions._resolve import resolve
from BayesianDLL._active_model import _active_model
from BayesianDLL._plate import get_active_plates


def test_model_registration_and_log_probability(normal_model):
    model = normal_model
    assert _active_model._active_model is None
    assert set(model.params) == {"mean"}
    assert set(model.observed_params) == {"data"}
    assert set(model.graph.edges) == {("mean", "data")}
    x = torch.tensor([0.7])
    expected = torch.distributions.Normal(0.0, 2.0).log_prob(x).sum()
    expected += torch.distributions.Normal(x, 1.0).log_prob(torch.tensor([1.0, 2.0, 2.2])).sum()
    torch.testing.assert_close(model.log_prob("mean", x), expected)
    torch.testing.assert_close(model.params["mean"].constrained_value, torch.zeros(1))


@pytest.mark.parametrize("many", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_temporary_values_are_restored(normal_model, many, raises):
    model = normal_model
    original = model.params["mean"].unconstrained_value.clone()
    context = model.temporarily_set_many({"mean": torch.ones(1)}) if many else model.temporarily_set("mean", torch.ones(1))
    def use_context():
        with context:
            torch.testing.assert_close(model.params["mean"].constrained_value, torch.ones(1))
            if raises:
                raise ValueError("test exception")
    if raises:
        with pytest.raises(ValueError, match="test exception"):
            use_context()
    else:
        use_context()
    torch.testing.assert_close(model.params["mean"].unconstrained_value, original)


def test_hierarchical_model_gradients_with_deterministic_chain():
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 2.0), torch.tensor([0.4]))
        scale = RandomParameter("scale", Exponential(1.5), torch.tensor([1.2]))
        child = RandomParameter("child", Normal(mean, scale), torch.tensor([0.7]))
        doubled = DeterministicParameter("doubled", lambda x: 2 * x, lambda x: {"child": 2 * torch.ones_like(x)}, [child])
        shifted = DeterministicParameter("shifted", lambda x: x + 1, lambda x: {"doubled": torch.ones_like(x)}, [doubled])
        ObservedParameter("data", Normal(shifted, scale), torch.tensor([1.0, 2.0, 3.0]))
    values = {name: p.unconstrained_value.detach().clone().requires_grad_() for name, p in model.params.items()}
    with model.temporarily_set_many(values):
        expected = torch.autograd.grad(model.model_log_prob(), tuple(values.values()))
        actual = model.joint_grad_log_prob()
        for name, grad in zip(values, expected):
            torch.testing.assert_close(actual[name], grad)
            torch.testing.assert_close(model.grad_log_prob(name, values[name]), grad)


def test_regression_matrix_derivative():
    design = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    with Model() as model:
        weights = RandomParameter("weights", Normal(0.0, 2.0), shape=2)
        mu = DeterministicParameter("mu", lambda w: design @ w, lambda w: {"weights": design}, [weights])
        ObservedParameter("data", Normal(mu, 1.0), torch.tensor([1.0, 2.0, 4.0]))
    torch.testing.assert_close(model.joint_grad_log_prob()["weights"], design.T @ torch.tensor([1.0, 2.0, 4.0]))
    weights.set_constrained_value(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(mu.constrained_value, torch.tensor([1.0, 3.0, 5.0]))


def test_nested_plates_capture_shape_and_metadata():
    with Model():
        with plate("groups", 2):
            with plate("observations", 3):
                p = RandomParameter("p", Normal(0.0, 1.0), shape=4)
                observed = ObservedParameter("data", Normal(p, 1.0), torch.zeros(2, 3, 4))
            scalar = RandomParameter("scalar", Normal(0.0, 1.0), torch.tensor(2.0))
    assert p.constrained_value.shape == (2, 3, 4)
    assert scalar.constrained_value.shape == (2, 1)
    assert [(p.name, p.size, p.dim) for p in observed.plates] == [("groups", 2, -1), ("observations", 3, -2)]
    assert get_active_plates() == []


def test_contexts_cleanup_on_exception():
    with pytest.raises(ValueError):
        with Model():
            with plate("data", 2, dim=-3):
                assert get_active_plates()[0].dim == -3
                raise ValueError("stop")
    assert _active_model._active_model is None
    assert get_active_plates() == []


def test_parameter_initialization_and_setters():
    with Model():
        positive = RandomParameter("positive", Exponential(1.0))
        simplex = RandomParameter("simplex", Dirichlet(torch.ones(3)))
        discrete = RandomParameter("discrete", Bernoulli(0.5))
    assert positive.constrained_value.item() == 1.0
    torch.testing.assert_close(simplex.constrained_value, torch.full((3,), 1 / 3))
    assert discrete.constrained_value.item() == 0
    positive.set_constrained_value(torch.tensor([2.0]))
    torch.testing.assert_close(positive.unconstrained_value, torch.tensor([2.0]).log())
    positive.set_unconstrained_value(torch.tensor([0.0]))
    assert positive.constrained_value.item() == 1.0
    for setter in (positive.set_constrained_value, positive.set_unconstrained_value):
        with pytest.raises(TypeError, match="Tensor"):
            setter(1.0)


@pytest.mark.parametrize("factory", [
    lambda: RandomParameter("x", Normal(0.0, 1.0)),
    lambda: ObservedParameter("x", Normal(0.0, 1.0), torch.ones(1)),
    lambda: DeterministicParameter("x", lambda: 1, lambda: {}, []),
])
def test_parameters_require_active_model(factory):
    with pytest.raises(RuntimeError, match="active model"):
        factory()


def test_missing_dependency_is_rejected():
    with Model():
        other = RandomParameter("other", Normal(0.0, 1.0))
    with pytest.raises(RuntimeError, match="not in the computation graph"):
        with Model():
            RandomParameter("x", Normal(other, 1.0))


def test_variational_parameter_clamping_and_resolution():
    p = VariationalParameter("scale", torch.tensor(1.0), min=0.1, max=2.0)
    assert p.value.shape == (1,)
    p.set_new_value(torch.tensor(-3.0))
    torch.testing.assert_close(resolve(p), torch.tensor([0.1]))
    p.set_new_value(torch.tensor(4.0))
    torch.testing.assert_close(resolve(p), torch.tensor([2.0]))
    torch.testing.assert_close(resolve([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="not of type"):
        resolve(object())


def test_condition_basic_and_immutability():
    with Model() as model:
        x = RandomParameter("x", Normal(0.0, 1.0))
        y = ObservedParameter("y", Normal(x, 1.0), torch.tensor([2.0]))

    assert "x" in model.params
    assert "x" not in model.observed_params

    # Condition x to 1.5
    conditioned = condition(model, {"x": 1.5})

    # Original model is untouched
    assert "x" in model.params
    assert "x" not in model.observed_params
    assert _active_model._active_model is None
    assert conditioned is not model

    # Conditioned model has x as observed
    assert "x" not in conditioned.params
    assert "x" in conditioned.observed_params
    assert conditioned.graph.nodes["x"]["type"] == "observed"
    torch.testing.assert_close(conditioned.observed_params["x"].observed_values, torch.tensor([1.5]))
    torch.testing.assert_close(conditioned.observed_params["x"].constrained_value, torch.tensor([1.5]))

    # Joint log prob
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(1.5))
        + torch.distributions.Normal(1.5, 1.0).log_prob(torch.tensor(2.0))
    )
    torch.testing.assert_close(conditioned.model_log_prob(), expected)


def test_condition_calling_conventions():
    with Model() as model:
        x = RandomParameter("x", Normal(0.0, 1.0))
        y = RandomParameter("y", Normal(x, 1.0))
        ObservedParameter("z", Normal(y, 1.0), torch.tensor([0.0]))

    # 1. Via dict
    m1 = condition(model, {"x": 0.5})
    assert "x" in m1.observed_params

    # 2. Via kwargs
    m2 = condition(model, x=0.5)
    assert "x" in m2.observed_params

    # 3. Via parameter object as key
    m3 = condition(model, {x: 0.5})
    assert "x" in m3.observed_params

    # 4. Via model method
    m4 = model.condition({"x": 0.5})
    assert "x" in m4.observed_params
    m5 = model.condition(x=0.5)
    assert "x" in m5.observed_params


def test_condition_deterministic_downstream_and_gradients():
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0), torch.tensor([1.0]))
        b = RandomParameter("b", Normal(0.0, 1.0), torch.tensor([2.0]))
        det = DeterministicParameter("det", lambda a, b: a + 2 * b, lambda a, b: {"a": torch.ones_like(a), "b": 2 * torch.ones_like(b)}, [a, b])
        ObservedParameter("obs", Normal(det, 1.0), torch.tensor([5.0]))

    # Condition on a = 3.0
    cond = condition(model, a=3.0)
    assert "a" in cond.observed_params
    assert "b" in cond.params

    # Check that det can resolve its value in cond
    # det = 3.0 + 2 * 2.0 = 7.0
    torch.testing.assert_close(cond.deterministic_params["det"].constrained_value, torch.tensor([7.0]))

    # Gradients should only include remaining latent params ('b')
    grads = cond.joint_grad_log_prob()
    assert set(grads.keys()) == {"b"}
    assert "a" not in grads


def test_condition_update_observed_parameter():
    with Model() as model:
        x = RandomParameter("x", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(x, 1.0), torch.tensor([1.0]))

    # Update y's observed values
    cond = condition(model, y=torch.tensor([3.0]))
    torch.testing.assert_close(cond.observed_params["y"].observed_values, torch.tensor([3.0]))
    # Original unchanged
    torch.testing.assert_close(model.observed_params["y"].observed_values, torch.tensor([1.0]))


def test_condition_data_object_updates_copy_without_observing_it():
    with Model() as model:
        x = Data("x", torch.tensor([1.0, 2.0]))
        slope = RandomParameter("slope", Normal(0.0, 1.0), torch.tensor([1.0]))
        mean = DeterministicParameter(
            "mean",
            lambda slope, x: slope * x,
            lambda slope, x: {"slope": x},
            [slope, x],
        )
        ObservedParameter("y", Normal(mean, 1.0), torch.tensor([1.0, 2.0]))

    conditioned = condition(model, {x: torch.tensor([2.0, 4.0])})

    assert conditioned is not model
    assert conditioned.data["x"] is not x
    torch.testing.assert_close(model.data["x"].value, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(conditioned.data["x"].value, torch.tensor([2.0, 4.0]))
    assert conditioned.graph.nodes["x"]["type"] == "data"
    assert "x" not in conditioned.params
    assert "x" not in conditioned.observed_params
    torch.testing.assert_close(conditioned.deterministic_params["mean"].constrained_value, torch.tensor([2.0, 4.0]))


def test_condition_data_by_name_and_model_method():
    with Model() as model:
        x = Data("x", torch.tensor([1.0, 2.0]))
        ObservedParameter("y", Normal(x, 1.0), torch.tensor([1.0, 2.0]))

    by_name = condition(model, x=torch.tensor([3.0, 4.0]))
    by_method = model.condition({x: torch.tensor([5.0, 6.0])})

    torch.testing.assert_close(by_name.data["x"].value, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(by_method.data["x"].value, torch.tensor([5.0, 6.0]))


def test_condition_data_uses_data_event_shape_validation():
    with Model() as model:
        features = Data("features", torch.zeros(3, 2), event_ndim=1)

    with pytest.raises(ValueError, match="requires event shape"):
        condition(model, {features: torch.zeros(3, 3)})

    torch.testing.assert_close(model.data["features"].value, torch.zeros(3, 2))


def test_condition_sampling():
    with Model() as model:
        x = RandomParameter("x", Normal(0.0, 1.0))
        y = RandomParameter("y", Normal(x, 1.0))
        ObservedParameter("obs", Normal(y, 1.0), torch.tensor([0.0]))

    cond = condition(model, x=1.0)
    trace = cond.sample(n_samples=5, warmup_length=5, n_chains=1, progress_bar=False)
    assert "y" in trace
    assert "x" not in trace


def test_condition_validation_errors():
    with Model() as model:
        x = RandomParameter("x", Exponential(1.0))
        det = DeterministicParameter("det", lambda x: x * 2, lambda x: {"x": 2 * torch.ones_like(x)}, [x])
        ObservedParameter("y", Normal(det, 1.0), torch.tensor([1.0]))

    with pytest.raises(TypeError, match="must be an instance of Model"):
        condition("not_a_model", x=1.0)

    with pytest.raises(TypeError, match="must be a dict"):
        condition(model, "not_a_dict")

    with pytest.raises(KeyError, match="Variable 'unknown' not found"):
        condition(model, unknown=1.0)

    with pytest.raises(ValueError, match="Cannot condition deterministic parameter 'det'"):
        condition(model, det=2.0)

    with pytest.raises(ValueError, match="outside its distribution's state space"):
        condition(model, x=-1.0)
