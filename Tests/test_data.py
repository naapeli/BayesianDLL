import cloudpickle
import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Normal
from BayesianDLL.Samplers import SamplingBlock
from BayesianDLL._plate import get_active_plates


def test_updated_regression_data_reaches_log_prob_gradients_and_sampler():
    with Model() as model:
        x = Data("x", [0.0, 1.0, 2.0])
        y = Data("y", [1.0, 2.0, 3.0])
        slope = RandomParameter("slope", Normal(0.0, 4.0))
        mu = DeterministicParameter(
            "mu", lambda m, x: m * x,
            lambda m, x: {"slope": x}, [slope, x],
        )
        with plate("observations", 3):
            observed = ObservedParameter("likelihood", Normal(mu, 1.0), y)
    block = SamplingBlock(model, ["slope"])
    old_logp = model.model_log_prob().clone()
    x.set_value([2.0, 3.0, 4.0, 5.0])
    y.set_value([4.0, 5.0, 7.0, 8.0])
    theta = torch.tensor([0.5], requires_grad=True)
    slope.set_unconstrained_value(theta)
    expected = torch.distributions.Normal(0.0, 2.0).log_prob(theta).sum()
    expected += torch.distributions.Normal(theta * x.value, 1.0).log_prob(y.value).sum()
    expected_grad, = torch.autograd.grad(expected, theta)
    torch.testing.assert_close(model.model_log_prob(), expected)
    torch.testing.assert_close(model.log_prob("slope", theta), expected)
    torch.testing.assert_close(model.joint_grad_log_prob()["slope"], expected_grad)
    assert not torch.isclose(old_logp, expected)
    assert observed.observed_values is y.value
    assert set(model.params) == {"slope"}
    assert {("x", "mu"), ("y", "likelihood")} <= set(model.graph.edges)
    # An existing sampling block and a serialized worker model see new data.
    torch.testing.assert_close(block.sampler.log_target(theta), expected)
    copied = cloudpickle.loads(cloudpickle.dumps(model))
    torch.testing.assert_close(copied.model_log_prob(), expected)
    copied.data["y"].set_value([1.0, 1.0, 1.0])
    torch.testing.assert_close(copied.observed_params["likelihood"].observed_values, torch.ones(3))
    torch.testing.assert_close(y.value, torch.tensor([4.0, 5.0, 7.0, 8.0]))


def test_data_can_parameterize_distribution_directly():
    with Model() as model:
        center = Data("center", 0.0)
        parameter = RandomParameter("parameter", Normal(center, 1.0))
    center.set_value(3.0)
    expected = torch.distributions.Normal(3.0, 1.0).log_prob(parameter.constrained_value).sum()
    torch.testing.assert_close(model.model_log_prob(), expected)
    torch.testing.assert_close(model.joint_grad_log_prob()["parameter"], torch.tensor([3.0]))
    assert ("center", "parameter") in model.graph.edges


def test_data_validation_and_storage():
    with pytest.raises(RuntimeError, match="active model"):
        Data("x", [1.0])
    source = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    with Model() as model:
        data = Data("x", source, event_ndim=1)
        with pytest.raises(ValueError, match="already exists"):
            Data("x", source)
    source.detach().fill_(0)
    torch.testing.assert_close(data.value, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert not data.value.requires_grad
    data.set_value([3, 4])
    assert data.value.dtype == torch.float32
    with pytest.raises(ValueError, match="requires event shape"):
        data.set_value([1.0])
    torch.testing.assert_close(model.data["x"].value, torch.tensor([3.0, 4.0], dtype=torch.float32))


@pytest.mark.parametrize("event_shape", [(), (3,), (2, 3)])
def test_data_preserves_events_and_allows_different_batch_shapes(event_shape):
    with Model():
        data = Data("x", torch.zeros((4,) + event_shape, dtype=torch.float32),
                    event_ndim=len(event_shape))
    for batch_shape in [(7,), (2, 5), (), (0,)]:
        replacement = torch.ones(batch_shape + event_shape, requires_grad=True)
        data.set_value(replacement)
        assert data.event_shape == event_shape
        assert data.batch_shape == batch_shape
        assert data.value.dtype == torch.float32
        assert data.value.device == replacement.device
        assert not data.value.requires_grad
        replacement.detach().zero_()
        torch.testing.assert_close(data.value, torch.ones(batch_shape + event_shape, dtype=torch.float32))


@pytest.mark.parametrize("shape", [(5, 2, 4), (5, 3, 2), (3,), ()])
def test_data_rejects_changed_or_missing_event_dimensions(shape):
    with Model():
        data = Data("x", torch.ones(4, 2, 3), event_ndim=2)
    with pytest.raises(ValueError, match="requires event shape"):
        data.set_value(torch.zeros(shape))
    torch.testing.assert_close(data.value, torch.ones(4, 2, 3))


@pytest.mark.parametrize("event_ndim, error", [(-1, ValueError), (3, ValueError),
                                            (1.5, TypeError), (True, TypeError)])
def test_data_validates_event_ndim_before_registration(event_ndim, error):
    with Model() as model:
        with pytest.raises(error, match="event_ndim"):
            Data("x", torch.zeros(4, 2), event_ndim=event_ndim)
    assert "x" not in model.data
    assert "x" not in model.graph


def test_data_backed_plate_size_is_dynamic_and_integer_size_is_constant():
    with Model():
        data = Data("x", torch.zeros(4, 2), event_ndim=1)
        with plate("dynamic", data):
            dynamic_info = get_active_plates()[0]
        with plate("constant", 4):
            constant_info = get_active_plates()[0]

    assert dynamic_info.size == 4
    assert constant_info.size == 4
    data.set_value(torch.zeros(7, 2))
    assert dynamic_info.size == 7
    assert constant_info.size == 4
