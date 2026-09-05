import itertools

import pytest
import torch

from BayesianDLL.Distributions import (
    ContinuousPositive, ContinuousRange, ContinuousReal, ContinuousSimplex,
    DiscretePositive, DiscreteRange, JointStateSpace, Union,
)
from BayesianDLL.Distributions._transforms import (
    IdentityTransform, InverseSoftPlusTransform, LogitTransform, LogTransform,
    SoftMaxTransform,
)


@pytest.mark.parametrize("transform", [
    IdentityTransform(), LogTransform(2.0), LogTransform(2.0, "smaller"),
    InverseSoftPlusTransform(2.0), InverseSoftPlusTransform(2.0, "smaller"),
    LogitTransform(-2.0, 3.0),
])
def test_elementwise_transform_round_trip_and_jacobian(transform):
    z = torch.tensor([-1.2, 0.0, 1.4], requires_grad=True)
    x = transform.inverse(z)
    torch.testing.assert_close(transform.forward(x), z)
    jacobian = torch.autograd.functional.jacobian(transform.inverse, z)
    torch.testing.assert_close(transform.derivative(z), jacobian.diag())
    torch.testing.assert_close(transform.log_abs_det_jacobian(z), torch.linalg.slogdet(jacobian).logabsdet)
    log_det = transform.log_abs_det_jacobian(z)
    expected = torch.autograd.grad(log_det, z)[0] if log_det.requires_grad else torch.zeros_like(z)
    torch.testing.assert_close(transform.grad_log_abs_det_jacobian(z), expected)


def test_softmax_round_trip_and_full_jacobian():
    transform = SoftMaxTransform()
    z = torch.tensor([-2.0, 0.3, 1.0])
    x = transform.inverse(z)
    assert ContinuousSimplex().contains(x)
    torch.testing.assert_close(transform.inverse(transform.forward(x)), x)
    torch.testing.assert_close(transform.inverse(z + 1000), x)
    torch.testing.assert_close(transform.derivative(z), torch.autograd.functional.jacobian(transform.inverse, z))


@pytest.mark.parametrize("space,inside,outside", [
    (ContinuousReal(), [-1.0, 0.0, 3.0], [float("inf")]),
    (ContinuousPositive(), [0.1, 2.0], [0.0, 1.0]),
    (ContinuousRange(-1, 2), [-1.0, 2.0], [2.1]),
    (ContinuousSimplex(), [0.2, 0.3, 0.5], [0.2, 0.3, 0.4]),
    (DiscreteRange(1, 3), 2.0, 2.5),
    (DiscretePositive(), 3.0, 0.0),
])
def test_state_space_membership(space, inside, outside):
    assert space.contains(torch.tensor(inside))
    assert not space.contains(torch.tensor(outside))
    assert space.is_continuous() != space.is_discrete()


def test_discrete_enumeration():
    space = DiscreteRange(2, 4)
    assert list(space) == [2, 3, 4]
    assert len(space) == 3
    assert space[0] == 2
    assert space[-1] == 4
    torch.testing.assert_close(space.values, torch.tensor([2, 3, 4]))
    assert list(itertools.islice(DiscretePositive(), 4)) == [1, 2, 3, 4]


def test_union_and_joint_spaces():
    union = Union(ContinuousRange(-2, -1), ContinuousPositive())
    assert union.is_continuous()
    assert union.contains(torch.tensor(-1.5))
    assert union.contains(torch.tensor(1.0))
    assert not union.contains(torch.tensor(0.0))
    joint = JointStateSpace(
        {"x": ContinuousReal(), "y": ContinuousPositive()},
        {"x": (0, 2, (2,)), "y": (2, 3, (1,))},
    )
    assert joint.is_continuous()
    assert joint.contains(torch.tensor([-3.0, 0.0, 1.0]))
    assert not joint.contains(torch.tensor([-3.0, 0.0, -1.0]))
