"""OrderedTransform, the ContinuousOrdered state space and the Ordered distribution."""

import math

import pytest
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp, Index
from BayesianDLL.Distributions import (
    ContinuousOrdered, ContinuousPositive, Dirichlet, Exponential, HalfNormal, Mixture, Normal, Ordered,
    OrderedTransform,
)
from BayesianDLL.Evaluation import gelman_rubin


def _y(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


# ------------------------------------------------------------ transform
@pytest.mark.parametrize("K", [1, 2, 4])
def test_ordered_transform_round_trip_and_order(K):
    transform = OrderedTransform()
    y = _y(K)
    x = transform.inverse(y)
    assert ContinuousOrdered().contains(x)
    torch.testing.assert_close(transform.forward(x), y)
    torch.testing.assert_close(transform.inverse(transform.forward(x)), x)
    # x_0 = y_0, x_k - x_{k-1} = exp(y_k)
    torch.testing.assert_close(x[0], y[0])
    torch.testing.assert_close(x[1:] - x[:-1], torch.exp(y[1:]))


@pytest.mark.parametrize("K", [1, 3, 5])
def test_ordered_transform_jacobian_against_autograd(K):
    transform = OrderedTransform()
    y = _y(K, seed=1)
    jac = torch.autograd.functional.jacobian(transform.inverse, y)
    torch.testing.assert_close(transform.derivative(y), jac)
    cotangent = _y(K, seed=2)
    torch.testing.assert_close(transform.vjp(y, cotangent), jac.T @ cotangent)
    torch.testing.assert_close(transform.log_abs_det_jacobian(y), torch.linalg.slogdet(jac)[1])
    torch.testing.assert_close(transform.log_abs_det_jacobian(y), y[1:].sum())
    y_grad = y.clone().requires_grad_()
    expected, = torch.autograd.grad(transform.log_abs_det_jacobian(y_grad), y_grad, allow_unused=True)
    expected = torch.zeros_like(y) if expected is None else expected
    torch.testing.assert_close(transform.grad_log_abs_det_jacobian(y), expected)


def test_ordered_transform_batched_equals_per_row():
    transform = OrderedTransform()
    y = _y(5, 3, 4, seed=3)
    cotangent = _y(5, 3, 4, seed=4)
    x = transform.inverse(y)
    jac = transform.derivative(y)
    vjp = transform.vjp(y, cotangent)
    grad = transform.grad_log_abs_det_jacobian(y)
    for c in range(5):
        for b in range(3):
            torch.testing.assert_close(x[c, b], transform.inverse(y[c, b]))
            torch.testing.assert_close(jac[c, b], transform.derivative(y[c, b]))
            torch.testing.assert_close(vjp[c, b], transform.vjp(y[c, b], cotangent[c, b]))
            torch.testing.assert_close(grad[c, b], transform.grad_log_abs_det_jacobian(y[c, b]))
    torch.testing.assert_close(transform.forward(x), y)


def test_ordered_transform_float32():
    y = _y(4).float()
    x = OrderedTransform().inverse(y)
    assert x.dtype == torch.float32
    assert OrderedTransform().vjp(y, torch.ones_like(y)).dtype == torch.float32


# ------------------------------------------------------------ state space
def test_continuous_ordered_contains():
    space = ContinuousOrdered()
    assert space.contains(torch.tensor([-1.0, 0.0, 2.0]))
    assert space.contains(torch.tensor([[-1.0, 0.0], [3.0, 4.0]]))  # batched rows
    assert not space.contains(torch.tensor([0.0, 0.0, 1.0]))  # not strict
    assert not space.contains(torch.tensor([[-1.0, 0.0], [4.0, 3.0]]))
    assert not space.contains(torch.tensor([0.0, float("inf")]))
    positive = ContinuousOrdered(ContinuousPositive())
    assert positive.contains(torch.tensor([0.5, 2.0]))
    assert not positive.contains(torch.tensor([-0.5, 2.0]))


# ------------------------------------------------------------ distribution
def test_ordered_distribution_log_prob_and_gradient_against_autograd():
    distribution = Ordered(Normal(torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64), 1.5), 3)
    assert distribution.event_shape == (3,)
    z = _y(3, seed=5)
    z_grad = z.clone().requires_grad_()
    x = torch.cumsum(torch.cat([z_grad[:1], torch.exp(z_grad[1:])]), 0)
    reference = torch.distributions.Normal(torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64), 1.5).log_prob(x).sum() + z_grad[1:].sum()
    expected, = torch.autograd.grad(reference, z_grad)
    torch.testing.assert_close(distribution._log_prob_unconstrained(z), reference.detach())
    torch.testing.assert_close(distribution._log_prob_grad_unconstrained(z), expected)


def test_ordered_positive_base_stays_positive_and_ordered():
    distribution = Ordered(Exponential(1.0), 3)
    z = _y(3, seed=6)
    x = distribution.transform.inverse(z)
    assert distribution.state_space.contains(x)
    torch.testing.assert_close(distribution.transform.forward(x), z)
    z_grad = z.clone().requires_grad_()
    x_grad = torch.exp(torch.cumsum(torch.cat([z_grad[:1], torch.exp(z_grad[1:])]), 0))
    reference = (-x_grad).sum() + torch.log(torch.abs(torch.det(torch.autograd.functional.jacobian(
        lambda v: torch.exp(torch.cumsum(torch.cat([v[:1], torch.exp(v[1:])]), 0)), z_grad, create_graph=True))))
    expected, = torch.autograd.grad(reference, z_grad)
    torch.testing.assert_close(distribution._log_prob_unconstrained(z), reference.detach())
    torch.testing.assert_close(distribution._log_prob_grad_unconstrained(z), expected)


def test_ordered_rejects_decreasing_or_multivariate_bases():
    with pytest.raises(ValueError):
        Ordered(Dirichlet(torch.ones(3)), 3)


def test_ordered_sample_is_sorted():
    draws = Ordered(Normal(0.0, 1.0), 4).sample(1000)
    assert draws.shape == (1000, 4)
    assert ContinuousOrdered().contains(draws)


def _ordered_hierarchical_model():
    """An ordered vector whose base location is itself random, observed with noise."""
    data = torch.tensor([[-1.2, 0.4, 2.1], [-0.8, 0.1, 1.7]], dtype=torch.float64)
    with Model() as model:
        loc = RandomParameter("loc", Normal(0.0, 2.0))
        scale = RandomParameter("scale", HalfNormal(2.0))
        theta = RandomParameter("theta", Ordered(Normal(loc, scale), 3))
        noise = RandomParameter("noise", HalfNormal(1.0))
        ObservedParameter("y", Normal(theta, noise), data)
    return model


def test_ordered_model_gradient_against_autograd():
    model = _ordered_hierarchical_model()
    model.compile()
    names = list(model.params)
    values = {name: _y(*model.params[name].unconstrained_value.shape, seed=10 + i) for i, name in enumerate(names)}
    for name, value in values.items():
        model.params[name].set_unconstrained_value(value)
    grads = model.joint_grad_log_prob(names)

    leaves = {name: value.clone().requires_grad_() for name, value in values.items()}
    loc = leaves["loc"]
    scale = torch.exp(leaves["scale"])
    noise = torch.exp(leaves["noise"])
    z = leaves["theta"]
    theta = torch.cumsum(torch.cat([z[:1], torch.exp(z[1:])]), 0)
    normal = torch.distributions.Normal
    half = torch.distributions.HalfNormal
    data = torch.tensor([[-1.2, 0.4, 2.1], [-0.8, 0.1, 1.7]], dtype=torch.float64)
    logp = (normal(0.0, 2.0).log_prob(loc).sum()
            + half(2.0).log_prob(scale).sum() + leaves["scale"].sum()
            + half(1.0).log_prob(noise).sum() + leaves["noise"].sum()
            + normal(loc, scale).log_prob(theta).sum() + z[1:].sum()
            + normal(theta, noise).log_prob(data).sum())
    expected = torch.autograd.grad(logp, [leaves[name] for name in names])
    for name, grad in zip(names, expected):
        torch.testing.assert_close(grads[name], grad)
    torch.testing.assert_close(torch.as_tensor(model.model_log_prob()).reshape(()), logp.detach())


def test_ordered_model_batched_equals_per_chain():
    model = _ordered_hierarchical_model()
    model.compile()
    names = list(model.params)
    n_chains = 4
    values = {name: _y(n_chains, *model.params[name].unconstrained_value.shape, seed=20 + i) for i, name in enumerate(names)}
    per_chain_logp, per_chain_grads = [], []
    for c in range(n_chains):
        for name in names:
            model.params[name].set_unconstrained_value(values[name][c])
        per_chain_logp.append(torch.as_tensor(model.model_log_prob()).reshape(()))
        per_chain_grads.append(model.joint_grad_log_prob(names))
    with model.batched(n_chains, values=values):
        assert model._batch_mode == "tensor", model.batch_fallback_reason
        logp = model.model_log_prob()
        grads = model.joint_grad_log_prob(names)
    torch.testing.assert_close(logp, torch.stack(per_chain_logp))
    for name in names:
        torch.testing.assert_close(grads[name], torch.stack([g[name] for g in per_chain_grads]))


def test_ordered_model_to_float32():
    model = _ordered_hierarchical_model()
    model.to(dtype=torch.float32)
    model.compile()
    assert model.params["theta"].unconstrained_value.dtype == torch.float32
    grads = model.joint_grad_log_prob(list(model.params))
    assert grads["theta"].dtype == torch.float32
    assert torch.isfinite(grads["theta"]).all()


@pytest.mark.parametrize("chain_method", ["vectorized", "parallel"])
def test_nuts_ordered_standard_normal_matches_order_statistics(chain_method):
    with Model() as model:
        RandomParameter("x", Ordered(Normal(0.0, 1.0), 3))
    result = model.sample(1500, 500, n_chains=4, seed=3, progress_bar=False, chain_method=chain_method)
    draws = result["x"].reshape(-1, 3)
    assert ContinuousOrdered().contains(draws)
    # Expected order statistics of three standard normals: -3 / (2 sqrt(pi)), 0, 3 / (2 sqrt(pi)).
    expected = torch.tensor([-1.5 / math.sqrt(math.pi), 0.0, 1.5 / math.sqrt(math.pi)], dtype=draws.dtype)
    torch.testing.assert_close(draws.mean(0), expected, atol=0.06, rtol=0)


def _two_cluster_data():
    generator = torch.Generator().manual_seed(0)
    return torch.cat([
        -1.0 + 0.5 * torch.randn(120, generator=generator, dtype=torch.float64),
        2.0 + 1.0 * torch.randn(80, generator=generator, dtype=torch.float64),
    ])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nuts_ordered_mixture_converges_in_every_chain(seed):
    data = _two_cluster_data()
    with Model() as model:
        weights = RandomParameter("weights", Dirichlet(torch.full((2,), 2.0, dtype=torch.float64)))
        means = RandomParameter("means", Ordered(Normal(0.0, 5.0), 2))
        log_stds = RandomParameter("log_stds", Normal(0.0, 1.0), shape=2)
        stds = Exp("stds", log_stds)
        components = [Normal(Index(f"mean{k}", means, [k]), Index(f"std{k}", stds, [k])) for k in range(2)]
        with plate("data", data.shape[0]):
            ObservedParameter("y", Mixture(components, weights), data)
    result = model.sample(300, 300, n_chains=4, seed=seed, progress_bar=False, chain_method="vectorized")
    per_chain_means = result["means"].mean(1)  # (chains, 2)
    assert torch.all((per_chain_means[:, 0] + 1.0).abs() < 0.25)
    assert torch.all((per_chain_means[:, 1] - 2.0).abs() < 0.5)
    rhat = gelman_rubin({"means": result["means"]})["means"]
    assert torch.all(rhat < 1.05)
