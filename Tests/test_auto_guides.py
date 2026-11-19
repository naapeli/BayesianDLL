"""Automatic guides, the Laplace approximation and minibatch variational inference."""

import math

import pytest
import torch

from BayesianDLL import (
    AutoMultivariateNormal, AutoNormal, Data, DeterministicParameter, Model, ObservedParameter,
    RandomParameter, laplace_approximation, plate,
)
from BayesianDLL.Distributions import Bernoulli, Beta, HalfNormal, Normal
from BayesianDLL.Samplers import SamplingResult
from BayesianDLL.Variational import BBVI, elbo


def _two_latent_model():
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.5))
        DeterministicParameter("twice", lambda m: 2 * m, lambda m: {"mean": 2 * torch.ones_like(m)}, [mean])
        with plate("data", 4):
            ObservedParameter("y", Normal(mean, sigma), torch.tensor([0.3, 1.2, 0.8, 1.9]))
    return model


@pytest.mark.parametrize("guide_class", [AutoNormal, AutoMultivariateNormal])
def test_auto_guide_gradients_match_autograd(guide_class):
    model = _two_latent_model()
    guide = guide_class(model, init_scale=0.4)
    table = guide.variational_parameters()
    # Move away from the symmetric initialisation so every gradient entry matters.
    for parameter in table.values():
        parameter.set_unconstrained_value(parameter.unconstrained_value + 0.1 * torch.randn_like(parameter.value).tril() if parameter.value.ndim == 2 else parameter.unconstrained_value + 0.1 * torch.randn_like(parameter.value))
        parameter.value.requires_grad_(True)
    value, grads = elbo(model, guide, n_samples=5)
    expected = torch.autograd.grad(value, [table[key].value for key in grads])
    for key, reference in zip(grads, expected):
        if key == "scale_tril":
            reference = torch.tril(reference)
        torch.testing.assert_close(grads[key], reference, msg=key)


def test_auto_normal_recovers_conjugate_posterior(normal_model):
    guide = AutoNormal(normal_model)
    BBVI(normal_model, guide, n_samples=8, epochs=800, lr=0.03, verbose=False)
    assert guide.locs["mean"].value.item() == pytest.approx(1.6, abs=0.05)
    assert guide.scales["mean"].value.item() == pytest.approx(math.sqrt(4 / 13), rel=0.15)
    trace = guide.sample(500, n_chains=2)
    assert isinstance(trace, SamplingResult)
    assert trace["mean"].shape == (2, 500, 1)
    summary = trace.summary()
    assert "mean[0]" in summary.index


@pytest.mark.integration
def test_auto_multivariate_normal_captures_posterior_correlation():
    """Linear regression with known noise: the posterior is Gaussian with correlated coefficients."""
    x = torch.linspace(0.5, 1.5, 12)
    y = 1.0 + 2.0 * x + 0.3 * torch.randn(12)
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 3.0))
        b = RandomParameter("b", Normal(0.0, 3.0))
        xs = Data("x", x)
        mu = DeterministicParameter(
            "mu", lambda a, b, x: a + b * x,
            lambda a, b, x: {"a": torch.ones_like(x), "b": x}, [a, b, xs],
        )
        with plate("data", xs):
            ObservedParameter("y", Normal(mu, 0.3), y)
    design = torch.stack([torch.ones_like(x), x], dim=1)
    precision = design.T @ design / 0.09 + torch.eye(2) / 9
    covariance = torch.linalg.inv(precision)
    mean = covariance @ design.T @ y / 0.09

    guide = AutoMultivariateNormal(model)
    BBVI(model, guide, n_samples=8, epochs=2500, lr=0.02, verbose=False)
    torch.testing.assert_close(guide.loc, mean, atol=0.05, rtol=0)
    torch.testing.assert_close(guide.covariance, covariance, atol=0.25 * covariance.abs().max().item(), rtol=0)
    correlation = guide.covariance[0, 1] / guide.covariance.diagonal().prod().sqrt()
    assert correlation.item() < -0.8  # strongly negatively correlated intercept and slope
    # The triangular structure is preserved and the diagonal stays positive.
    assert torch.equal(guide.scale_tril, torch.tril(guide.scale_tril))
    assert (torch.diagonal(guide.scale_tril) > 0).all()
    trace = guide.sample(300)
    assert trace["mu"].shape == (1, 300, 12)


def test_auto_guides_reject_discrete_latents():
    with Model() as model:
        RandomParameter("z", Bernoulli(0.3))
    for guide_class in (AutoNormal, AutoMultivariateNormal):
        with pytest.raises(ValueError, match="enumerat"):
            guide_class(model)
    with pytest.raises(ValueError, match="enumerat"):
        laplace_approximation(model)


def test_laplace_is_exact_for_conjugate_normal(normal_model):
    approximation = laplace_approximation(normal_model, map_epochs=50)
    torch.testing.assert_close(approximation.mean, torch.tensor([1.6]), atol=1e-8, rtol=0)
    torch.testing.assert_close(approximation.cov, torch.tensor([[4 / 13]]), atol=1e-6, rtol=0)
    y = torch.tensor([1.0, 2.0, 2.2])
    covariance = torch.eye(3) + 4.0
    exact = torch.distributions.MultivariateNormal(torch.zeros(3), covariance).log_prob(y)
    assert approximation.log_marginal_likelihood.item() == pytest.approx(exact.item(), abs=1e-6)
    assert normal_model.params["mean"].constrained_value.item() == pytest.approx(1.6, abs=1e-8)
    trace = approximation.sample(2000, n_chains=2)
    assert trace["mean"].shape == (2, 2000, 1)
    assert trace["mean"].mean().item() == pytest.approx(1.6, abs=0.03)
    assert trace["mean"].std().item() == pytest.approx(math.sqrt(4 / 13), rel=0.05)


def test_laplace_is_approximately_right_for_beta_bernoulli():
    data = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    a, b = 2.0, 2.0
    with Model() as model:
        p = RandomParameter("p", Beta(a, b))
        with plate("data", len(data)):
            ObservedParameter("y", Bernoulli(p), data)
    approximation = laplace_approximation(model, map_epochs=200, lr=0.05)
    k, n = data.sum().item(), len(data)
    # In the logit space the mode of p is (a + k) / (a + b + n).
    p_mode = torch.sigmoid(approximation.mean).item()
    assert p_mode == pytest.approx((a + k) / (a + b + n), abs=1e-6)
    exact = math.lgamma(a + k) + math.lgamma(b + n - k) - math.lgamma(a + b + n) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    assert approximation.log_marginal_likelihood.item() == pytest.approx(exact, abs=0.05)
    trace = approximation.sample(4000)
    posterior_mean = (a + k) / (a + b + n)
    assert trace["p"].mean().item() == pytest.approx(posterior_mean, abs=0.02)


def test_laplace_warns_when_negative_hessian_is_not_positive_definite():
    with Model() as model:
        RandomParameter("x", Normal(0.0, 1.0))
    model.joint_grad_log_prob = lambda names=None, nodes=None: {"x": model.params["x"].unconstrained_value.clone()}
    with pytest.warns(UserWarning, match="not positive definite"):
        laplace_approximation(model, map_epochs=0, newton_steps=0)


@pytest.mark.integration
def test_minibatch_bbvi_matches_full_batch_optimum():
    torch.manual_seed(3)
    n, batch = 5000, 100
    x = torch.randn(n)
    y = 0.5 + 1.5 * x + torch.randn(n)

    def build():
        with Model() as model:
            a = RandomParameter("a", Normal(0.0, 5.0))
            b = RandomParameter("b", Normal(0.0, 5.0))
            xs = Data("x", x)
            mu = DeterministicParameter(
                "mu", lambda a, b, x: a + b * x,
                lambda a, b, x: {"a": torch.ones_like(x), "b": x}, [a, b, xs],
            )
            with plate("observations", xs):
                ObservedParameter("y", Normal(mu, 1.0), y)
        return model

    # The full-batch optimum of a mean-field Normal guide for this Gaussian
    # posterior: the exact posterior mean and the scales 1 / sqrt(precision_ii).
    design = torch.stack([torch.ones_like(x), x], dim=1)
    precision = design.T @ design + torch.eye(2) / 25
    mean = torch.linalg.solve(precision, design.T @ y)
    scales = precision.diagonal().rsqrt()

    model = build()
    guide = AutoNormal(model)
    BBVI(model, guide, n_samples=4, epochs=1500, lr=0.01, lrd=0.999, batch_size=batch, verbose=False)
    torch.testing.assert_close(guide.loc, mean, atol=0.03, rtol=0)
    torch.testing.assert_close(guide.scale, scales, atol=0.01, rtol=0)
    # The data are restored after every minibatch.
    assert model.data["x"].value.shape == (n,)
    assert model.observed_params["y"].observed_values.shape == (n,)


def test_minibatch_requires_data_plate(normal_model):
    guide = AutoNormal(normal_model)
    with pytest.raises(ValueError, match="plate"):
        BBVI(normal_model, guide, epochs=1, batch_size=2, verbose=False)


def test_minibatch_scales_likelihood():
    """With B = N the minibatch ELBO equals the full ELBO; the scaled ELBO is unbiased."""
    from BayesianDLL.Variational._minibatch import Subsampler
    from BayesianDLL.Variational._ELBO import scaled_log_joint
    x = torch.arange(6.0)
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        xs = Data("x", x)
        mu = DeterministicParameter("mu", lambda a, x: a * x, lambda a, x: {"a": x}, [a, xs])
        with plate("obs", xs):
            ObservedParameter("y", Normal(mu, 1.0), torch.ones(6))
    full = model.model_log_prob()
    subsampler = Subsampler(model, 3)
    estimates = []
    for _ in range(400):
        with subsampler.minibatch() as scales:
            assert scales == {"y": 2.0}
            assert model.data["x"].value.shape == (3,)
            estimates.append(scaled_log_joint(model, scales))
    assert torch.stack(estimates).mean().item() == pytest.approx(full.item(), rel=0.02)


@pytest.mark.parametrize("outer_sized_by_data", [False, True])
def test_minibatch_subsamples_inner_nested_plate(outer_sized_by_data):
    """The subsampled axis is the plate's position in the value shape, not its sizing dim.

    The inner plate of a nested pair sits at position 1 of the observed values
    (shape (groups, n)), while its sizing Data is 1-d.
    """
    from BayesianDLL.Variational._minibatch import Subsampler
    from BayesianDLL.Variational._ELBO import scaled_log_joint

    n_groups, n = 3, 8
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n_groups, n, dtype=torch.float64)
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        group_sizes = Data("groups", torch.zeros(n_groups, dtype=torch.float64))
        xs = Data("x", x)
        mu = DeterministicParameter("mu", lambda a, x: a * x, lambda a, x: {"a": x}, [a, xs])
        with plate("groups", group_sizes if outer_sized_by_data else n_groups):
            with plate("obs", xs):
                ObservedParameter("y", Normal(mu, 1.0), y)

    subsampler = Subsampler(model, 4, subsample_plate="obs")
    a_value = model.params["a"].constrained_value
    with subsampler.minibatch(generator=torch.Generator().manual_seed(1)) as scales:
        x_batch = model.data["x"].value
        y_batch = model.observed_params["y"].observed_values
        assert x_batch.shape == (4,)
        assert y_batch.shape == (n_groups, 4)
        # Every row of the batch is the same subset of columns as x.
        columns = [int(torch.nonzero(x == value)[0]) for value in x_batch]
        assert torch.equal(y_batch, y[:, columns])
        expected = (
            torch.distributions.Normal(0.0, 1.0).log_prob(a_value).sum()
            + scales["y"] * torch.distributions.Normal(a_value * x_batch, 1.0).log_prob(y_batch).sum()
        )
        assert scaled_log_joint(model, scales).item() == pytest.approx(expected.item(), rel=1e-10)
    assert torch.equal(model.data["x"].value, x)
    assert torch.equal(model.observed_params["y"].observed_values, y)


def test_minibatch_subsamples_outer_nested_plate():
    """Subsampling the outer plate of a nested pair slices axis 0 of the observations."""
    from BayesianDLL.Variational._minibatch import Subsampler
    from BayesianDLL.Variational._ELBO import scaled_log_joint

    n, n_inner = 8, 3
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, n_inner, dtype=torch.float64)
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        xs = Data("x", x)
        mu = DeterministicParameter("mu", lambda a, x: (a * x).unsqueeze(-1), lambda a, x: {"a": x.unsqueeze(-1)}, [a, xs])
        with plate("obs", xs):
            with plate("repeats", n_inner):
                ObservedParameter("y", Normal(mu, 1.0), y)

    subsampler = Subsampler(model, 4, subsample_plate="obs")
    a_value = model.params["a"].constrained_value
    with subsampler.minibatch(generator=torch.Generator().manual_seed(1)) as scales:
        x_batch = model.data["x"].value
        y_batch = model.observed_params["y"].observed_values
        assert x_batch.shape == (4,)
        assert y_batch.shape == (4, n_inner)
        rows = [int(torch.nonzero(x == value)[0]) for value in x_batch]
        assert torch.equal(y_batch, y[rows])
        expected = (
            torch.distributions.Normal(0.0, 1.0).log_prob(a_value).sum()
            + scales["y"] * torch.distributions.Normal((a_value * x_batch).unsqueeze(-1), 1.0).log_prob(y_batch).sum()
        )
        assert scaled_log_joint(model, scales).item() == pytest.approx(expected.item(), rel=1e-10)
