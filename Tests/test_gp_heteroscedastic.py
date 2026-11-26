"""Heteroscedastic (per-observation) noise variances in ExactGP, dense and with inducing points."""

import warnings

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    RBF,
    ExactGP,
    KroneckerGP,
    Matern52,
    ToeplitzGP,
    exact_gp_predictive,
    stable_cholesky,
)

DTYPE = torch.float64
C = 4


def _finite_difference_check(model, eps=1e-6, rtol=1e-5, atol=1e-6):
    gradients = model.joint_grad_log_prob()
    for name, parameter in model.params.items():
        start = parameter.unconstrained_value.clone()
        numerical = torch.zeros_like(start).reshape(-1)
        for index in range(start.numel()):
            shift = torch.zeros_like(start).reshape(-1)
            shift[index] = eps
            parameter.set_unconstrained_value(start + shift.reshape(start.shape))
            upper = model.model_log_prob()
            parameter.set_unconstrained_value(start - shift.reshape(start.shape))
            lower = model.model_log_prob()
            numerical[index] = (upper - lower) / (2 * eps)
        parameter.set_unconstrained_value(start)
        assert gradients[name].shape == start.shape, name
        torch.testing.assert_close(gradients[name].reshape(-1), numerical, rtol=rtol, atol=atol, msg=f"gradient of {name}")
    return gradients


def _assert_batched_equals_loop(model, seed=0, scale=0.2):
    generator = torch.Generator().manual_seed(seed)
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        values[name] = u.unsqueeze(0) + scale * torch.randn((C, *u.shape), generator=generator, dtype=u.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(C, values=values, mode="auto"):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            batched_lp = model.model_log_prob()
            batched_grads = model.joint_grad_log_prob()
    with model.batched(C, values=values, mode="loop"):
        loop_lp = model.model_log_prob()
        loop_grads = model.joint_grad_log_prob()
    torch.testing.assert_close(batched_lp, loop_lp, atol=1e-10, rtol=1e-10)
    for name, gradient in loop_grads.items():
        torch.testing.assert_close(batched_grads[name], gradient, atol=1e-10, rtol=1e-10, msg=f"gradient of {name}")


def _data(n=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=generator, dtype=DTYPE) * 2.0
    noise = 0.02 + 0.3 * x[:, 0] ** 2
    y = torch.sin(2 * x[:, 0]) + x[:, 1] + noise.sqrt() * torch.randn(n, generator=generator, dtype=DTYPE)
    return x, y, noise


def _model(noise_style, inducing=None, method=None, n_outputs=None, n=12):
    x, y, noise = _data(n)
    if n_outputs is not None:
        y = torch.stack([y + 0.1 * r for r in range(n_outputs)])
    with Model() as model:
        inputs = Data("x", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=2,
                                          initial_value=torch.tensor([-0.2, 0.3], dtype=DTYPE))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.4], dtype=DTYPE))
        if noise_style == "vector":
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), shape=n, initial_value=noise.log())
            noise_variance = Exp("noise", log_noise)
        elif noise_style == "per_output":
            initial = torch.stack([noise * (1 + r) for r in range(n_outputs)]).log()
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), shape=(n_outputs, n), initial_value=initial)
            noise_variance = Exp("noise", log_noise)
        elif noise_style == "output_scalar":
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), shape=(n_outputs, 1),
                                        initial_value=torch.linspace(-2.0, -1.0, n_outputs, dtype=DTYPE).reshape(-1, 1))
            noise_variance = Exp("noise", log_noise)
        elif noise_style == "function":
            # The noise variance as a function of the inputs: σ²(x) = exp(a + b x_0).
            a = RandomParameter("a", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.5], dtype=DTYPE))
            b = RandomParameter("b", Normal(0.0, 1.0), initial_value=torch.tensor([1.2], dtype=DTYPE))
            from BayesianDLL import DeterministicParameter
            noise_variance = DeterministicParameter(
                "noise", lambda a_, b_, x_: torch.exp(a_ + b_ * x_[..., 0]),
                vjp_func=lambda g, a_, b_, x_: {
                    "a": (g * torch.exp(a_ + b_ * x_[..., 0])).sum(-1, keepdim=True),
                    "b": (g * torch.exp(a_ + b_ * x_[..., 0]) * x_[..., 0]).sum(-1, keepdim=True),
                },
                inputs=[a, b, inputs],
            )
        else:
            noise_variance = Data("noise", noise)
        kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.1], dtype=DTYPE))
        function = ExactGP("f", inputs, kernel, noise_variance=noise_variance, mean=mean,
                           inducing_points=inducing, method=method, n_outputs=n_outputs)
        ObservedParameter("y", function, y)
    return model, function, x, y


def _dense_log_likelihood(function, x, y, noise_diagonal):
    values = function._named_parameter_values()
    covariance = function.kernel(x, x, parameter_values=values) + torch.diag(noise_diagonal)
    _, (jitter, _, _) = stable_cholesky(covariance, jitter=function.jitter, return_jitter=True)
    covariance = covariance + jitter * torch.eye(x.shape[0], dtype=DTYPE)
    mean = values["mean"].expand(x.shape[0])
    return torch.distributions.MultivariateNormal(mean, covariance).log_prob(y)


# ----------------------------------------------------------------------------
# Dense ExactGP
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("noise_style", ["vector", "data", "function"])
def test_dense_log_likelihood_matches_multivariate_normal(noise_style):
    model, function, x, y = _model(noise_style)
    noise = torch.as_tensor(model.deterministic_params["noise"].constrained_value if noise_style != "data"
                            else function.noise_variance.value)
    torch.testing.assert_close(function.log_pdf(y), _dense_log_likelihood(function, x, y, noise))


@pytest.mark.parametrize("noise_style", ["vector", "function"])
def test_dense_gradients_match_finite_differences(noise_style):
    model, _, _, _ = _model(noise_style)
    _finite_difference_check(model)


def test_dense_noise_gradient_matches_autograd():
    model, function, x, y = _model("vector")
    noise = model.deterministic_params["noise"].constrained_value.clone().requires_grad_(True)
    _dense_log_likelihood(function, x, y, noise).backward()
    torch.testing.assert_close(function.log_pdf_param_grads(y)["noise"], noise.grad)


def test_constant_noise_equal_to_a_scalar_gives_the_same_likelihood():
    x, y, _ = _data()
    with Model():
        inputs = Data("x", x)
        scalar = ExactGP("a", inputs, RBF(0.5, 1.2), noise_variance=0.1)
        vector = ExactGP("b", inputs, RBF(0.5, 1.2), noise_variance=torch.full((x.shape[0],), 0.1, dtype=DTYPE))
    torch.testing.assert_close(vector.log_pdf(y), scalar.log_pdf(y))


@pytest.mark.parametrize("noise_style", ["vector", "function"])
def test_dense_batched_matches_chain_by_chain(noise_style):
    model, _, _, _ = _model(noise_style)
    _assert_batched_equals_loop(model)


def test_wrong_noise_shape_is_rejected():
    x, _, _ = _data()
    with Model():
        inputs = Data("x", x)
        with pytest.raises(ValueError, match="one value per observation"):
            ExactGP("f", inputs, RBF(0.5), noise_variance=torch.ones(5, dtype=DTYPE))
        with pytest.raises(ValueError, match="one value per observation"):
            ExactGP("g", inputs, RBF(0.5), noise_variance=torch.ones(3, x.shape[0], dtype=DTYPE),
                    inducing_points=x[:3], n_outputs=2)


@pytest.mark.parametrize("structure", ["kronecker", "toeplitz"])
def test_structured_gps_reject_per_observation_noise(structure):
    grid = torch.linspace(0.0, 1.0, 6, dtype=DTYPE)
    with Model():
        with pytest.raises(ValueError, match="heteroscedastic"):
            if structure == "kronecker":
                KroneckerGP("f", [grid, grid[:3]], [RBF(0.5), RBF(0.5)], noise_variance=torch.full((6, 3), 0.1, dtype=DTYPE))
            else:
                ToeplitzGP("f", Data("x", grid), RBF(0.5), noise_variance=torch.full((6,), 0.1, dtype=DTYPE))


# ----------------------------------------------------------------------------
# Inducing points
# ----------------------------------------------------------------------------


def _sparse_reference(function, x, z, y, noise_diagonal, method):
    """Dense formula of the VFE / DTC / FITC log likelihood with Λ = diag(σ²) (+ FITC correction)."""
    values = function._named_parameter_values()
    k_nn = function.kernel(x, x, parameter_values=values)
    k_mm = function.kernel(z, z, parameter_values=values)
    k_nm = function.kernel(x, z, parameter_values=values)
    _, (jitter, _, _) = stable_cholesky(k_mm, jitter=function.jitter, return_jitter=True)
    k_mm = k_mm + jitter * torch.eye(z.shape[0], dtype=DTYPE)
    q_nn = k_nm @ torch.linalg.solve(k_mm, k_nm.T)
    lam = noise_diagonal
    if method == "fitc":
        lam = lam + torch.diagonal(k_nn - q_nn)
    value = torch.distributions.MultivariateNormal(values["mean"].expand(x.shape[0]), q_nn + torch.diag(lam)).log_prob(y)
    if method == "vfe":
        value = value - 0.5 * (torch.diagonal(k_nn - q_nn) / noise_diagonal).sum()
    return value


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_log_likelihood_matches_dense_formula(method):
    x, _, _ = _data()
    model, function, x, y = _model("vector", inducing=x[:5].clone(), method=method)
    noise = model.deterministic_params["noise"].constrained_value
    torch.testing.assert_close(function.log_pdf(y), _sparse_reference(function, x, x[:5], y, noise, method))


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gradients_match_finite_differences(method):
    x, y, noise = _data()
    z0 = x[:5].clone()
    with Model() as model:
        inputs = Data("x", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=2,
                                          initial_value=torch.tensor([-0.2, 0.3], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), shape=x.shape[0], initial_value=noise.log())
        z = RandomParameter("z", Normal(z0, 1.0), shape=tuple(z0.shape), initial_value=z0 + 0.05)
        function = ExactGP("f", inputs, Matern52(Exp("lengthscale", log_lengthscale), 1.3), inducing_points=z,
                           noise_variance=Exp("noise", log_noise), mean=0.2, method=method)
        ObservedParameter("y", function, y)
    _finite_difference_check(model)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
@pytest.mark.parametrize("noise_style", ["vector", "per_output", "output_scalar"])
def test_independent_outputs_equal_the_sum_of_single_output_gps(method, noise_style):
    x, _, _ = _data()
    model, function, x, y = _model(noise_style, inducing=x[:5].clone(), method=method, n_outputs=3)
    noise = model.deterministic_params["noise"].constrained_value
    noise = noise.expand(3, x.shape[0]) if noise.ndim == 2 else noise.expand(3, -1)
    expected = sum(_sparse_reference(function, x, x[:5], y[r], noise[r], method) for r in range(3))
    torch.testing.assert_close(function.log_pdf(y), expected)
    _finite_difference_check(model)


@pytest.mark.parametrize("method", ["vfe", "fitc"])
@pytest.mark.parametrize("noise_style", ["vector", "per_output"])
def test_sparse_batched_matches_chain_by_chain(method, noise_style):
    x, _, _ = _data()
    n_outputs = 2 if noise_style == "per_output" else None
    model, _, _, _ = _model(noise_style, inducing=x[:5].clone(), method=method, n_outputs=n_outputs)
    _assert_batched_equals_loop(model)


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------


def test_dense_predictive_matches_the_analytic_posterior():
    x, y, noise = _data(n=10)
    kernel = RBF(torch.tensor([0.6, 0.9], dtype=DTYPE), 1.1)
    with Model():
        function = ExactGP("f", Data("x", x), kernel, noise_variance=noise, jitter=1e-10)
    new = torch.rand(4, 2, generator=torch.Generator().manual_seed(3), dtype=DTYPE) * 2.0
    extra = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=DTYPE)
    torch.manual_seed(0)
    draws = exact_gp_predictive(function, {}, y, new, n_samples=40000, observation_noise=extra)[0]
    covariance = kernel(x, x) + torch.diag(noise)
    cross = kernel(new, x)
    mean = cross @ torch.linalg.solve(covariance, y)
    expected = kernel(new, new) - cross @ torch.linalg.solve(covariance, cross.T) + torch.diag(extra)
    torch.testing.assert_close(draws.mean(0), mean, atol=0.02, rtol=0)
    torch.testing.assert_close(torch.cov(draws.T), expected, atol=0.02, rtol=0)


@pytest.mark.parametrize("n_outputs", [None, 2])
def test_sparse_predictive_equals_dense_when_inducing_points_are_the_inputs(n_outputs):
    x, y, noise = _data(n=10)
    kernel = RBF(torch.tensor([0.6, 0.9], dtype=DTYPE), 1.1)
    with Model():
        inputs = Data("x", x)
        dense = ExactGP("dense", inputs, kernel, noise_variance=noise, jitter=1e-10)
        if n_outputs is None:
            sparse = ExactGP("sparse", inputs, kernel, inducing_points=x, noise_variance=noise, jitter=1e-10)
        else:
            per_output = torch.stack([noise, 2 * noise])
            sparse = ExactGP("sparse", inputs, kernel, inducing_points=x, noise_variance=per_output, jitter=1e-10,
                             n_outputs=2)
            dense_2 = ExactGP("dense_2", inputs, kernel, noise_variance=2 * noise, jitter=1e-10)
    new = torch.rand(3, 2, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    trace = {"dummy": torch.zeros(1, 1, 1, dtype=DTYPE)}
    torch.manual_seed(0)
    expected = exact_gp_predictive(dense, trace, y, new, n_samples=2000)
    if n_outputs is None:
        torch.manual_seed(0)
        actual = exact_gp_predictive(sparse, trace, y, new, n_samples=2000)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        return
    observations = torch.stack([y, y - 0.3])
    torch.manual_seed(0)
    actual = exact_gp_predictive(sparse, trace, observations, new, n_samples=2000)
    assert actual.shape == (1, 2000, 2, 3)
    expected_2 = exact_gp_predictive(dense_2, trace, y - 0.3, new, n_samples=4000)
    torch.testing.assert_close(actual[:, :, 0].mean(1), expected.mean(1), atol=0.03, rtol=0)
    torch.testing.assert_close(actual[:, :, 1].mean(1), expected_2.mean(1), atol=0.03, rtol=0)
    torch.testing.assert_close(actual[:, :, 1].var(1), expected_2.var(1), atol=0.02, rtol=0.1)


def test_posterior_predictive_with_data_noise_at_new_inputs():
    x, y, noise = _data(n=10)
    with Model() as model:
        inputs = Data("x", x)
        noise_data = Data("noise", noise)
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.1], dtype=DTYPE))
        function = ExactGP("f", inputs, RBF(0.7, Exp("variance", log_variance)), noise_variance=noise_data)
        ObservedParameter("y", function, y)
    new = torch.rand(3, 2, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    inputs.set_value(new)
    with pytest.raises(ValueError, match="Data.set_value"):
        model.predict(samples_per_step=5)
    new_noise = torch.tensor([1e-8, 1e-8, 4.0], dtype=DTYPE)
    noise_data.set_value(new_noise)
    torch.manual_seed(0)
    prediction = model.predict(samples_per_step=20000)
    observations = prediction["y"][0]
    latent = prediction["f"][0]
    assert observations.shape == (20000, 3)
    # The latent draws condition on the training noise; the observations add the new noise.
    torch.testing.assert_close((observations - latent).var(0), new_noise, atol=0.15, rtol=0.05)
    values = {"log_variance": model.params["log_variance"].constrained_value}
    kernel = RBF(0.7, values["log_variance"].exp())
    cross = kernel(new, x)
    mean = cross @ torch.linalg.solve(kernel(x, x) + torch.diag(noise), y)
    torch.testing.assert_close(latent.mean(0), mean, atol=0.03, rtol=0)
