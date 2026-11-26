"""Spectral mixture and change-point kernels: values, hyperparameter derivatives, input VJPs and batching."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    RBF,
    ChangePoint,
    ExactGP,
    Linear,
    Matern32,
    Periodic,
    SpectralMixture,
    ToeplitzGP,
)

DTYPE = torch.float64
C = 4


def _name(name):
    return SimpleNamespace(name=name)


# ----------------------------------------------------------------------------
# Reference formulas (plain torch, differentiable with autograd)
# ----------------------------------------------------------------------------


def _reference_spectral_mixture(x1, x2, weights, frequencies, lengthscales):
    count, features = weights.shape[0], x1.shape[-1]
    frequencies = frequencies.reshape(count, -1).expand(count, features)
    lengthscales = lengthscales.reshape(count, -1).expand(count, features)
    tau = x1[:, None, :] - x2[None, :, :]
    result = 0.0
    for q in range(count):
        component = torch.ones(tau.shape[:2], dtype=x1.dtype)
        for d in range(features):
            component = component * torch.exp(-0.5 * tau[..., d] ** 2 / lengthscales[q, d] ** 2)
            component = component * torch.cos(2 * torch.pi * frequencies[q, d] * tau[..., d])
        result = result + weights[q] * component
    return result


def _reference_change_point(x1, x2, location, steepness, before, after, dimension=0, first=None, second=None):
    s1 = torch.sigmoid(steepness * (x1[:, dimension] - location))
    s2 = torch.sigmoid(steepness * (x2[:, dimension] - location))
    first = before(x1, x2) if first is None else first
    second = after(x1, x2) if second is None else second
    return (1 - s1)[:, None] * (1 - s2)[None, :] * first + s1[:, None] * s2[None, :] * second


def _rbf(lengthscale, variance):
    def kernel(a, b):
        return variance * torch.exp(-0.5 * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / lengthscale ** 2)
    return kernel


def _linear(variance):
    return lambda a, b: variance * a @ b.T


_X1 = torch.tensor([[0.0, 0.2], [0.4, -0.1], [1.0, 0.8], [0.3, 0.3], [-0.6, 0.5]], dtype=DTYPE)
_X2 = torch.tensor([[0.1, 0.0], [0.9, 0.5], [0.3, 0.3], [-0.2, 1.1]], dtype=DTYPE)
_SM_VALUES = {
    "weights": torch.tensor([0.7, 0.4, 1.1], dtype=DTYPE),
    "frequencies": torch.tensor([[0.3, 0.1], [1.2, 0.6], [0.05, 0.9]], dtype=DTYPE),
    "lengthscales": torch.tensor([[0.8, 1.4], [0.5, 0.9], [2.0, 0.7]], dtype=DTYPE),
}


def _sm_values(shared):
    values = {key: value.clone() for key, value in _SM_VALUES.items()}
    if shared:
        values["frequencies"] = values["frequencies"][:, 0].clone()
        values["lengthscales"] = values["lengthscales"][:, 1].clone()
    return values


def _sm_kernel():
    return SpectralMixture(_name("weights"), _name("frequencies"), _name("lengthscales"))


# ----------------------------------------------------------------------------
# Spectral mixture
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shared", [False, True])
def test_spectral_mixture_matches_direct_formula(shared):
    values = _sm_values(shared)
    kernel = _sm_kernel()
    expected = _reference_spectral_mixture(_X1, _X2, values["weights"], values["frequencies"], values["lengthscales"])
    torch.testing.assert_close(kernel(_X1, _X2, parameter_values=values), expected)
    torch.testing.assert_close(kernel.diag(_X1, parameter_values=values), torch.full((5,), values["weights"].sum().item(), dtype=DTYPE))


def test_spectral_mixture_one_component_is_a_cosine_times_rbf():
    x = torch.linspace(0.0, 3.0, 7, dtype=DTYPE)
    kernel = SpectralMixture(torch.tensor([1.3], dtype=DTYPE), torch.tensor([0.5], dtype=DTYPE), torch.tensor([0.8], dtype=DTYPE))
    tau = x[:, None] - x[None, :]
    torch.testing.assert_close(kernel(x, x), RBF(0.8, 1.3)(x, x) * torch.cos(torch.pi * tau))


@pytest.mark.parametrize("shared", [False, True])
def test_spectral_mixture_derivatives_match_autograd(shared):
    values = _sm_values(shared)
    kernel = _sm_kernel()
    analytic = kernel.derivatives(_X1, _X2, parameter_values=values)
    names = list(values)
    jacobians = torch.autograd.functional.jacobian(
        lambda *args: _reference_spectral_mixture(_X1, _X2, *args), tuple(values[name] for name in names)
    )
    for name, jacobian in zip(names, jacobians):
        expected = jacobian.movedim(0, -1).movedim(0, -1)  # parameter axes first
        assert analytic[name].shape == (*values[name].shape, 5, 4), name
        torch.testing.assert_close(analytic[name], expected, msg=f"derivative of {name}")
    diagonal = kernel.diag_derivatives(_X1, parameter_values=values)
    full = kernel.derivatives(_X1, _X1, parameter_values=values)
    for name in names:
        torch.testing.assert_close(diagonal[name], torch.diagonal(full[name], dim1=-2, dim2=-1), msg=name)


@pytest.mark.parametrize("shared", [False, True])
def test_spectral_mixture_input_vjp_matches_autograd(shared):
    values = _sm_values(shared)
    kernel = _sm_kernel()
    cotangent = torch.randn(5, 4, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    square = torch.randn(5, 5, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    leaf = _X1.clone().requires_grad_(True)
    reference = _reference_spectral_mixture(leaf, _X2, values["weights"], values["frequencies"], values["lengthscales"])
    (cotangent * reference).sum().backward()
    torch.testing.assert_close(kernel.input_vjp(_X1, _X2, cotangent, parameter_values=values), leaf.grad)
    leaf = _X1.clone().requires_grad_(True)
    reference = _reference_spectral_mixture(leaf, leaf, values["weights"], values["frequencies"], values["lengthscales"])
    (square * reference).sum().backward()
    torch.testing.assert_close(kernel.input_vjp_symmetric(_X1, square, parameter_values=values), leaf.grad)
    torch.testing.assert_close(kernel.diag_input_vjp(_X1, torch.ones(5, dtype=DTYPE), parameter_values=values), torch.zeros_like(_X1))


def test_spectral_mixture_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="SpectralMixture frequencies"):
        SpectralMixture(torch.ones(3, dtype=DTYPE), torch.ones(2, dtype=DTYPE), torch.ones(3, dtype=DTYPE))(_X1, _X2)
    with pytest.raises(ValueError, match="SpectralMixture lengthscales"):
        SpectralMixture(torch.ones(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE), torch.ones(2, 3, dtype=DTYPE))(_X1, _X2)


# ----------------------------------------------------------------------------
# Change point
# ----------------------------------------------------------------------------


def _cp_kernel(dimension=0):
    before = RBF(_name("lengthscale"), 1.3)
    after = Linear(_name("slope")) + Periodic(0.7, 0.4, 1.1)
    return ChangePoint(before, after, _name("location"), _name("steepness"), dimension=dimension)


_CP_VALUES = {
    "lengthscale": torch.tensor([0.6], dtype=DTYPE),
    "slope": torch.tensor([0.8], dtype=DTYPE),
    "location": torch.tensor([0.35], dtype=DTYPE),
    "steepness": torch.tensor([4.0], dtype=DTYPE),
}


def _cp_reference(x1, x2, values, dimension=0):
    periodic = Periodic(0.7, 0.4, 1.1)(x1.detach(), x2.detach())
    if x1.requires_grad or x2.requires_grad:
        # A differentiable periodic kernel for the input gradients.
        def periodic_kernel(a, b):
            distance = torch.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) + 1e-30)
            return 0.4 * torch.exp(-2 * torch.sin(torch.pi * distance / 1.1) ** 2 / 0.7 ** 2)
        periodic = periodic_kernel(x1, x2)
    first = _rbf(values["lengthscale"], 1.3)(x1, x2)
    second = _linear(values["slope"])(x1, x2) + periodic
    return _reference_change_point(x1, x2, values["location"], values["steepness"], None, None, dimension, first, second)


@pytest.mark.parametrize("dimension", [0, -1])
def test_change_point_matches_direct_formula(dimension):
    kernel = _cp_kernel(dimension)
    expected = _cp_reference(_X1, _X2, _CP_VALUES, dimension)
    torch.testing.assert_close(kernel(_X1, _X2, parameter_values=_CP_VALUES), expected)
    torch.testing.assert_close(kernel.diag(_X1, parameter_values=_CP_VALUES), torch.diagonal(kernel(_X1, _X1, parameter_values=_CP_VALUES)))


def test_change_point_switches_between_the_kernels():
    x = torch.tensor([[-5.0], [-4.0], [4.0], [5.0]], dtype=DTYPE)
    kernel = ChangePoint(RBF(1.0, 2.0), RBF(1.0, 0.5), location=0.0, steepness=50.0)
    covariance = kernel(x, x)
    torch.testing.assert_close(covariance[:2, :2], RBF(1.0, 2.0)(x[:2], x[:2]))
    torch.testing.assert_close(covariance[2:, 2:], RBF(1.0, 0.5)(x[2:], x[2:]))
    torch.testing.assert_close(covariance[:2, 2:], torch.zeros(2, 2, dtype=DTYPE))


@pytest.mark.parametrize("dimension", [0, 1])
def test_change_point_derivatives_match_autograd(dimension):
    kernel = _cp_kernel(dimension)
    names = list(_CP_VALUES)
    analytic = kernel.derivatives(_X1, _X2, parameter_values=_CP_VALUES)
    jacobians = torch.autograd.functional.jacobian(
        lambda *args: _cp_reference(_X1, _X2, dict(zip(names, args)), dimension), tuple(_CP_VALUES[n] for n in names)
    )
    for name, jacobian in zip(names, jacobians):
        torch.testing.assert_close(analytic[name], jacobian.movedim(-1, 0), msg=f"derivative of {name}")
    diagonal = kernel.diag_derivatives(_X1, parameter_values=_CP_VALUES)
    full = kernel.derivatives(_X1, _X1, parameter_values=_CP_VALUES)
    for name in names:
        torch.testing.assert_close(diagonal[name], torch.diagonal(full[name], dim1=-2, dim2=-1), msg=name)


@pytest.mark.parametrize("dimension", [0, 1])
def test_change_point_input_vjps_match_autograd(dimension):
    kernel = _cp_kernel(dimension)
    generator = torch.Generator().manual_seed(2)
    cotangent = torch.randn(5, 4, generator=generator, dtype=DTYPE)
    square = torch.randn(5, 5, generator=generator, dtype=DTYPE)
    weights = torch.randn(5, generator=generator, dtype=DTYPE)
    leaf = _X1.clone().requires_grad_(True)
    (cotangent * _cp_reference(leaf, _X2, _CP_VALUES, dimension)).sum().backward()
    torch.testing.assert_close(kernel.input_vjp(_X1, _X2, cotangent, parameter_values=_CP_VALUES), leaf.grad)
    leaf = _X1.clone().requires_grad_(True)
    (square * _cp_reference(leaf, leaf, _CP_VALUES, dimension)).sum().backward()
    torch.testing.assert_close(kernel.input_vjp_symmetric(_X1, square, parameter_values=_CP_VALUES), leaf.grad)
    # The diagonal depends on x through the sigmoids and the linear kernel.
    leaf = _X1.clone().requires_grad_(True)
    (weights * torch.diagonal(_cp_reference(leaf, leaf, _CP_VALUES, dimension))).sum().backward()
    torch.testing.assert_close(kernel.diag_input_vjp(_X1, weights, parameter_values=_CP_VALUES), leaf.grad)


def test_change_point_validates_arguments():
    with pytest.raises(TypeError, match="Kernel"):
        ChangePoint(RBF(1.0), lambda a, b: a, 0.0)
    with pytest.raises(ValueError, match="out of range"):
        ChangePoint(RBF(1.0), RBF(1.0), 0.0, dimension=3)(_X1, _X2)


# ----------------------------------------------------------------------------
# Batched form
# ----------------------------------------------------------------------------


_BATCH = {
    "spectral_mixture": (_sm_kernel, lambda b, g: {
        "weights": 0.5 + torch.rand(b, 3, generator=g, dtype=DTYPE),
        "frequencies": torch.rand(b, 3, 2, generator=g, dtype=DTYPE),
        "lengthscales": 0.5 + torch.rand(b, 3, generator=g, dtype=DTYPE),
    }),
    "change_point": (lambda: _cp_kernel(1), lambda b, g: {
        "lengthscale": 0.5 + torch.rand(b, 1, generator=g, dtype=DTYPE),
        "slope": 0.5 + torch.rand(b, 1, generator=g, dtype=DTYPE),
        "location": torch.randn(b, 1, generator=g, dtype=DTYPE),
        "steepness": 1.0 + torch.rand(b, 1, generator=g, dtype=DTYPE),
    }),
}


@pytest.mark.parametrize("name", list(_BATCH))
def test_batched_methods_match_per_element_calls(name):
    factory, draw = _BATCH[name]
    kernel = factory()
    generator = torch.Generator().manual_seed(1)
    batch = 3
    values = draw(batch, generator)
    x1 = torch.randn(batch, 4, 2, generator=generator, dtype=DTYPE)
    x2 = torch.randn(5, 2, generator=generator, dtype=DTYPE)
    cotangent = torch.randn(batch, 4, 5, generator=generator, dtype=DTYPE)
    weights = torch.randn(batch, 4, generator=generator, dtype=DTYPE)
    covariance = kernel.batch_evaluate(x1, x2.unsqueeze(0), values)
    derivatives = kernel.batch_derivatives(x1, x2.unsqueeze(0), values)
    diagonal = kernel.batch_diag(x1, values)
    diagonal_derivatives = kernel.batch_diag_derivatives(x1, values)
    vjp = kernel.batch_input_vjp(x1, x2.unsqueeze(0), cotangent, values)
    diag_vjp = kernel.batch_diag_input_vjp(x1, weights, values)
    for b in range(batch):
        single = {key: value[b] for key, value in values.items()}
        torch.testing.assert_close(covariance[b], kernel(x1[b], x2, parameter_values=single))
        for key, derivative in kernel.derivatives(x1[b], x2, parameter_values=single).items():
            torch.testing.assert_close(derivatives[key].expand(batch, *derivative.shape)[b], derivative)
        torch.testing.assert_close(diagonal.expand(batch, -1)[b], kernel.diag(x1[b], parameter_values=single))
        for key, derivative in kernel.diag_derivatives(x1[b], parameter_values=single).items():
            torch.testing.assert_close(diagonal_derivatives[key].expand(batch, *derivative.shape)[b], derivative)
        torch.testing.assert_close(vjp[b], kernel.input_vjp(x1[b], x2, cotangent[b], parameter_values=single))
        torch.testing.assert_close(diag_vjp[b], kernel.diag_input_vjp(x1[b], weights[b], parameter_values=single))


# ----------------------------------------------------------------------------
# In GP models
# ----------------------------------------------------------------------------


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


def _series(n=30):
    x = torch.linspace(0.0, 6.0, n, dtype=DTYPE)
    y = torch.where(x < 3.0, torch.sin(3 * x), 0.3 * x - 0.9) + 0.05 * torch.randn(n, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    return x, y


def _model(kernel_name, learnable_inputs=False, inducing=None, method=None):
    x, y = _series()
    with Model() as model:
        if learnable_inputs:
            inputs = RandomParameter("x", Normal(x, 0.1), shape=x.shape[0], initial_value=x + 0.01)
        else:
            inputs = Data("x", x)
        if kernel_name == "spectral_mixture":
            log_weights = RandomParameter("log_weights", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([0.2, -0.5], dtype=DTYPE))
            frequencies = RandomParameter("frequencies", Normal(0.5, 0.5), shape=2, initial_value=torch.tensor([0.45, 0.1], dtype=DTYPE))
            log_lengthscales = RandomParameter("log_lengthscales", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([0.3, 0.8], dtype=DTYPE))
            kernel = SpectralMixture(Exp("weights", log_weights), frequencies, Exp("lengthscales", log_lengthscales))
        else:
            location = RandomParameter("location", Normal(3.0, 1.0), initial_value=torch.tensor([2.8], dtype=DTYPE))
            log_steepness = RandomParameter("log_steepness", Normal(1.0, 1.0), initial_value=torch.tensor([1.2], dtype=DTYPE))
            log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), initial_value=torch.tensor([-0.5], dtype=DTYPE))
            kernel = ChangePoint(Matern32(Exp("lengthscale", log_lengthscale), 1.0), Linear(0.2, 1.0) + 0.1,
                                 location, Exp("steepness", log_steepness))
        function = ExactGP("f", inputs, kernel, noise_variance=0.05, inducing_points=inducing, method=method)
        ObservedParameter("y", function, y)
    return model, function


@pytest.mark.parametrize("kernel_name", ["spectral_mixture", "change_point"])
def test_exact_gp_gradients_match_finite_differences(kernel_name):
    model, _ = _model(kernel_name)
    _finite_difference_check(model)
    _assert_batched_equals_loop(model)


@pytest.mark.parametrize("kernel_name", ["spectral_mixture", "change_point"])
def test_learnable_inputs_with_fitc_match_finite_differences(kernel_name):
    x, _ = _series()
    model, _ = _model(kernel_name, learnable_inputs=True, inducing=x[::5].clone(), method="fitc")
    _finite_difference_check(model)


def test_toeplitz_gp_with_spectral_mixture_matches_exact_gp():
    x, y = _series(40)
    kernel = SpectralMixture(torch.tensor([1.0, 0.3], dtype=DTYPE), torch.tensor([0.45, 0.0], dtype=DTYPE),
                             torch.tensor([1.5, 0.4], dtype=DTYPE))
    with Model():
        toeplitz = ToeplitzGP("t", Data("x", x), kernel, noise_variance=0.05)
        exact = ExactGP("e", Data("x2", x), kernel, noise_variance=0.05)
    torch.testing.assert_close(toeplitz.log_pdf(y), exact.log_pdf(y), atol=1e-8, rtol=0)
