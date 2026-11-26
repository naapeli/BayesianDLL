"""Hilbert-space approximate latent GPs (LatentGP with n_basis) and kernel spectral densities."""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP._process import _gp_predictive as gp_predictive
from BayesianDLL.GP import (
    RBF,
    Constant,
    LatentGP,
    LinearMean,
    Matern,
    Matern32,
    Matern52,
    Periodic,
    SpectralMixture,
)

DTYPE = torch.float64
C = 4


def _name(name):
    return SimpleNamespace(name=name)


# ----------------------------------------------------------------------------
# Spectral densities
# ----------------------------------------------------------------------------


_DENSITY_KERNELS = {
    "rbf": lambda: RBF(0.7, 1.3),
    "matern32": lambda: Matern32(0.6, 1.2),
    "matern52": lambda: Matern52(0.5, 0.9),
    "matern_fractional": lambda: Matern(0.6, 1.2, nu=2.2),
    "spectral_mixture": lambda: SpectralMixture(torch.tensor([0.7, 0.4], dtype=DTYPE), torch.tensor([0.3, 1.2], dtype=DTYPE),
                                                torch.tensor([0.8, 0.5], dtype=DTYPE)),
    "scaled_sum": lambda: Constant(2.0) * RBF(0.7, 1.0) + Matern52(0.9, 0.3) * 0.5,
}


@pytest.mark.parametrize("name", list(_DENSITY_KERNELS))
def test_spectral_density_integrates_to_the_kernel(name):
    kernel = _DENSITY_KERNELS[name]()
    omega = torch.linspace(-80.0, 80.0, 160001, dtype=DTYPE)
    tau = torch.tensor([0.0, 0.3, 1.1], dtype=DTYPE)
    density = kernel.spectral_density(omega)
    covariance = (density * torch.cos(omega * tau[:, None])).sum(-1) * (omega[1] - omega[0]) / (2 * math.pi)
    expected = kernel(tau, torch.zeros(1, dtype=DTYPE))[:, 0]
    torch.testing.assert_close(covariance, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("kind", ["rbf", "matern52", "spectral_mixture"])
def test_spectral_density_in_two_dimensions_integrates_to_the_kernel(kind):
    lengthscale = torch.tensor([0.7, 1.1], dtype=DTYPE)
    kernel = {
        "rbf": lambda: RBF(lengthscale, 1.3),
        "matern52": lambda: Matern52(lengthscale, 1.3),
        "spectral_mixture": lambda: SpectralMixture(torch.tensor([0.9], dtype=DTYPE), torch.tensor([[0.2, 0.1]], dtype=DTYPE),
                                                    lengthscale.reshape(1, 2)),
    }[kind]()
    axis = torch.linspace(-40.0, 40.0, 1601, dtype=DTYPE)
    grid = torch.cartesian_prod(axis, axis)
    tau = torch.tensor([[0.4, -0.2]], dtype=DTYPE)
    density = kernel.spectral_density(grid)
    covariance = (density * torch.cos(grid @ tau[0])).sum() * (axis[1] - axis[0]) ** 2 / (2 * math.pi) ** 2
    torch.testing.assert_close(covariance, kernel(tau, torch.zeros(1, 2, dtype=DTYPE))[0, 0], atol=2e-3, rtol=0)


_DERIVATIVE_CASES = {
    "rbf": (lambda: RBF(_name("l"), _name("v")), {"l": [0.7], "v": [1.3]}),
    "rbf_ard": (lambda: RBF(_name("l"), _name("v")), {"l": [0.7, 1.4], "v": [1.3]}),
    "matern32_ard": (lambda: Matern32(_name("l"), _name("v")), {"l": [0.6, 0.9], "v": [1.2]}),
    "matern_fractional": (lambda: Matern(_name("l"), _name("v"), nu=0.8), {"l": [0.6], "v": [1.2]}),
    "spectral_mixture": (lambda: SpectralMixture(_name("w"), _name("mu"), _name("l")),
                         {"w": [0.7, 0.4], "mu": [[0.3, 0.1], [1.2, 0.6]], "l": [0.8, 0.5]}),
    "spectral_mixture_per_feature": (lambda: SpectralMixture(_name("w"), _name("mu"), _name("l")),
                                     {"w": [0.7, 0.4], "mu": [0.3, 1.2], "l": [[0.8, 1.1], [0.5, 0.6]]}),
    "scaled_sum": (lambda: Constant(_name("c")) * RBF(_name("l"), 1.0) + Matern52(_name("l"), _name("v")),
                   {"c": [2.0], "l": [0.7], "v": [0.4]}),
}


@pytest.mark.parametrize("case", list(_DERIVATIVE_CASES))
def test_spectral_density_derivatives_match_finite_differences(case):
    factory, raw = _DERIVATIVE_CASES[case]
    kernel = factory()
    values = {key: torch.tensor(value, dtype=DTYPE) for key, value in raw.items()}
    omega = torch.tensor([[0.0, 0.5], [1.3, -0.7], [2.5, 3.0], [0.2, 4.1]], dtype=DTYPE)
    batched = {key: value.unsqueeze(0) for key, value in values.items()}
    analytic = kernel.batch_spectral_density_derivatives(omega, batched)
    step = 1e-6
    for key, value in values.items():
        numerical = torch.zeros(*value.shape, omega.shape[0], dtype=DTYPE)
        for index in range(value.numel()):
            plus = {k: v.clone() for k, v in values.items()}
            minus = {k: v.clone() for k, v in values.items()}
            plus[key].reshape(-1)[index] += step
            minus[key].reshape(-1)[index] -= step
            difference = kernel.spectral_density(omega, plus) - kernel.spectral_density(omega, minus)
            numerical.reshape(-1, omega.shape[0])[index] = difference / (2 * step)
        torch.testing.assert_close(analytic[key][0], numerical, rtol=1e-6, atol=1e-9, msg=f"derivative of {key}")


def test_batched_spectral_density_matches_per_element_calls():
    kernel = SpectralMixture(_name("w"), _name("mu"), _name("l")) + RBF(_name("ls"), 0.5)
    generator = torch.Generator().manual_seed(0)
    values = {
        "w": 0.5 + torch.rand(3, 2, generator=generator, dtype=DTYPE),
        "mu": torch.rand(3, 2, generator=generator, dtype=DTYPE),
        "l": 0.5 + torch.rand(3, 2, 2, generator=generator, dtype=DTYPE),
        "ls": 0.5 + torch.rand(3, 2, generator=generator, dtype=DTYPE),
    }
    omega = torch.rand(5, 2, generator=generator, dtype=DTYPE) * 3
    density = kernel.batch_spectral_density(omega, values)
    derivatives = kernel.batch_spectral_density_derivatives(omega, values)
    for b in range(3):
        single = {key: value[b] for key, value in values.items()}
        torch.testing.assert_close(density[b], kernel.spectral_density(omega, single))
        for key, derivative in kernel.batch_spectral_density_derivatives(omega, {k: v[b:b + 1] for k, v in values.items()}).items():
            torch.testing.assert_close(derivatives[key][b], derivative[0])


@pytest.mark.parametrize("kernel", [Periodic(1.0), RBF(1.0) * Matern52(1.0), RBF(1.0) + 0.5])
def test_kernels_without_a_closed_form_spectral_density_raise(kernel):
    with pytest.raises(NotImplementedError):
        kernel.spectral_density(torch.ones(3, 1, dtype=DTYPE))


# ----------------------------------------------------------------------------
# Basis and approximation quality
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_factory", [lambda: RBF(0.5, 1.2), lambda: Matern52(0.6, 0.8)])
def test_basis_expansion_approximates_the_kernel(kernel_factory):
    x = torch.linspace(-2.0, 3.0, 40, dtype=DTYPE)
    with Model():
        f = LatentGP("f", Data("x", x), kernel_factory(), n_basis=80, boundary_factor=2.0)
    basis = f._basis(x.reshape(1, -1, 1))[0]
    density = f.gp_kernel.spectral_density(f._basis_frequencies)
    approximation = basis @ torch.diag(density) @ basis.T
    assert (approximation - f.gp_kernel(x, x)).abs().max() < 1e-2


def test_basis_expansion_in_two_dimensions_approximates_the_kernel():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(30, 2, generator=generator, dtype=DTYPE) * torch.tensor([2.0, 4.0], dtype=DTYPE)
    kernel = RBF(torch.tensor([0.6, 1.2], dtype=DTYPE), 1.1)
    with Model():
        f = LatentGP("f", Data("x", x), kernel, n_basis=(24, 20), boundary_factor=3.0)
    assert f.gp_latent.constrained_value.shape == (480,)
    basis = f._basis(x.unsqueeze(0))[0]
    approximation = basis @ torch.diag(kernel.spectral_density(f._basis_frequencies)) @ basis.T
    assert (approximation - kernel(x, x)).abs().max() < 1e-3


# ----------------------------------------------------------------------------
# The latent node: values and gradients
# ----------------------------------------------------------------------------


def _series(n=25, d=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=generator, dtype=DTYPE) * 3.0 - 1.0
    y = torch.sin(2 * x[:, 0]) + (x[:, 1] if d > 1 else 0.0) + 0.1 * torch.randn(n, generator=generator, dtype=DTYPE)
    return (x[:, 0] if d == 1 else x), y


def _kernel(kind, lengthscale, variance):
    if kind == "rbf":
        return RBF(lengthscale, variance)
    if kind == "matern32":
        return Matern32(lengthscale, variance)
    if kind == "spectral_mixture":
        return SpectralMixture(torch.tensor([0.6, 0.3], dtype=DTYPE) * variance, torch.tensor([0.2, 0.7], dtype=DTYPE),
                               lengthscale.expand(2) if hasattr(lengthscale, "expand") else lengthscale)
    return Constant(variance) * RBF(lengthscale, 1.0)


def _model(kind="rbf", d=1, learnable_inputs=False, likelihood="normal", n_basis=12, mean="linear", centred=False):
    x, y = _series(d=d)
    with Model() as model:
        if learnable_inputs:
            inputs = RandomParameter("x", Normal(x, 0.2), shape=tuple(x.shape), initial_value=x + 0.03)
        else:
            inputs = Data("x", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=d,
                                          initial_value=torch.linspace(-0.4, 0.2, d, dtype=DTYPE))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.3], dtype=DTYPE))
        lengthscale, variance = Exp("lengthscale", log_lengthscale), Exp("variance", log_variance)
        if kind == "spectral_mixture":
            kernel = SpectralMixture(Exp("weights", RandomParameter("log_weights", Normal(0.0, 1.0), shape=2,
                                                                     initial_value=torch.tensor([0.1, -0.6], dtype=DTYPE))),
                                     RandomParameter("frequencies", Normal(0.3, 0.3), shape=2,
                                                     initial_value=torch.tensor([0.2, 0.6], dtype=DTYPE)),
                                     Exp("lengthscales", RandomParameter("log_lengthscales", Normal(0.0, 1.0), shape=2,
                                                                          initial_value=torch.tensor([0.1, -0.2], dtype=DTYPE))))
        else:
            kernel = _kernel(kind, lengthscale, variance)
        if mean == "linear":
            coefficients = RandomParameter("coefficients", Normal(0.0, 1.0), shape=d, initial_value=torch.full((d,), 0.3, dtype=DTYPE))
            mean_function = LinearMean(coefficients, RandomParameter("intercept", Normal(0.0, 1.0)))
        else:
            mean_function = None
        f = LatentGP("f", inputs, kernel, mean=mean_function, n_basis=n_basis, boundary_factor=1.6, centred=centred)
        f.gp_latent.set_unconstrained_value(torch.randn(f.gp_latent.unconstrained_value.shape,
                                                         generator=torch.Generator().manual_seed(4), dtype=DTYPE))
        if likelihood == "normal":
            ObservedParameter("y", Normal(f, 0.3), y)
        else:
            ObservedParameter("y", Bernoulli(logits=f), (y > 0).to(DTYPE))
    if centred:
        # Weights of a typical size under their prior (standard deviations sqrt(S)).
        sd = model.deterministic_params["f_basis_sd"].constrained_value
        f.gp_latent.set_unconstrained_value(sd * f.gp_latent.constrained_value)
    return model, f, x


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


def test_forward_matches_the_basis_expansion():
    model, f, x = _model("rbf", mean=None)
    lengthscale = model.deterministic_params["lengthscale"].constrained_value
    variance = model.deterministic_params["variance"].constrained_value
    boundary = 1.6 * (x.max() - x.min()) / 2
    centre = (x.max() + x.min()) / 2
    j = torch.arange(1, 13, dtype=DTYPE)
    omega = torch.pi * j / (2 * boundary)
    basis = torch.sin(omega * (x[:, None] - centre + boundary)) / boundary.sqrt()
    density = variance * math.sqrt(2 * math.pi) * lengthscale * torch.exp(-0.5 * (lengthscale * omega) ** 2)
    expected = basis @ (density.sqrt() * f.gp_latent.constrained_value)
    torch.testing.assert_close(f.constrained_value, expected)


@pytest.mark.parametrize("likelihood", ["normal", "bernoulli"])
@pytest.mark.parametrize("kind", ["rbf", "matern32", "spectral_mixture", "scaled_rbf"])
def test_gradients_match_finite_differences(kind, likelihood):
    model, _, _ = _model(kind, likelihood=likelihood)
    _finite_difference_check(model)


@pytest.mark.parametrize("likelihood", ["normal", "bernoulli"])
@pytest.mark.parametrize("kind", ["matern32", "spectral_mixture", "scaled_rbf"])
def test_centred_gradients_match_finite_differences(kind, likelihood):
    model, f, _ = _model(kind, likelihood=likelihood, centred=True)
    assert f.gp_latent.name == "f_weights" and "f_basis_sd" in model.deterministic_params
    _finite_difference_check(model)


def test_centred_with_learnable_inputs_matches_finite_differences():
    model, _, _ = _model("rbf", d=2, learnable_inputs=True, n_basis=(6, 5), centred=True)
    _finite_difference_check(model)


def test_centred_is_a_reparameterisation_of_the_non_centred_form():
    # With beta = sqrt(S) z both forms give the same function, and the log densities
    # differ by the log Jacobian sum(log sqrt(S)) of z -> beta.
    model, f, _ = _model("matern32")
    centred_model, centred_f, _ = _model("matern32", centred=True)
    root = centred_model.deterministic_params["f_basis_sd"].constrained_value
    centred_f.gp_latent.set_unconstrained_value(root * f.gp_latent.constrained_value)
    torch.testing.assert_close(centred_f.constrained_value, f.constrained_value)
    torch.testing.assert_close(centred_model.model_log_prob(), model.model_log_prob() - root.log().sum())


@pytest.mark.parametrize("kind", ["rbf", "matern32"])
def test_two_dimensional_ard_with_learnable_inputs_matches_finite_differences(kind):
    model, _, _ = _model(kind, d=2, learnable_inputs=True, n_basis=(6, 5))
    _finite_difference_check(model)


def test_vjp_matches_autograd_oracle():
    model, f, _ = _model("matern32", d=2, learnable_inputs=True, n_basis=(5, 4))
    values = [value.detach().clone() for value in f.input_values()]
    names = [node.name for node in f.inputs]
    cotangent = torch.randn(f.constrained_value.shape, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    hand = f.vjp(cotangent, values)
    leaves = {name: value.clone().requires_grad_(True) for name, value in zip(names, values)}
    x, z, lengthscale, variance = leaves["x"], leaves["f_white"], leaves["lengthscale"], leaves["variance"]
    centre, boundary = f._basis_centre, f._basis_boundary
    basis = None
    omegas = []
    for d, m in enumerate((5, 4)):
        omega = torch.pi * torch.arange(1, m + 1, dtype=DTYPE) / (2 * boundary[d])
        omegas.append(omega)
        values_d = torch.sin(omega * (x[:, d:d + 1] - centre[d] + boundary[d])) / boundary[d].sqrt()
        basis = values_d if basis is None else (basis[:, :, None] * values_d[:, None, :]).reshape(x.shape[0], -1)
    grid = torch.cartesian_prod(*omegas)
    nu, dims = 1.5, 2
    constant = dims * math.log(2) + dims / 2 * math.log(math.pi) + math.lgamma(nu + dims / 2) - math.lgamma(nu) + nu * math.log(2 * nu)
    density = variance * math.exp(constant) * lengthscale.prod() * (2 * nu + ((grid * lengthscale) ** 2).sum(-1)) ** -(nu + dims / 2)
    mean = x @ leaves["coefficients"] + leaves["intercept"]
    value = mean + basis @ (density.sqrt() * z)
    torch.testing.assert_close(value.detach(), f.constrained_value)
    oracle = torch.autograd.grad((value * cotangent).sum(), list(leaves.values()))
    for name, expected in zip(leaves, oracle):
        torch.testing.assert_close(torch.as_tensor(hand[name]).reshape(expected.shape), expected, msg=f"VJP of {name}")


@pytest.mark.parametrize("kind", ["rbf", "spectral_mixture"])
@pytest.mark.parametrize("learnable_inputs", [False, True])
@pytest.mark.parametrize("centred", [False, True])
def test_batched_matches_chain_by_chain(kind, learnable_inputs, centred):
    model, _, _ = _model(kind, learnable_inputs=learnable_inputs, centred=centred)
    generator = torch.Generator().manual_seed(0)
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        values[name] = u.unsqueeze(0) + 0.2 * torch.randn((C, *u.shape), generator=generator, dtype=u.dtype)
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


def test_follows_model_to_float32():
    model, f, _ = _model("rbf", d=2, n_basis=(5, 4))
    reference = model.joint_grad_log_prob()
    model.to(dtype=torch.float32)
    assert f.constrained_value.dtype == torch.float32
    for name, gradient in model.joint_grad_log_prob().items():
        assert gradient.dtype == torch.float32
        torch.testing.assert_close(gradient.double(), reference[name], rtol=1e-3, atol=1e-3)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------


def _trace(model, n_draws=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    draws = {name: [] for name in [*model.params, *model.deterministic_params]}
    for _ in range(n_draws):
        for parameter in model.params.values():
            u = parameter.unconstrained_value
            parameter.set_unconstrained_value(u + 0.1 * torch.randn(u.shape, generator=generator, dtype=u.dtype))
        for name in draws:
            node = model.params.get(name) or model.deterministic_params[name]
            draws[name].append(node.constrained_value.clone())
    return {name: torch.stack(values).unsqueeze(0) for name, values in draws.items()}


@pytest.mark.parametrize("centred", [False, True])
def test_prediction_evaluates_the_basis_at_new_inputs(centred):
    model, f, x = _model("rbf", centred=centred)
    trace = _trace(model)
    at_training = gp_predictive(f, trace, x, n_samples=2)
    torch.testing.assert_close(at_training[:, 0], trace["f"][0])
    torch.testing.assert_close(at_training[:, 1], trace["f"][0])
    grid = torch.linspace(-1.0, 2.0, 11, dtype=DTYPE)
    predictions = gp_predictive(f, trace, grid, n_samples=1, observation_noise=0.0)
    model.data["x"].set_value(grid)
    result = model.posterior_predicative(trace, samples_per_step=2, progress_bar=False)
    torch.testing.assert_close(result["f"].reshape(3, 11), predictions[:, 0])
    assert result["y"].shape == (3, 2, 11)


def test_validates_arguments():
    x = torch.linspace(0.0, 1.0, 5, dtype=DTYPE)
    with Model():
        inputs = Data("x", x)
        with pytest.raises(TypeError, match="spectral density"):
            LatentGP("a", inputs, Periodic(0.5), n_basis=10)
        with pytest.raises(ValueError, match="not both"):
            LatentGP("b", inputs, RBF(0.5), n_basis=10, inducing_points=x[:2])
        with pytest.raises(ValueError, match="needs n_basis"):
            LatentGP("c", inputs, RBF(0.5), boundary_factor=2.0)
        with pytest.raises(ValueError, match="one per input feature"):
            LatentGP("d", inputs, RBF(0.5), n_basis=(4, 5))
        with pytest.raises(ValueError, match="at least 1"):
            LatentGP("e", inputs, RBF(0.5), n_basis=4, boundary_factor=0.5)
        with pytest.raises(ValueError, match="one value per basis function"):
            LatentGP("f", inputs, RBF(0.5), n_basis=4, latent=RandomParameter("z", Normal(0.0, 1.0), shape=5))
        f = LatentGP("g", inputs, RBF(0.5), n_basis=7)
        assert isinstance(f, LatentGP) and f.gp_latent.constrained_value.shape == (7,)
        with pytest.raises(ValueError, match="centred needs n_basis"):
            LatentGP("h", inputs, RBF(0.5), centred=True)
        with pytest.raises(ValueError, match="latent cannot be given"):
            LatentGP("i", inputs, RBF(0.5), n_basis=4, centred=True, latent=RandomParameter("w", Normal(0.0, 1.0), shape=4))
        # A kernel without model parameters gives the weights fixed prior standard deviations.
        fixed = LatentGP("j", inputs, RBF(0.5), n_basis=7, centred=True)
        root = RBF(0.5).spectral_density(fixed._basis_frequencies).sqrt()
        torch.testing.assert_close(fixed.gp_latent.distribution.sigma, root)
