import pytest
import torch
from types import SimpleNamespace

from BayesianDLL import (
    Data,
    DeterministicParameter,
    Guide,
    Model,
    ObservedParameter,
    RandomParameter,
    VariationalParameter,
    condition,
    plate,
)
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Exponential, Normal, Uniform
from BayesianDLL.GP import (
    Constant,
    ExactGP,
    Linear,
    LatentGP,
    Matern,
    Matern32,
    Periodic,
    RBF,
    WhiteNoise,
    gp_predictive,
    exact_gp_predictive,
    Matern52,
    RationalQuadratic,
    SparseGP,
    sparse_gp_predictive,
    stable_cholesky,
)
from BayesianDLL.Variational import elbo


def build_gp_model(n=4):
    x = torch.linspace(0.0, 1.0, n)
    y = torch.sin(6 * x)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Exponential(1.0))
        variance = RandomParameter("variance", Exponential(1.0))
        latent = LatentGP("function", inputs, RBF(lengthscale, variance))
        ObservedParameter("observations", Normal(latent, 0.1 ** 0.5), y)
    return model, inputs, latent


def build_exact_gp_model(n=4):
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    y = torch.sin(6 * x)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Uniform(0.2, 1.0))
        variance = RandomParameter("variance", Uniform(0.25, 2.0))
        noise = RandomParameter("noise", Uniform(0.01, 0.2))
        function = ExactGP(
            "function",
            inputs,
            RBF(lengthscale, variance),
            noise_variance=noise,
        )
        ObservedParameter("observations", function, y)
    return model, inputs, function, y


def test_kernel_composition_and_matern_shapes():
    x = torch.linspace(0.0, 1.0, 5)
    kernel = RBF(0.4, 1.2) + Matern32(0.7, 0.3)
    covariance = kernel(x, x)
    assert covariance.shape == (5, 5)
    torch.testing.assert_close(covariance, covariance.transpose(-2, -1))
    assert torch.linalg.eigvalsh(covariance).min() > 0
    noisy = WhiteNoise(0.2)(x, x)
    torch.testing.assert_close(noisy, 0.2 * torch.eye(5))


def test_periodic_kernel_repeats_at_its_period():
    x1 = torch.tensor([[0.0], [0.4], [1.0]])
    x2 = torch.tensor([[0.2], [0.8]])
    kernel = Periodic(lengthscale=0.7, variance=1.3, period=2.0)

    torch.testing.assert_close(kernel(x1, x2), kernel(x1 + 2.0, x2))
    torch.testing.assert_close(kernel(x1, x2), kernel(x1, x2 + 2.0))
    torch.testing.assert_close(torch.diag(kernel(x1, x1)), torch.full((3,), 1.3))


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(SimpleNamespace(name="lengthscale"), 1.3),
        Matern(SimpleNamespace(name="lengthscale"), 1.3, nu=0.8),
        Periodic(SimpleNamespace(name="lengthscale"), 1.3, 1.7),
    ],
)
def test_kernel_lengthscale_derivative_matches_finite_difference(kernel):
    x = torch.tensor([[0.0], [0.4], [1.0]])
    parameter_values = {"lengthscale": torch.tensor([0.9])}
    analytic = kernel.derivatives(x, x, parameter_values=parameter_values)["lengthscale"][0]
    step = 1e-5
    plus = kernel(x, x, parameter_values={"lengthscale": torch.tensor([0.9 + step])})
    minus = kernel(x, x, parameter_values={"lengthscale": torch.tensor([0.9 - step])})
    finite_difference = (plus - minus) / (2 * step)
    torch.testing.assert_close(analytic, finite_difference, atol=2e-5, rtol=2e-5)


def _finite_difference_kernel_derivative(kernel, parameter_values, name, x1, x2, step=1e-5):
    value = parameter_values[name]
    derivatives = []
    for index in range(value.numel()):
        plus_values = {key: tensor.clone() for key, tensor in parameter_values.items()}
        minus_values = {key: tensor.clone() for key, tensor in parameter_values.items()}
        plus_values[name].reshape(-1)[index] += step
        minus_values[name].reshape(-1)[index] -= step
        plus = kernel(x1, x2, parameter_values=plus_values)
        minus = kernel(x1, x2, parameter_values=minus_values)
        derivatives.append((plus - minus) / (2 * step))
    return torch.stack(derivatives)


@pytest.mark.parametrize(
    "kernel,parameter_values,name",
    [
        (Constant(SimpleNamespace(name="variance")), {"variance": torch.tensor([1.3])}, "variance"),
        (WhiteNoise(SimpleNamespace(name="variance")), {"variance": torch.tensor([0.7])}, "variance"),
        (Linear(SimpleNamespace(name="variance")), {"variance": torch.tensor([1.1])}, "variance"),
        (Linear(1.1, SimpleNamespace(name="offset")), {"offset": torch.tensor([0.2, -0.1])}, "offset"),
        (
            RBF(SimpleNamespace(name="lengthscale"), SimpleNamespace(name="variance")),
            {"lengthscale": torch.tensor([0.8]), "variance": torch.tensor([1.2])},
            "variance",
        ),
        (
            Matern(SimpleNamespace(name="lengthscale"), SimpleNamespace(name="variance"), nu=0.8),
            {"lengthscale": torch.tensor([0.8]), "variance": torch.tensor([1.2])},
            "variance",
        ),
        (
            Periodic(
                SimpleNamespace(name="lengthscale"),
                SimpleNamespace(name="variance"),
                1.7,
            ),
            {"lengthscale": torch.tensor([0.8]), "variance": torch.tensor([1.2])},
            "variance",
        ),
        (
            Periodic(
                0.8,
                SimpleNamespace(name="variance"),
                SimpleNamespace(name="period"),
            ),
            {"variance": torch.tensor([1.2]), "period": torch.tensor([1.7])},
            "period",
        ),
    ],
)
def test_remaining_kernel_derivatives_match_finite_difference(kernel, parameter_values, name):
    x1 = torch.tensor([[0.0, 0.2], [0.4, -0.1], [1.0, 0.8]])
    x2 = torch.tensor([[0.1, 0.0], [0.9, 0.5]])
    parameter_values = {key: value.to(dtype=x1.dtype) for key, value in parameter_values.items()}
    analytic = kernel.derivatives(x1, x2, parameter_values=parameter_values)[name]
    finite_difference = _finite_difference_kernel_derivative(kernel, parameter_values, name, x1, x2)
    torch.testing.assert_close(analytic, finite_difference, atol=3e-5, rtol=3e-5)


def test_ard_lengthscale_derivative_matches_finite_difference():
    lengthscale = SimpleNamespace(name="lengthscale")
    kernel = RBF(lengthscale, 1.2)
    x1 = torch.tensor([[0.0, 0.2], [0.4, -0.1], [1.0, 0.8]])
    values = {"lengthscale": torch.tensor([0.8, 1.1])}
    analytic = kernel.derivatives(x1, x1, parameter_values=values)["lengthscale"]
    finite_difference = _finite_difference_kernel_derivative(kernel, values, "lengthscale", x1, x1)
    torch.testing.assert_close(analytic, finite_difference, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("operation", ["sum", "product"])
def test_composite_kernel_derivatives_match_finite_difference(operation):
    lengthscale = SimpleNamespace(name="lengthscale")
    variance = SimpleNamespace(name="variance")
    base = RBF(lengthscale, 1.1)
    other = Matern32(0.7, variance)
    kernel = base + other if operation == "sum" else base * other
    x = torch.tensor([[0.0], [0.3], [1.0]])
    values = {"lengthscale": torch.tensor([0.8]), "variance": torch.tensor([0.9])}
    for name in values:
        analytic = kernel.derivatives(x, x, parameter_values=values)[name]
        finite_difference = _finite_difference_kernel_derivative(kernel, values, name, x, x)
        torch.testing.assert_close(analytic, finite_difference, atol=3e-5, rtol=3e-5)


def test_gp_adds_whitened_latent_and_has_finite_gradients():
    model, _, latent = build_gp_model()
    assert isinstance(latent, LatentGP)
    assert isinstance(latent, DeterministicParameter)
    assert "function_white" in model.params
    assert latent.constrained_value.shape == (4,)
    gradients = model.joint_grad_log_prob()
    assert set(gradients) == {"function_white", "lengthscale", "variance"}
    assert all(torch.isfinite(value).all() for value in gradients.values())


def test_condition_can_fix_latent_gp_hyperparameters():
    x = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    y = torch.sin(2 * torch.pi * x)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(-1.0, 0.5 ** 0.5))
        log_variance = RandomParameter("log_variance", Normal(0.0, 0.5 ** 0.5))
        log_noise = RandomParameter("log_noise", Normal(-2.5, 0.5 ** 0.5))
        lengthscale = Exp("lengthscale", log_lengthscale)
        variance = Exp("variance", log_variance)
        noise = Exp("noise", log_noise)
        LatentGP("function", inputs, RBF(lengthscale, variance))
        ObservedParameter("observations", Normal(model.deterministic_params["function"], noise), y)

    fixed = condition(
        model,
        {
            "log_lengthscale": torch.tensor([-1.0], dtype=x.dtype),
            "log_variance": torch.tensor([0.0], dtype=x.dtype),
            "log_noise": torch.tensor([-2.5], dtype=x.dtype),
        },
    )
    assert set(fixed.params) == {"function_white"}
    assert set(fixed.observed_params) == {"log_lengthscale", "log_variance", "log_noise", "observations"}


def test_gp_accepts_an_explicit_latent_variable():
    x = torch.linspace(0.0, 1.0, 4)
    with Model() as model:
        inputs = Data("inputs", x)
        z = RandomParameter("z", Normal(0.0, 1.0), shape=x.numel())
        function = LatentGP("function", inputs, RBF(0.5, 1.0), latent=z)

    assert model.params["z"] is z
    assert "function_white" not in model.params
    assert function.gp_latent is z


def test_gp_cholesky_derivative_matches_finite_difference():
    model, _, latent = build_gp_model()
    parameter = model.params["lengthscale"]
    original = parameter.constrained_value.clone()
    step = 1e-5
    analytic = latent.derivative("lengthscale")
    parameter.set_constrained_value(original + step)
    plus = latent.constrained_value.clone()
    parameter.set_constrained_value(original - step)
    minus = latent.constrained_value.clone()
    parameter.set_constrained_value(original)
    finite_difference = (plus - minus) / (2 * step)
    torch.testing.assert_close(analytic.reshape(-1), finite_difference, atol=3e-5, rtol=3e-5)


def test_gp_variance_gradient_matches_full_model_finite_difference():
    x = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    y = torch.sin(6 * x)
    with Model() as model:
        inputs = Data("inputs", x)
        z = RandomParameter(
            "z",
            Normal(0.0, 1.0),
            initial_value=torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64),
            shape=4,
        )
        variance = RandomParameter(
            "variance",
            Exponential(1.0),
            initial_value=torch.tensor([1.2], dtype=torch.float64),
        )
        function = LatentGP("function", inputs, RBF(0.5, variance), latent=z)
        # The literal likelihood variance must not be confused with the GP
        # hyperparameter named "variance".
        ObservedParameter("observations", Normal(function, 0.1 ** 0.5), y)

    unconstrained = variance.unconstrained_value.clone()
    step = 1e-5
    analytic = model.joint_grad_log_prob(["variance"])["variance"]
    variance.set_unconstrained_value(unconstrained + step)
    plus = model.model_log_prob()
    variance.set_unconstrained_value(unconstrained - step)
    minus = model.model_log_prob()
    variance.set_unconstrained_value(unconstrained)
    finite_difference = (plus - minus) / (2 * step)
    torch.testing.assert_close(analytic.reshape(()), finite_difference, atol=3e-4, rtol=3e-4)


@pytest.mark.integration
def test_gp_mcmc_runs_with_latent_function():
    model, _, _ = build_gp_model(n=3)
    trace = model.sample(6, 6, n_chains=1, progress_bar=False, check_convergence=False, seed=0)
    assert trace["function_white"].shape == (1, 6, 3)
    assert torch.isfinite(trace["function"] if "function" in trace else trace.deterministic_trace["function"]).all()


@pytest.mark.integration
def test_gp_mcmc_promotes_float32_inputs_to_inference_dtype():
    x = torch.linspace(0.0, 1.0, 3, dtype=torch.float32)
    y = torch.sin(6 * x)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Exponential(1.0))
        variance = RandomParameter("variance", Exponential(1.0))
        function = LatentGP("function", inputs, RBF(lengthscale, variance))
        ObservedParameter("observations", Normal(function, 0.1 ** 0.5), y)
    # A two-step warmup leaves the step size untuned (every post-warmup step
    # diverged); ten warmup steps are still cheap.
    trace = model.sample(2, 10, n_chains=1, progress_bar=False, check_convergence=False, seed=0)
    assert trace["function"].dtype == trace["function_white"].dtype


def test_gp_predictive_samples_from_trace():
    model, inputs, latent = build_gp_model(n=3)
    trace = model.sample(4, 4, n_chains=1, progress_bar=False, check_convergence=False, seed=0)
    new_inputs = torch.linspace(-0.2, 1.2, 5)
    samples = gp_predictive(latent, trace, new_inputs, n_samples=3)
    assert samples.shape == (4, 3, 5)
    assert torch.isfinite(samples).all()


def test_exact_gp_marginal_likelihood_has_finite_gradients():
    model, _, _, _ = build_exact_gp_model()
    gradients = model.joint_grad_log_prob()
    assert set(gradients) == {"lengthscale", "variance", "noise"}
    assert all(torch.isfinite(value).all() for value in gradients.values())


def test_exact_gp_marginal_likelihood_matches_multivariate_normal():
    model, inputs, function, y = build_exact_gp_model(n=3)
    actual = function.log_pdf(y)
    covariance = function.kernel(inputs.value, inputs.value)
    covariance = covariance + function.noise_variance.constrained_value.reshape(())[None, None] * torch.eye(3, dtype=torch.float64)
    expected = torch.distributions.MultivariateNormal(
        torch.zeros(3, dtype=torch.float64),
        covariance_matrix=covariance + function.jitter * torch.eye(3, dtype=torch.float64),
    ).log_prob(y)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.integration
def test_exact_gp_mcmc_and_predictive_samples():
    model, inputs, function, y = build_exact_gp_model(n=3)
    trace = model.sample(
        5,
        5,
        n_chains=1,
        progress_bar=False,
        check_convergence=False,
        start_point_variance=0.2,
        max_depth=4,
        delta=0.8,
    )
    samples = exact_gp_predictive(
        function,
        trace,
        y,
        torch.linspace(-0.2, 1.2, 4, dtype=torch.float64),
        n_samples=2,
    )
    assert samples.shape == (5, 2, 4)
    assert torch.isfinite(samples).all()


def test_whitened_gp_is_usable_in_variational_inference():
    model, _, _ = build_gp_model(n=3)
    with Guide() as guide:
        RandomParameter(
            "function_white",
            Normal(
                VariationalParameter("loc", torch.zeros(3)),
                VariationalParameter("std", torch.ones(3), min=1e-3),
            ),
            shape=3,
        )
        RandomParameter(
            "lengthscale",
            Exponential(VariationalParameter("rate", torch.ones(1), min=1e-3)),
        )
        RandomParameter(
            "variance",
            Exponential(VariationalParameter("rate", torch.ones(1), min=1e-3)),
        )
    value, gradients = elbo(model, guide, n_samples=2)
    assert torch.isfinite(value)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())


def _finite_difference_check(model, eps=1e-6):
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
        torch.testing.assert_close(
            gradients[name].reshape(-1), numerical, rtol=1e-5, atol=1e-6, msg=f"gradient of {name}"
        )


@pytest.mark.parametrize("kernel_name", ["rbf", "periodic"])
def test_latent_gp_gradient_includes_scale_dependent_jitter(kernel_name):
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    y = torch.sin(2 * torch.pi * x)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), initial_value=torch.tensor([0.2]))
        # A variance above one makes the jitter depend on the variance.
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([1.5]))
        lengthscale = Exp("lengthscale", log_lengthscale)
        variance = Exp("variance", log_variance)
        if kernel_name == "rbf":
            kernel = RBF(lengthscale, variance)
        else:
            kernel = Periodic(lengthscale, variance, RandomParameter("period", Uniform(0.5, 1.5)))
        function = LatentGP("function", inputs, kernel, jitter=1e-4)
        ObservedParameter("observations", Normal(function, 0.05 ** 0.5), y)
    model.params["function_white"].set_unconstrained_value(torch.linspace(-1.0, 1.0, 8, dtype=torch.float64))
    _finite_difference_check(model)


def test_exact_gp_gradient_includes_scale_dependent_jitter():
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    y = torch.sin(2 * torch.pi * x)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), initial_value=torch.tensor([0.2]))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([1.5]))
        log_noise = RandomParameter("log_noise", Normal(0.0, 1.0), initial_value=torch.tensor([0.3]))
        function = ExactGP(
            "function",
            inputs,
            RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance)),
            noise_variance=Exp("noise", log_noise),
            jitter=1e-4,
        )
        ObservedParameter("observations", function, y)
    _finite_difference_check(model)


# ----------------------------------------------------------------------------
# ARD length-scales
# ----------------------------------------------------------------------------

_ARD_X1 = torch.tensor([[0.0, 0.2, -0.3], [0.4, -0.1, 0.5], [1.0, 0.8, 0.1], [0.3, 0.3, 0.3]], dtype=torch.float64)
_ARD_X2 = torch.tensor([[0.1, 0.0, 0.2], [0.9, 0.5, -0.4]], dtype=torch.float64)
_ARD_LENGTHSCALE = torch.tensor([0.7, 1.3, 0.4], dtype=torch.float64)


def _direct_ard_kernel(kind, x1, x2, lengthscale, variance=1.4, period=1.7):
    """Reference formulas, written element by element."""
    result = torch.zeros(x1.shape[0], x2.shape[0], dtype=x1.dtype)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            delta = x1[i] - x2[j]
            r = torch.sqrt(((delta / lengthscale) ** 2).sum())
            if kind == "rbf":
                value = torch.exp(-0.5 * r ** 2)
            elif kind == "matern12":
                value = torch.exp(-r)
            elif kind == "matern32":
                value = (1 + 3 ** 0.5 * r) * torch.exp(-(3 ** 0.5) * r)
            elif kind == "matern52":
                value = (1 + 5 ** 0.5 * r + 5 / 3 * r ** 2) * torch.exp(-(5 ** 0.5) * r)
            else:
                value = torch.exp(-2 * (torch.sin(torch.pi * delta.abs() / period) ** 2 / lengthscale ** 2).sum())
            result[i, j] = variance * value
    return result


def _ard_kernel(kind, lengthscale):
    if kind == "rbf":
        return RBF(lengthscale, 1.4)
    if kind == "matern12":
        return Matern(lengthscale, 1.4, nu=0.5)
    if kind == "matern32":
        return Matern32(lengthscale, 1.4)
    if kind == "matern52":
        return Matern52(lengthscale, 1.4)
    return Periodic(lengthscale, 1.4, 1.7)


ARD_KINDS = ["rbf", "matern12", "matern32", "matern52", "periodic"]


@pytest.mark.parametrize("kind", ARD_KINDS)
def test_ard_kernel_values_match_direct_formula(kind):
    kernel = _ard_kernel(kind, _ARD_LENGTHSCALE)
    expected = _direct_ard_kernel(kind, _ARD_X1, _ARD_X2, _ARD_LENGTHSCALE)
    torch.testing.assert_close(kernel(_ARD_X1, _ARD_X2), expected)
    torch.testing.assert_close(kernel.diag(_ARD_X1), torch.diagonal(kernel(_ARD_X1, _ARD_X1)))


@pytest.mark.parametrize("kind", ARD_KINDS + ["matern_fractional"])
def test_ard_lengthscale_derivatives_match_finite_differences(kind):
    name = SimpleNamespace(name="lengthscale")
    kernel = Matern(name, 1.4, nu=0.8) if kind == "matern_fractional" else _ard_kernel(kind, name)
    values = {"lengthscale": _ARD_LENGTHSCALE.clone()}
    analytic = kernel.derivatives(_ARD_X1, _ARD_X2, parameter_values=values)["lengthscale"]
    assert analytic.shape == (3, 4, 2)
    finite_difference = _finite_difference_kernel_derivative(kernel, values, "lengthscale", _ARD_X1, _ARD_X2, step=1e-6)
    torch.testing.assert_close(analytic, finite_difference, atol=1e-7, rtol=1e-6)
    # The diagonal derivatives used by the sparse GP agree with the full ones.
    full = kernel.derivatives(_ARD_X1, _ARD_X1, parameter_values=values)["lengthscale"]
    diag = kernel.diag_derivatives(_ARD_X1, parameter_values=values)["lengthscale"]
    torch.testing.assert_close(diag, torch.diagonal(full, dim1=-2, dim2=-1))


@pytest.mark.parametrize("kind", ["rbf", "matern32", "matern52", "periodic"])
def test_ard_lengthscale_derivatives_match_autograd(kind):
    name = SimpleNamespace(name="lengthscale")
    kernel = _ard_kernel(kind, name)
    analytic = kernel.derivatives(_ARD_X1, _ARD_X2, parameter_values={"lengthscale": _ARD_LENGTHSCALE})["lengthscale"]
    jacobian = torch.autograd.functional.jacobian(
        lambda lengthscale: _direct_ard_kernel(kind, _ARD_X1, _ARD_X2, lengthscale), _ARD_LENGTHSCALE
    )
    torch.testing.assert_close(analytic, jacobian.permute(2, 0, 1))


@pytest.mark.parametrize("kind", ["rbf", "matern32", "periodic"])
def test_scalar_lengthscale_on_multidimensional_inputs_has_one_derivative(kind):
    name = SimpleNamespace(name="lengthscale")
    kernel = _ard_kernel(kind, name)
    values = {"lengthscale": torch.tensor([0.8], dtype=torch.float64)}
    analytic = kernel.derivatives(_ARD_X1, _ARD_X2, parameter_values=values)["lengthscale"]
    assert analytic.shape == (1, 4, 2)
    finite_difference = _finite_difference_kernel_derivative(kernel, values, "lengthscale", _ARD_X1, _ARD_X2, step=1e-6)
    torch.testing.assert_close(analytic, finite_difference, atol=1e-7, rtol=1e-6)


def test_ard_lengthscale_with_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="one value per feature"):
        RBF(torch.tensor([0.5, 0.6]))(_ARD_X1, _ARD_X1)
    with pytest.raises(ValueError, match="one value per feature"):
        Periodic(torch.tensor([0.5, 0.6]))(_ARD_X1, _ARD_X1)


def _ard_data(n=10):
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(n, 3, generator=generator, dtype=torch.float64)
    y = torch.sin(4 * x[:, 0]) + 0.5 * x[:, 1] + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    return x, y


def _ard_lengthscale_node(style):
    initial = torch.tensor([-0.4, 0.3, 0.1], dtype=torch.float64)
    if style == "shape":
        return RandomParameter("lengthscale", Exponential(1.0), shape=3, initial_value=initial.exp())
    if style == "plate":
        with plate("features", 3):
            return RandomParameter("lengthscale", Exponential(1.0), initial_value=initial.exp())
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=3, initial_value=initial)
    return Exp("lengthscale", log_lengthscale)


@pytest.mark.parametrize("style", ["shape", "plate", "deterministic"])
@pytest.mark.parametrize("kind", ["rbf", "matern32", "periodic"])
def test_exact_gp_with_ard_lengthscales_matches_finite_differences(style, kind):
    x, y = _ard_data()
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = _ard_lengthscale_node(style)
        variance = RandomParameter("variance", Exponential(1.0), initial_value=torch.tensor([1.3], dtype=torch.float64))
        noise = RandomParameter("noise", Exponential(1.0), initial_value=torch.tensor([0.2], dtype=torch.float64))
        kernel = {"rbf": RBF, "matern32": Matern32}.get(kind, lambda l, v: Periodic(l, v, 1.5))(lengthscale, variance)
        ObservedParameter("observations", ExactGP("function", inputs, kernel, noise_variance=noise), y)
    gradients = model.joint_grad_log_prob()
    for name, parameter in model.params.items():
        assert gradients[name].shape == parameter.unconstrained_value.shape
    _finite_difference_check(model)


def test_exact_gp_ard_gradient_matches_autograd_oracle():
    x, y = _ard_data()
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Exponential(1.0), shape=3, initial_value=torch.tensor([0.6, 1.4, 0.9], dtype=torch.float64))
        function = ExactGP("function", inputs, RBF(lengthscale, 1.2), noise_variance=0.1, jitter=1e-9)
        ObservedParameter("observations", function, y)
    analytic = function.log_pdf_param_grads(y)["lengthscale"]
    assert analytic.shape == (3,)

    def log_likelihood(lengthscale_value):
        difference = (x[:, None, :] - x[None, :, :]) / lengthscale_value
        covariance = 1.2 * torch.exp(-0.5 * difference.square().sum(-1)) + (0.1 + 1e-9) * torch.eye(x.shape[0], dtype=x.dtype)
        return torch.distributions.MultivariateNormal(torch.zeros_like(y), covariance).log_prob(y)

    oracle = torch.autograd.functional.jacobian(log_likelihood, lengthscale.constrained_value.clone())
    torch.testing.assert_close(analytic, oracle)


def test_latent_gp_with_ard_lengthscales_matches_finite_differences():
    x, y = _ard_data(n=6)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = _ard_lengthscale_node("deterministic")
        function = LatentGP("function", inputs, Matern32(lengthscale, 1.1))
        ObservedParameter("observations", Normal(function, 0.3), y)
    model.params["function_white"].set_unconstrained_value(torch.linspace(-1.0, 1.0, 6, dtype=torch.float64))
    _finite_difference_check(model)


@pytest.mark.integration
def test_exact_gp_ard_nuts_assigns_a_long_lengthscale_to_an_irrelevant_input():
    generator = torch.Generator().manual_seed(11)
    x = torch.rand(40, 2, generator=generator, dtype=torch.float64) * 4.0
    # Only the first feature affects the response.
    y = torch.sin(2.0 * x[:, 0]) + 0.1 * torch.randn(40, generator=generator, dtype=torch.float64)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.5), shape=2)
        lengthscale = Exp("lengthscale", log_lengthscale)
        log_noise = RandomParameter("log_noise", Normal(-3.0, 1.0))
        function = ExactGP("function", inputs, RBF(lengthscale, 1.0), noise_variance=Exp("noise", log_noise))
        ObservedParameter("observations", function, y)
    trace = model.sample(150, 150, n_chains=2, progress_bar=False, check_convergence=False, seed=5)
    median = trace["log_lengthscale"].reshape(-1, 2).median(dim=0).values
    # The relevant feature has a short length-scale (the sine period is ~3);
    # the irrelevant one is pushed far beyond the data range.
    assert median[0] < 0.5
    assert median[1] > median[0] + 1.5


# ----------------------------------------------------------------------------
# Sparse (inducing-point) GP
# ----------------------------------------------------------------------------


def _sparse_model(method, inducing=None, n=15, mean_parameter=True):
    generator = torch.Generator().manual_seed(7)
    x = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    y = torch.sin(3 * x[:, 0]) + x[:, 1] + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    inducing = x[:6] if inducing is None else inducing
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([-0.3, 0.4], dtype=torch.float64))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.6], dtype=torch.float64))
        log_noise = RandomParameter("log_noise", Normal(0.0, 1.0), initial_value=torch.tensor([-1.5], dtype=torch.float64))
        period = RandomParameter("period", Uniform(1.0, 3.0), initial_value=torch.tensor([1.8], dtype=torch.float64))
        mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=torch.float64)) if mean_parameter else 0.0
        lengthscale = Exp("lengthscale", log_lengthscale)
        kernel = RBF(lengthscale, Exp("variance", log_variance)) * Periodic(1.2, 1.0, period) + Matern(lengthscale, 0.3, nu=1.5)
        function = SparseGP("function", inputs, kernel, inducing_points=inducing, noise_variance=Exp("noise", log_noise), mean=mean, method=method)
        ObservedParameter("observations", function, y)
    return model, function, x, y


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_gradients_match_finite_differences(method):
    model, _, _, _ = _sparse_model(method)
    gradients = model.joint_grad_log_prob()
    assert gradients["log_lengthscale"].shape == (2,)
    _finite_difference_check(model)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_log_likelihood_matches_dense_formula(method):
    model, function, x, y = _sparse_model(method)
    values = function._named_parameter_values()
    z = x[:6]
    k_nn = function.kernel(x, x, parameter_values=values)
    k_mm = function.kernel(z, z, parameter_values=values)
    k_nm = function.kernel(x, z, parameter_values=values)
    # stable_cholesky scales the jitter with the largest diagonal entry.
    _, (jitter, _, _) = stable_cholesky(k_mm, jitter=function.jitter, return_jitter=True)
    k_mm = k_mm + jitter * torch.eye(6, dtype=x.dtype)
    q_nn = k_nm @ torch.linalg.solve(k_mm, k_nm.T)
    noise = values["noise"].reshape(())
    covariance = q_nn + noise * torch.eye(x.shape[0], dtype=x.dtype)
    if method == "fitc":
        covariance = covariance + torch.diag(torch.diagonal(k_nn - q_nn))
    expected = torch.distributions.MultivariateNormal(values["mean"].expand(x.shape[0]), covariance).log_prob(y)
    if method == "vfe":
        expected = expected - 0.5 * torch.trace(k_nn - q_nn) / noise
    torch.testing.assert_close(function.log_pdf(y), expected)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_equals_exact_gp_when_inducing_points_are_the_inputs(method):
    x, y = _ard_data(n=12)
    kernel = RBF(torch.tensor([0.5, 0.9, 1.4], dtype=torch.float64), 1.3)
    with Model():
        inputs = Data("inputs", x)
        exact = ExactGP("exact", inputs, kernel, noise_variance=0.1, jitter=1e-12)
        sparse = SparseGP("sparse", inputs, kernel, inducing_points=x, noise_variance=0.1, method=method, jitter=1e-12)
    torch.testing.assert_close(sparse.log_pdf(y), exact.log_pdf(y), atol=1e-7, rtol=1e-7)


def test_sparse_gp_vfe_is_a_lower_bound_on_the_exact_likelihood():
    x, y = _ard_data(n=30)
    kernel = RBF(torch.tensor([0.5, 0.9, 1.4], dtype=torch.float64), 1.3)
    with Model():
        inputs = Data("inputs", x)
        exact = ExactGP("exact", inputs, kernel, noise_variance=0.1, jitter=1e-10)
        sparse = SparseGP("sparse", inputs, kernel, inducing_points=x[:5], noise_variance=0.1, jitter=1e-10)
    assert sparse.log_pdf(y) < exact.log_pdf(y)


def test_sparse_gp_validates_its_arguments():
    x, _ = _ard_data(n=5)
    with Model() as model:
        inputs = Data("inputs", x)
        with pytest.raises(ValueError, match="method"):
            SparseGP("f", inputs, RBF(1.0), inducing_points=x, method="svgp")
        # Learnable inducing inputs are accepted (see Tests/test_gp_structured.py).
        z = RandomParameter("z", Normal(0.0, 1.0), shape=(2, 3))
        SparseGP("f", inputs, RBF(1.0), inducing_points=z)


def test_sparse_gp_scales_to_many_observations():
    import time

    n, m = 2000, 30
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, generator=generator, dtype=torch.float64)
    y = torch.sin(4 * x[:, 0]) + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=3)
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
        function = SparseGP(
            "function", inputs, RBF(Exp("lengthscale", log_lengthscale), 1.0),
            inducing_points=x[:m], noise_variance=Exp("noise", log_noise),
        )
        ObservedParameter("observations", function, y)
    start = time.perf_counter()
    log_probability = model.model_log_prob()
    gradients = model.joint_grad_log_prob()
    elapsed = time.perf_counter() - start
    print(f"SparseGP n={n}, m={m}: log density + gradient in {elapsed:.3f} s")
    assert torch.isfinite(log_probability)
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())


def test_sparse_gp_predictive_matches_exact_gp_when_inducing_points_are_the_inputs():
    x, y = _ard_data(n=12)
    kernel = RBF(torch.tensor([0.5, 0.9, 1.4], dtype=torch.float64), 1.3)
    with Model():
        inputs = Data("inputs", x)
        exact = ExactGP("exact", inputs, kernel, noise_variance=0.1, jitter=1e-10)
        sparse = SparseGP("sparse", inputs, kernel, inducing_points=x, noise_variance=0.1, jitter=1e-10)
    new_inputs = torch.rand(4, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    trace = {"dummy": torch.zeros(1, 1, 1, dtype=torch.float64)}
    torch.manual_seed(0)
    exact_samples = exact_gp_predictive(exact, trace, y, new_inputs, n_samples=2000)
    torch.manual_seed(0)
    sparse_samples = sparse_gp_predictive(sparse, trace, y, new_inputs, n_samples=2000)
    assert sparse_samples.shape == (1, 2000, 4)
    torch.testing.assert_close(sparse_samples.mean(dim=1), exact_samples.mean(dim=1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sparse_samples, exact_samples, atol=1e-4, rtol=1e-4)


@pytest.mark.integration
def test_sparse_gp_nuts_and_predictive():
    model, function, x, y = _sparse_model("vfe", n=40)
    trace = model.sample(10, 10, n_chains=1, progress_bar=False, check_convergence=False, seed=0)
    samples = sparse_gp_predictive(function, trace, y, torch.rand(5, 2, dtype=torch.float64), n_samples=3, observation_noise=0.01)
    assert samples.shape == (10, 3, 5)
    assert torch.isfinite(samples).all()


# ----------------------------------------------------------------------------
# Pure-torch Matérn and the rational-quadratic kernel
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.3, 0.8, 1.0, 2.0, 3.7])
def test_power_bessel_k_matches_scipy(nu):
    from scipy.special import kv

    from BayesianDLL.GP._kernels import _power_bessel_k

    a = torch.logspace(-8, 2.8, 300, dtype=torch.float64)
    for order in (nu, nu - 1.0):
        expected = a ** nu * torch.as_tensor(kv(order, a.numpy()), dtype=torch.float64)
        actual = _power_bessel_k(nu, order, a)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=0.0)


def _scipy_matern(x1, x2, lengthscale, variance, nu):
    """The previous scipy-based implementation, kept as a test oracle."""
    from scipy.special import gamma, kv

    s = (((x1[:, None, :] - x2[None, :, :]) / lengthscale) ** 2).sum(-1).sqrt()
    a = (2 * nu) ** 0.5 * s
    safe = torch.where(a == 0, torch.ones_like(a), a)
    value = 2 ** (1 - nu) / gamma(nu) * safe ** nu * torch.as_tensor(kv(nu, safe.numpy()), dtype=a.dtype)
    return variance * torch.where(a == 0, torch.ones_like(value), value)


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.5, 3.7, 6.5])
def test_matern_matches_scipy_reference(nu):
    kernel = Matern(_ARD_LENGTHSCALE, 1.4, nu=nu)
    x2 = torch.cat([_ARD_X2, _ARD_X1[:1]])  # includes a zero distance
    expected = _scipy_matern(_ARD_X1, x2, _ARD_LENGTHSCALE, 1.4, nu)
    torch.testing.assert_close(kernel(_ARD_X1, x2), expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 4.5])
def test_matern_half_integer_closed_form_agrees_with_quadrature(nu):
    from BayesianDLL.GP._kernels import _half_integer_matern, _half_integer_order, _power_bessel_k

    a = torch.logspace(-4, 2, 100, dtype=torch.float64)
    value, slope = _half_integer_matern(_half_integer_order(nu), a)
    coefficient = 2 ** (1 - nu) / torch.exp(torch.lgamma(torch.tensor(nu, dtype=torch.float64)))
    torch.testing.assert_close(value, coefficient * _power_bessel_k(nu, nu, a), rtol=1e-9, atol=1e-300)
    torch.testing.assert_close(slope, -coefficient * _power_bessel_k(nu, nu - 1, a), rtol=1e-9, atol=1e-300)


@pytest.mark.parametrize("nu", [0.3, 1.0, 2.0, 3.7, 3.5])
def test_general_matern_ard_derivatives_match_finite_differences(nu):
    kernel = Matern(SimpleNamespace(name="lengthscale"), SimpleNamespace(name="variance"), nu=nu)
    values = {"lengthscale": _ARD_LENGTHSCALE.clone(), "variance": torch.tensor([1.4], dtype=torch.float64)}
    x2 = torch.cat([_ARD_X2, _ARD_X1[:1]])
    for name in values:
        analytic = kernel.derivatives(_ARD_X1, x2, parameter_values=values)[name]
        finite_difference = _finite_difference_kernel_derivative(kernel, values, name, _ARD_X1, x2, step=1e-6)
        torch.testing.assert_close(analytic, finite_difference, atol=1e-7, rtol=1e-6)


def test_matern_runs_in_float32():
    x = _ARD_X1.to(torch.float32)
    for nu in (0.8, 2.5):
        covariance = Matern(_ARD_LENGTHSCALE.to(torch.float32), 1.0, nu=nu)(x, x)
        assert covariance.dtype == torch.float32
        expected = _scipy_matern(_ARD_X1, _ARD_X1, _ARD_LENGTHSCALE, 1.0, nu).to(torch.float32)
        torch.testing.assert_close(covariance, expected, rtol=1e-5, atol=1e-6)


def test_rational_quadratic_matches_direct_formula():
    kernel = RationalQuadratic(_ARD_LENGTHSCALE, 1.4, 0.7)
    r2 = (((_ARD_X1[:, None, :] - _ARD_X2[None, :, :]) / _ARD_LENGTHSCALE) ** 2).sum(-1)
    torch.testing.assert_close(kernel(_ARD_X1, _ARD_X2), 1.4 * (1 + r2 / 1.4) ** -0.7)
    # Large alpha approaches the RBF kernel.
    torch.testing.assert_close(
        RationalQuadratic(_ARD_LENGTHSCALE, 1.4, 1e7)(_ARD_X1, _ARD_X2), RBF(_ARD_LENGTHSCALE, 1.4)(_ARD_X1, _ARD_X2),
        rtol=1e-6, atol=1e-7,
    )


@pytest.mark.parametrize("lengthscale", [torch.tensor([0.8], dtype=torch.float64), _ARD_LENGTHSCALE])
def test_rational_quadratic_derivatives_match_finite_differences(lengthscale):
    kernel = RationalQuadratic(SimpleNamespace(name="lengthscale"), SimpleNamespace(name="variance"), SimpleNamespace(name="alpha"))
    values = {
        "lengthscale": lengthscale.clone(),
        "variance": torch.tensor([1.4], dtype=torch.float64),
        "alpha": torch.tensor([0.7], dtype=torch.float64),
    }
    for name in values:
        analytic = kernel.derivatives(_ARD_X1, _ARD_X2, parameter_values=values)[name]
        assert analytic.shape == (values[name].numel(), 4, 2)
        finite_difference = _finite_difference_kernel_derivative(kernel, values, name, _ARD_X1, _ARD_X2, step=1e-6)
        torch.testing.assert_close(analytic, finite_difference, atol=1e-7, rtol=1e-6)


def test_exact_gp_with_rational_quadratic_matches_finite_differences():
    x, y = _ard_data()
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = _ard_lengthscale_node("deterministic")
        alpha = RandomParameter("alpha", Exponential(1.0), initial_value=torch.tensor([0.9], dtype=torch.float64))
        noise = RandomParameter("noise", Exponential(1.0), initial_value=torch.tensor([0.2], dtype=torch.float64))
        kernel = RationalQuadratic(lengthscale, 1.2, alpha)
        ObservedParameter("observations", ExactGP("function", inputs, kernel, noise_variance=noise), y)
    _finite_difference_check(model)
