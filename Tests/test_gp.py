import pytest
import torch
from types import SimpleNamespace

from BayesianDLL import (
    Data,
    DeterministicParameter,
    MeanFieldGuide,
    Model,
    ObservedParameter,
    RandomParameter,
    VariationalParameter,
    condition,
)
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Exponential, Normal, Uniform
from BayesianDLL.GP import (
    Constant,
    ExactGP,
    GaussianProcess,
    ExactGaussianProcess,
    Linear,
    LatentGP,
    Matern,
    Matern32,
    Periodic,
    RBF,
    WhiteNoise,
    gp_predictive,
    exact_gp_predictive,
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
        ObservedParameter("observations", Normal(latent, 0.1), y)
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


def test_gp_legacy_names_are_compatible_aliases():
    assert GaussianProcess is LatentGP
    assert ExactGaussianProcess is ExactGP


def test_condition_can_fix_latent_gp_hyperparameters():
    x = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    y = torch.sin(2 * torch.pi * x)
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(-1.0, 0.5))
        log_variance = RandomParameter("log_variance", Normal(0.0, 0.5))
        log_noise = RandomParameter("log_noise", Normal(-2.5, 0.5))
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
        function = GaussianProcess("function", inputs, RBF(0.5, 1.0), latent=z)

    assert model.params["z"] is z
    assert "function_white" not in model.params
    assert function.latent is z
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
        function = GaussianProcess("function", inputs, RBF(0.5, variance), latent=z)
        # The literal likelihood variance must not be confused with the GP
        # hyperparameter named "variance".
        ObservedParameter("observations", Normal(function, 0.1), y)

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
    trace = model.sample(6, 6, n_chains=1, progress_bar=False, check_convergence=False)
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
        function = GaussianProcess("function", inputs, RBF(lengthscale, variance))
        ObservedParameter("observations", Normal(function, 0.1), y)
    trace = model.sample(2, 2, n_chains=1, progress_bar=False, check_convergence=False)
    assert trace["function"].dtype == trace["function_white"].dtype


def test_gp_predictive_samples_from_trace():
    model, inputs, latent = build_gp_model(n=3)
    trace = model.sample(4, 4, n_chains=1, progress_bar=False, check_convergence=False)
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
    with MeanFieldGuide() as guide:
        RandomParameter(
            "function_white",
            Normal(
                VariationalParameter("loc", torch.zeros(3)),
                VariationalParameter("var", torch.ones(3), min=1e-3),
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
