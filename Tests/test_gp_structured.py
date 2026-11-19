"""Learnable inducing inputs, Kronecker-grid and Toeplitz GPs."""

import time

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    Constant,
    ExactGP,
    Kernel,
    KroneckerGP,
    Linear,
    Matern,
    Matern32,
    Matern52,
    Periodic,
    RationalQuadratic,
    RBF,
    SparseGP,
    ToeplitzGP,
    WhiteNoise,
    exact_gp_predictive,
    kronecker_gp_predictive,
    levinson_solve,
    sparse_gp_predictive,
    toeplitz_gp_predictive,
)

DTYPE = torch.float64


def _finite_difference_check(model, eps=1e-6, rtol=1e-5, atol=1e-6, names=None):
    gradients = model.joint_grad_log_prob()
    for name, parameter in model.params.items():
        if names is not None and name not in names:
            continue
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


# ----------------------------------------------------------------------------
# Kernel input gradients
# ----------------------------------------------------------------------------

_LS = torch.tensor([0.7, 1.3], dtype=DTYPE)
_INPUT_KERNELS = {
    "rbf": lambda: RBF(_LS, 1.3),
    "rbf_scalar": lambda: RBF(0.6, 1.1),
    "periodic": lambda: Periodic(0.8, 1.2, 1.7),
    "periodic_ard": lambda: Periodic(_LS, 1.2, 1.7),
    "matern12": lambda: Matern(_LS, 1.1, nu=0.5),
    "matern32": lambda: Matern32(_LS, 1.0),
    "matern52": lambda: Matern52(0.9, 1.0),
    "matern_general": lambda: Matern(_LS, 1.0, nu=1.3),
    "rational_quadratic": lambda: RationalQuadratic(_LS, 1.2, 0.8),
    "linear": lambda: Linear(0.7, torch.tensor([0.1, -0.2], dtype=DTYPE)),
    "constant": lambda: Constant(0.5),
    "white": lambda: WhiteNoise(0.1),
    "sum": lambda: RBF(_LS, 1.3) + Linear(0.7, 0.1),
    "product": lambda: Periodic(_LS, 1.2, 1.7) * Matern52(0.9, 1.0) * Constant(0.5),
}


def _fd_input_gradient(function, x, eps=1e-6):
    gradient = torch.zeros_like(x)
    for index in range(x.numel()):
        shift = torch.zeros_like(x).reshape(-1)
        shift[index] = eps
        shift = shift.reshape(x.shape)
        gradient.reshape(-1)[index] = (function(x + shift) - function(x - shift)) / (2 * eps)
    return gradient


@pytest.mark.parametrize("name", list(_INPUT_KERNELS))
def test_kernel_input_vjp_matches_finite_differences(name):
    kernel = _INPUT_KERNELS[name]()
    generator = torch.Generator().manual_seed(3)
    x1 = torch.randn(5, 2, generator=generator, dtype=DTYPE)
    x2 = torch.randn(4, 2, generator=generator, dtype=DTYPE)
    cross_cotangent = torch.randn(5, 4, generator=generator, dtype=DTYPE)
    square_cotangent = torch.randn(5, 5, generator=generator, dtype=DTYPE)
    torch.testing.assert_close(
        kernel.input_vjp(x1, x2, cross_cotangent),
        _fd_input_gradient(lambda a: (cross_cotangent * kernel(a, x2)).sum(), x1),
        rtol=1e-6, atol=1e-8,
    )
    torch.testing.assert_close(
        kernel.input_vjp_symmetric(x1, square_cotangent),
        _fd_input_gradient(lambda a: (square_cotangent * kernel(a, a)).sum(), x1),
        rtol=1e-6, atol=1e-8,
    )


@pytest.mark.parametrize("name", ["rbf", "periodic", "periodic_ard", "rational_quadratic", "linear", "sum"])
def test_kernel_input_vjp_matches_autograd(name):
    kernel = _INPUT_KERNELS[name]()
    generator = torch.Generator().manual_seed(4)
    x1 = torch.randn(6, 2, generator=generator, dtype=DTYPE)
    x2 = torch.randn(3, 2, generator=generator, dtype=DTYPE)
    cotangent = torch.randn(6, 3, generator=generator, dtype=DTYPE)
    leaf = x1.clone().requires_grad_(True)
    (cotangent * kernel(leaf, x2)).sum().backward()
    torch.testing.assert_close(kernel.input_vjp(x1, x2, cotangent), leaf.grad)


# ----------------------------------------------------------------------------
# Learnable inducing inputs
# ----------------------------------------------------------------------------

_SPARSE_KERNELS = {
    "rbf_ard": lambda ls: RBF(ls, 1.4),
    "matern52": lambda ls: Matern52(ls, 0.9),
    "rq_plus_linear": lambda ls: RationalQuadratic(ls, 1.1, 1.5) + Linear(0.3, 0.1),
    "periodic_times_matern": lambda ls: Periodic(ls, 1.2, 2.5) * Matern(ls, 1.0, nu=1.5),
}


def _learnable_inducing_model(method, kernel_name, deterministic=False, n=14, m=5):
    generator = torch.Generator().manual_seed(11)
    x = torch.rand(n, 2, generator=generator, dtype=DTYPE) * 2.0
    y = torch.sin(2 * x[:, 0]) + x[:, 1] + 0.1 * torch.randn(n, generator=generator, dtype=DTYPE)
    z0 = torch.rand(m, 2, generator=generator, dtype=DTYPE) * 2.0
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([-0.2, 0.3], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-1.0], dtype=DTYPE))
        if deterministic:
            raw = RandomParameter("z_raw", Normal(0.0, 1.0), shape=tuple(z0.shape), initial_value=z0 - 1.0)
            inducing = DeterministicParameter(
                "z", lambda r: r + 1.0, lambda r: {"z_raw": torch.ones_like(r)}, [raw],
            )
        else:
            inducing = RandomParameter("z", Normal(z0, 1.0), shape=tuple(z0.shape), initial_value=z0)
        kernel = _SPARSE_KERNELS[kernel_name](Exp("lengthscale", log_lengthscale))
        function = SparseGP(
            "f", inputs, kernel, inducing_points=inducing, noise_variance=Exp("noise", log_noise),
            mean=RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.1], dtype=DTYPE)),
            method=method,
        )
        ObservedParameter("y", function, y)
    return model, function, x, y


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
@pytest.mark.parametrize("kernel_name", list(_SPARSE_KERNELS))
def test_learnable_inducing_gradients_match_finite_differences(method, kernel_name):
    model, _, _, _ = _learnable_inducing_model(method, kernel_name)
    gradients = _finite_difference_check(model)
    assert gradients["z"].shape == (5, 2)


def test_learnable_inducing_through_a_deterministic_node():
    model, _, _, _ = _learnable_inducing_model("vfe", "rbf_ard", deterministic=True)
    _finite_difference_check(model)


def test_learnable_inducing_gradient_includes_scale_dependent_jitter():
    """A variance above one makes the jitter depend on diag(K_mm), which depends on Z for Linear."""
    generator = torch.Generator().manual_seed(2)
    x = torch.rand(10, 1, generator=generator, dtype=DTYPE)
    y = torch.sin(4 * x[:, 0])
    z0 = torch.tensor([[0.1], [0.5], [2.0]], dtype=DTYPE)
    with Model() as model:
        inputs = Data("inputs", x)
        z = RandomParameter("z", Normal(z0, 1.0), shape=tuple(z0.shape), initial_value=z0)
        function = SparseGP("f", inputs, RBF(0.5, 1.5) + Linear(1.0, 0.0), inducing_points=z, noise_variance=0.1, method="vfe", jitter=1e-3)
        ObservedParameter("y", function, y)
    _finite_difference_check(model)


def _autograd_sparse_log_likelihood(method, x, y, z, lengthscale, variance, noise, mean, jitter):
    """Dense reference written with autograd-able torch ops (test oracle)."""
    def rbf(a, b):
        return variance * torch.exp(-0.5 * (((a[:, None, :] - b[None, :, :]) / lengthscale) ** 2).sum(-1))
    k_mm = rbf(z, z)
    scale = torch.clamp(torch.diagonal(k_mm).abs().max(), min=1.0)
    k_mm = k_mm + jitter * scale * torch.eye(z.shape[0], dtype=DTYPE)
    k_nm = rbf(x, z)
    q_nn = k_nm @ torch.linalg.solve(k_mm, k_nm.T)
    k_nn_diag = variance * torch.ones(x.shape[0], dtype=DTYPE)
    covariance = q_nn + noise * torch.eye(x.shape[0], dtype=DTYPE)
    if method == "fitc":
        covariance = covariance + torch.diag(k_nn_diag - torch.diagonal(q_nn))
    value = torch.distributions.MultivariateNormal(mean.expand(x.shape[0]), covariance).log_prob(y)
    if method == "vfe":
        value = value - 0.5 * (k_nn_diag - torch.diagonal(q_nn)).sum() / noise
    return value


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_learnable_inducing_gradient_matches_autograd_oracle(method):
    generator = torch.Generator().manual_seed(5)
    x = torch.rand(12, 2, generator=generator, dtype=DTYPE)
    y = torch.cos(3 * x[:, 0]) - x[:, 1]
    z0 = torch.rand(4, 2, generator=generator, dtype=DTYPE)
    lengthscale = torch.tensor([0.4, 0.8], dtype=DTYPE)
    with Model():
        inputs = Data("inputs", x)
        z = RandomParameter("z", Normal(0.0, 1.0), shape=tuple(z0.shape), initial_value=z0)
        function = SparseGP("f", inputs, RBF(lengthscale, 1.2), inducing_points=z, noise_variance=0.05, mean=0.3, method=method)
    leaf = z0.clone().requires_grad_(True)
    reference = _autograd_sparse_log_likelihood(
        method, x, y, leaf, lengthscale, 1.2, 0.05, torch.tensor(0.3, dtype=DTYPE), function.jitter
    )
    reference.backward()
    torch.testing.assert_close(function.log_pdf(y), reference.detach())
    torch.testing.assert_close(function.log_pdf_param_grads(y)["z"], leaf.grad)


def _bound_at(x, y, z, lengthscale=0.3, noise=0.01):
    with Model():
        function = SparseGP("f", Data("inputs", x), RBF(lengthscale, 1.0), inducing_points=z, noise_variance=noise)
    return function.log_pdf(y)


def test_learning_inducing_points_improves_the_vfe_bound():
    """MAP over Z (prior centred on bad clustered locations) raises the VFE bound."""
    generator = torch.Generator().manual_seed(0)
    x = torch.linspace(0.0, 4.0, 80, dtype=DTYPE).reshape(-1, 1)
    y = torch.sin(3.0 * x[:, 0]) + 0.1 * torch.randn(80, generator=generator, dtype=DTYPE)
    z_bad = torch.linspace(0.0, 0.5, 8, dtype=DTYPE).reshape(-1, 1)
    with Model() as model:
        inputs = Data("inputs", x)
        z = RandomParameter("z", Normal(z_bad, 5.0), shape=tuple(z_bad.shape), initial_value=z_bad.clone())
        function = SparseGP("f", inputs, RBF(0.3, 1.0), inducing_points=z, noise_variance=0.01, method="vfe")
        ObservedParameter("y", function, y)
    fixed_bound = _bound_at(x, y, z_bad)
    model.find_MAP(lr=0.05, epochs=400, verbose=False)
    learned = model.params["z"].constrained_value.clone()
    learned_bound = _bound_at(x, y, learned)
    exact = None
    with Model():
        exact = ExactGP("e", Data("inputs", x), RBF(0.3, 1.0), noise_variance=0.01).log_pdf(y)
    print(f"VFE bound: fixed {fixed_bound.item():.1f}, learned {learned_bound.item():.1f}, exact {exact.item():.1f}")
    assert learned_bound > fixed_bound + 100.0
    assert learned_bound <= exact + 1e-6
    # The learned inducing inputs spread over the data range.
    assert learned.max() > 3.0


def test_learnable_inducing_predictive_reads_inducing_draws():
    model, function, x, y = _learnable_inducing_model("vfe", "rbf_ard")
    trace = {
        "log_lengthscale": torch.zeros(1, 2, 2, dtype=DTYPE),
        "lengthscale": torch.ones(1, 2, 2, dtype=DTYPE),
        "noise": torch.full((1, 2, 1), 0.1, dtype=DTYPE),
        "mean": torch.zeros(1, 2, 1, dtype=DTYPE),
        "z": torch.stack([x[:5], x[5:10]]).unsqueeze(0),
    }
    samples = sparse_gp_predictive(function, trace, y, x[:3], n_samples=4)
    assert samples.shape == (2, 4, 3)
    assert torch.isfinite(samples).all()


@pytest.mark.integration
def test_learnable_inducing_vectorized_chains_match_chain_by_chain():
    model, _, _, _ = _learnable_inducing_model("vfe", "matern52")
    with model.batched(3):
        assert model._batch_mode == "tensor"
        values = model.joint_grad_log_prob()
        assert values["z"].shape == (3, 5, 2)
    trace = model.sample(5, 5, n_chains=2, chain_method="vectorized", progress_bar=False, check_convergence=False, seed=0)
    assert trace["z"].shape == (2, 5, 5, 2)


# ----------------------------------------------------------------------------
# Kronecker GP
# ----------------------------------------------------------------------------


class _Column(Kernel):
    """Test helper: apply a kernel to selected input columns."""

    def __init__(self, kernel, columns):
        super().__init__(*kernel.parameters)
        self.kernel = kernel
        self.columns = columns

    def evaluate(self, x1, x2, parameter_values=None):
        return self.kernel.evaluate(x1[:, self.columns], x2[:, self.columns], parameter_values=parameter_values)

    def derivatives(self, x1, x2, parameter_values=None):
        return self.kernel.derivatives(x1[:, self.columns], x2[:, self.columns], parameter_values=parameter_values)


def _grid_data(sizes=(7, 5)):
    generator = torch.Generator().manual_seed(1)
    grids = [torch.linspace(0.0, 2.0, s, dtype=DTYPE) + 0.05 * torch.rand(s, generator=generator, dtype=DTYPE) for s in sizes]
    points = torch.cartesian_prod(*grids).reshape(-1, len(grids))
    y = torch.sin(2 * points[:, 0]) * torch.cos(points[:, -1]) + 0.1 * torch.randn(points.shape[0], generator=generator, dtype=DTYPE)
    return grids, points, y


def _kronecker_and_exact(sizes=(7, 5), factor_kernels=None):
    grids, points, y = _grid_data(sizes)
    with Model() as kronecker_model:
        log_l1 = RandomParameter("log_l1", Normal(0.0, 1.0), initial_value=torch.tensor([-0.4], dtype=DTYPE))
        log_l2 = RandomParameter("log_l2", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.7], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.5], dtype=DTYPE))
        mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
        kernels = factor_kernels(Exp("l1", log_l1), Exp("l2", log_l2), Exp("variance", log_variance)) if factor_kernels else [
            RBF(Exp("l1", log_l1), Exp("variance", log_variance)), Matern52(Exp("l2", log_l2), 1.0)]
        kronecker = KroneckerGP("f", [Data(f"grid{p}", g) for p, g in enumerate(grids)], kernels,
                                noise_variance=Exp("noise", log_noise), mean=mean, jitter=0.0)
        ObservedParameter("y", kronecker, y.reshape(sizes))
    with Model() as exact_model:
        log_l1 = RandomParameter("log_l1", Normal(0.0, 1.0), initial_value=torch.tensor([-0.4], dtype=DTYPE))
        log_l2 = RandomParameter("log_l2", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([0.7], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.5], dtype=DTYPE))
        mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
        factors = factor_kernels(Exp("l1", log_l1), Exp("l2", log_l2), Exp("variance", log_variance)) if factor_kernels else [
            RBF(Exp("l1", log_l1), Exp("variance", log_variance)), Matern52(Exp("l2", log_l2), 1.0)]
        kernel = _Column(factors[0], [0])
        for p, factor in enumerate(factors[1:], start=1):
            kernel = kernel * _Column(factor, [p])
        exact = ExactGP("f", Data("points", points), kernel, noise_variance=Exp("noise", log_noise), mean=mean, jitter=1e-14)
        ObservedParameter("y", exact, y)
    return kronecker_model, kronecker, exact_model, exact, grids, points, y


def test_kronecker_log_likelihood_matches_exact_gp():
    kronecker_model, kronecker, exact_model, exact, _, _, y = _kronecker_and_exact()
    torch.testing.assert_close(kronecker.log_pdf(y), exact.log_pdf(y), atol=1e-8, rtol=0)
    torch.testing.assert_close(kronecker.log_pdf(y.reshape(7, 5)), exact.log_pdf(y), atol=1e-8, rtol=0)
    torch.testing.assert_close(kronecker_model.model_log_prob(), exact_model.model_log_prob(), atol=1e-8, rtol=0)


def test_kronecker_three_factor_grid_matches_exact_gp():
    def kernels(l1, l2, variance):
        return [RBF(l1, variance), Periodic(l2, 1.0, 1.5), RationalQuadratic(0.8, 1.0, 2.0)]
    kronecker_model, kronecker, exact_model, exact, _, _, y = _kronecker_and_exact((4, 3, 5), kernels)
    torch.testing.assert_close(kronecker.log_pdf(y), exact.log_pdf(y), atol=1e-8, rtol=0)
    kronecker_gradients = kronecker_model.joint_grad_log_prob()
    exact_gradients = exact_model.joint_grad_log_prob()
    for name in exact_gradients:
        torch.testing.assert_close(kronecker_gradients[name], exact_gradients[name], atol=1e-7, rtol=1e-7)


def test_kronecker_gradients_match_finite_differences_and_exact_gp():
    kronecker_model, _, exact_model, _, _, _, _ = _kronecker_and_exact()
    gradients = _finite_difference_check(kronecker_model)
    exact_gradients = exact_model.joint_grad_log_prob()
    for name in exact_gradients:
        torch.testing.assert_close(gradients[name], exact_gradients[name], atol=1e-7, rtol=1e-7)


def test_kronecker_ard_factor_with_multidimensional_inputs():
    generator = torch.Generator().manual_seed(9)
    grid_a = torch.rand(4, 2, generator=generator, dtype=DTYPE)
    grid_b = torch.linspace(0, 1, 3, dtype=DTYPE)
    with Model() as model:
        log_l = RandomParameter("log_l", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([0.1, -0.3], dtype=DTYPE))
        function = KroneckerGP("f", [grid_a, grid_b], [RBF(Exp("l", log_l), 1.2), Matern32(0.5, 1.0)], noise_variance=0.1)
        y = torch.randn(4, 3, generator=generator, dtype=DTYPE)
        ObservedParameter("y", function, y)
    _finite_difference_check(model)
    index = torch.cartesian_prod(torch.arange(4), torch.arange(3))
    points = torch.cat([grid_a[index[:, 0]], grid_b[index[:, 1]].reshape(-1, 1)], dim=1)
    kernel = _Column(RBF(torch.exp(torch.tensor([0.1, -0.3], dtype=DTYPE)), 1.2), [0, 1]) * _Column(Matern32(0.5, 1.0), [2])
    with Model():
        exact = ExactGP("e", Data("p", points), kernel, noise_variance=0.1 + function.jitter, jitter=1e-14)
    torch.testing.assert_close(function.log_pdf(y), exact.log_pdf(y.reshape(-1)), atol=1e-8, rtol=0)


def test_kronecker_predictive_matches_exact_gp():
    _, kronecker, _, exact, grids, points, y = _kronecker_and_exact()
    trace = {
        "l1": torch.tensor([[[0.6]], [[0.8]]], dtype=DTYPE).reshape(1, 2, 1),
        "l2": torch.tensor([1.1, 0.9], dtype=DTYPE).reshape(1, 2, 1),
        "variance": torch.tensor([1.5, 2.0], dtype=DTYPE).reshape(1, 2, 1),
        "noise": torch.tensor([0.05, 0.08], dtype=DTYPE).reshape(1, 2, 1),
        "mean": torch.tensor([0.1, -0.1], dtype=DTYPE).reshape(1, 2, 1),
    }
    new_points = torch.rand(6, 2, generator=torch.Generator().manual_seed(4), dtype=DTYPE) * 2.0
    torch.manual_seed(0)
    exact_samples = exact_gp_predictive(exact, trace, y, new_points, n_samples=3)
    torch.manual_seed(0)
    kronecker_samples = kronecker_gp_predictive(kronecker, trace, y, new_points, n_samples=3)
    torch.testing.assert_close(kronecker_samples, exact_samples, atol=1e-6, rtol=1e-6)
    # A new grid is flattened in row-major order.
    new_grid = [torch.tensor([0.3, 1.2], dtype=DTYPE), torch.tensor([0.1, 0.5, 1.9], dtype=DTYPE)]
    grid_points = torch.cartesian_prod(*new_grid)
    torch.manual_seed(1)
    from_grid = kronecker_gp_predictive(kronecker, trace, y, new_grid, n_samples=2)
    torch.manual_seed(1)
    from_points = exact_gp_predictive(exact, trace, y, grid_points, n_samples=2)
    torch.testing.assert_close(from_grid, from_points, atol=1e-6, rtol=1e-6)
    # Diagonal mode: same marginal means and variances.
    torch.manual_seed(2)
    diagonal = kronecker_gp_predictive(kronecker, trace, y, new_points, n_samples=20000, diagonal=True)
    torch.manual_seed(2)
    full = kronecker_gp_predictive(kronecker, trace, y, new_points, n_samples=20000)
    torch.testing.assert_close(diagonal.mean(1), full.mean(1), atol=0.03, rtol=0)
    torch.testing.assert_close(diagonal.var(1), full.var(1), atol=0.01, rtol=0.05)


def test_kronecker_rejects_wrong_sizes():
    with Model():
        function = KroneckerGP("f", [torch.zeros(3, dtype=DTYPE), torch.zeros(2, dtype=DTYPE)], [RBF(1.0), RBF(1.0)])
        with pytest.raises(ValueError, match="one value per grid point"):
            function.log_pdf(torch.zeros(5, dtype=DTYPE))
        with pytest.raises(ValueError, match="one kernel per grid factor"):
            KroneckerGP("g", [torch.zeros(3, dtype=DTYPE)], [RBF(1.0), RBF(1.0)])


def test_kronecker_sample_shape():
    with Model():
        function = KroneckerGP("f", [torch.linspace(0, 1, 4, dtype=DTYPE), torch.linspace(0, 1, 3, dtype=DTYPE)], [RBF(0.5), RBF(0.5)], noise_variance=0.1)
    assert function.sample(5).shape == (5, 4, 3)


def test_kronecker_timing_on_a_100_by_100_grid():
    sizes = (100, 100)
    grids = [torch.linspace(0.0, 1.0, s, dtype=DTYPE) for s in sizes]
    points = torch.cartesian_prod(*grids)
    y = torch.sin(6 * points[:, 0]) * torch.cos(4 * points[:, 1]) + 0.1 * torch.randn(points.shape[0], dtype=DTYPE)
    with Model() as model:
        log_l = RandomParameter("log_l", Normal(-1.5, 1.0), shape=2, initial_value=torch.tensor([-1.5, -1.5], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.0], dtype=DTYPE))
        l = Exp("l", log_l)
        # Each factor reads its own entry of the length-scale vector.
        l1 = DeterministicParameter("l1", lambda v: v[..., :1], lambda v: {"l": torch.tensor([[1.0, 0.0]], dtype=DTYPE)}, [l])
        l2 = DeterministicParameter("l2", lambda v: v[..., 1:], lambda v: {"l": torch.tensor([[0.0, 1.0]], dtype=DTYPE)}, [l])
        function = KroneckerGP("f", grids, [RBF(l1, 1.0), RBF(l2, 1.0)], noise_variance=Exp("noise", log_noise))
        ObservedParameter("y", function, y.reshape(sizes))
    start = time.perf_counter()
    log_probability = model.model_log_prob()
    gradients = model.joint_grad_log_prob()
    elapsed = time.perf_counter() - start
    print(f"KroneckerGP 100x100 grid (N=10000): log density + gradient in {elapsed:.3f} s")
    assert torch.isfinite(log_probability)
    assert all(torch.isfinite(g).all() for g in gradients.values())


# ----------------------------------------------------------------------------
# Toeplitz GP
# ----------------------------------------------------------------------------


def test_levinson_solve_matches_dense_solve():
    n = 40
    lags = torch.arange(n, dtype=DTYPE)
    column = torch.exp(-0.5 * (lags / 6.0) ** 2)
    column[0] += 0.1
    matrix = column[(lags[:, None] - lags[None, :]).abs().long()]
    rhs = torch.randn(n, 3, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    solution, log_det = levinson_solve(column, rhs)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    torch.testing.assert_close(log_det, torch.logdet(matrix))
    with pytest.raises(RuntimeError, match="positive definite"):
        levinson_solve(torch.tensor([1.0, 2.0, 0.0], dtype=DTYPE), torch.ones(3, dtype=DTYPE))


def test_toeplitz_inverse_diagonal_sums():
    from BayesianDLL.GP._structured import _toeplitz_inverse_diagonal_sums
    n = 25
    lags = torch.arange(n, dtype=DTYPE)
    column = 1.3 * torch.exp(-lags / 4.0)
    column[0] += 0.2
    matrix = column[(lags[:, None] - lags[None, :]).abs().long()]
    inverse = torch.linalg.inv(matrix)
    expected = torch.stack([torch.diagonal(inverse, offset=-k).sum() for k in range(n)])
    first_column, _ = levinson_solve(column, torch.eye(n, dtype=DTYPE)[:, 0])
    torch.testing.assert_close(_toeplitz_inverse_diagonal_sums(first_column), expected)


def _toeplitz_and_exact(kernel_factory, n=60, variance_init=0.4):
    x = torch.linspace(-1.0, 3.0, n, dtype=DTYPE)
    generator = torch.Generator().manual_seed(8)
    y = torch.sin(3 * x) + 0.2 * torch.randn(n, generator=generator, dtype=DTYPE)
    models = []
    for cls in (ToeplitzGP, ExactGP):
        with Model() as model:
            log_l = RandomParameter("log_l", Normal(0.0, 1.0), initial_value=torch.tensor([-0.5], dtype=DTYPE))
            log_variance = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([variance_init], dtype=DTYPE))
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.0], dtype=DTYPE))
            mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.3], dtype=DTYPE))
            kernel = kernel_factory(Exp("l", log_l), Exp("variance", log_variance))
            function = cls("f", Data("x", x), kernel, noise_variance=Exp("noise", log_noise), mean=mean)
            ObservedParameter("y", function, y)
        models.append((model, function))
    return models, x, y


_TOEPLITZ_KERNELS = {
    "rbf": lambda l, v: RBF(l, v),
    "matern32": lambda l, v: Matern32(l, v),
    "matern_general": lambda l, v: Matern(l, v, nu=1.7),
    "rq_plus_periodic": lambda l, v: RationalQuadratic(l, v, 1.5) + Periodic(0.9, 0.3, 1.3),
}


@pytest.mark.parametrize("kernel_name", list(_TOEPLITZ_KERNELS))
@pytest.mark.parametrize("variance_init", [-0.5, 0.8])  # below / above the jitter-scale threshold
def test_toeplitz_matches_exact_gp(kernel_name, variance_init):
    ((toeplitz_model, toeplitz), (exact_model, exact)), _, y = _toeplitz_and_exact(_TOEPLITZ_KERNELS[kernel_name], variance_init=variance_init)
    torch.testing.assert_close(toeplitz.log_pdf(y), exact.log_pdf(y), atol=1e-8, rtol=0)
    toeplitz_gradients = _finite_difference_check(toeplitz_model)
    exact_gradients = exact_model.joint_grad_log_prob()
    for name in exact_gradients:
        torch.testing.assert_close(toeplitz_gradients[name], exact_gradients[name], atol=1e-7, rtol=1e-7)


def test_toeplitz_predictive_matches_exact_gp():
    ((_, toeplitz), (_, exact)), x, y = _toeplitz_and_exact(_TOEPLITZ_KERNELS["rbf"])
    trace = {
        "l": torch.tensor([0.5, 0.7], dtype=DTYPE).reshape(1, 2, 1),
        "variance": torch.tensor([1.0, 1.3], dtype=DTYPE).reshape(1, 2, 1),
        "noise": torch.tensor([0.05, 0.1], dtype=DTYPE).reshape(1, 2, 1),
        "mean": torch.tensor([0.0, 0.2], dtype=DTYPE).reshape(1, 2, 1),
    }
    new = torch.tensor([-1.5, 0.33, 2.71, 3.4], dtype=DTYPE)
    torch.manual_seed(0)
    exact_samples = exact_gp_predictive(exact, trace, y, new, n_samples=3)
    torch.manual_seed(0)
    toeplitz_samples = toeplitz_gp_predictive(toeplitz, trace, y, new, n_samples=3)
    torch.testing.assert_close(toeplitz_samples, exact_samples, atol=1e-6, rtol=1e-6)


def test_toeplitz_validates_inputs():
    with Model():
        x = Data("x", torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE))
        with pytest.raises(ValueError, match="equally spaced"):
            ToeplitzGP("f", x, RBF(1.0))
        with pytest.raises(TypeError, match="stationary"):
            ToeplitzGP("g", Data("x2", torch.linspace(0, 1, 3, dtype=DTYPE)), RBF(1.0) + Linear(1.0))


def test_toeplitz_timing_against_exact_gp():
    n = 1500
    x = torch.linspace(0.0, 30.0, n, dtype=DTYPE)
    y = torch.sin(x) + 0.1 * torch.randn(n, dtype=DTYPE)
    for cls in (ToeplitzGP, ExactGP):
        with Model() as model:
            log_l = RandomParameter("log_l", Normal(0.0, 1.0))
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
            function = cls("f", Data("x", x), Matern52(Exp("l", log_l), 1.0), noise_variance=Exp("noise", log_noise))
            ObservedParameter("y", function, y)
        start = time.perf_counter()
        model.model_log_prob()
        model.joint_grad_log_prob()
        print(f"{cls.__name__} n={n}: log density + gradient in {time.perf_counter() - start:.3f} s")


def test_structured_gps_follow_model_to_float32():
    with Model() as model:
        log_l = RandomParameter("log_l", Normal(0.0, 1.0))
        l = Exp("l", log_l)
        grid = [torch.linspace(0, 1, 6, dtype=DTYPE), torch.linspace(0, 1, 4, dtype=DTYPE)]
        ObservedParameter("y_grid", KroneckerGP("k", grid, [RBF(l, 1.0), RBF(0.5, 1.0)], noise_variance=0.1), torch.zeros(6, 4, dtype=DTYPE))
        x = Data("x", torch.linspace(0, 1, 20, dtype=DTYPE))
        ObservedParameter("y_line", ToeplitzGP("t", x, Matern52(l, 1.0), noise_variance=0.1), torch.zeros(20, dtype=DTYPE))
        z = RandomParameter("z", Normal(0.0, 1.0), shape=(3, 1), initial_value=torch.tensor([[0.1], [0.5], [0.9]], dtype=DTYPE))
        ObservedParameter("y_sparse", SparseGP("s", x, RBF(l, 1.0), inducing_points=z, noise_variance=0.1), torch.zeros(20, dtype=DTYPE))
    reference = model.joint_grad_log_prob()
    model.to(dtype=torch.float32)
    value = model.model_log_prob()
    gradients = model.joint_grad_log_prob()
    assert value.dtype == torch.float32
    for name, gradient in gradients.items():
        assert gradient.dtype == torch.float32
        torch.testing.assert_close(gradient.double(), reference[name], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("structure", ["toeplitz", "kronecker"])
def test_structured_gp_rejects_a_per_input_vector_mean(structure):
    x = torch.linspace(0.0, 1.0, 6, dtype=DTYPE)
    train_mean = torch.linspace(1.0, 2.0, 6, dtype=DTYPE)
    with Model():
        with pytest.raises(ValueError, match="mean function"):
            if structure == "toeplitz":
                ToeplitzGP("f", Data("x", x), RBF(0.4, 1.3), noise_variance=0.1, mean=train_mean)
            else:
                KroneckerGP("f", [Data("x", x)], [RBF(0.4, 1.3)], noise_variance=0.1, mean=train_mean)


def _toeplitz_column(kind, n=50):
    lags = torch.arange(n, dtype=DTYPE)
    nugget = (lags == 0).to(DTYPE)
    if kind == "exponential":
        return 1.3 * torch.exp(-lags / 4.0) + 0.2 * nugget
    if kind == "rbf":
        return torch.exp(-0.5 * (lags / 3.0) ** 2) + 1e-2 * nugget
    if kind == "matern52":
        scaled = 5 ** 0.5 * lags / 5.0
        return (1 + scaled + scaled ** 2 / 3) * torch.exp(-scaled) + 1e-3 * nugget
    # quasi-periodic: periodic times RBF
    return torch.exp(-2 * torch.sin(torch.pi * lags / 12) ** 2 / 0.8 ** 2 - 0.5 * (lags / 20) ** 2) + 0.05 * nugget


def _dense_toeplitz(column):
    lags = torch.arange(column.shape[-1])
    return column[(lags[:, None] - lags[None, :]).abs()]


@pytest.mark.parametrize("kind", ["exponential", "rbf", "matern52", "quasi_periodic"])
def test_gohberg_semencul_diagonal_sums_match_dense_inverse(kind):
    """D_k = Σ_i (T⁻¹)_{i+k,i} from the first column of T⁻¹ matches a dense inverse (n = 50, κ up to 1e4)."""
    from BayesianDLL.GP._structured import _toeplitz_inverse_diagonal_sums

    column = _toeplitz_column(kind)
    n = column.shape[0]
    inverse = torch.linalg.inv(_dense_toeplitz(column))
    expected = torch.stack([torch.diagonal(inverse, offset=-k).sum() for k in range(n)])
    first_column, _ = levinson_solve(column, torch.eye(n, dtype=DTYPE)[:, 0])
    actual = _toeplitz_inverse_diagonal_sums(first_column)
    # Measured: max error <= 1e-13 relative to the largest sum.
    assert (actual - expected).abs().max() <= 1e-10 * expected.abs().max()


@pytest.mark.parametrize("condition", [1e2, 1e6, 1e10])
def test_levinson_accuracy_against_cholesky(condition):
    """Residual and log det of the Levinson solve against a dense Cholesky (RBF plus nugget, n = 50)."""
    n = 50
    lags = torch.arange(n, dtype=DTYPE)
    base = torch.exp(-0.5 * (lags / 5.0) ** 2)
    column = base.clone()
    column[0] += torch.linalg.eigvalsh(_dense_toeplitz(base)).max() / condition
    matrix = _dense_toeplitz(column)
    assert torch.linalg.cond(matrix).item() == pytest.approx(condition, rel=0.05)
    rhs = torch.randn(n, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    solution, log_det = levinson_solve(column, rhs)
    factor = torch.linalg.cholesky(matrix)
    cholesky_solution = torch.cholesky_solve(rhs.reshape(-1, 1), factor).reshape(-1)
    residual = torch.linalg.norm(matrix @ solution - rhs) / torch.linalg.norm(rhs)
    cholesky_residual = torch.linalg.norm(matrix @ cholesky_solution - rhs) / torch.linalg.norm(rhs)
    eps = torch.finfo(DTYPE).eps
    # Measured residuals 4e-15, 4e-11, 5e-7 (Cholesky: 4e-15, 4e-11, 3e-7): about κ ε, like Cholesky.
    assert residual <= 10 * eps * condition
    assert residual <= 10 * cholesky_residual
    # Measured |Δ log det| 3e-14, 4e-10, 4e-7.
    assert abs(log_det - 2 * torch.log(torch.diagonal(factor)).sum()) <= 10 * eps * condition
