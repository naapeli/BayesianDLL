"""Learnable training inputs X: ExactGP with inducing points (VFE, DTC, FITC), KroneckerGP grid axes, ToeplitzGP (rejected).

Also the diagonal input VJP of every kernel and an inducing-point ExactGP with several
independent output dimensions that share X (``n_outputs``), the building
block of a GP latent variable model.
"""

import warnings

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Affine, Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    Constant,
    ExactGP,
    Kernel,
    KroneckerGP,
    Linear,
    LinearMean,
    Matern,
    Matern32,
    Matern52,
    Periodic,
    RationalQuadratic,
    RBF,
    ToeplitzGP,
    WhiteNoise,
    exact_gp_predictive,
)

DTYPE = torch.float64
C = 4


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


def _assert_batched_equals_loop(model, seed=0, scale=0.1):
    generator = torch.Generator().manual_seed(seed)
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        values[name] = u.unsqueeze(0) + scale * torch.randn((C, *u.shape), generator=generator, dtype=u.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fallback to the loop mode warns
        with model.batched(C, values=values, mode="auto"):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            batched_lp = model.model_log_prob()
            batched_grads = model.joint_grad_log_prob()
    with model.batched(C, values=values, mode="loop"):
        loop_lp = model.model_log_prob()
        loop_grads = model.joint_grad_log_prob()
    assert batched_lp.shape == (C,)
    torch.testing.assert_close(batched_lp, loop_lp, atol=1e-10, rtol=1e-10)
    for name, gradient in loop_grads.items():
        assert batched_grads[name].shape == gradient.shape, name
        torch.testing.assert_close(batched_grads[name], gradient, atol=1e-10, rtol=1e-10, msg=f"gradient of {name}")


# ----------------------------------------------------------------------------
# Diagonal input VJP of the kernels
# ----------------------------------------------------------------------------


_LS = torch.tensor([0.7, 1.3], dtype=DTYPE)
_OFFSET = torch.tensor([0.1, -0.2], dtype=DTYPE)


class _UserLinear(Kernel):
    """A user kernel with only the unbatched ``evaluate`` / ``input_vjp`` (exercises the generic fallback)."""

    def evaluate(self, x1, x2, parameter_values=None):
        return 0.8 * (x1 - 0.3) @ (x2 - 0.3).transpose(-2, -1)

    def input_vjp(self, x1, x2, cotangent, parameter_values=None):
        return 0.8 * cotangent @ (x2 - 0.3)


_DIAG_KERNELS = {
    "rbf": lambda: RBF(_LS, 1.3),
    "periodic": lambda: Periodic(0.8, 1.2, 1.7),
    "periodic_ard": lambda: Periodic(_LS, 1.2, 1.7),
    "matern12": lambda: Matern(_LS, 1.1, nu=0.5),
    "matern32": lambda: Matern32(_LS, 1.0),
    "matern52": lambda: Matern52(0.9, 1.0),
    "matern_general": lambda: Matern(_LS, 1.0, nu=1.3),
    "rational_quadratic": lambda: RationalQuadratic(_LS, 1.2, 0.8),
    "constant": lambda: Constant(0.5),
    "white": lambda: WhiteNoise(0.1),
    "linear": lambda: Linear(0.7, _OFFSET),
    "linear_scalar_offset": lambda: Linear(1.4, 0.3),
    "rbf_plus_linear": lambda: RBF(_LS, 1.3) + Linear(0.7, 0.1),
    "linear_times_linear": lambda: Linear(0.7, _OFFSET) * Linear(1.1, 0.4),
    "rbf_times_linear_plus_constant": lambda: RBF(_LS, 1.3) * Linear(0.7, _OFFSET) * Constant(0.5) + Linear(0.2),
    "user_kernel": lambda: _UserLinear(),
    "user_times_linear": lambda: _UserLinear() * Linear(0.5, 0.2),
}


def _pairwise_diagonal(kernel, x):
    """``k(x_i, x_i)`` for every row, evaluated through ``kernel(x, x)`` with autograd-able ops."""
    return torch.diagonal(kernel(x, x))


@pytest.mark.parametrize("name", list(_DIAG_KERNELS))
def test_diag_input_vjp_matches_autograd_and_finite_differences(name):
    kernel = _DIAG_KERNELS[name]()
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(6, 2, generator=generator, dtype=DTYPE)
    g = torch.randn(6, generator=generator, dtype=DTYPE)
    result = kernel.diag_input_vjp(x, g)
    assert result.shape == x.shape
    # Autograd oracle on the diagonal of the full matrix; at zero distance the
    # stationary kernels' sqrt has no autograd derivative, and k(x, x) is the
    # constant variance there, so the oracle is zero.
    leaf = x.clone().requires_grad_(True)
    diagonal = _pairwise_diagonal(kernel, leaf)
    if diagonal.requires_grad:
        (gradient,) = torch.autograd.grad((g * diagonal).sum(), leaf, allow_unused=True)
        gradient = torch.zeros_like(x) if gradient is None else torch.nan_to_num(gradient, nan=0.0)
    else:
        gradient = torch.zeros_like(x)
    torch.testing.assert_close(result, gradient, rtol=1e-10, atol=1e-12)
    # Finite differences of diag(K(x, x)), both arguments moving.
    eps = 1e-6
    numerical = torch.zeros_like(x)
    for index in range(x.numel()):
        shift = torch.zeros_like(x).reshape(-1)
        shift[index] = eps
        shift = shift.reshape(x.shape)
        upper = (g * kernel.diag(x + shift)).sum()
        lower = (g * kernel.diag(x - shift)).sum()
        numerical.reshape(-1)[index] = (upper - lower) / (2 * eps)
    torch.testing.assert_close(result, numerical, rtol=1e-6, atol=1e-8)


def test_linear_diag_input_vjp_closed_form():
    x = torch.randn(5, 2, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    g = torch.linspace(-1.0, 1.0, 5, dtype=DTYPE)
    expected = 2 * 0.7 * (x - _OFFSET) * g[:, None]
    torch.testing.assert_close(Linear(0.7, _OFFSET).diag_input_vjp(x, g), expected)


def test_batched_diag_input_vjp_matches_per_element_calls():
    generator = torch.Generator().manual_seed(2)
    with Model():
        log_v = RandomParameter("log_v", Normal(0.0, 1.0))
        offset = RandomParameter("c", Normal(0.0, 1.0), shape=2)
        kernel = RBF(0.8, 1.2) * Linear(Exp("v", log_v), offset) + Linear(0.3)
    x = torch.randn(C, 5, 2, generator=generator, dtype=DTYPE)
    g = torch.randn(C, 5, generator=generator, dtype=DTYPE)
    values = {"v": torch.rand(C, 1, generator=generator, dtype=DTYPE) + 0.5,
              "c": torch.randn(C, 2, generator=generator, dtype=DTYPE)}
    batched = kernel.batch_diag_input_vjp(x, g, values)
    for b in range(C):
        single = kernel.diag_input_vjp(x[b], g[b], parameter_values={"v": values["v"][b], "c": values["c"][b]})
        torch.testing.assert_close(batched[b], single, atol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------------
# ExactGP with inducing points and learnable training inputs
# ----------------------------------------------------------------------------


_SPARSE_KERNELS = {
    "rbf": lambda ls: RBF(ls, 1.3),
    "matern52": lambda ls: Matern52(ls, 0.9),
    "linear": lambda ls: Linear(0.8, 0.2),
    "rbf_plus_linear": lambda ls: RBF(ls, 1.1) + Linear(0.4, torch.tensor([0.1, -0.3], dtype=DTYPE)),
}


def _sparse_data(n=10, m=4, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=generator, dtype=DTYPE) * 2.0 - 1.0
    y = torch.sin(2 * x[:, 0]) + x[:, 1] + 0.1 * torch.randn(n, generator=generator, dtype=DTYPE)
    z = torch.rand(m, 2, generator=generator, dtype=DTYPE) * 2.0 - 1.0
    return x, y, z


def _mean(kind):
    if kind == "zero":
        return None
    w = RandomParameter("w", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([0.4, -0.7], dtype=DTYPE))
    b = RandomParameter("b", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
    return LinearMean(w, b)


def _sparse_model(method, kernel_name, mean_kind, learnable=True, deterministic=False, learnable_z=False):
    x, y, z0 = _sparse_data()
    with Model() as model:
        if deterministic:
            raw = RandomParameter("x_raw", Normal(0.0, 1.0), shape=tuple(x.shape), initial_value=(x - 0.1) / 1.5)
            log_scale = RandomParameter("log_scale", Normal(0.0, 1.0), initial_value=torch.tensor([0.4], dtype=DTYPE))
            inputs = Affine("x", raw, loc=0.1, scale=Exp("scale", log_scale))
        elif learnable:
            inputs = RandomParameter("x", Normal(0.0, 1.0), shape=tuple(x.shape), initial_value=x.clone())
        else:
            inputs = Data("x", x)
        log_l = RandomParameter("log_l", Normal(0.0, 1.0), shape=2, initial_value=torch.tensor([-0.3, 0.2], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-1.2], dtype=DTYPE))
        z = RandomParameter("z", Normal(z0, 1.0), shape=tuple(z0.shape), initial_value=z0.clone()) if learnable_z else z0
        gp = ExactGP("f", inputs, _SPARSE_KERNELS[kernel_name](Exp("l", log_l)), inducing_points=z,
                     noise_variance=Exp("noise", log_noise), mean=_mean(mean_kind), method=method)
        ObservedParameter("y", gp, y)
    return model, gp, x, y, z0


@pytest.mark.parametrize("mean_kind", ["zero", "linear"])
@pytest.mark.parametrize("kernel_name", list(_SPARSE_KERNELS))
@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_input_gradients_match_finite_differences(method, kernel_name, mean_kind):
    model, _, x, _, _ = _sparse_model(method, kernel_name, mean_kind)
    gradients = _finite_difference_check(model)
    assert gradients["x"].shape == x.shape


def _dense_sparse_log_likelihood(gp, method, x, y, z):
    """Dense reference: the kernel and mean evaluated with autograd-able torch ops (test oracle)."""
    kernel, mean = gp.kernel, gp.mean
    k_mm = kernel(z, z)
    scale = torch.clamp(torch.diagonal(k_mm).abs().max(), min=1.0)
    k_mm = k_mm + gp.jitter * scale * torch.eye(z.shape[0], dtype=DTYPE)
    k_nm = kernel(x, z)
    q_nn = k_nm @ torch.linalg.solve(k_mm, k_nm.T)
    k_nn_diag = torch.diagonal(kernel(x, x))
    noise = torch.as_tensor(gp.noise_variance.constrained_value).reshape(())
    covariance = q_nn + noise * torch.eye(x.shape[0], dtype=DTYPE)
    if method == "fitc":
        covariance = covariance + torch.diag(k_nn_diag - torch.diagonal(q_nn))
    m = mean(x)
    value = torch.distributions.MultivariateNormal(m.expand(x.shape[0]), covariance).log_prob(y)
    if method == "vfe":
        value = value - 0.5 * (k_nn_diag - torch.diagonal(q_nn)).sum() / noise
    return value


@pytest.mark.parametrize("mean_kind", ["zero", "linear"])
@pytest.mark.parametrize("kernel_name", ["rbf", "linear", "rbf_plus_linear"])
@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_input_gradient_matches_autograd_oracle(method, kernel_name, mean_kind):
    _, gp, x, y, z = _sparse_model(method, kernel_name, mean_kind)
    leaf = x.clone().requires_grad_(True)
    reference = _dense_sparse_log_likelihood(gp, method, leaf, y, z)
    (expected,) = torch.autograd.grad(reference, leaf)
    torch.testing.assert_close(gp.log_pdf(y), reference.detach(), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(gp.log_pdf_param_grads(y)["x"], expected, atol=1e-9, rtol=1e-8)


def test_sparse_gp_inputs_through_a_deterministic_node():
    model, _, _, _, _ = _sparse_model("fitc", "rbf_plus_linear", "linear", deterministic=True)
    gradients = _finite_difference_check(model)
    assert {"x_raw", "log_scale"} <= set(gradients)


def test_sparse_gp_learnable_inputs_and_inducing_points_together():
    model, _, _, _, _ = _sparse_model("vfe", "rbf_plus_linear", "linear", learnable_z=True)
    gradients = _finite_difference_check(model)
    assert {"x", "z"} <= set(gradients)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
@pytest.mark.parametrize("kernel_name", ["matern52", "rbf_plus_linear"])
def test_sparse_gp_learnable_inputs_batched_equals_chain_by_chain(method, kernel_name):
    model, _, _, _, _ = _sparse_model(method, kernel_name, "linear", learnable_z=True)
    _assert_batched_equals_loop(model)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_sparse_gp_learnable_inputs_at_fixed_values_match_fixed_inputs(method):
    learnable, learnable_gp, _, y, _ = _sparse_model(method, "rbf_plus_linear", "linear")
    fixed, fixed_gp, _, _, _ = _sparse_model(method, "rbf_plus_linear", "linear", learnable=False)
    torch.testing.assert_close(learnable_gp.log_pdf(y), fixed_gp.log_pdf(y), atol=1e-12, rtol=0)
    learnable_gradients = learnable.joint_grad_log_prob()
    fixed_gradients = fixed.joint_grad_log_prob()
    assert set(learnable_gradients) == set(fixed_gradients) | {"x"}
    for name, gradient in fixed_gradients.items():
        torch.testing.assert_close(learnable_gradients[name], gradient, atol=1e-12, rtol=0, msg=name)


# ----------------------------------------------------------------------------
# ExactGP with inducing points and several independent outputs sharing X (n_outputs)
# ----------------------------------------------------------------------------


def _multi_output_model(method, n_outputs=3, learnable=True, seed=5):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(9, 2, generator=generator, dtype=DTYPE) * 2.0 - 1.0
    y = torch.stack([torch.sin((r + 1) * x[:, 0]) + 0.3 * r * x[:, 1] for r in range(n_outputs)])
    y = y + 0.1 * torch.randn(y.shape, generator=generator, dtype=DTYPE)
    z0 = torch.rand(4, 2, generator=generator, dtype=DTYPE) * 2.0 - 1.0
    with Model() as model:
        inputs = RandomParameter("x", Normal(0.0, 1.0), shape=tuple(x.shape), initial_value=x.clone()) if learnable else Data("x", x)
        log_l = RandomParameter("log_l", Normal(0.0, 1.0), initial_value=torch.tensor([-0.2], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-1.5], dtype=DTYPE))
        z = RandomParameter("z", Normal(z0, 1.0), shape=tuple(z0.shape), initial_value=z0.clone())
        gp = ExactGP("f", inputs, RBF(Exp("l", log_l), 1.2) + Linear(0.3), inducing_points=z,
                     noise_variance=Exp("noise", log_noise), mean=_mean("linear"), method=method, n_outputs=n_outputs)
        ObservedParameter("y", gp, y)
    return model, gp, x, y, z0


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_multi_output_sparse_gp_is_the_sum_of_single_outputs(method):
    _, gp, x, y, _ = _multi_output_model(method)
    assert gp.event_shape == (3, 9)
    total = 0.0
    for r in range(3):
        with Model():
            single = ExactGP("s", Data("x", x), gp.kernel, inducing_points=gp.inducing_points.constrained_value,
                             noise_variance=gp.noise_variance.constrained_value, mean=gp.mean, method=method)
            total = total + single.log_pdf(y[r])
    torch.testing.assert_close(gp.log_pdf(y), total, atol=1e-10, rtol=1e-10)
    assert gp.log_pdf_grad(y).shape == y.shape


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
def test_multi_output_sparse_gp_gradients_match_finite_differences(method):
    model, _, _, _, _ = _multi_output_model(method)
    gradients = _finite_difference_check(model)
    assert {"x", "z", "log_l", "log_noise", "w", "b"} <= set(gradients)


@pytest.mark.parametrize("method", ["vfe", "fitc"])
def test_multi_output_sparse_gp_batched_equals_chain_by_chain(method):
    model, _, _, _, _ = _multi_output_model(method)
    _assert_batched_equals_loop(model)


def test_multi_output_sparse_gp_sample_and_predictive_shapes():
    _, gp, x, y, z = _multi_output_model("vfe")
    assert gp.sample(5).shape == (5, 3, 9)
    trace = {
        "l": torch.full((1, 2, 1), 0.8, dtype=DTYPE), "noise": torch.full((1, 2, 1), 0.05, dtype=DTYPE),
        "w": torch.zeros(1, 2, 2, dtype=DTYPE), "b": torch.zeros(1, 2, 1, dtype=DTYPE),
        "z": torch.stack([z, z + 0.1]).unsqueeze(0), "x": torch.stack([x, x + 0.05]).unsqueeze(0),
    }
    samples = exact_gp_predictive(gp, trace, y, x[:4], n_samples=6)
    assert samples.shape == (2, 6, 3, 4)
    # Each output's predictive equals a single-output inducing-point ExactGP on that output.
    with Model():
        single = ExactGP("s", Data("x", x), gp.kernel, inducing_points=z, noise_variance=0.05, mean=gp.mean)
    single_trace = {key: trace[key] for key in ("l", "w", "b", "z")}
    single_trace["noise"] = trace["noise"]
    torch.manual_seed(0)
    multi = exact_gp_predictive(gp, trace, y, x[:4], n_samples=40000)
    for r in range(3):
        torch.manual_seed(1)
        reference = exact_gp_predictive(single, {k: v[:, :1] for k, v in single_trace.items()}, y[r], x[:4], n_samples=40000)
        torch.testing.assert_close(multi[0, :, r].mean(0), reference[0].mean(0), atol=0.03, rtol=0)


def test_sparse_gp_predictive_reads_learnable_inputs_from_the_trace():
    _, gp, x, y, z = _sparse_model("vfe", "rbf", "zero")
    trace = {"l": torch.full((1, 1, 2), 0.7, dtype=DTYPE), "noise": torch.full((1, 1, 1), 1e-8, dtype=DTYPE),
             "x": (x + 0.3).reshape(1, 1, *x.shape)}
    torch.manual_seed(0)
    shifted = exact_gp_predictive(gp, trace, y, x + 0.3, n_samples=1)
    with Model():
        reference_gp = ExactGP("r", Data("xr", x + 0.3), RBF(torch.full((2,), 0.7, dtype=DTYPE), 1.3),
                               inducing_points=z, noise_variance=1e-8)
    torch.manual_seed(0)
    reference = exact_gp_predictive(reference_gp, {"unused": torch.zeros(1, 1, 1)}, y, x + 0.3, n_samples=1)
    torch.testing.assert_close(shifted, reference, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------
# KroneckerGP with a learnable grid axis
# ----------------------------------------------------------------------------


def _kronecker_model(mean_kind, learnable=True, factor_kernel="rbf"):
    generator = torch.Generator().manual_seed(8)
    g0 = torch.linspace(-1.0, 1.0, 5, dtype=DTYPE) + 0.05 * torch.rand(5, generator=generator, dtype=DTYPE)
    g1 = torch.linspace(0.0, 2.0, 4, dtype=DTYPE)
    points = torch.cartesian_prod(g0, g1)
    y = (torch.sin(2 * points[:, 0]) * torch.cos(points[:, 1]) + 0.1 * torch.randn(20, generator=generator, dtype=DTYPE)).reshape(5, 4)
    kernels = {"rbf": lambda l, v: RBF(l, v), "rbf_plus_linear": lambda l, v: RBF(l, v) + Linear(0.3, 0.2)}
    with Model() as model:
        axis = RandomParameter("g0", Normal(0.0, 1.0), shape=5, initial_value=g0.clone()) if learnable else Data("g0", g0)
        log_l = RandomParameter("log_l", Normal(0.0, 1.0), initial_value=torch.tensor([-0.4], dtype=DTYPE))
        log_v = RandomParameter("log_v", Normal(0.0, 1.0), initial_value=torch.tensor([0.3], dtype=DTYPE))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-2.0], dtype=DTYPE))
        gp = KroneckerGP("f", [axis, Data("g1", g1)],
                         [kernels[factor_kernel](Exp("l", log_l), Exp("v", log_v)), Matern52(0.9, 1.0)],
                         noise_variance=Exp("noise", log_noise), mean=_mean(mean_kind))
        ObservedParameter("y", gp, y)
    return model, gp, g0, g1, y


@pytest.mark.parametrize("mean_kind", ["zero", "linear"])
@pytest.mark.parametrize("factor_kernel", ["rbf", "rbf_plus_linear"])
def test_kronecker_learnable_axis_gradients_match_finite_differences(mean_kind, factor_kernel):
    model, _, g0, _, _ = _kronecker_model(mean_kind, factor_kernel=factor_kernel)
    gradients = _finite_difference_check(model)
    assert gradients["g0"].shape == g0.shape


@pytest.mark.parametrize("mean_kind", ["zero", "linear"])
def test_kronecker_learnable_axis_gradient_matches_autograd_oracle(mean_kind):
    _, gp, g0, g1, y = _kronecker_model(mean_kind, factor_kernel="rbf_plus_linear")
    leaf = g0.clone().requires_grad_(True)
    k0 = gp.kernels[0](leaf, leaf)
    k1 = gp.kernels[1](g1, g1)
    noise = gp.noise_variance.constrained_value.reshape(()) + gp.jitter
    covariance = torch.kron(k0, k1) + noise * torch.eye(20, dtype=DTYPE)
    points = torch.cat([leaf[torch.arange(5).repeat_interleave(4)].reshape(-1, 1), g1.repeat(5).reshape(-1, 1)], dim=1)
    mean = gp.mean(points)
    reference = torch.distributions.MultivariateNormal(mean.expand(20), covariance).log_prob(y.reshape(-1))
    (expected,) = torch.autograd.grad(reference, leaf)
    torch.testing.assert_close(gp.log_pdf(y), reference.detach(), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(gp.log_pdf_param_grads(y)["g0"], expected, atol=1e-8, rtol=1e-7)


def test_kronecker_multidimensional_learnable_axis():
    generator = torch.Generator().manual_seed(9)
    grid_a = torch.rand(4, 2, generator=generator, dtype=DTYPE)
    grid_b = torch.linspace(0, 1, 3, dtype=DTYPE)
    with Model() as model:
        axis = RandomParameter("a", Normal(0.0, 1.0), shape=(4, 2), initial_value=grid_a.clone())
        w = RandomParameter("w", Normal(0.0, 1.0), shape=3, initial_value=torch.tensor([0.5, -0.2, 0.3], dtype=DTYPE))
        function = KroneckerGP("f", [axis, grid_b], [RBF(torch.tensor([0.6, 0.9], dtype=DTYPE), 1.2), Matern32(0.5, 1.0)],
                               noise_variance=0.1, mean=LinearMean(w, 0.1))
        ObservedParameter("y", function, torch.randn(4, 3, generator=generator, dtype=DTYPE))
    gradients = _finite_difference_check(model)
    assert gradients["a"].shape == (4, 2)


def test_kronecker_learnable_axis_batched_equals_chain_by_chain():
    model, _, _, _, _ = _kronecker_model("linear", factor_kernel="rbf_plus_linear")
    _assert_batched_equals_loop(model)


def test_kronecker_learnable_axis_at_fixed_values_matches_fixed_axis():
    learnable, learnable_gp, _, _, y = _kronecker_model("linear")
    fixed, fixed_gp, _, _, _ = _kronecker_model("linear", learnable=False)
    torch.testing.assert_close(learnable_gp.log_pdf(y), fixed_gp.log_pdf(y), atol=1e-12, rtol=0)
    learnable_gradients = learnable.joint_grad_log_prob()
    fixed_gradients = fixed.joint_grad_log_prob()
    assert set(learnable_gradients) == set(fixed_gradients) | {"g0"}
    for name, gradient in fixed_gradients.items():
        torch.testing.assert_close(learnable_gradients[name], gradient, atol=1e-12, rtol=0, msg=name)


# ----------------------------------------------------------------------------
# ToeplitzGP
# ----------------------------------------------------------------------------


def test_toeplitz_gp_rejects_learnable_inputs():
    x = torch.linspace(0.0, 1.0, 8, dtype=DTYPE)
    with Model():
        inputs = RandomParameter("x", Normal(0.0, 1.0), shape=8, initial_value=x)
        with pytest.raises(TypeError, match="Use ExactGP"):
            ToeplitzGP("f", inputs, RBF(0.5, 1.0), noise_variance=0.1)
        spacing = RandomParameter("h", Normal(0.0, 1.0), initial_value=torch.tensor([1.0], dtype=DTYPE))
        with pytest.raises(TypeError, match="fixed, equally spaced"):
            ToeplitzGP("g", Affine("xs", Data("grid", x), scale=spacing), RBF(0.5, 1.0), noise_variance=0.1)
        # Fixed inputs still work.
        ToeplitzGP("ok", Data("xd", x), RBF(0.5, 1.0), noise_variance=0.1)


def test_exact_gp_predictive_reads_learnable_inputs_from_the_trace():
    x, y, _ = _sparse_data()
    with Model():
        inputs = RandomParameter("x", Normal(0.0, 1.0), shape=tuple(x.shape), initial_value=x.clone())
        gp = ExactGP("f", inputs, RBF(0.7, 1.3), noise_variance=0.05)
    with Model():
        reference_gp = ExactGP("r", Data("xr", x + 0.3), RBF(0.7, 1.3), noise_variance=0.05)
    torch.manual_seed(0)
    shifted = exact_gp_predictive(gp, {"x": (x + 0.3).reshape(1, 1, *x.shape)}, y, x[:3], n_samples=2)
    torch.manual_seed(0)
    reference = exact_gp_predictive(reference_gp, {"unused": torch.zeros(1, 1, 1)}, y, x[:3], n_samples=2)
    torch.testing.assert_close(shifted, reference)
