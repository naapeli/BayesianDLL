"""SparseLatentGP: a whitened inducing-point (DTC) latent GP with learnable inducing inputs."""

import warnings

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP import (
    RBF,
    LatentGP,
    Matern52,
    Periodic,
    SparseLatentGP,
    sparse_latent_gp_predictive,
    stable_cholesky,
)

DTYPE = torch.float64
C = 4


def _data(n=14, d=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=generator, dtype=DTYPE) * 2 - 1
    labels = (x[:, 0] * x[:, -1] + 0.3 * torch.sin(3 * x[:, 0]) > 0).to(DTYPE)
    targets = torch.sin(2 * x[:, 0]) + x[:, -1]
    return x, labels, targets


_KERNELS = {
    "rbf": lambda l, v: RBF(l, v),
    "matern52": lambda l, v: Matern52(l, v),
    "ard_rbf": lambda l, v: RBF(l, v),
    "rbf_times_periodic": lambda l, v: RBF(l, v) * Periodic(1.1, 1.0, 1.7),
}


def _model(kernel_name="rbf", likelihood="bernoulli", m=5, learnable_z=True, log_variance=0.3, jitter=1e-6, seed=0):
    x, labels, targets = _data(seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    z0 = torch.rand(m, 2, generator=generator, dtype=DTYPE) * 2 - 1
    shape = 2 if kernel_name == "ard_rbf" else 1
    with Model() as model:
        log_lengthscale = RandomParameter(
            "log_lengthscale", Normal(-0.5, 0.5), shape=shape,
            initial_value=torch.full((shape,), -0.4, dtype=DTYPE) + 0.1 * torch.arange(shape, dtype=DTYPE),
        )
        log_var = RandomParameter("log_variance", Normal(0.0, 1.0), initial_value=torch.tensor([log_variance], dtype=DTYPE))
        kernel = _KERNELS[kernel_name](Exp("lengthscale", log_lengthscale), Exp("variance", log_var))
        z = RandomParameter("z", Normal(z0, 0.5), shape=(m, 2), initial_value=z0.clone()) if learnable_z else z0
        v = RandomParameter("v", Normal(0.0, 1.0), shape=m,
                            initial_value=torch.randn(m, generator=generator, dtype=DTYPE))
        mean = RandomParameter("mean", Normal(0.0, 1.0), initial_value=torch.tensor([0.2], dtype=DTYPE))
        f = SparseLatentGP("f", Data("x", x), kernel, z, v, mean=mean, jitter=jitter)
        if likelihood == "bernoulli":
            ObservedParameter("y", Bernoulli(logits=f), labels)
        else:
            ObservedParameter("y", Normal(f, 0.3), targets)
    return model, f, x


def _dense_value(f, x, z, v, kernel_values, mean):
    k_mm = f.gp_kernel(z, z, parameter_values=kernel_values)
    k_nm = f.gp_kernel(x, z, parameter_values=kernel_values)
    _, (jitter, _, _) = stable_cholesky(k_mm, jitter=f.gp_jitter, return_jitter=True)
    k_mm = k_mm + jitter * torch.eye(z.shape[0], dtype=x.dtype)
    u = torch.linalg.cholesky(k_mm) @ v  # u ~ N(0, K_mm)
    return mean + k_nm @ torch.linalg.solve(k_mm, u)


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
        torch.testing.assert_close(gradients[name].reshape(-1), numerical, rtol=1e-5, atol=1e-6, msg=f"gradient of {name}")


# ----------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", ["rbf", "matern52", "ard_rbf"])
def test_forward_matches_dense_formula(kernel_name):
    model, f, x = _model(kernel_name)
    values = {name: model.deterministic_params[name].constrained_value for name in ("lengthscale", "variance")}
    expected = _dense_value(f, x, model.params["z"].constrained_value, model.params["v"].constrained_value,
                            values, model.params["mean"].constrained_value)
    torch.testing.assert_close(f.constrained_value, expected)


def test_forward_with_scale_dependent_jitter_matches_dense_formula():
    model, f, x = _model("rbf", log_variance=1.5, jitter=1e-4)  # variance > 1: the jitter scales with K_mm
    values = {name: model.deterministic_params[name].constrained_value for name in ("lengthscale", "variance")}
    expected = _dense_value(f, x, model.params["z"].constrained_value, model.params["v"].constrained_value,
                            values, model.params["mean"].constrained_value)
    torch.testing.assert_close(f.constrained_value, expected)


def test_equals_latent_gp_when_inducing_points_are_the_inputs():
    x = torch.linspace(-1.0, 1.0, 7, dtype=DTYPE)
    white = torch.randn(7, generator=torch.Generator().manual_seed(2), dtype=DTYPE)
    kernel = RBF(0.4, 1.3)
    with Model():
        inputs = Data("x", x)
        v = RandomParameter("v", Normal(0.0, 1.0), shape=7, initial_value=white)
        sparse = SparseLatentGP("sparse", inputs, kernel, x.clone(), v, mean=0.5, jitter=1e-10)
        latent = LatentGP("latent", inputs, kernel, latent=v, mean=0.5, jitter=1e-10)
        # f_sparse = K L⁻ᵀ v = L v - jitter L⁻ᵀ v: equal up to the jitter.
        torch.testing.assert_close(sparse.constrained_value, latent.constrained_value, atol=1e-7, rtol=0)


def test_fixed_one_dimensional_inducing_points_and_default_latent():
    x = torch.linspace(0.0, 3.0, 20, dtype=DTYPE)
    z = torch.linspace(0.0, 3.0, 6, dtype=DTYPE)
    with Model() as model:
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.5))
        f = SparseLatentGP("f", Data("x", x), Matern52(Exp("lengthscale", log_lengthscale), 1.0), Data("z", z))
        ObservedParameter("y", Normal(f, 0.2), torch.sin(2 * x))
    assert "f_white" in model.params and model.params["f_white"].constrained_value.shape == (6,)
    assert f.constrained_value.shape == (20,)
    _finite_difference_check(model)


def test_validates_arguments():
    x = torch.linspace(0.0, 1.0, 5, dtype=DTYPE)
    with Model():
        inputs = Data("x", x)
        v = RandomParameter("v", Normal(0.0, 1.0), shape=4)
        with pytest.raises(ValueError, match="one value per inducing input"):
            SparseLatentGP("f", inputs, RBF(0.5, 1.0), x[:3], v)
        with pytest.raises(TypeError):
            SparseLatentGP("g", inputs, lambda a, b: a, x[:4], v)


# ----------------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("likelihood", ["bernoulli", "normal"])
@pytest.mark.parametrize("kernel_name", ["rbf", "matern52", "ard_rbf", "rbf_times_periodic"])
def test_joint_gradient_matches_finite_differences(kernel_name, likelihood):
    model, _, _ = _model(kernel_name, likelihood)
    gradients = model.joint_grad_log_prob()
    assert set(gradients) == {"log_lengthscale", "log_variance", "z", "v", "mean"}
    assert gradients["z"].shape == (5, 2)
    _finite_difference_check(model)


def test_joint_gradient_with_scale_dependent_jitter_and_fixed_inducing_points():
    model, _, _ = _model("matern52", "normal", learnable_z=False, log_variance=1.5, jitter=1e-4)
    assert "z" not in model.joint_grad_log_prob()
    _finite_difference_check(model)


@pytest.mark.parametrize("kernel_name", ["rbf", "matern52", "ard_rbf"])
def test_vjp_matches_autograd_oracle(kernel_name):
    model, f, x = _model(kernel_name, log_variance=1.2, jitter=1e-3)
    inputs = [value.detach().clone() for value in f.input_values()]
    names = [getattr(node, "name", None) for node in f.inputs]
    cotangent = torch.randn(x.shape[0], generator=torch.Generator().manual_seed(5), dtype=DTYPE)
    hand = f.vjp(cotangent, inputs)

    leaves = {name: value.clone().requires_grad_(True) for name, value in zip(names, inputs) if name not in ("x",)}
    v, z, mean = leaves["v"], leaves["z"], leaves["mean"]
    kernel_values = {"lengthscale": leaves["lengthscale"], "variance": leaves["variance"]}
    def kernel(a, b):
        # Written out directly; sqrt(r² + 1e-30) keeps autograd finite at r = 0.
        scaled = (a.unsqueeze(-2) - b.unsqueeze(-3)) / kernel_values["lengthscale"].reshape(-1)
        r = torch.sqrt(scaled.square().sum(-1) + 1e-30)
        if kernel_name == "matern52":
            s = 5 ** 0.5 * r
            return kernel_values["variance"] * (1 + s + s ** 2 / 3) * torch.exp(-s)
        return kernel_values["variance"] * torch.exp(-0.5 * r ** 2)

    k_mm, k_nm = kernel(z, z), kernel(x, z)
    torch.testing.assert_close(k_nm.detach(), f.gp_kernel(x, z.detach(), parameter_values={k: t.detach() for k, t in kernel_values.items()}))
    scale = torch.clamp(torch.diagonal(k_mm).max(), min=1.0)
    factor = torch.linalg.cholesky(k_mm + f.gp_jitter * scale * torch.eye(z.shape[0], dtype=DTYPE))
    value = mean + k_nm @ torch.linalg.solve_triangular(factor.T, v.unsqueeze(-1), upper=True).squeeze(-1)
    torch.testing.assert_close(value.detach(), f.constrained_value)
    oracle = torch.autograd.grad((value * cotangent).sum(), list(leaves.values()))
    for (name, _), expected in zip(leaves.items(), oracle):
        torch.testing.assert_close(torch.as_tensor(hand[name]).reshape(expected.shape), expected, msg=f"VJP of {name}")


# ----------------------------------------------------------------------------
# Vectorized chains and dtype
# ----------------------------------------------------------------------------


def _assert_batched_equals_loop(model, seed=0, scale=0.3):
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
    assert set(batched_grads) == set(loop_grads)
    for name, gradient in loop_grads.items():
        assert batched_grads[name].shape == gradient.shape, name
        torch.testing.assert_close(batched_grads[name], gradient, atol=1e-10, rtol=1e-10, msg=f"gradient of {name}")


@pytest.mark.parametrize("likelihood", ["bernoulli", "normal"])
@pytest.mark.parametrize("kernel_name", ["rbf", "matern52", "ard_rbf", "rbf_times_periodic"])
def test_batched_matches_chain_by_chain(kernel_name, likelihood):
    model, _, _ = _model(kernel_name, likelihood)
    _assert_batched_equals_loop(model)


def test_batched_with_scale_dependent_jitter_and_fixed_inducing_points():
    model, _, _ = _model("rbf", "bernoulli", learnable_z=False, log_variance=1.5, jitter=1e-4)
    _assert_batched_equals_loop(model)


def test_batched_with_only_the_whitened_values_latent():
    x, labels, _ = _data()
    with Model() as model:
        v = RandomParameter("v", Normal(0.0, 1.0), shape=4)
        f = SparseLatentGP("f", Data("x", x), RBF(0.6, 1.5), x[:4].clone(), v)
        ObservedParameter("y", Bernoulli(logits=f), labels)
    _assert_batched_equals_loop(model)


def test_vectorized_nuts_runs_without_fallback():
    model, _, _ = _model("rbf", "bernoulli", m=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message=".*cannot be evaluated for all chains.*")
        trace = model.sample(30, 30, n_chains=3, chain_method="vectorized", max_depth=4, seed=0, progress_bar=False,
                             check_convergence=False)
    assert trace["z"].shape == (3, 30, 4, 2)
    assert torch.isfinite(trace["f"]).all()


def test_follows_model_to_float32():
    model, f, _ = _model("ard_rbf", "bernoulli", jitter=1e-4)
    reference = model.joint_grad_log_prob()
    model.to(dtype=torch.float32)
    assert model.model_log_prob().dtype == torch.float32
    assert f.constrained_value.dtype == torch.float32
    gradients = model.joint_grad_log_prob()
    for name, gradient in gradients.items():
        assert gradient.dtype == torch.float32, name
        torch.testing.assert_close(gradient.double(), reference[name], rtol=1e-3, atol=1e-3)


def test_fixed_tensor_inducing_points_follow_model_to_float32():
    model, f, _ = _model("rbf", "normal", learnable_z=False, jitter=1e-4)
    model.to(dtype=torch.float32)
    assert f.constrained_value.dtype == torch.float32
    assert all(g.dtype == torch.float32 for g in model.joint_grad_log_prob().values())


# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------


def _trace_from_values(model, n_draws=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    trace = {}
    draws = {name: [] for name in [*model.params, *model.deterministic_params]}
    for _ in range(n_draws):
        for parameter in model.params.values():
            u = parameter.unconstrained_value
            parameter.set_unconstrained_value(u + 0.1 * torch.randn(u.shape, generator=generator, dtype=u.dtype))
        for name in draws:
            node = model.params.get(name) or model.deterministic_params[name]
            draws[name].append(node.constrained_value.clone())
    for name, values in draws.items():
        trace[name] = torch.stack(values).unsqueeze(0)  # (1 chain, n_draws, *shape)
    return trace


@pytest.mark.parametrize("kernel_name", ["rbf", "ard_rbf"])
def test_predictive_at_training_inputs_reproduces_the_traced_function(kernel_name):
    model, f, x = _model(kernel_name)
    trace = _trace_from_values(model)
    predictions = sparse_latent_gp_predictive(f, trace, x, n_samples=2, chunk_size=2)
    assert predictions.shape == (3, 2, x.shape[0])
    torch.testing.assert_close(predictions[:, 0], trace["f"][0])
    torch.testing.assert_close(predictions[:, 1], trace["f"][0])


def test_predictive_conditional_variance_vanishes_at_the_inducing_inputs():
    model, f, _ = _model("rbf", learnable_z=False)
    trace = _trace_from_values(model, n_draws=2)
    z = f.gp_inducing_points
    far = torch.tensor([[5.0, 5.0]], dtype=DTYPE)
    torch.manual_seed(0)
    at_z = sparse_latent_gp_predictive(f, trace, z, n_samples=200, conditional_variance=True)
    deterministic = sparse_latent_gp_predictive(f, trace, z)
    torch.testing.assert_close(at_z.std(dim=1), torch.zeros_like(at_z[:, 0]), atol=1e-2, rtol=0)
    torch.testing.assert_close(at_z.mean(dim=1), deterministic[:, 0], atol=1e-2, rtol=0)
    far_draws = sparse_latent_gp_predictive(f, trace, far, n_samples=2000, conditional_variance=True)
    variance = trace["variance"].reshape(-1)
    torch.testing.assert_close(far_draws.var(dim=1).reshape(-1), variance, rtol=0.15, atol=0)


def test_predictive_with_guide_samples():
    from BayesianDLL import Guide
    from BayesianDLL.Variational import BBVI

    model, f, x = _model("rbf", m=4)
    guide = Guide(model, default_family="mean_field")
    BBVI(model, guide, epochs=5, lr=0.01, verbose=False)
    trace = guide.sample(10, n_chains=2)
    grid = torch.rand(7, 2, dtype=DTYPE)
    predictions = sparse_latent_gp_predictive(f, trace, grid)
    assert predictions.shape == (20, 1, 7)
    assert torch.isfinite(predictions).all()


def test_a_per_input_vector_mean_is_rejected():
    x, _, _ = _data(n=5)
    with Model():
        with pytest.raises(ValueError, match="mean function"):
            SparseLatentGP("f", Data("x", x), RBF(0.5, 1.0), x[:3].clone(), mean=torch.linspace(-1.0, 1.0, 5, dtype=DTYPE))
