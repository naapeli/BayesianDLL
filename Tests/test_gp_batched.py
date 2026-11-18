"""Vectorized chains: every GP node is evaluated as one batched tensor computation.

For each node the batched log density and gradient (``Model.batch`` in
tensor mode) must equal the chain-by-chain evaluation to 1e-10 in float64,
without falling back to the loop mode.
"""

import time
import warnings

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP import (
    RBF,
    Constant,
    ExactGP,
    Kernel,
    KroneckerGP,
    LatentGP,
    Linear,
    Matern,
    Matern32,
    Matern52,
    Periodic,
    RationalQuadratic,
    SparseGP,
    ToeplitzGP,
    WhiteNoise,
    batched_stable_cholesky,
)

DTYPE = torch.float64
C = 5


def _random_unconstrained(model, n_chains, seed=0, scale=0.3):
    generator = torch.Generator().manual_seed(seed)
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        noise = torch.randn((n_chains, *u.shape), generator=generator, dtype=u.dtype)
        values[name] = u.unsqueeze(0) + scale * noise
    return values


def _assert_batched_equals_loop(model, seed=0, scale=0.3):
    values = _random_unconstrained(model, C, seed, scale)
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
    return batched_lp, batched_grads


def _hyperparameters(shape=1):
    log_lengthscale = RandomParameter("log_lengthscale", Normal(-0.5, 0.5), shape=shape)
    log_variance = RandomParameter("log_variance", Normal(0.0, 0.5))
    log_noise = RandomParameter("log_noise", Normal(-2.0, 0.5))
    return Exp("lengthscale", log_lengthscale), Exp("variance", log_variance), Exp("noise", log_noise)


_KERNELS = {
    "rbf": lambda l, v: RBF(l, v),
    "matern12": lambda l, v: Matern(l, v, nu=0.5),
    "matern_general": lambda l, v: Matern(l, v, nu=1.3),
    "periodic": lambda l, v: Periodic(l, v, 1.4),
    "rq_plus_linear": lambda l, v: RationalQuadratic(l, v, 1.5) + Linear(0.3, 0.1),
    "product_with_constant": lambda l, v: Matern52(l, 1.0) * Constant(v) + WhiteNoise(0.01),
}


# ----------------------------------------------------------------------------
# Kernels in batched form
# ----------------------------------------------------------------------------


_BATCH_KERNELS = {
    "rbf": lambda p: RBF(p["l"], p["v"]),
    "periodic": lambda p: Periodic(p["l"], p["v"], p["p"]),
    "periodic_scalar": lambda p: Periodic(p["s"], p["v"], p["p"]),
    "matern32": lambda p: Matern32(p["l"], p["v"]),
    "matern_general": lambda p: Matern(p["l"], p["v"], nu=1.7),
    "rational_quadratic": lambda p: RationalQuadratic(p["l"], p["v"], p["a"]),
    "linear": lambda p: Linear(p["v"], p["c"]),
    "constant_white": lambda p: Constant(p["v"]) + WhiteNoise(p["a"]),
    "product": lambda p: RBF(p["l"], p["v"]) * Periodic(p["s"], 1.0, p["p"]),
}


class _Name:
    def __init__(self, name):
        self.name = name


@pytest.mark.parametrize("name", list(_BATCH_KERNELS))
def test_kernel_batched_methods_match_per_element_calls(name):
    nodes = {key: _Name(key) for key in ("l", "v", "p", "s", "a", "c")}
    kernel = _BATCH_KERNELS[name](nodes)
    generator = torch.Generator().manual_seed(1)
    batch = 3
    values = {
        "l": 0.5 + torch.rand(batch, 2, generator=generator, dtype=DTYPE),
        "v": 0.5 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
        "p": 1.0 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
        "s": 0.5 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
        "a": 0.5 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
        "c": torch.randn(batch, 2, generator=generator, dtype=DTYPE),
    }
    values = {p.name: values[p.name] for p in kernel.parameters}
    x1 = torch.randn(batch, 4, 2, generator=generator, dtype=DTYPE)
    x2 = torch.randn(5, 2, generator=generator, dtype=DTYPE)  # shared by the batch
    cotangent = torch.randn(batch, 4, 5, generator=generator, dtype=DTYPE)
    square = torch.randn(batch, 4, 4, generator=generator, dtype=DTYPE)
    covariance = kernel.batch_evaluate(x1, x2.unsqueeze(0), values)
    derivatives = kernel.batch_derivatives(x1, x2.unsqueeze(0), values)
    diagonal = kernel.batch_diag(x1, values)
    diagonal_derivatives = kernel.batch_diag_derivatives(x1, values)
    vjp = kernel.batch_input_vjp(x1, x2.unsqueeze(0), cotangent, values)
    symmetric = kernel.batch_input_vjp_symmetric(x1, square, values)
    for b in range(batch):
        single = {key: value[b] for key, value in values.items()}
        torch.testing.assert_close(covariance[b], kernel(x1[b], x2, parameter_values=single))
        for key, derivative in kernel.derivatives(x1[b], x2, parameter_values=single).items():
            torch.testing.assert_close(derivatives[key].expand(batch, *derivative.shape)[b], derivative)
        torch.testing.assert_close(diagonal.expand(batch, -1)[b], kernel.diag(x1[b], parameter_values=single))
        for key, derivative in kernel.diag_derivatives(x1[b], parameter_values=single).items():
            torch.testing.assert_close(diagonal_derivatives[key].expand(batch, *derivative.shape)[b], derivative)
        torch.testing.assert_close(vjp[b], kernel.input_vjp(x1[b], x2, cotangent[b], parameter_values=single))
        torch.testing.assert_close(symmetric[b], kernel.input_vjp_symmetric(x1[b], square[b], parameter_values=single))


def test_user_kernel_with_only_unbatched_methods_is_looped_inside_the_batch():
    class Scaled(Kernel):
        def __init__(self, scale):
            super().__init__(scale)
            self.scale = scale
            self.inner = RBF(0.7, 1.0)

        def evaluate(self, x1, x2, parameter_values=None):
            return parameter_values[self.scale.name].reshape(()) * self.inner(x1, x2)

        def derivatives(self, x1, x2, parameter_values=None):
            return {self.scale.name: self.inner(x1, x2).unsqueeze(0)}

    x = torch.linspace(0, 1, 6, dtype=DTYPE)
    y = torch.sin(4 * x)
    with Model() as model:
        log_scale = RandomParameter("log_scale", Normal(0.0, 1.0))
        ObservedParameter("y", ExactGP("f", Data("x", x), Scaled(Exp("scale", log_scale)), noise_variance=0.1), y)
    _assert_batched_equals_loop(model)


def test_batched_stable_cholesky_uses_a_jitter_per_matrix():
    good = torch.eye(3, dtype=DTYPE) * 2.0
    singular = torch.ones(3, 3, dtype=DTYPE) * 4.0  # needs more jitter than the first try
    factor, (jitter, scale, index) = batched_stable_cholesky(torch.stack([good, singular, good]), jitter=1e-12)
    assert jitter[0] == jitter[2] and jitter[1] > jitter[0]
    for b, matrix in enumerate([good, singular, good]):
        single, (single_jitter, _, _) = batched_stable_cholesky(matrix.unsqueeze(0), jitter=1e-12)
        torch.testing.assert_close(factor[b], single[0])
        torch.testing.assert_close(jitter[b], single_jitter[0])


# ----------------------------------------------------------------------------
# GP nodes
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", list(_KERNELS))
def test_exact_gp_batched_matches_chain_by_chain(kernel_name):
    generator = torch.Generator().manual_seed(2)
    x = torch.rand(12, 2, generator=generator, dtype=DTYPE)
    y = torch.sin(3 * x[:, 0]) + x[:, 1]
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters(shape=2)
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        f = ExactGP("f", Data("x", x), _KERNELS[kernel_name](lengthscale, variance), noise_variance=noise, mean=mean)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model)


def test_exact_gp_batched_with_scale_dependent_jitter():
    x = torch.linspace(0, 1, 10, dtype=DTYPE)
    y = torch.sin(6 * x)
    with Model() as model:
        log_variance = RandomParameter("log_variance", Normal(1.5, 0.5))  # variance > 1: jitter depends on K
        f = ExactGP("f", Data("x", x), RBF(0.3, Exp("variance", log_variance)), noise_variance=0.05, jitter=1e-4)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model)


@pytest.mark.parametrize("kernel_name", ["rbf", "matern_general", "rq_plus_linear"])
def test_latent_gp_batched_matches_chain_by_chain(kernel_name):
    x = torch.linspace(-1.0, 1.0, 10, dtype=DTYPE)
    labels = (torch.sin(3 * x) > 0).to(DTYPE)
    with Model() as model:
        lengthscale, variance, _ = _hyperparameters()
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        f = LatentGP("f", Data("x", x), _KERNELS[kernel_name](lengthscale, variance), mean=mean)
        ObservedParameter("y", Bernoulli(logits=f), labels)
    _assert_batched_equals_loop(model, scale=0.5)


def test_latent_gp_batched_with_regression_likelihood_and_ard():
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(9, 2, generator=generator, dtype=DTYPE)
    y = torch.cos(2 * x[:, 0]) * x[:, 1]
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters(shape=2)
        f = LatentGP("f", Data("x", x), RBF(lengthscale, variance), jitter=1e-5)
        ObservedParameter("y", Normal(f, 0.2), y)
    _assert_batched_equals_loop(model)


@pytest.mark.parametrize("method", ["vfe", "dtc", "fitc"])
@pytest.mark.parametrize("kernel_name", ["rbf", "periodic", "rq_plus_linear"])
def test_sparse_gp_batched_with_learnable_inducing_points(method, kernel_name):
    generator = torch.Generator().manual_seed(4)
    x = torch.rand(20, 2, generator=generator, dtype=DTYPE)
    y = torch.sin(3 * x[:, 0]) - x[:, 1]
    z0 = torch.rand(5, 2, generator=generator, dtype=DTYPE)
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters(shape=2)
        z = RandomParameter("z", Normal(z0, 0.5), shape=(5, 2), initial_value=z0)
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        f = SparseGP("f", Data("x", x), _KERNELS[kernel_name](lengthscale, variance), inducing_points=z,
                     noise_variance=noise, mean=mean, method=method)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model, scale=0.1)


def test_sparse_gp_batched_with_fixed_one_dimensional_inducing_points():
    x = torch.linspace(0, 3, 30, dtype=DTYPE)
    y = torch.sin(2 * x)
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters()
        z = RandomParameter("z", Normal(torch.linspace(0, 3, 6, dtype=DTYPE), 0.3), shape=6,
                            initial_value=torch.linspace(0, 3, 6, dtype=DTYPE))
        f = SparseGP("f", Data("x", x), Matern52(lengthscale, variance), inducing_points=z, noise_variance=noise)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model, scale=0.1)


def test_kronecker_gp_batched_matches_chain_by_chain():
    grids = [torch.linspace(0, 1, 6, dtype=DTYPE), torch.linspace(0, 2, 5, dtype=DTYPE), torch.linspace(0, 1, 3, dtype=DTYPE)]
    points = torch.cartesian_prod(*grids)
    y = (torch.sin(3 * points[:, 0]) * torch.cos(points[:, 1]) + points[:, 2]).reshape(6, 5, 3)
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters()
        log_l2 = RandomParameter("log_l2", Normal(0.0, 0.5))
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        f = KroneckerGP("f", [Data(f"g{p}", g) for p, g in enumerate(grids)],
                        [RBF(lengthscale, variance), Matern32(Exp("l2", log_l2), 1.0), Periodic(0.8, 1.0, 1.3)],
                        noise_variance=noise, mean=mean)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model)


@pytest.mark.parametrize("kernel_name", ["rbf", "matern_general", "rq_plus_linear"])
def test_toeplitz_gp_batched_matches_chain_by_chain(kernel_name):
    if kernel_name == "rq_plus_linear":
        factory = lambda l, v: RationalQuadratic(l, v, 1.5) + Periodic(0.7, 0.3, 1.1)
    else:
        factory = _KERNELS[kernel_name]
    x = torch.linspace(0, 5, 40, dtype=DTYPE)
    y = torch.sin(2 * x) + 0.1 * torch.cos(7 * x)
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters()
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        f = ToeplitzGP("f", Data("x", x), factory(lengthscale, variance), noise_variance=noise, mean=mean)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model)


def test_toeplitz_gp_batched_with_scale_dependent_jitter():
    x = torch.linspace(0, 5, 30, dtype=DTYPE)
    y = torch.sin(2 * x)
    with Model() as model:
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))  # chains above and below one
        f = ToeplitzGP("f", Data("x", x), RBF(0.8, Exp("variance", log_variance)), noise_variance=0.05, jitter=1e-4)
        ObservedParameter("y", f, y)
    _assert_batched_equals_loop(model, scale=1.0)


@pytest.mark.integration
def test_vectorized_nuts_runs_gp_nodes_without_fallback():
    x = torch.linspace(0, 1, 15, dtype=DTYPE)
    y = torch.sin(6 * x)
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters()
        ObservedParameter("y", ExactGP("f", Data("x", x), RBF(lengthscale, variance), noise_variance=noise), y)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*cannot be evaluated for all chains.*")
        trace = model.sample(10, 10, n_chains=4, chain_method="vectorized", progress_bar=False,
                             check_convergence=False, seed=0)
    assert trace["log_lengthscale"].shape[:2] == (4, 10)


# ----------------------------------------------------------------------------
# Timing: vectorized chains against the chain-by-chain loop
# ----------------------------------------------------------------------------


def _time(model, values, mode, repeats=2):
    with model.batched(8, values=values, mode=mode, check=False):
        model.joint_grad_log_prob()
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            model.model_log_prob()
            model.joint_grad_log_prob()
            best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("node,n", [("toeplitz", 1000), ("exact", 300)])
def test_timing_vectorized_against_loop(node, n):
    x = torch.linspace(0, 20, n, dtype=DTYPE)
    y = torch.sin(x) + 0.1 * torch.randn(n, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    cls = ToeplitzGP if node == "toeplitz" else ExactGP
    with Model() as model:
        lengthscale, variance, noise = _hyperparameters()
        ObservedParameter("y", cls("f", Data("x", x), Matern52(lengthscale, variance), noise_variance=noise), y)
    values = _random_unconstrained(model, 8, scale=0.1)
    tensor_time = _time(model, values, "tensor")
    loop_time = _time(model, values, "loop")
    print(f"{cls.__name__} n={n}, C=8: vectorized {1000 * tensor_time:.0f} ms, "
          f"chain-by-chain loop {1000 * loop_time:.0f} ms (x{loop_time / tensor_time:.1f})")


def test_plain_callable_with_batched_inducing_points_falls_back_to_the_loop_mode():
    """The one documented unsupported case: a plain callable cannot evaluate per-chain inputs."""
    x = torch.linspace(0, 1, 8, dtype=DTYPE)
    y = torch.sin(4 * x)
    z0 = torch.linspace(0, 1, 3, dtype=DTYPE)
    with Model() as model:
        z = RandomParameter("z", Normal(z0, 0.3), shape=3, initial_value=z0)
        kernel = RBF(0.4, 1.0) + (lambda a, b: 0.1 * torch.ones(a.shape[0], b.shape[0], dtype=DTYPE))
        ObservedParameter("y", SparseGP("f", Data("x", x), kernel, inducing_points=z, noise_variance=0.1), y)
    with pytest.warns(UserWarning, match="chain by chain"):
        with model.batched(3, mode="auto"):
            assert model._batch_mode == "loop"
