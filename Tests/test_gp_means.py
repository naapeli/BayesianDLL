"""GP mean functions: values, hand-written derivatives, use in every GP class and in the predictives."""

import warnings

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    RBF,
    ConstantMean,
    ExactGP,
    KroneckerGP,
    LatentGP,
    LinearMean,
    Matern52,
    MeanFunction,
    SparseGP,
    SparseLatentGP,
    SumMean,
    ToeplitzGP,
    ZeroMean,
    exact_gp_predictive,
    gp_predictive,
    kronecker_gp_predictive,
    sparse_gp_predictive,
    sparse_latent_gp_predictive,
    stable_cholesky,
    toeplitz_gp_predictive,
)

DTYPE = torch.float64
C = 4


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


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
    return gradients


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
    torch.testing.assert_close(batched_lp, loop_lp, atol=1e-10, rtol=1e-10)
    for name, gradient in loop_grads.items():
        assert batched_grads[name].shape == gradient.shape, name
        torch.testing.assert_close(batched_grads[name], gradient, atol=1e-10, rtol=1e-10, msg=f"gradient of {name}")


def _make_mean(kind, features):
    """A mean function with learnable parameters (created in the active model)."""
    if kind == "constant":
        return ConstantMean(RandomParameter("c", Normal(0.0, 1.0), initial_value=torch.tensor([0.3], dtype=DTYPE)))
    if kind == "linear":
        w = RandomParameter("w", Normal(0.0, 1.0), shape=features,
                            initial_value=torch.linspace(0.4, -0.6, features, dtype=DTYPE))
        b = RandomParameter("b", Normal(0.0, 1.0), initial_value=torch.tensor([-0.2], dtype=DTYPE))
        return LinearMean(w, b)
    if kind == "linear_deterministic":  # coefficients through a DeterministicParameter, plus a constant part
        log_w = RandomParameter("log_w", Normal(0.0, 1.0), shape=features,
                                initial_value=torch.linspace(-0.3, 0.2, features, dtype=DTYPE))
        return LinearMean(Exp("w", log_w), 0.1) + ConstantMean(
            RandomParameter("c", Normal(0.0, 1.0), initial_value=torch.tensor([0.25], dtype=DTYPE)))
    raise ValueError(kind)


_MEAN_KINDS = ["constant", "linear", "linear_deterministic"]


def _data(n=9, features=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, features, generator=generator, dtype=DTYPE) * 2 - 1
    y = torch.sin(2 * x[:, 0]) + 0.5 * x[:, -1] + 0.1 * torch.randn(n, generator=generator, dtype=DTYPE)
    return x, y


def _kernel():
    log_l = RandomParameter("log_l", Normal(0.0, 1.0), initial_value=torch.tensor([-0.3], dtype=DTYPE))
    return RBF(Exp("l", log_l), 1.1)


def _noise():
    log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0), initial_value=torch.tensor([-1.5], dtype=DTYPE))
    return Exp("noise", log_noise)


def _model(gp_class, mean_kind, learnable_inputs=False, learnable_z=False, method="vfe"):
    if gp_class in ("toeplitz", "kronecker"):
        return _structured_model(gp_class, mean_kind)
    x, y = _data()
    with Model() as model:
        if learnable_inputs:
            inputs = RandomParameter("x_in", Normal(x, 0.3), shape=tuple(x.shape), initial_value=x.clone())
        else:
            inputs = Data("x", x)
        mean = _make_mean(mean_kind, x.shape[1])
        z0 = x[:4].clone() + 0.05
        z = RandomParameter("z", Normal(z0, 0.5), shape=tuple(z0.shape), initial_value=z0.clone()) if learnable_z else z0
        if gp_class == "exact":
            ObservedParameter("y", ExactGP("f", inputs, _kernel(), noise_variance=_noise(), mean=mean), y)
        elif gp_class == "sparse":
            ObservedParameter("y", SparseGP("f", inputs, _kernel(), z, noise_variance=_noise(), mean=mean, method=method), y)
        elif gp_class == "latent":
            f = LatentGP("f", inputs, _kernel(), mean=mean, jitter=1e-8)
            ObservedParameter("y", Normal(f, 0.3), y)
        elif gp_class == "sparse_latent":
            f = SparseLatentGP("f", inputs, _kernel(), z, mean=mean, jitter=1e-8)
            ObservedParameter("y", Normal(f, 0.3), y)
        else:
            raise ValueError(gp_class)
    for name, parameter in model.params.items():
        if name.endswith("_white"):
            parameter.set_unconstrained_value(torch.linspace(-0.8, 0.9, parameter.unconstrained_value.numel(), dtype=DTYPE))
    return model


def _structured_model(gp_class, mean_kind):
    if gp_class == "toeplitz":
        x = torch.linspace(-1.0, 2.0, 12, dtype=DTYPE)
        y = torch.sin(2 * x) + 0.3 * x
        with Model() as model:
            mean = _make_mean(mean_kind, 1)
            ObservedParameter("y", ToeplitzGP("f", Data("x", x), _kernel(), noise_variance=_noise(), mean=mean), y)
        return model
    grids = [torch.linspace(-1.0, 1.0, 4, dtype=DTYPE), torch.linspace(0.0, 2.0, 3, dtype=DTYPE)]
    points = torch.cartesian_prod(*grids)
    y = torch.sin(2 * points[:, 0]) + 0.4 * points[:, 1]
    with Model() as model:
        mean = _make_mean(mean_kind, 2)
        gp = KroneckerGP("f", [Data("g0", grids[0]), Data("g1", grids[1])], [_kernel(), Matern52(0.8, 1.0)],
                         noise_variance=_noise(), mean=mean)
        ObservedParameter("y", gp, y.reshape(4, 3))
    return model


# ----------------------------------------------------------------------------
# Mean functions on their own
# ----------------------------------------------------------------------------


def _named_parameters(features):
    with Model():
        w = RandomParameter("w", Normal(0.0, 1.0), shape=features)
        b = RandomParameter("b", Normal(0.0, 1.0))
        c = RandomParameter("c", Normal(0.0, 1.0))
    return w, b, c


_UNIT_MEANS = {
    "zero": lambda w, b, c: ZeroMean(),
    "constant": lambda w, b, c: ConstantMean(c),
    "constant_number": lambda w, b, c: ConstantMean(1.7),
    "linear": lambda w, b, c: LinearMean(w, b),
    "linear_fixed": lambda w, b, c: LinearMean(torch.tensor([0.5, -1.0, 2.0][: w.unconstrained_value.numel()], dtype=DTYPE), 0.3),
    "sum": lambda w, b, c: LinearMean(w) + c + ConstantMean(b) + 0.5,
    "shared": lambda w, b, c: LinearMean(w, c) + ConstantMean(c),  # one node in two parts
}


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("mean_name", list(_UNIT_MEANS))
def test_mean_values_and_derivatives_match_autograd_and_finite_differences(mean_name, features):
    w, b, c = _named_parameters(features)
    mean = _UNIT_MEANS[mean_name](w, b, c)
    generator = torch.Generator().manual_seed(features)
    x = torch.randn(6, features, generator=generator, dtype=DTYPE)
    if features == 1:
        x = x[:, 0]  # 1-D inputs with a scalar-like coefficient
    values = {
        "w": torch.randn(features, generator=generator, dtype=DTYPE),
        "b": torch.randn(1, generator=generator, dtype=DTYPE),
        "c": torch.randn(1, generator=generator, dtype=DTYPE),
    }
    cotangent = torch.randn(6, generator=generator, dtype=DTYPE)
    m = mean(x, parameter_values=values)
    assert m.shape == (6,)

    # Reference value.
    features_matrix = x.reshape(6, -1)
    expected = {
        "zero": lambda: torch.zeros(6, dtype=DTYPE),
        "constant": lambda: values["c"].expand(6),
        "constant_number": lambda: torch.full((6,), 1.7, dtype=DTYPE),
        "linear": lambda: features_matrix @ values["w"] + values["b"],
        "linear_fixed": lambda: features_matrix @ torch.tensor([0.5, -1.0, 2.0][:features], dtype=DTYPE) + 0.3,
        "sum": lambda: features_matrix @ values["w"] + values["c"] + values["b"] + 0.5,
        "shared": lambda: features_matrix @ values["w"] + 2 * values["c"],
    }[mean_name]()
    torch.testing.assert_close(m, expected)

    # Parameter derivatives against autograd and finite differences.
    leaves = {name: value.clone().requires_grad_(True) for name, value in values.items()}
    _backward((cotangent * mean(x, parameter_values=leaves)).sum())
    derivatives = mean.derivatives(x, parameter_values=values)
    for parameter in mean.parameters:
        name = parameter.name
        analytic = (derivatives[name] * cotangent).sum(-1).reshape(-1)
        torch.testing.assert_close(analytic, leaves[name].grad.reshape(-1), msg=f"d/d{name}")
        for index in range(values[name].numel()):
            shifted = {k: v.clone() for k, v in values.items()}
            shifted[name].view(-1)[index] += 1e-6
            upper = (cotangent * mean(x, parameter_values=shifted)).sum()
            shifted[name].view(-1)[index] -= 2e-6
            lower = (cotangent * mean(x, parameter_values=shifted)).sum()
            torch.testing.assert_close(analytic[index], (upper - lower) / 2e-6, rtol=1e-6, atol=1e-8)
    assert set(derivatives) == {p.name for p in mean.parameters}

    # Input VJP against autograd.
    x_leaf = x.clone().requires_grad_(True)
    _backward((cotangent * mean(x_leaf, parameter_values=values)).sum())
    vjp = mean.input_vjp(x, cotangent, parameter_values=values)
    assert vjp.shape == features_matrix.shape
    expected_vjp = torch.zeros_like(x) if x_leaf.grad is None else x_leaf.grad
    torch.testing.assert_close(vjp.reshape(-1), expected_vjp.reshape(-1))


def _backward(output):
    """Autograd oracle; a mean that does not depend on the leaves has no graph (zero gradient)."""
    if output.requires_grad:
        output.backward()


def test_batched_mean_matches_per_element_evaluation():
    w, b, c = _named_parameters(2)
    mean = LinearMean(w, b) + ConstantMean(c)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(3, 5, 2, generator=generator, dtype=DTYPE)
    values = {"w": torch.randn(3, 2, generator=generator, dtype=DTYPE),
              "b": torch.randn(3, 1, generator=generator, dtype=DTYPE),
              "c": torch.randn(1, 1, generator=generator, dtype=DTYPE)}  # shared by the batch
    cotangent = torch.randn(3, 5, generator=generator, dtype=DTYPE)
    batched = mean.batch_evaluate(x, values)
    derivatives = mean.batch_derivatives(x, values)
    vjp = mean.batch_input_vjp(x, cotangent, values)
    for index in range(3):
        single = {"w": values["w"][index], "b": values["b"][index], "c": values["c"][0]}
        torch.testing.assert_close(batched[index], mean(x[index], parameter_values=single))
        torch.testing.assert_close(vjp[index], mean.input_vjp(x[index], cotangent[index], parameter_values=single))
        for name, derivative in mean.derivatives(x[index], parameter_values=single).items():
            batch_derivative = derivatives[name]
            torch.testing.assert_close(batch_derivative[index if batch_derivative.shape[0] > 1 else 0], derivative)


class _UnbatchedQuadratic(MeanFunction):
    """A user mean that only implements the unbatched API (the batched defaults loop)."""

    def __init__(self, scale):
        super().__init__(scale)
        self.scale = scale

    def evaluate(self, x, parameter_values=None):
        scale = parameter_values[self.scale.name].reshape(())
        return scale * torch.as_tensor(x).reshape(-1).square()

    def derivatives(self, x, parameter_values=None):
        return {self.scale.name: torch.as_tensor(x).reshape(1, -1).square()}

    def input_vjp(self, x, cotangent, parameter_values=None):
        scale = parameter_values[self.scale.name].reshape(())
        return (2 * scale * cotangent * torch.as_tensor(x).reshape(-1)).reshape(-1, 1)


def test_user_mean_function_with_the_unbatched_api_in_an_exact_gp():
    x, y = _data(features=1)
    with Model() as model:
        scale = RandomParameter("scale", Normal(0.0, 1.0), initial_value=torch.tensor([0.4], dtype=DTYPE))
        inputs = RandomParameter("x_in", Normal(x[:, 0], 0.3), shape=x.shape[0], initial_value=x[:, 0].clone())
        ObservedParameter("y", ExactGP("f", inputs, _kernel(), noise_variance=_noise(), mean=_UnbatchedQuadratic(scale)), y)
    _finite_difference_check(model)


def test_linear_mean_checks_the_number_of_coefficients():
    mean = LinearMean(torch.tensor([1.0, 2.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="one value per input feature"):
        mean(torch.zeros(4, 3, dtype=DTYPE))
    with pytest.raises(ValueError, match="scalar"):
        LinearMean(1.0, torch.tensor([1.0, 2.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="scalar"):
        ConstantMean(torch.tensor([1.0, 2.0], dtype=DTYPE))


def test_numbers_and_sums_convert_to_mean_functions():
    mean = 1.5 + LinearMean(2.0)
    assert isinstance(mean, SumMean)
    torch.testing.assert_close(mean(torch.tensor([0.0, 1.0], dtype=DTYPE)), torch.tensor([1.5, 3.5], dtype=DTYPE))
    assert mean.parameters == ()


# ----------------------------------------------------------------------------
# Gradients of every GP class
# ----------------------------------------------------------------------------


_GP_CASES = [
    ("exact", {}),
    ("exact", {"learnable_inputs": True}),
    ("sparse", {"method": "vfe"}),
    ("sparse", {"method": "fitc", "learnable_z": True}),
    ("sparse", {"method": "vfe", "learnable_z": True}),
    ("latent", {}),
    ("latent", {"learnable_inputs": True}),
    ("sparse_latent", {"learnable_z": True}),
    ("sparse_latent", {"learnable_inputs": True, "learnable_z": True}),
    ("toeplitz", {}),
    ("kronecker", {}),
]


def _case_id(case):
    gp_class, options = case
    return "-".join([gp_class, *(f"{k}={v}" for k, v in options.items())])


@pytest.mark.parametrize("mean_kind", _MEAN_KINDS)
@pytest.mark.parametrize("case", _GP_CASES, ids=_case_id)
def test_gp_gradients_with_mean_functions_match_finite_differences(case, mean_kind):
    gp_class, options = case
    model = _model(gp_class, mean_kind, **options)
    gradients = _finite_difference_check(model)
    mean_names = {"constant": {"c"}, "linear": {"w", "b"}, "linear_deterministic": {"log_w", "c"}}[mean_kind]
    assert mean_names <= set(gradients)
    for name in mean_names:
        assert bool(gradients[name].abs().sum() > 0), name


@pytest.mark.parametrize("case", _GP_CASES, ids=_case_id)
def test_gp_with_a_linear_mean_batched_equals_chain_by_chain(case):
    gp_class, options = case
    _assert_batched_equals_loop(_model(gp_class, "linear", **options))


@pytest.mark.parametrize("gp_class", ["sparse", "sparse_latent"])
def test_inducing_points_get_no_mean_term(gp_class):
    """The mean is evaluated at X: the Z gradient is the same with and without a mean function."""
    with_mean = _model(gp_class, "linear", learnable_z=True)
    gradients = with_mean.joint_grad_log_prob()
    # Reference: a zero-mean GP on the observations shifted by the (fixed) mean at X.
    x, y = _data()
    w = with_mean.params["w"].constrained_value
    b = with_mean.params["b"].constrained_value
    m = x @ w + b
    with Model() as reference:
        z0 = x[:4].clone() + 0.05
        z = RandomParameter("z", Normal(z0, 0.5), shape=tuple(z0.shape), initial_value=z0.clone())
        RandomParameter("w", Normal(0.0, 1.0), shape=2, initial_value=w.clone())
        RandomParameter("b", Normal(0.0, 1.0), initial_value=b.clone())
        if gp_class == "sparse":
            ObservedParameter("y", SparseGP("f", Data("x", x), _kernel(), z, noise_variance=_noise()), y - m)
        else:
            f = SparseLatentGP("f", Data("x", x), _kernel(), z, jitter=1e-8)
            ObservedParameter("y", Normal(f, 0.3), y - m)
    if gp_class == "sparse_latent":
        reference.params["f_white"].set_unconstrained_value(with_mean.params["f_white"].unconstrained_value.clone())
    torch.testing.assert_close(gradients["z"], reference.joint_grad_log_prob()["z"])


# ----------------------------------------------------------------------------
# Scalar means behave as before; ZeroMean is a fast path
# ----------------------------------------------------------------------------


def test_scalar_means_behave_as_before():
    x, y = _data()
    log_probs, gradients = {}, {}
    for label in ("zero_default", "zero_number", "scalar_node"):
        with Model() as model:
            kwargs = {}
            if label == "zero_number":
                kwargs["mean"] = 0.0
            elif label == "scalar_node":
                kwargs["mean"] = RandomParameter("m", Normal(0.0, 1.0), initial_value=torch.tensor([0.4], dtype=DTYPE))
            gp = ExactGP("f", Data("x", x), _kernel(), noise_variance=_noise(), **kwargs)
            ObservedParameter("y", gp, y)
        log_probs[label] = model.model_log_prob()
        gradients[label] = _finite_difference_check(model)
        if label == "scalar_node":
            # The old formula: a scalar mean gets Σ α with α = (K + σ² I)⁻¹ (y - m), plus its Normal(0, 1) prior -m.
            alpha = gp._components(y)["alpha"][0]
            torch.testing.assert_close(gradients[label]["m"].reshape(()), alpha.sum() - 0.4)
            assert isinstance(gp.mean, ConstantMean)
    torch.testing.assert_close(log_probs["zero_default"], log_probs["zero_number"])
    for name in gradients["zero_default"]:
        torch.testing.assert_close(gradients["zero_default"][name], gradients["zero_number"][name])


def test_scalar_node_mean_in_latent_gp_behaves_as_before():
    x, y = _data()
    with Model() as model:
        offset = RandomParameter("offset", Normal(0.0, 1.0), initial_value=torch.tensor([0.4], dtype=DTYPE))
        f = LatentGP("f", Data("x", x), _kernel(), mean=offset, jitter=1e-8)
        ObservedParameter("y", Normal(f, 0.3), y)
    gradients = _finite_difference_check(model)
    residual = (y - f.constrained_value) / 0.09
    torch.testing.assert_close(gradients["offset"].reshape(()), residual.sum() + (-0.4))  # likelihood + Normal(0, 1) prior


def test_zero_mean_default_adds_no_parameters_or_graph_edges():
    x, y = _data()
    with Model() as model:
        exact = ExactGP("f", Data("x", x), RBF(0.5, 1.0), noise_variance=0.1)
        other = ExactGP("g", Data("x2", x), RBF(0.5, 1.0), noise_variance=0.1)
        latent = LatentGP("h", Data("x3", x), RBF(0.5, 1.0))
        sparse = SparseLatentGP("s", Data("x4", x), RBF(0.5, 1.0), x[:3].clone())
    for gp in (exact, other):
        assert isinstance(gp.mean, ZeroMean)
        assert gp._mean_parameter_nodes == []
        assert gp.parameters == {"x"} or gp.parameters == {"x2"}
    assert exact.mean is not other.mean  # a fresh ZeroMean per GP
    assert [node.name for node in latent.inputs] == ["x3", "h_white"]
    assert [node.name for node in sparse.inputs] == ["x4", "s_white"]
    assert isinstance(latent.gp_mean, ZeroMean) and isinstance(sparse.gp_mean, ZeroMean)
    assert set(model.graph.predecessors("h")) == {"x3", "h_white"}


# ----------------------------------------------------------------------------
# Predictions with a LinearMean read per draw from the trace
# ----------------------------------------------------------------------------


_TRACE_W = torch.tensor([[0.8, -0.4], [-1.2, 0.3], [0.1, 1.5]], dtype=DTYPE)
_TRACE_B = torch.tensor([[0.5], [-0.7], [1.9]], dtype=DTYPE)
_LENGTHSCALE, _VARIANCE, _NOISE = 0.7, 1.3, 0.05


def _linear_mean_nodes(features=2):
    # Current node values differ from every traced draw.
    w = RandomParameter("w", Normal(0.0, 1.0), shape=features, initial_value=torch.full((features,), 5.0, dtype=DTYPE))
    b = RandomParameter("b", Normal(0.0, 1.0), initial_value=torch.tensor([-5.0], dtype=DTYPE))
    return w, b


def _trace(features=2, **extra):
    trace = {"w": _TRACE_W[:, :features].unsqueeze(0), "b": _TRACE_B.unsqueeze(0)}
    trace.update({name: value.unsqueeze(0) for name, value in extra.items()})
    return trace


def _rbf(a, b):
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return _VARIANCE * torch.exp(-0.5 * torch.cdist(a, b).square() / _LENGTHSCALE ** 2)


def _linear(x, draw, features=2):
    return x.reshape(x.shape[0], -1) @ _TRACE_W[draw, :features] + _TRACE_B[draw, 0]


def _gaussian_reference(means, covariances, n_samples, jitter, seed):
    """Draws ``mean + ε Lᵀ`` with the predictives' order of random numbers."""
    torch.manual_seed(seed)
    outputs = []
    for mean, covariance in zip(means, covariances):
        covariance = (covariance + covariance.transpose(-2, -1)) / 2
        factor = stable_cholesky(covariance, jitter=jitter)
        outputs.append(mean + torch.randn((n_samples, mean.numel()), dtype=DTYPE) @ factor.transpose(-2, -1))
    return torch.stack(outputs)


def test_exact_and_toeplitz_predictive_with_a_linear_mean_match_a_dense_reference():
    x = torch.linspace(-1.0, 2.0, 8, dtype=DTYPE)
    y = torch.sin(2 * x) + 0.7 * x
    new = torch.tensor([-1.3, 0.25, 0.9, 2.4], dtype=DTYPE)
    with Model():
        w, b = _linear_mean_nodes(1)
        exact = ExactGP("f", Data("x", x), RBF(_LENGTHSCALE, _VARIANCE), noise_variance=_NOISE, mean=LinearMean(w, b), jitter=1e-12)
    with Model():
        w, b = _linear_mean_nodes(1)
        toeplitz = ToeplitzGP("f", Data("x", x), RBF(_LENGTHSCALE, _VARIANCE), noise_variance=_NOISE, mean=LinearMean(w, b), jitter=1e-12)
    trace = _trace(features=1)
    means, covariances = [], []
    for draw in range(3):
        sigma = _rbf(x, x) + _NOISE * torch.eye(8, dtype=DTYPE)
        cross = _rbf(new, x)
        means.append(_linear(new, draw, 1) + cross @ torch.linalg.solve(sigma, y - _linear(x, draw, 1)))
        covariances.append(_rbf(new, new) - cross @ torch.linalg.solve(sigma, cross.T))
    for gp, predictive, jitter in ((exact, exact_gp_predictive, 1e-12), (toeplitz, toeplitz_gp_predictive, 1e-10)):
        torch.manual_seed(5)
        actual = predictive(gp, trace, y, new, n_samples=2)
        expected = _gaussian_reference(means, covariances, 2, jitter, seed=5)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


def test_kronecker_predictive_with_a_linear_mean_matches_a_dense_reference():
    grids = [torch.linspace(-1.0, 1.0, 4, dtype=DTYPE), torch.linspace(0.0, 2.0, 3, dtype=DTYPE)]
    points = torch.cartesian_prod(*grids)
    y = torch.sin(2 * points[:, 0]) + 0.4 * points[:, 1]
    new = torch.tensor([[0.1, 0.5], [-0.8, 1.7], [1.2, 2.2]], dtype=DTYPE)
    with Model():
        w, b = _linear_mean_nodes(2)
        gp = KroneckerGP("f", [Data("g0", grids[0]), Data("g1", grids[1])], [RBF(_LENGTHSCALE, _VARIANCE), RBF(_LENGTHSCALE, 1.0)],
                         noise_variance=_NOISE, mean=LinearMean(w, b), jitter=0.0)
    trace = _trace()
    means, covariances = [], []
    for draw in range(3):
        sigma = _rbf(points, points) + _NOISE * torch.eye(12, dtype=DTYPE)
        cross = _rbf(new, points)
        means.append(_linear(new, draw) + cross @ torch.linalg.solve(sigma, y - _linear(points, draw)))
        covariances.append(_rbf(new, new) - cross @ torch.linalg.solve(sigma, cross.T))
    torch.manual_seed(2)
    actual = kronecker_gp_predictive(gp, trace, y, new, n_samples=2)
    torch.testing.assert_close(actual, _gaussian_reference(means, covariances, 2, 1e-10, seed=2), atol=1e-7, rtol=0)
    # A new grid (one array per factor) is the same as its points in row-major order.
    torch.manual_seed(2)
    on_grid = kronecker_gp_predictive(gp, trace, y, [grids[0][:2], grids[1]], n_samples=1)
    torch.manual_seed(2)
    as_points = kronecker_gp_predictive(gp, trace, y, torch.cartesian_prod(grids[0][:2], grids[1]), n_samples=1)
    torch.testing.assert_close(on_grid, as_points)


def test_sparse_predictive_with_a_linear_mean_matches_a_dense_reference():
    x, y = _data(n=10)
    z = x[:4].clone() + 0.05
    new = torch.tensor([[0.1, 0.5], [-0.8, 0.7], [0.6, -0.2]], dtype=DTYPE)
    with Model():
        w, b = _linear_mean_nodes(2)
        gp = SparseGP("f", Data("x", x), RBF(_LENGTHSCALE, _VARIANCE), z, noise_variance=_NOISE, mean=LinearMean(w, b), jitter=1e-12)
    trace = _trace()
    means, covariances = [], []
    k_mm, k_mn, k_ms = _rbf(z, z), _rbf(z, x), _rbf(z, new)
    sigma_u = torch.linalg.inv(k_mm + k_mn @ k_mn.T / _NOISE)
    for draw in range(3):
        means.append(_linear(new, draw) + k_ms.T @ sigma_u @ k_mn @ (y - _linear(x, draw)) / _NOISE)
        covariances.append(_rbf(new, new) - k_ms.T @ torch.linalg.solve(k_mm, k_ms) + k_ms.T @ sigma_u @ k_ms)
    torch.manual_seed(3)
    actual = sparse_gp_predictive(gp, trace, y, new, n_samples=2)
    torch.testing.assert_close(actual, _gaussian_reference(means, covariances, 2, 1e-12, seed=3), atol=1e-6, rtol=0)


def test_latent_predictive_with_a_linear_mean_matches_a_dense_reference():
    x, _ = _data(n=6)
    new = torch.tensor([[0.1, 0.5], [-0.8, 0.7], [0.6, -0.2]], dtype=DTYPE)
    with Model():
        w, b = _linear_mean_nodes(2)
        gp = LatentGP("f", Data("x", x), RBF(_LENGTHSCALE, _VARIANCE), mean=LinearMean(w, b), jitter=1e-12)
    functions = torch.stack([torch.sin(3 * x[:, 0]) + d for d in range(3)])
    trace = _trace(f=functions)
    means, covariances = [], []
    k, cross = _rbf(x, x), _rbf(new, x)
    for draw in range(3):
        means.append(_linear(new, draw) + cross @ torch.linalg.solve(k, functions[draw] - _linear(x, draw)))
        covariances.append(_rbf(new, new) - cross @ torch.linalg.solve(k, cross.T) + 0.01 * torch.eye(3, dtype=DTYPE))
    torch.manual_seed(4)
    actual = gp_predictive(gp, trace, new, n_samples=2, observation_noise=0.01)
    torch.testing.assert_close(actual, _gaussian_reference(means, covariances, 2, 1e-12, seed=4), atol=1e-6, rtol=0)


def test_sparse_latent_predictive_with_a_linear_mean_matches_a_dense_reference():
    x, _ = _data(n=6)
    z = x[:4].clone() + 0.05
    new = torch.tensor([[0.1, 0.5], [-0.8, 0.7], [0.6, -0.2]], dtype=DTYPE)
    with Model():
        w, b = _linear_mean_nodes(2)
        gp = SparseLatentGP("f", Data("x", x), RBF(_LENGTHSCALE, _VARIANCE), z, mean=LinearMean(w, b), jitter=1e-12)
    white = torch.randn(3, 4, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    trace = _trace(f_white=white)
    factor = torch.linalg.cholesky(_rbf(z, z))
    expected = torch.stack([
        _linear(new, d) + _rbf(new, z) @ torch.linalg.solve_triangular(factor.T, white[d].reshape(-1, 1), upper=True).reshape(-1)
        for d in range(3)
    ])
    actual = sparse_latent_gp_predictive(gp, trace, new, n_samples=2, chunk_size=2)
    assert actual.shape == (3, 2, 3)
    torch.testing.assert_close(actual, expected.unsqueeze(1).expand(3, 2, 3), atol=1e-8, rtol=0)
