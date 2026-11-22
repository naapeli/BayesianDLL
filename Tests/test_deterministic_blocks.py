"""Deterministic building blocks: forward values and hand-written VJPs.

The reverse pass of the model is compared with ``torch.autograd`` (used only
as a test oracle) on the unconstrained joint log density.
"""

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import (
    Add,
    Affine,
    Concat,
    Divide,
    Elementwise,
    Exp,
    Index,
    Linear,
    MatMul,
    Multiply,
    Softmax,
    Softplus,
    Stack,
    Subtract,
    Sum,
    Tanh,
)
from BayesianDLL.Deterministic._blocks import CholeskyCovariance, CovarianceFromCholesky
from BayesianDLL.Distributions import (
    Bernoulli,
    Categorical,
    HalfNormal,
    LKJCholesky,
    MultivariateNormal,
    Normal,
)


# ----------------------------------------------------------------- helpers
def randomize(model, scale=0.7):
    for parameter in model.params.values():
        parameter.set_unconstrained_value(scale * torch.randn_like(parameter.unconstrained_value))


def autograd_grads(model):
    names = list(model.params)
    leaves = {
        name: model.params[name].unconstrained_value.detach().clone().requires_grad_(True)
        for name in names
    }
    with model.temporarily_set_many(leaves):
        log_prob = model.model_log_prob()
        grads = torch.autograd.grad(log_prob, [leaves[name] for name in names], allow_unused=True)
    return {
        name: torch.zeros_like(leaves[name]) if grad is None else grad
        for name, grad in zip(names, grads)
    }


def assert_grads_match(model):
    randomize(model)
    expected = autograd_grads(model)
    actual = model.joint_grad_log_prob()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], msg=f"gradient of {name}")


def normal(name, shape):
    return RandomParameter(name, Normal(0.0, 1.0), shape=shape)


def observe(name, mean, sigma=0.8):
    with torch.no_grad():
        value = mean.constrained_value
    ObservedParameter(name, Normal(mean, sigma), torch.randn(value.shape))


# ----------------------------------------------------------------- elementwise arithmetic
def test_arithmetic_blocks_forward_and_broadcast_gradients():
    data = torch.randn(3, 4)
    with Model() as model:
        a = normal("a", (3, 1))
        b = normal("b", (4,))
        c = normal("c", (3, 4))
        x = Data("x", data)
        s = Add("s", a, b, x, 2.0)
        d = Subtract("d", s, c)
        m = Multiply("m", d, a, torch.full((4,), 0.5))
        q = Divide("q", m, Softplus("pos", c))
        observe("y", q)

    with torch.no_grad():
        av, bv, cv = a.constrained_value, b.constrained_value, c.constrained_value
        expected = ((av + bv + data + 2.0) - cv) * av * 0.5 / torch.nn.functional.softplus(cv)
    torch.testing.assert_close(q.constrained_value, expected)
    assert_grads_match(model)


def test_divide_by_node_and_constant_numerator():
    with Model() as model:
        b = RandomParameter("b", HalfNormal(1.0), shape=(5,))
        q = Divide("q", 3.0, b)
        r = Divide("r", b, 2)
        observe("y", Add("t", q, r))
    torch.testing.assert_close(q.constrained_value, 3.0 / b.constrained_value)
    torch.testing.assert_close(r.constrained_value, b.constrained_value / 2)
    assert_grads_match(model)


def test_repeated_input_accumulates_gradient():
    with Model() as model:
        x = normal("x", (4,))
        square = Multiply("square", x, x)
        observe("y", Add("z", square, x))
    torch.testing.assert_close(square.constrained_value, x.constrained_value ** 2)
    assert_grads_match(model)
    torch.testing.assert_close(square.derivative("x"), 2 * x.constrained_value)


def test_affine_non_centered_with_scalar_nodes_and_constants():
    with Model() as model:
        mu = normal("mu", (1,))
        tau = RandomParameter("tau", HalfNormal(1.0))
        z = normal("z", (6,))
        theta = Affine("theta", z, loc=mu, scale=tau)
        shifted = Affine("shifted", theta, loc=torch.arange(6.0), scale=2)
        default = Affine("default", z)
        observe("y", shifted)
    torch.testing.assert_close(theta.constrained_value, mu.constrained_value + tau.constrained_value * z.constrained_value)
    torch.testing.assert_close(default.constrained_value, z.constrained_value)
    assert_grads_match(model)
    # Elementwise local derivatives stay available (used by the ELBO).
    torch.testing.assert_close(theta.derivative("z"), tau.constrained_value.expand(6))


def test_unary_nonlinearities():
    with Model() as model:
        x = normal("x", (2, 3))
        t = Tanh("t", x)
        sp = Softplus("sp", x)
        observe("y", Add("s", t, sp))
    torch.testing.assert_close(t.constrained_value, torch.tanh(x.constrained_value))
    torch.testing.assert_close(sp.constrained_value, torch.nn.functional.softplus(x.constrained_value))
    assert_grads_match(model)


def test_elementwise_user_function_with_two_inputs():
    with Model() as model:
        x = normal("x", (5,))
        w = normal("w", (1,))
        f = Elementwise(
            "f",
            lambda x, w: torch.sin(x) * w ** 2,
            lambda x, w: (torch.cos(x) * w ** 2, 2 * w * torch.sin(x)),
            x,
            w,
        )
        observe("y", f)
    torch.testing.assert_close(f.constrained_value, torch.sin(x.constrained_value) * w.constrained_value ** 2)
    assert_grads_match(model)


# ----------------------------------------------------------------- softmax, reductions, shapes
@pytest.mark.parametrize("dim", [0, -1])
def test_softmax_forward_and_gradient(dim):
    with Model() as model:
        x = normal("x", (3, 4))
        s = Softmax("s", x, dim=dim)
        observe("y", Multiply("scaled", s, torch.arange(1.0, 5.0)))
    torch.testing.assert_close(s.constrained_value, torch.softmax(x.constrained_value, dim=dim))
    assert_grads_match(model)


@pytest.mark.parametrize("dim, keepdim", [(None, False), (None, True), (1, False), (0, True), ((0, 2), False)])
def test_sum_forward_and_gradient(dim, keepdim):
    with Model() as model:
        x = normal("x", (2, 3, 4))
        s = Sum("s", x, dim=dim, keepdim=keepdim)
        observe("y", s)
    value = x.constrained_value
    expected = value.sum() if dim is None else value.sum(dim=dim, keepdim=keepdim)
    if dim is None and keepdim:
        expected = expected.reshape(1, 1, 1)
    torch.testing.assert_close(s.constrained_value, expected)
    assert_grads_match(model)


def test_concat_and_stack():
    with Model() as model:
        a = normal("a", (2, 3))
        b = normal("b", (2, 1))
        cat = Concat("cat", a, b, torch.zeros(2, 2), dim=-1)
        c = normal("c", (4,))
        st = Stack("st", c, Exp("ec", c), 1.0, dim=1)
        observe("y1", cat)
        observe("y2", st)
    torch.testing.assert_close(
        cat.constrained_value, torch.cat([a.constrained_value, b.constrained_value, torch.zeros(2, 2)], dim=-1)
    )
    cv = c.constrained_value
    torch.testing.assert_close(st.constrained_value, torch.stack([cv, torch.exp(cv), torch.ones(4)], dim=1))
    assert_grads_match(model)


# ----------------------------------------------------------------- linear algebra
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((5, 3), (3,)),  # matrix-vector
        ((3,), (3, 2)),  # vector-matrix
        ((4,), (4,)),  # dot product
        ((5, 3), (3, 2)),  # matrix-matrix
        ((2, 5, 3), (3, 4)),  # batched with a broadcast operand
        ((2, 1, 4, 3), (5, 3, 2)),  # broadcast batch dimensions on both sides
    ],
)
def test_matmul_forward_and_gradient(shape_a, shape_b):
    with Model() as model:
        a = normal("a", shape_a)
        b = normal("b", shape_b)
        product = MatMul("product", a, b)
        observe("y", product)
    torch.testing.assert_close(product.constrained_value, a.constrained_value @ b.constrained_value)
    assert_grads_match(model)


def test_matmul_with_data_and_dense_derivative_fallback():
    design = torch.randn(6, 3)
    with Model() as model:
        x = Data("x", design)
        w = normal("w", (3,))
        mean = MatMul("mean", x, w)
        observe("y", mean)
    assert_grads_match(model)
    # derivative() falls back to the dense Jacobian built from the VJP.
    torch.testing.assert_close(mean.derivative("w"), design)
    assert set(mean.derivatives()) == {"x", "w"}


def test_cholesky_covariance_with_lkj_and_multivariate_normal():
    with Model() as model:
        L_corr = RandomParameter("L_corr", LKJCholesky(3, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0), shape=(3,))
        L = CholeskyCovariance("L", L_corr, sigma)
        mu = normal("mu", (3,))
        ObservedParameter("y", MultivariateNormal(mu, scale_tril=L), torch.randn(10, 3))
    randomize(model)
    torch.testing.assert_close(L.constrained_value, torch.diag(sigma.constrained_value) @ L_corr.constrained_value)
    assert_grads_match(model)


def test_covariance_from_cholesky_with_multivariate_normal():
    with Model() as model:
        L_corr = RandomParameter("L_corr", LKJCholesky(3, 1.5))
        sigma = RandomParameter("sigma", HalfNormal(1.0), shape=(3,))
        cov = CovarianceFromCholesky("cov", CholeskyCovariance("L", L_corr, sigma))
        ObservedParameter("y", MultivariateNormal(torch.zeros(3), cov), torch.randn(8, 3))
    randomize(model)
    Lv = torch.diag(sigma.constrained_value) @ L_corr.constrained_value
    torch.testing.assert_close(cov.constrained_value, Lv @ Lv.T)
    assert_grads_match(model)


# ----------------------------------------------------------------- indexing
def test_index_forward_and_gradient_with_constant_index():
    idx = torch.tensor([0, 2, 2, 1, 0, 2])
    with Model() as model:
        theta = normal("theta", (3,))
        picked = Index("picked", theta, idx)
        observe("y", picked)
    torch.testing.assert_close(picked.constrained_value, theta.constrained_value[idx])
    assert_grads_match(model)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_index_of_matrix_along_dim_with_2d_index(dim):
    idx = torch.tensor([[0, 1], [1, 1], [0, 0]])
    with Model() as model:
        beta = normal("beta", (2, 2, 3) if dim == 1 else (2, 3, 2))
        picked = Index("picked", beta, idx, dim=dim)
        observe("y", picked)
    value = beta.constrained_value
    expected = torch.index_select(value, dim % 3, idx.reshape(-1)).reshape(
        *value.shape[: dim % 3], *idx.shape, *value.shape[dim % 3 + 1:]
    )
    torch.testing.assert_close(picked.constrained_value, expected)
    assert_grads_match(model)


def test_index_affine_likelihood_chain_matches_explicit_torch_expression():
    group = torch.tensor([0, 1, 1, 3, 2, 0, 3, 3])
    covariate = torch.randn(8)
    y = torch.randn(8)
    with Model() as model:
        g = Data("group", group)
        x = Data("x", covariate)
        mu = normal("mu", (1,))
        tau = RandomParameter("tau", HalfNormal(1.0))
        z = normal("z", (4,))
        slope = normal("slope", (1,))
        theta = Affine("theta", z, loc=mu, scale=tau)
        mean = Add("mean", Index("theta_obs", theta, g), Multiply("slope_x", slope, x))
        ObservedParameter("y", Normal(mean, 0.5), y)
    randomize(model)

    leaves = {name: p.unconstrained_value.detach().clone().requires_grad_(True) for name, p in model.params.items()}
    tau_value = torch.exp(leaves["tau"])  # HalfNormal uses a log transform
    theta_value = leaves["mu"] + tau_value * leaves["z"]
    mean_value = theta_value[group] + leaves["slope"] * covariate
    log_prob = (
        torch.distributions.Normal(mean_value, 0.5).log_prob(y).sum()
        + torch.distributions.Normal(0.0, 1.0).log_prob(leaves["mu"]).sum()
        + torch.distributions.Normal(0.0, 1.0).log_prob(leaves["z"]).sum()
        + torch.distributions.Normal(0.0, 1.0).log_prob(leaves["slope"]).sum()
        + torch.distributions.HalfNormal(1.0).log_prob(tau_value).sum()
        + leaves["tau"].sum()
    )
    names = list(leaves)
    expected = dict(zip(names, torch.autograd.grad(log_prob, [leaves[n] for n in names])))
    actual = model.joint_grad_log_prob()
    for name in names:
        torch.testing.assert_close(actual[name], expected[name])


def test_index_with_mutable_data_index_predicts_new_groups():
    with Model() as model:
        g = Data("group", torch.tensor([0, 1, 1]))
        theta = normal("theta", (4,))
        picked = Index("picked", theta, g)
        observe("y", picked)
    theta_value = theta.constrained_value
    torch.testing.assert_close(picked.constrained_value, theta_value[[0, 1, 1]])

    new_index = torch.tensor([3, 2, 3, 0, 1])
    g.set_value(new_index)
    torch.testing.assert_close(picked.constrained_value, theta_value[new_index])
    model.observed_params["y"].observed_values = torch.randn(5)
    assert_grads_match(model)


def test_index_rejects_float_constant_index():
    with Model():
        theta = normal("theta", (3,))
        with pytest.raises(TypeError, match="integer"):
            Index("picked", theta, torch.tensor([0.0, 1.0]))


def test_blocks_reject_unknown_argument_types():
    with Model():
        x = normal("x", (2,))
        with pytest.raises(TypeError, match="model node, a VariationalParameter, a tensor or a number"):
            Add("bad", x, "not a node")


# ----------------------------------------------------------------- larger compositions
def test_one_hidden_layer_network_gradient():
    inputs = torch.randn(7, 1)
    with Model() as model:
        x = Data("x", inputs)
        w1 = normal("w1", (1, 5))
        b1 = normal("b1", (5,))
        w2 = normal("w2", (5,))
        b2 = normal("b2", (1,))
        hidden = Tanh("hidden", Add("pre", MatMul("xw1", x, w1), b1))
        out = Add("out", MatMul("hw2", hidden, w2), b2)
        ObservedParameter("y", Normal(out, 0.3), torch.randn(7))
    randomize(model)
    with torch.no_grad():
        expected = torch.tanh(inputs @ w1.constrained_value + b1.constrained_value) @ w2.constrained_value + b2.constrained_value
    torch.testing.assert_close(out.constrained_value, expected)
    assert_grads_match(model)


def test_softmax_regression_with_categorical_logits():
    features = torch.randn(9, 2)
    labels = torch.randint(0, 3, (9,)).double()
    with Model() as model:
        x = Data("x", features)
        w = normal("w", (2, 3))
        b = normal("b", (3,))
        logits = Add("logits", MatMul("xw", x, w), b)
        probabilities = Softmax("probabilities", logits)
        ObservedParameter("y", Categorical(logits=logits), labels)
        ObservedParameter("y_probs", Categorical(probs=probabilities), labels)
    assert_grads_match(model)


def test_linear_and_unary_transforms_use_vjp():
    design = torch.randn(5, 3)
    with Model() as model:
        x = Data("x", design)
        w = normal("w", (3,))
        b = normal("b", (5,))
        log_s = normal("log_s", (1,))
        mean = Linear("mean", x, w, intercept=b)
        ObservedParameter("y", Normal(mean, Exp("s", log_s)), torch.randn(5))
    assert mean.vjp_func is not None
    assert_grads_match(model)


def test_custom_deterministic_with_vjp_only():
    with Model() as model:
        x = normal("x", (4,))
        cumulative = DeterministicParameter(
            "cumulative",
            lambda v: torch.cumsum(v, 0),
            inputs=[x],
            vjp_func=lambda g, v: {"x": torch.flip(torch.cumsum(torch.flip(g, (0,)), 0), (0,))},
        )
        observe("y", cumulative)
    assert_grads_match(model)
    torch.testing.assert_close(cumulative.derivative("x"), torch.tril(torch.ones(4, 4)))


def test_deterministic_requires_a_derivative():
    with Model():
        x = normal("x", (1,))
        with pytest.raises(TypeError, match="derivative_func or vjp_func"):
            DeterministicParameter("bad", lambda v: v, None, [x])


# ----------------------------------------------------------------- enumeration
def test_enumerated_index_and_weighted_reverse_pass():
    data = torch.cat([torch.randn(5) - 2.0, torch.randn(5) + 2.0])
    with Model() as model:
        means = RandomParameter("means", Normal(0.0, 3.0), shape=(2,))
        scale = RandomParameter("scale", HalfNormal(1.0))
        with plate("data", 10):
            z = RandomParameter("z", Bernoulli(0.4), shape=(), marginalize=True)
            centre = Index("centre", means, z)
            ObservedParameter("y", Normal(Affine("loc", centre, scale=1.0), scale), data)
    randomize(model)

    # Marginal gradient (enumeration calls _reverse_pass with term weights).
    assert_grads_match(model)

    # Weighted reverse pass for a fixed assignment equals autograd of the
    # weighted elementwise log density.
    z.set_constrained_value(torch.tensor([0.0, 1.0] * 5))
    weights = torch.rand(10)
    leaves = {name: p.unconstrained_value.detach().clone().requires_grad_(True) for name, p in model.params.items()}
    with model.temporarily_set_many(leaves):
        obs = model.observed_params["y"]
        weighted = (weights * obs.distribution.log_pdf(obs.observed_values)).sum()
        expected = torch.autograd.grad(weighted, [leaves["means"], leaves["scale"]])
    actual = model._reverse_pass(["means", "scale"], active_terms={"y"}, term_weights={"y": weights})
    torch.testing.assert_close(actual["means"], expected[0])
    torch.testing.assert_close(actual["scale"], expected[1])


# ----------------------------------------------------------------- Linear shapes
# (n_features, input layout, coefficient shape). A one-element coefficient with
# a design matrix of shape (N, 1) is a coefficient vector, not a scalar slope.
LINEAR_CASES = [
    (1, "vector", ()),       # scalar slope, inputs (N,)
    (1, "vector", (1,)),     # one-element slope, inputs (N,)
    (1, "matrix", (1,)),     # design matrix (N, 1), coefficients (1,)
    (1, "matrix", ()),       # design matrix (N, 1), scalar coefficient
    (3, "matrix", (3,)),     # design matrix (N, 3), coefficients (3,)
]


def _linear_model(n_features, layout, coefficient_shape, random_inputs):
    n_obs = 6
    shape = (n_obs,) if layout == "vector" else (n_obs, n_features)
    design = torch.randn(shape)
    with Model() as model:
        x = normal("x", shape) if random_inputs else Data("x", design)
        w = normal("w", coefficient_shape)
        b = normal("b", ())
        mean = Linear("mean", x, w, intercept=b)
        ObservedParameter("y", Normal(mean, 0.7), torch.randn(n_obs))
    return model, mean


@pytest.mark.parametrize("random_inputs", [False, True], ids=["data", "random_inputs"])
@pytest.mark.parametrize("n_features, layout, coefficient_shape", LINEAR_CASES, ids=lambda v: str(v).replace(" ", ""))
def test_linear_shapes_and_gradients(n_features, layout, coefficient_shape, random_inputs):
    model, mean = _linear_model(n_features, layout, coefficient_shape, random_inputs)
    randomize(model)
    with torch.no_grad():
        x = (model.params["x"].constrained_value if random_inputs else mean.regression_inputs.constrained_value)
        w = model.params["w"].constrained_value
        b = model.params["b"].constrained_value
        expected = (x * w.reshape(()) if layout == "vector" else x @ w.reshape(-1)) + b
    assert mean.constrained_value.shape == (6,)
    torch.testing.assert_close(mean.constrained_value, expected)
    assert_grads_match(model)


@pytest.mark.parametrize("random_inputs", [False, True], ids=["data", "random_inputs"])
@pytest.mark.parametrize("n_features, layout, coefficient_shape", LINEAR_CASES, ids=lambda v: str(v).replace(" ", ""))
def test_batched_linear_matches_autograd(n_features, layout, coefficient_shape, random_inputs):
    model, mean = _linear_model(n_features, layout, coefficient_shape, random_inputs)
    n_chains = 3
    values = {name: 0.7 * torch.randn((n_chains, *p.unconstrained_value.shape)) for name, p in model.params.items()}
    expected_lp, expected_grads, expected_means = [], {name: [] for name in values}, []
    for chain in range(n_chains):
        leaves = {name: value[chain].clone().requires_grad_(True) for name, value in values.items()}
        with model.temporarily_set_many(leaves):
            lp = model.model_log_prob()
            grads = torch.autograd.grad(lp, list(leaves.values()))
            expected_means.append(mean.constrained_value.detach().clone())
        expected_lp.append(lp.detach())
        for name, grad in zip(leaves, grads):
            expected_grads[name].append(grad)
    with model.batched(n_chains, values=values):
        assert model._batch_mode == "tensor", model.batch_fallback_reason
        lp = model.model_log_prob()
        grads = model.joint_grad_log_prob(list(values))
        batched_mean = mean.constrained_value.clone()
    assert batched_mean.shape == (n_chains, 6)
    torch.testing.assert_close(batched_mean, torch.stack(expected_means))
    torch.testing.assert_close(lp, torch.stack(expected_lp))
    for name in values:
        torch.testing.assert_close(grads[name], torch.stack(expected_grads[name]), msg=f"gradient of {name}")


@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_matrix_coefficients_matches_autograd(has_bias):
    torch.manual_seed(42)
    x_val = torch.randn(6, 4, dtype=torch.float64)
    w_val = torch.randn(4, 5, dtype=torch.float64)
    b_val = torch.randn(5, dtype=torch.float64)
    with Model() as model:
        x = Data("x", x_val)
        w = RandomParameter("w", Normal(0.0, 1.0), shape=(4, 5), initial_value=w_val)
        b = RandomParameter("b", Normal(0.0, 1.0), shape=(5,), initial_value=b_val) if has_bias else 0.0
        h = Tanh("h", Linear("lin", x, w, intercept=b))
        with plate("obs", 6):
            ObservedParameter("y", Normal(h, 1.0), torch.zeros(6, 5, dtype=torch.float64))
    assert_grads_match(model)


def test_batched_linear_matrix_coefficients_matches_autograd():
    torch.manual_seed(42)
    x_val = torch.randn(6, 4, dtype=torch.float64)
    with Model() as model:
        x = Data("x", x_val)
        w = RandomParameter("w", Normal(0.0, 1.0), shape=(4, 5), initial_value=torch.randn(4, 5, dtype=torch.float64))
        b = RandomParameter("b", Normal(0.0, 1.0), shape=(5,), initial_value=torch.randn(5, dtype=torch.float64))
        h = Tanh("h", Linear("lin", x, w, intercept=b))
        with plate("obs", 6):
            ObservedParameter("y", Normal(h, 1.0), torch.zeros(6, 5, dtype=torch.float64))
    n_chains = 3
    values = {name: 0.7 * torch.randn((n_chains, *p.unconstrained_value.shape), dtype=torch.float64) for name, p in model.params.items()}
    expected_lp, expected_grads = [], {name: [] for name in values}
    for chain in range(n_chains):
        leaves = {name: value[chain].clone().requires_grad_(True) for name, value in values.items()}
        with model.temporarily_set_many(leaves):
            lp = model.model_log_prob()
            grads = torch.autograd.grad(lp, list(leaves.values()))
        expected_lp.append(lp.detach())
        for name, grad in zip(leaves, grads):
            expected_grads[name].append(grad)
    with model.batched(n_chains, values=values):
        lp = model.model_log_prob()
        grads = model.joint_grad_log_prob(list(values))
    torch.testing.assert_close(lp, torch.stack(expected_lp))
    for name in values:
        torch.testing.assert_close(grads[name], torch.stack(expected_grads[name]), msg=f"gradient of {name}")

