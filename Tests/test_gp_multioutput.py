"""Multi-output GPs: intrinsic and linear coregionalisation kernels."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import (
    RBF,
    ExactGP,
    IntrinsicCoregionalization,
    LatentGP,
    LinearCoregionalization,
    Matern32,
    exact_gp_predictive,
)

DTYPE = torch.float64
P = 3


def _name(name):
    return SimpleNamespace(name=name)


def _stacked(n=6, d=1, seed=0, column=-1, outputs=P):
    """Inputs of all outputs stacked output-major with the index in ``column``."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=generator, dtype=DTYPE) * 2.0
    index = torch.arange(outputs, dtype=DTYPE).repeat_interleave(n).unsqueeze(-1)
    features = x.repeat(outputs, 1)
    return torch.cat([features, index], dim=-1) if column == -1 else torch.cat([index, features], dim=-1)


def _reference(inputs, w, kappa, kernels):
    """Explicit ``Σ_q B_q[i, j] k_q(x, x')`` with autograd-friendly operations."""
    index = inputs[:, -1].round().long()
    features = inputs[:, :-1]
    total = 0
    for q, kernel in enumerate(kernels):
        matrix = w[q] @ w[q].T + torch.diag(kappa[q])
        total = total + matrix[index][:, index] * kernel(features, features)
    return total


# ----------------------------------------------------------------------------
# The kernel
# ----------------------------------------------------------------------------


def test_matches_the_explicit_sum_and_the_kronecker_structure():
    generator = torch.Generator().manual_seed(1)
    w = [torch.randn(P, 2, generator=generator, dtype=DTYPE), torch.randn(P, 1, generator=generator, dtype=DTYPE)]
    kappa = [torch.rand(P, generator=generator, dtype=DTYPE), torch.rand(P, generator=generator, dtype=DTYPE)]
    kernels = [RBF(0.7, 1.2), Matern32(0.4, 0.8)]
    inputs = _stacked()
    kernel = LinearCoregionalization(kernels, w, kappa)
    torch.testing.assert_close(kernel(inputs), _reference(inputs, w, kappa, kernels))
    torch.testing.assert_close(kernel.diag(inputs), torch.diagonal(kernel(inputs)))
    # ICM: B ⊗ K_x for output-major stacking.
    icm = IntrinsicCoregionalization(kernels[0], w[0], kappa[0])
    x = inputs[:6, :-1]
    matrix = w[0] @ w[0].T + torch.diag(kappa[0])
    torch.testing.assert_close(icm(inputs), torch.kron(matrix, kernels[0](x, x)))
    torch.testing.assert_close(icm.coregionalization_matrices()[0], matrix)
    # The index column can come first.
    first = LinearCoregionalization([kernels[0]], [w[0]], output_dimension=0)
    torch.testing.assert_close(first(_stacked(column=0)), IntrinsicCoregionalization(kernels[0], w[0])(inputs))


def _values(batch, generator):
    return {
        "w1": torch.randn(batch, P, 2, generator=generator, dtype=DTYPE),
        "w2": torch.randn(batch, P, 1, generator=generator, dtype=DTYPE),
        "k1": 0.2 + torch.rand(batch, P, generator=generator, dtype=DTYPE),
        "k2": 0.2 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),  # scalar, shared by the outputs
        "l": 0.5 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
        "v": 0.5 + torch.rand(batch, 1, generator=generator, dtype=DTYPE),
    }


def _kernel():
    shapes = {"w1": (P, 2), "w2": (P, 1), "k1": (P,), "k2": (), "l": (), "v": ()}
    with Model():
        nodes = {key: RandomParameter(key, Normal(0.0, 1.0), shape=shape) for key, shape in shapes.items()}
    return LinearCoregionalization(
        [RBF(nodes["l"], nodes["v"]), Matern32(0.6, nodes["v"])], [nodes["w1"], nodes["w2"]], [nodes["k1"], nodes["k2"]]
    )


def test_hand_derivatives_match_finite_differences_in_batched_form():
    generator = torch.Generator().manual_seed(2)
    batch = 3
    kernel = _kernel()
    values = _values(batch, generator)
    x1 = torch.cat([torch.randn(batch, 5, 2, generator=generator, dtype=DTYPE),
                    torch.randint(0, P, (batch, 5, 1), generator=generator).to(DTYPE)], dim=-1)
    x2 = torch.cat([torch.randn(batch, 4, 2, generator=generator, dtype=DTYPE),
                    torch.randint(0, P, (batch, 4, 1), generator=generator).to(DTYPE)], dim=-1)
    cotangent = torch.randn(batch, 5, 4, generator=generator, dtype=DTYPE)
    diag_cotangent = torch.randn(batch, 5, generator=generator, dtype=DTYPE)
    eps = 1e-6

    # Batch-summed scalar functions of the parameters.
    full = lambda v: (kernel.batch_evaluate(x1, x2, v) * cotangent).sum()
    diagonal = lambda v: (kernel.batch_diag(x1, v) * diag_cotangent).sum()
    derivatives = kernel.batch_derivatives(x1, x2, values)
    diag_derivatives = kernel.batch_diag_derivatives(x1, values)
    for name, value in values.items():
        assert derivatives[name].shape == (batch, *value.shape[1:], 5, 4), name
        assert diag_derivatives[name].shape == (batch, *value.shape[1:], 5), name
        hand = (derivatives[name] * cotangent.reshape(batch, *([1] * (value.ndim - 1)), 5, 4)).sum(dim=(-2, -1))
        expected = torch.zeros_like(value)
        diag_hand = (diag_derivatives[name] * diag_cotangent.reshape(batch, *([1] * (value.ndim - 1)), 5)).sum(dim=-1)
        diag_expected = torch.zeros_like(value)
        for index in range(value.numel()):
            shift = torch.zeros_like(value).reshape(-1)
            shift[index] = eps
            shift = shift.reshape(value.shape)
            plus, minus = {**values, name: value + shift}, {**values, name: value - shift}
            expected.reshape(-1)[index] = (full(plus) - full(minus)) / (2 * eps)
            diag_expected.reshape(-1)[index] = (diagonal(plus) - diagonal(minus)) / (2 * eps)
        torch.testing.assert_close(hand.sum(dim=0), expected.sum(dim=0), rtol=1e-6, atol=1e-8, msg=name)
        torch.testing.assert_close(diag_hand.sum(dim=0), diag_expected.sum(dim=0), rtol=1e-6, atol=1e-8, msg=name)

    # Input gradients: the index column gets exactly zero.
    vjp = kernel.batch_input_vjp(x1, x2, cotangent, values)
    assert vjp.shape == x1.shape and bool((vjp[..., -1] == 0).all())
    for column in range(2):
        shift = torch.zeros_like(x1)
        shift[:, :, column] = eps
        expected = ((kernel.batch_evaluate(x1 + shift, x2, values) - kernel.batch_evaluate(x1 - shift, x2, values))
                    * cotangent).sum(dim=(-2, -1)) / (2 * eps)
        torch.testing.assert_close(vjp[:, :, column].sum(dim=1), expected, rtol=1e-6, atol=1e-8)
    square = torch.randn(batch, 5, 5, generator=generator, dtype=DTYPE)
    symmetric = kernel.batch_input_vjp_symmetric(x1, square, values)
    diag_vjp = kernel.batch_diag_input_vjp(x1, diag_cotangent, values)
    assert bool((symmetric[..., -1] == 0).all()) and bool((diag_vjp[..., -1] == 0).all())
    for column in range(2):
        shift = torch.zeros_like(x1)
        shift[:, :, column] = eps
        expected = ((kernel.batch_evaluate(x1 + shift, x1 + shift, values) - kernel.batch_evaluate(x1 - shift, x1 - shift, values))
                    * square).sum(dim=(-2, -1)) / (2 * eps)
        torch.testing.assert_close(symmetric[:, :, column].sum(dim=1), expected, rtol=1e-6, atol=1e-8)
        expected = ((kernel.batch_diag(x1 + shift, values) - kernel.batch_diag(x1 - shift, values)) * diag_cotangent).sum(dim=-1) / (2 * eps)
        torch.testing.assert_close(diag_vjp[:, :, column].sum(dim=1), expected, rtol=1e-6, atol=1e-8)


def test_batched_matches_per_element_calls():
    generator = torch.Generator().manual_seed(3)
    kernel = _kernel()
    values = _values(3, generator)
    x = _stacked(5, 2)[:, [0, 1, 2]].unsqueeze(0).expand(3, -1, -1)
    x = x + 0.1 * torch.randn(x.shape, generator=generator, dtype=DTYPE) * torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE)
    covariance = kernel.batch_evaluate(x, x, values)
    derivatives = kernel.batch_derivatives(x, x, values)
    for b in range(3):
        single = {key: value[b] for key, value in values.items()}
        torch.testing.assert_close(covariance[b], kernel(x[b], parameter_values=single))
        for key, derivative in kernel.derivatives(x[b], x[b], parameter_values=single).items():
            torch.testing.assert_close(derivatives[key][b], derivative)


def test_validates_arguments_and_indices():
    w = torch.ones(P, 1, dtype=DTYPE)
    with pytest.raises(TypeError):
        LinearCoregionalization([lambda a, b: a], [w])
    with pytest.raises(ValueError, match="one weight matrix per kernel"):
        LinearCoregionalization([RBF(1.0), RBF(1.0)], [w])
    with pytest.raises(ValueError, match="same number of rows"):
        LinearCoregionalization([RBF(1.0), RBF(1.0)], [w, torch.ones(2, 1, dtype=DTYPE)])
    with pytest.raises(ValueError, match="matrices"):
        IntrinsicCoregionalization(RBF(1.0), torch.ones(P, dtype=DTYPE))
    with pytest.raises(ValueError, match="kappa"):
        IntrinsicCoregionalization(RBF(1.0), w, kappa=torch.ones(2, dtype=DTYPE))
    kernel = IntrinsicCoregionalization(RBF(1.0), w)
    bad = torch.tensor([[0.0, 1.0], [0.5, float(P)]], dtype=DTYPE)
    with pytest.raises(ValueError, match="output indices"):
        kernel(bad)
    with pytest.raises(ValueError, match="output-index column"):
        kernel(torch.ones(3, 1, dtype=DTYPE))


# ----------------------------------------------------------------------------
# GP nodes
# ----------------------------------------------------------------------------


def _data(n=7, seed=5):
    inputs = _stacked(n, 1, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    latent = torch.sin(2 * inputs[:, 0]) * (1.0 + 0.5 * inputs[:, 1])
    return inputs, latent + 0.1 * torch.randn(inputs.shape[0], generator=generator, dtype=DTYPE)


def _model(kind="exact", learnable_inputs=False):
    inputs_value, y = _data()
    with Model() as model:
        if learnable_inputs:
            inputs = RandomParameter("x", Normal(inputs_value, 0.2), shape=tuple(inputs_value.shape), initial_value=inputs_value + 0.02)
        else:
            inputs = Data("x", inputs_value)
        w = RandomParameter("w", Normal(0.0, 1.0), shape=(P, 2), initial_value=torch.tensor([[0.5, -0.2], [0.8, 0.3], [-0.4, 0.6]], dtype=DTYPE))
        kappa = Exp("kappa", RandomParameter("log_kappa", Normal(-1.0, 1.0), shape=P, initial_value=torch.tensor([-0.5, -1.0, -0.2], dtype=DTYPE)))
        lengthscale = Exp("lengthscale", RandomParameter("log_lengthscale", Normal(0.0, 1.0), initial_value=torch.tensor([0.1], dtype=DTYPE)))
        kernel = IntrinsicCoregionalization(RBF(lengthscale, 1.0), w, kappa)
        if kind == "exact":
            noise = Exp("noise", RandomParameter("log_noise", Normal(-2.0, 0.5), initial_value=torch.tensor([-1.5], dtype=DTYPE)))
            f = ExactGP("f", inputs, kernel, noise_variance=noise)
            ObservedParameter("y", f, y)
        else:
            f = LatentGP("f", inputs, kernel)
            f.gp_latent.set_unconstrained_value(torch.randn(f.gp_latent.unconstrained_value.shape,
                                                             generator=torch.Generator().manual_seed(4), dtype=DTYPE))
            ObservedParameter("y", Normal(f, 0.3), y)
    return model, f


def _finite_difference_check(model, eps=1e-6, rtol=1e-4, atol=1e-3):
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


@pytest.mark.parametrize("kind", ["exact", "latent"])
@pytest.mark.parametrize("learnable_inputs", [False, True])
def test_gp_gradients_match_finite_differences(kind, learnable_inputs):
    model, _ = _model(kind, learnable_inputs)
    _finite_difference_check(model)


@pytest.mark.parametrize("kind", ["exact", "latent"])
def test_batched_gp_matches_chain_by_chain(kind):
    model, _ = _model(kind)
    generator = torch.Generator().manual_seed(0)
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        values[name] = u.unsqueeze(0) + 0.2 * torch.randn((4, *u.shape), generator=generator, dtype=u.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(4, values=values, mode="auto"):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            batched_lp = model.model_log_prob()
            batched_grads = model.joint_grad_log_prob()
    with model.batched(4, values=values, mode="loop"):
        loop_lp = model.model_log_prob()
        loop_grads = model.joint_grad_log_prob()
    torch.testing.assert_close(batched_lp, loop_lp, atol=1e-10, rtol=1e-10)
    for name, gradient in loop_grads.items():
        torch.testing.assert_close(batched_grads[name], gradient, atol=1e-10, rtol=1e-10, msg=f"gradient of {name}")


def test_exact_gp_log_density_matches_the_dense_multivariate_normal():
    inputs, y = _data()
    generator = torch.Generator().manual_seed(8)
    w = torch.randn(P, 2, generator=generator, dtype=DTYPE)
    kappa = torch.tensor([0.3, 0.1, 0.5], dtype=DTYPE)
    with Model() as model:
        gp = ExactGP("f", Data("x", inputs), IntrinsicCoregionalization(RBF(0.8, 1.0), w, kappa), noise_variance=0.05)
        ObservedParameter("y", gp, y)
    covariance = _reference(inputs, [w], [kappa], [RBF(0.8, 1.0)]) + 0.05 * torch.eye(inputs.shape[0], dtype=DTYPE)
    expected = torch.distributions.MultivariateNormal(torch.zeros_like(y), covariance).log_prob(y)
    assert abs(float(model.model_log_prob()) - float(expected)) < 1e-4  # the GP adds a relative jitter


def test_prediction_for_a_new_output_uses_the_correlated_outputs():
    # Outputs 0 and 1 are the same function (W = 1, κ = 0); output 1 is observed only on the left half.
    x = torch.linspace(0.0, 3.0, 25, dtype=DTYPE)
    truth = torch.sin(x)
    rows = [(i, 0) for i in range(25)] + [(i, 1) for i in range(10)]
    inputs = torch.tensor([[float(x[i]), float(o)] for i, o in rows], dtype=DTYPE)
    y = torch.stack([truth[i] for i, _ in rows])
    with Model() as model:
        data = Data("x", inputs)
        kernel = IntrinsicCoregionalization(RBF(1.0, 1.0), torch.ones(2, 1, dtype=DTYPE))
        f = ExactGP("f", data, kernel, noise_variance=1e-4)
        ObservedParameter("y", f, y)
    new = torch.tensor([[float(v), 1.0] for v in x[15:]], dtype=DTYPE)
    draws = exact_gp_predictive(f, {}, y, new, n_samples=200)[0]
    assert draws.shape == (200, 10)
    assert (draws.mean(dim=0) - truth[15:]).abs().max() < 0.05
    assert draws.std(dim=0).max() < 0.2
