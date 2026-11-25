"""Gauss-Newton Laplace approximation: curvature, mode finding, subset Laplace and the linearised predictive."""

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, laplace_approximation, plate
from BayesianDLL.Deterministic import Exp, Linear, Tanh
from BayesianDLL.Distributions import Bernoulli, Binomial, HalfNormal, MultivariateNormal, Normal, Poisson
from BayesianDLL.Variational._ggn import GaussNewtonCurvature, NodeJacobians
from BayesianDLL.Variational._groups import LatentLayout
from BayesianDLL.Variational._laplace import finite_difference_hessian, ggn_hessian


def _regression(n=25, scale=0.3, prior=2.0, dynamic_plate=False):
    X = torch.randn(n, 3)
    y = X @ torch.tensor([1.0, -2.0, 0.5]) + 0.7 + scale * torch.randn(n)
    with Model() as model:
        x = Data("X", X, event_ndim=1 if dynamic_plate else 0)
        w = RandomParameter("w", Normal(0.0, prior), shape=(3,))
        b = RandomParameter("b", Normal(0.0, prior), shape=(1,))
        f = Linear("f", x, coefficients=w, intercept=b)
        with plate("obs", x if dynamic_plate else n):
            ObservedParameter("y", Normal(f, scale), y)
    return model, X, y


def _exact_posterior(X, y, scale, prior):
    design = torch.cat([X, torch.ones(len(y), 1)], dim=1)
    precision = design.T @ design / scale ** 2 + torch.eye(4) / prior ** 2
    return torch.linalg.solve(precision, design.T @ y / scale ** 2), precision


def test_ggn_equals_exact_hessian_for_linear_gaussian_model():
    model, X, y = _regression()
    layout = LatentLayout(model)
    flat = layout.current(model) + torch.randn(4)
    ggn = ggn_hessian(model, layout, flat)
    exact = finite_difference_hessian(model, layout, flat)
    torch.testing.assert_close(ggn, exact, atol=1e-6, rtol=1e-6)
    _, precision = _exact_posterior(X, y, 0.3, 2.0)
    torch.testing.assert_close(-ggn, precision, atol=1e-8, rtol=1e-8)


def test_ggn_laplace_is_exact_for_bayesian_linear_regression():
    model, X, y = _regression()
    approximation = laplace_approximation(model, map_epochs=0, hessian="ggn")
    mean, precision = _exact_posterior(X, y, 0.3, 2.0)
    torch.testing.assert_close(approximation.mean, mean, atol=1e-8, rtol=0)
    torch.testing.assert_close(approximation.cov, torch.linalg.inv(precision), atol=1e-8, rtol=1e-6)
    design = torch.cat([X, torch.ones(len(y), 1)], dim=1)
    evidence = torch.distributions.MultivariateNormal(
        torch.zeros(len(y)), 0.3 ** 2 * torch.eye(len(y)) + 4.0 * design @ design.T).log_prob(y)
    assert approximation.log_marginal_likelihood.item() == pytest.approx(evidence.item(), abs=1e-6)
    assert approximation.method == "ggn"


def test_ggn_matches_exact_hessian_for_canonical_links():
    X = torch.randn(30, 2)
    counts = torch.poisson(torch.exp(0.3 * X[:, 0] - 0.2 * X[:, 1]))
    labels = (torch.rand(30) < torch.sigmoid(X[:, 0] - X[:, 1])).to(torch.get_default_dtype())
    for likelihood in ("poisson", "bernoulli_logits", "binomial_logits"):
        with Model() as model:
            x = Data("X", X)
            w = RandomParameter("w", Normal(0.0, 1.5), shape=(2,))
            eta = Linear("eta", x, coefficients=w)
            with plate("obs", 30):
                if likelihood == "poisson":
                    ObservedParameter("y", Poisson(Exp("rate", eta)), counts)
                elif likelihood == "bernoulli_logits":
                    ObservedParameter("y", Bernoulli(logits=eta), labels)
                else:
                    ObservedParameter("y", Binomial(4, logits=eta), torch.clamp(counts, max=4))
        layout = LatentLayout(model)
        flat = torch.tensor([0.4, -0.3])
        torch.testing.assert_close(ggn_hessian(model, layout, flat), finite_difference_hessian(model, layout, flat), atol=1e-5, rtol=1e-5)


class _NoCurvatureNormal(Normal):
    def log_pdf_param_curvature(self, x):
        raise NotImplementedError


def test_terms_without_curvature_fall_back_to_the_exact_hessian():
    X = torch.randn(15, 2)
    y = X @ torch.tensor([1.0, -1.0]) + 0.2 * torch.randn(15)
    with Model() as model:
        w = RandomParameter("w", Normal(0.0, 1.0), shape=(2,))
        f = Linear("f", Data("X", X), coefficients=w)
        with plate("obs", 15):
            ObservedParameter("y", _NoCurvatureNormal(f, 0.5), y)
    layout = LatentLayout(model)
    flat = torch.tensor([0.3, 0.1])
    curvature = GaussNewtonCurvature(model, layout)
    assert curvature.fallback_terms == ["y"]
    torch.testing.assert_close(-curvature.matrix, finite_difference_hessian(model, layout, flat), atol=1e-6, rtol=1e-6)


def test_ggn_is_positive_semidefinite_for_hierarchical_models():
    groups = torch.arange(5).repeat_interleave(4)
    y = torch.randn(20) + torch.randn(5)[groups]
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 3.0))
        tau = RandomParameter("tau", HalfNormal(2.0))
        theta = RandomParameter("theta", Normal(mu, tau), shape=(5,))
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mean = DeterministicParameter("mean", lambda t: t[groups], None, [theta],
                                      vjp_func=lambda c, t: {"theta": torch.zeros_like(t).index_add_(0, groups, c)})
        with plate("obs", 20):
            ObservedParameter("y", Normal(mean, sigma), y)
    layout = LatentLayout(model)
    for _ in range(3):
        flat = layout.current(model) + torch.randn(layout.dim)
        with model.temporarily_set_many(layout.unpack(flat)):
            matrix = GaussNewtonCurvature(model, layout).matrix
        assert torch.isfinite(matrix).all()
        assert torch.linalg.eigvalsh(matrix).min() > -1e-8
    approximation = laplace_approximation(model, map_epochs=200, lr=0.05, hessian="ggn")
    assert torch.isfinite(approximation.cov).all()


def test_ggn_with_latent_covariance_and_vector_events():
    """A multivariate normal likelihood with latent mean and covariance (the covariance part is differentiated by finite differences)."""
    y = torch.randn(6, 2)
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 2.0), shape=(2,))
        s = RandomParameter("s", HalfNormal(1.0))
        cov = DeterministicParameter(
            "cov", lambda s: s * torch.eye(2) + 0.3, lambda s: {"s": torch.eye(2).reshape(-1, 1)}, [s],
        )
        with plate("obs", 6):
            ObservedParameter("y", MultivariateNormal(mu, covariance=cov), y)
    layout = LatentLayout(model)
    curvature = GaussNewtonCurvature(model, layout)
    assert curvature.fallback_terms == ["y"]
    assert torch.linalg.eigvalsh(curvature.matrix).min() > -1e-8
    with Model() as mean_only:
        mu = RandomParameter("mu", Normal(0.0, 2.0), shape=(2,))
        with plate("obs", 6):
            ObservedParameter("y", MultivariateNormal(mu, covariance=torch.tensor([[1.0, 0.4], [0.4, 2.0]])), y)
    layout = LatentLayout(mean_only)
    flat = torch.tensor([0.2, -0.1])
    curvature = GaussNewtonCurvature(mean_only, layout)
    assert curvature.fallback_terms == []
    torch.testing.assert_close(-curvature.matrix, finite_difference_hessian(mean_only, layout, flat), atol=1e-6, rtol=1e-6)


def test_node_jacobians_match_autograd():
    X = torch.randn(7, 1)
    with Model() as model:
        x = Data("X", X)
        w1 = RandomParameter("w1", Normal(0.0, 1.0), shape=(1, 3))
        b1 = RandomParameter("b1", Normal(0.0, 1.0), shape=(3,))
        w2 = RandomParameter("w2", Normal(0.0, 1.0), shape=(3,))
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        h = Tanh("h", Linear("lin", x, coefficients=w1, intercept=b1))
        f = Linear("f", h, coefficients=w2)
        with plate("obs", 7):
            ObservedParameter("y", Normal(f, sigma), torch.randn(7))
    layout = LatentLayout(model)
    flat = layout.current(model) + 0.3 * torch.randn(layout.dim)
    jacobians = NodeJacobians(model, layout)

    def forward(vector):
        values = layout.unpack(vector)
        w1_, b1_, w2_ = values["w1"], values["b1"], values["w2"]
        return torch.tanh(X @ w1_ + b1_) @ w2_

    with model.temporarily_set_many(layout.unpack(flat)):
        jacobian = jacobians.jacobian("f")
    expected = torch.autograd.functional.jacobian(forward, flat)
    torch.testing.assert_close(jacobian, expected, atol=1e-10, rtol=1e-8)


def test_ggn_levenberg_marquardt_finds_the_mode_from_a_poor_start():
    X = torch.randn(40, 2)
    counts = torch.poisson(torch.exp(0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.5))
    with Model() as model:
        w = RandomParameter("w", Normal(0.0, 2.0), shape=(2,), initial_value=torch.tensor([2.0, -2.0]))
        b = RandomParameter("b", Normal(0.0, 2.0), shape=(1,), initial_value=torch.tensor([-1.5]))
        eta = Linear("eta", Data("X", X), coefficients=w, intercept=b)
        with plate("obs", 40):
            ObservedParameter("y", Poisson(Exp("rate", eta)), counts)
    approximation = laplace_approximation(model, map_epochs=0, hessian="ggn", newton_steps=50)
    layout = approximation.layout
    assert float(model.joint_grad_log_prob()["w"].abs().max()) < 1e-7
    reference = laplace_approximation(model, map_epochs=0, hessian="fd")
    torch.testing.assert_close(approximation.mean, reference.mean, atol=1e-6, rtol=0)
    torch.testing.assert_close(approximation.cov, reference.cov, atol=1e-5, rtol=1e-4)


def test_subset_laplace_is_the_conditional_posterior():
    model, X, y = _regression()
    approximation = laplace_approximation(model, map_epochs=0, hessian="ggn", params=["w"])
    assert approximation.names == ["w"]
    mean, precision = _exact_posterior(X, y, 0.3, 2.0)
    # b is held at its starting value 0 (no refinement of the others), so condition on that.
    design = X
    conditional_precision = design.T @ design / 0.3 ** 2 + torch.eye(3) / 2.0 ** 2
    torch.testing.assert_close(approximation.cov, torch.linalg.inv(conditional_precision), atol=1e-8, rtol=1e-6)
    expected_mean = torch.linalg.solve(conditional_precision, design.T @ y / 0.3 ** 2)
    torch.testing.assert_close(approximation.mean, expected_mean, atol=1e-8, rtol=0)
    trace = approximation.sample(5, n_chains=1)
    assert trace["w"].shape == (1, 5, 3)
    assert torch.allclose(trace["b"], torch.zeros(1, 5, 1))
    for method in ("fd", "ggn"):
        again = laplace_approximation(model, map_epochs=0, hessian=method, params=[model.params["w"]])
        torch.testing.assert_close(again.cov, approximation.cov, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown latent"):
        laplace_approximation(model, map_epochs=0, params=["nothing"])
