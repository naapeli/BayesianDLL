"""Density oracles and analytic gradients for every exported distribution.

``torch.distributions`` and ``torch.autograd`` are used here only as reference
oracles; the library itself computes every gradient by hand.
"""

import math

import pytest
import torch
import torch.distributions as td

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Categorical, Cauchy, CorrCholeskyTransform, Dirichlet, DiscreteUniform, Exponential,
    Gamma, Geometric, HalfCauchy, HalfNormal, InvGamma, Laplace, LKJCholesky, LogNormal, LogTransform, Mixture,
    Multinomial, MultivariateNormal, NegativeBinomial, Normal, Poisson, StickBreakingTransform, StudentT,
    TransformedDistribution, Truncated, Uniform, ZeroInflated, Wishart, InverseWishart, PositiveDefiniteTransform,
)


t = torch.tensor
COV = [[2.0, 0.4], [0.4, 1.0]]


def _truncated_normal_reference(x):
    base = td.Normal(0.5, 1.5)
    log_z = torch.log(base.cdf(t(2.0)) - base.cdf(t(-1.0)))
    return base.log_prob(x) - log_z


def _lkj_point():
    return CorrCholeskyTransform(3).inverse(t([0.3, -0.5, 0.8]))


def continuous_case(name):
    cases = {
        "normal": lambda: (Normal(0.4, 2.0), td.Normal(0.4, 2.0), t([-1.0, 0.3, 2.0])),
        "half_normal": lambda: (HalfNormal(1.3), td.HalfNormal(1.3), t([0.1, 0.8, 2.5])),
        "log_normal": lambda: (LogNormal(0.2, 0.7), td.LogNormal(0.2, 0.7), t([0.3, 1.0, 2.5])),
        "multivariate_normal": lambda: (
            MultivariateNormal(t([0.2, -0.3]), t(COV)), td.MultivariateNormal(t([0.2, -0.3]), t(COV)),
            t([[0.1, 0.5], [1.2, -0.4]])),
        "mvn_scale_tril": lambda: (
            MultivariateNormal(t([0.2, -0.3]), scale_tril=torch.linalg.cholesky(t(COV))),
            td.MultivariateNormal(t([0.2, -0.3]), t(COV)), t([[0.1, 0.5], [1.2, -0.4]])),
        "beta": lambda: (Beta(2.0, 3.0), td.Beta(2.0, 3.0), t([0.1, 0.4, 0.8])),
        "exponential": lambda: (Exponential(1.3), td.Exponential(1.3), t([0.1, 0.8, 2.0])),
        "gamma": lambda: (Gamma(2.5, 1.5), td.Gamma(2.5, 1.5), t([0.2, 1.1, 3.0])),
        "uniform": lambda: (Uniform(-2.0, 3.0), td.Uniform(-2.0, 3.0), t([-1.0, 0.2, 2.0])),
        "inverse_gamma": lambda: (InvGamma(2.0, 3.0), td.InverseGamma(2.0, 3.0), t([0.3, 1.2, 3.0])),
        "half_cauchy": lambda: (HalfCauchy(1.7), td.HalfCauchy(1.7), t([0.1, 1.0, 3.0])),
        "cauchy": lambda: (Cauchy(0.5, 1.7), td.Cauchy(0.5, 1.7), t([-2.0, 0.4, 3.0])),
        "laplace": lambda: (Laplace(0.5, 1.7), td.Laplace(0.5, 1.7), t([-2.0, 0.4, 3.0])),
        "student_t": lambda: (StudentT(4.0, 0.5, 1.7), td.StudentT(4.0, 0.5, 1.7), t([-2.0, 0.4, 3.0])),
        "dirichlet": lambda: (Dirichlet(t([2.0, 3.0, 4.0])), td.Dirichlet(t([2.0, 3.0, 4.0])), t([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])),
        "lkj_cholesky": lambda: (LKJCholesky(3, 2.0), td.LKJCholesky(3, 2.0), _lkj_point()),
        "mixture": lambda: (
            Mixture([Normal(-1.0, 0.5), Normal(2.0, 2.0)], t([0.3, 0.7])),
            td.MixtureSameFamily(td.Categorical(t([0.3, 0.7])), td.Normal(t([-1.0, 2.0]), t([0.5, 2.0]))),
            t([-2.0, 0.5, 3.0])),
        "truncated_normal": lambda: (
            Truncated(Normal(0.5, 1.5), -1.0, 2.0),
            type("Ref", (), {"log_prob": staticmethod(_truncated_normal_reference)})(),
            t([-0.5, 0.4, 1.9])),
        "transformed": lambda: (TransformedDistribution(Normal(0.2, 0.7), LogTransform()), td.LogNormal(0.2, 0.7), t([0.3, 1.0, 2.5])),
    }
    return cases[name]()


WISHART_SCALE = [[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 1.5]]
WISHART_X = [[[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 4.0]], [[1.0, -0.2, 0.0], [-0.2, 0.5, 0.1], [0.0, 0.1, 0.8]]]


class _InverseWishartReference:
    """p_IW(X; df, S) = p_W(X^-1; df, S^-1) |X|^-(D+1)."""

    def __init__(self, df, scale):
        self.wishart = td.Wishart(torch.as_tensor(df), torch.linalg.inv(scale))

    def log_prob(self, x):
        return self.wishart.log_prob(torch.linalg.inv(x)) - 4 * torch.logdet(x)


def wishart_case(name):
    S, X = t(WISHART_SCALE), t(WISHART_X)
    if name == "wishart":
        return Wishart(5.0, S), td.Wishart(t(5.0), S), X
    if name == "wishart_tril":
        return Wishart(5.0, scale_tril=torch.linalg.cholesky(S)), td.Wishart(t(5.0), S), X
    if name == "inverse_wishart":
        return InverseWishart(6.0, S), _InverseWishartReference(6.0, S), X
    return InverseWishart(6.0, scale_tril=torch.linalg.cholesky(S)), _InverseWishartReference(6.0, S), X


WISHART_CASES = ["wishart", "wishart_tril", "inverse_wishart", "inverse_wishart_tril"]


@pytest.mark.parametrize("name", WISHART_CASES)
def test_wishart_family_density_and_value_gradient(name):
    dist, reference, X = wishart_case(name)
    torch.testing.assert_close(dist.log_pdf(X), reference.log_prob(X))
    torch.testing.assert_close(dist.pdf(X), reference.log_prob(X).exp())
    grad = dist.log_pdf_grad(X)
    torch.testing.assert_close(grad, grad.transpose(-2, -1))
    eps = 1e-6
    for i, j in [(0, 0), (1, 0), (2, 1), (2, 2)]:
        E = torch.zeros(3, 3)
        E[i, j] = E[j, i] = eps
        numeric = (dist.log_pdf(X + E) - dist.log_pdf(X - E)) / (2 * eps)
        torch.testing.assert_close((grad * E).sum((-2, -1)) / eps, numeric, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", WISHART_CASES)
def test_wishart_family_parameter_gradients(name):
    S = t(WISHART_SCALE)
    X = t(WISHART_X)
    family = Wishart if name.startswith("wishart") else InverseWishart
    df = torch.tensor(5.5, requires_grad=True)
    if name.endswith("tril"):
        L = torch.linalg.cholesky(S).requires_grad_()
        dist = family(df, scale_tril=L)
        expected_df, expected_L = torch.autograd.grad(dist.log_pdf(X).sum(), (df, L))
        grads = dist.log_pdf_param_grads(X)
        torch.testing.assert_close(grads["scale_tril"].sum(0), torch.tril(expected_L))
    else:
        dist = family(df, S)
        expected_df, = torch.autograd.grad(dist.log_pdf(X).sum(), df)
        grads = dist.log_pdf_param_grads(X)
        G = grads["scale"]
        torch.testing.assert_close(G, G.transpose(-2, -1))
        eps = 1e-6
        for i, j in [(0, 0), (1, 0), (2, 1)]:
            E = torch.zeros(3, 3)
            E[i, j] = E[j, i] = eps
            numeric = (family(df.detach(), S + E).log_pdf(X) - family(df.detach(), S - E).log_pdf(X)).sum() / (2 * eps)
            assert ((G * E).sum() / eps).item() == pytest.approx(numeric.item(), rel=1e-6)
    torch.testing.assert_close(grads["df"].sum(), expected_df)


@pytest.mark.parametrize("name", WISHART_CASES)
def test_wishart_family_unconstrained(name):
    dist, _, X = wishart_case(name)
    z = dist.transform.forward(X)
    assert z.shape == (2, 6)
    torch.testing.assert_close(dist.transform.inverse(z), X)
    torch.testing.assert_close(dist._log_prob_unconstrained(z), dist.log_pdf(X).sum() + dist.transform.log_abs_det_jacobian(z))
    z_grad = z.clone().requires_grad_()
    expected, = torch.autograd.grad(dist._log_prob_unconstrained(z_grad), z_grad)
    torch.testing.assert_close(dist._log_prob_grad_unconstrained(z), expected)


def test_positive_definite_transform_jacobian():
    T = PositiveDefiniteTransform(3)
    z = t([0.3, -0.4, 0.2, 0.7, -0.1, 0.5])
    jac = torch.autograd.functional.jacobian(T.inverse, z)
    torch.testing.assert_close(T.derivative(z), jac)
    rows, cols = torch.tril_indices(3, 3)
    torch.testing.assert_close(T.log_abs_det_jacobian(z), torch.linalg.slogdet(jac[rows, cols])[1])
    z_grad = z.clone().requires_grad_()
    expected, = torch.autograd.grad(T.log_abs_det_jacobian(z_grad), z_grad)
    torch.testing.assert_close(T.grad_log_abs_det_jacobian(z), expected)
    cotangent = torch.randn(3, 3)
    torch.testing.assert_close(T.vjp(z, cotangent), (jac * cotangent.unsqueeze(-1)).sum((0, 1)))
    assert T.unconstrained_shape((4, 3, 3)) == (4, 6)
    torch.testing.assert_close(T.inverse(torch.zeros(6)), torch.eye(3))


def test_model_gradient_through_wishart_prior():
    with Model() as model:
        cov = RandomParameter("cov", Wishart(4.0, t(WISHART_SCALE)))
        ObservedParameter("data", MultivariateNormal(torch.zeros(3), cov), t([[0.3, -0.2, 1.0], [1.5, 0.4, -0.7]]))
    assert model.params["cov"].unconstrained_value.shape == (6,)
    model.params["cov"].set_unconstrained_value(t([0.3, -0.4, 0.2, 0.7, -0.1, 0.5]))
    _assert_model_gradients_match_autograd(model)


CONTINUOUS = ["normal", "half_normal", "log_normal", "multivariate_normal", "mvn_scale_tril", "beta", "exponential",
              "gamma", "uniform", "inverse_gamma", "half_cauchy", "cauchy", "laplace", "student_t", "dirichlet",
              "lkj_cholesky", "mixture", "truncated_normal", "transformed"]


@pytest.mark.parametrize("name", CONTINUOUS)
def test_continuous_densities_match_torch(name):
    dist, reference, x = continuous_case(name)
    torch.testing.assert_close(dist.log_pdf(x), reference.log_prob(x))
    torch.testing.assert_close(dist.pdf(x), reference.log_prob(x).exp())


@pytest.mark.parametrize("name", [n for n in CONTINUOUS if n != "uniform"])
def test_value_gradients_match_autograd(name):
    dist, _, x = continuous_case(name)
    x.requires_grad_()
    expected, = torch.autograd.grad(dist.log_pdf(x).sum(), x)
    torch.testing.assert_close(dist.log_pdf_grad(x), expected)


@pytest.mark.parametrize("name", CONTINUOUS)
def test_unconstrained_density_and_gradient(name):
    dist, _, x = continuous_case(name)
    z = dist.transform.forward(x).detach()
    # round trip, change of variables, gradient vs autograd and finite differences
    torch.testing.assert_close(dist.transform.inverse(z), x)
    logp = dist._log_prob_unconstrained(z)
    assert logp.ndim == 0
    expected_logp = dist.log_pdf(dist.transform.inverse(z)).sum() + dist.transform.log_abs_det_jacobian(z)
    torch.testing.assert_close(logp, expected_logp)
    z_grad = z.clone().requires_grad_()
    expected, = torch.autograd.grad(dist._log_prob_unconstrained(z_grad), z_grad)
    actual = dist._log_prob_grad_unconstrained(z)
    torch.testing.assert_close(actual, expected)
    eps = 1e-6
    flat = z.reshape(-1)
    for i in range(flat.numel()):
        shift = torch.zeros_like(flat)
        shift[i] = eps
        numeric = (dist._log_prob_unconstrained((flat + shift).reshape(z.shape))
                   - dist._log_prob_unconstrained((flat - shift).reshape(z.shape))) / (2 * eps)
        assert actual.reshape(-1)[i].item() == pytest.approx(numeric.item(), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("transform,z", [
    (StickBreakingTransform(), t([0.3, -1.2, 0.7])),
    (CorrCholeskyTransform(4), t([0.3, -1.2, 0.7, 0.1, -0.4, 1.5])),
])
def test_vector_transforms_jacobians(transform, z):
    z = z.to(torch.float64)
    x = transform.inverse(z)
    torch.testing.assert_close(transform.forward(x), z)
    jac = torch.autograd.functional.jacobian(transform.inverse, z)
    torch.testing.assert_close(transform.derivative(z), jac)
    if isinstance(transform, StickBreakingTransform):
        square = jac[:-1]
        assert transform.unconstrained_shape((2, 4)) == (2, 3)
    else:
        rows, cols = torch.tril_indices(4, 4, offset=-1)
        square = jac[rows, cols]
        assert transform.unconstrained_shape((4, 4)) == (6,)
    torch.testing.assert_close(transform.log_abs_det_jacobian(z), torch.linalg.slogdet(square)[1])
    z_grad = z.clone().requires_grad_()
    expected, = torch.autograd.grad(transform.log_abs_det_jacobian(z_grad), z_grad)
    torch.testing.assert_close(transform.grad_log_abs_det_jacobian(z), expected)
    cotangent = torch.randn(x.shape)
    if isinstance(transform, CorrCholeskyTransform):
        cotangent = torch.tril(cotangent)
    torch.testing.assert_close(transform.vjp(z, cotangent), (jac * cotangent.unsqueeze(-1)).reshape(-1, z.numel()).sum(0))


def test_stick_breaking_zero_is_uniform_simplex():
    x = StickBreakingTransform().inverse(torch.zeros(2, 4))
    torch.testing.assert_close(x, torch.full((2, 5), 0.2))


PARAMETER_CASES = [
    (Normal, ["mean", "sigma"], [0.3, 1.7], [-0.5, 0.2, 1.5]),
    (HalfNormal, ["sigma"], [1.3], [0.2, 0.9, 2.0]),
    (LogNormal, ["mean", "sigma"], [0.3, 0.8], [0.5, 1.2, 3.0]),
    (Beta, ["a", "b"], [2.0, 3.0], [0.2, 0.4, 0.7]),
    (Exponential, ["rate"], [1.4], [0.2, 1.0, 3.0]),
    (Gamma, ["alpha", "beta"], [2.5, 1.5], [0.2, 1.0, 3.0]),
    (InvGamma, ["alpha", "beta"], [2.0, 3.0], [0.2, 1.0, 3.0]),
    (HalfCauchy, ["scale"], [1.4], [0.2, 1.0, 3.0]),
    (Cauchy, ["mean", "scale"], [0.3, 1.4], [-1.0, 0.5, 3.0]),
    (Laplace, ["mean", "scale"], [0.3, 1.4], [-1.0, 0.5, 3.0]),
    (StudentT, ["nu", "mean", "sigma"], [4.0, 0.3, 1.4], [-1.0, 0.5, 3.0]),
    (Uniform, ["low", "high"], [-1.0, 2.0], [-0.5, 0.5, 1.5]),
    (lambda low, high: Truncated(Normal(0.3, 1.2), low, high), ["low", "high"], [-1.0, 2.0], [-0.5, 0.5, 1.5]),
    (lambda m, s: Truncated(Normal(m, s), -1.0, 2.0), ["mean", "sigma"], [0.3, 1.2], [-0.5, 0.5, 1.5]),
    (lambda r: Truncated(Exponential(r), 0.5, 3.0), ["rate"], [0.8], [0.6, 1.5, 2.9]),
    (lambda m, s: Truncated(Laplace(m, s), high=1.0), ["mean", "scale"], [0.3, 1.2], [-3.0, 0.5, 0.9]),
    (lambda m, s: Truncated(Cauchy(m, s), low=-1.0), ["mean", "scale"], [0.3, 1.2], [-0.5, 0.5, 9.0]),
    (lambda m, s: TransformedDistribution(Normal(m, s), LogTransform()), ["mean", "sigma"], [0.3, 0.8], [0.5, 1.2, 3.0]),
    (Bernoulli, ["p"], [0.4], [0.0, 1.0, 1.0]),
    (lambda logits: Bernoulli(logits=logits), ["logits"], [-0.7], [0.0, 1.0, 1.0]),
    (Geometric, ["p"], [0.4], [1.0, 2.0, 4.0]),
    (lambda logits: Geometric(logits=logits), ["logits"], [0.3], [1.0, 2.0, 4.0]),
    (lambda p: Binomial(5, p), ["p"], [0.4], [0.0, 2.0, 5.0]),
    (lambda logits: Binomial(5, logits=logits), ["logits"], [0.4], [0.0, 2.0, 5.0]),
    (Poisson, ["rate"], [2.3], [0.0, 2.0, 5.0]),
    (lambda r, p: NegativeBinomial(r, p), ["total_count", "p"], [3.5, 0.4], [0.0, 2.0, 5.0]),
    (lambda r, logits: NegativeBinomial(r, logits=logits), ["total_count", "logits"], [3.5, -0.3], [0.0, 2.0, 5.0]),
    (lambda rate, gate: ZeroInflated(Poisson(rate), gate), ["rate", "gate"], [2.3, 0.3], [0.0, 2.0, 5.0, 0.0]),
    (DiscreteUniform, ["low", "high"], [1.0, 4.0], [1.0, 2.0, 4.0]),
]


@pytest.mark.parametrize("factory,keys,values,x", PARAMETER_CASES)
def test_parameter_gradients(factory, keys, values, x):
    params = [torch.tensor(v, requires_grad=True) for v in values]
    dist = factory(*params)
    x = torch.tensor(x)
    actual = dist.log_pdf_param_grads(x)
    if factory is DiscreteUniform:
        # the bounds enter log p only through -log(high - low + 1)
        expected = [torch.tensor(3 / 4.0), torch.tensor(-3 / 4.0)]
    else:
        expected = torch.autograd.grad(dist.log_pdf(x).sum(), params)
    for key, grad in zip(keys, expected):
        torch.testing.assert_close(actual[key].sum(), grad)


def test_dirichlet_and_categorical_parameter_gradients():
    alpha = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    x = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    expected, = torch.autograd.grad(Dirichlet(alpha).log_pdf(x).sum(), alpha)
    torch.testing.assert_close(Dirichlet(alpha).log_pdf_param_grads(x)["alpha"].sum(0), expected)

    for kind in ("probs", "logits"):
        param = torch.tensor([[0.2, 0.5, 0.3], [1.0, 2.0, 0.5]], requires_grad=True)
        dist = Categorical(**{kind: param})
        x = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])  # (3, 2) values for batch (2,)
        expected, = torch.autograd.grad(dist.log_pdf(x).sum(), param)
        torch.testing.assert_close(dist.log_pdf_param_grads(x)[kind].sum(0), expected)

        dist = Multinomial(6, **{kind: param})
        counts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 6.0, 0.0]])
        expected, = torch.autograd.grad(dist.log_pdf(counts).sum(), param)
        torch.testing.assert_close(dist.log_pdf_param_grads(counts)[kind], expected)


def test_lkj_parameter_gradient():
    eta = torch.tensor(2.5, requires_grad=True)
    dist = LKJCholesky(3, eta)
    x = _lkj_point()
    expected, = torch.autograd.grad(dist.log_pdf(x), eta)
    torch.testing.assert_close(dist.log_pdf_param_grads(x)["eta"], expected)


@pytest.mark.parametrize("parameterisation", ["covariance", "scale_tril"])
def test_multivariate_normal_batched_parameter_gradients(parameterisation):
    mu = torch.tensor([[0.2, -0.3, 0.1], [0.0, 0.5, -0.2]], requires_grad=True)  # batch of 2
    A = torch.tensor([[1.0, 0.2, 0.0], [0.3, 1.5, 0.1], [0.0, -0.4, 0.8]])
    cov = (A @ A.T + 0.5 * torch.eye(3)).requires_grad_()
    scale_tril = torch.linalg.cholesky(cov.detach()).requires_grad_()
    x = torch.tensor([[[0.1, 0.5, -1.0], [1.2, -0.4, 0.3]], [[0.0, 0.0, 0.0], [-0.5, 1.0, 2.0]]])  # (2, 2, 3)
    if parameterisation == "covariance":
        dist, matrix = MultivariateNormal(mu, cov), cov
    else:
        dist, matrix = MultivariateNormal(mu, scale_tril=scale_tril), scale_tril
    reference = td.MultivariateNormal(mu, scale_tril=torch.linalg.cholesky(cov)) if parameterisation == "covariance" else td.MultivariateNormal(mu, scale_tril=scale_tril)
    torch.testing.assert_close(dist.log_pdf(x), reference.log_prob(x))
    expected_mu, expected_matrix = torch.autograd.grad(dist.log_pdf(x).sum(), (mu, matrix))
    grads = dist.log_pdf_param_grads(x)
    torch.testing.assert_close(grads["mean"].sum(0), expected_mu)
    actual_matrix = grads[parameterisation].sum(dim=(0, 1))
    if parameterisation == "covariance":
        # Cholesky only reads the lower triangle, so autograd returns the
        # gradient folded onto it; the analytic one is the full symmetric matrix.
        torch.testing.assert_close(actual_matrix, actual_matrix.T)
        folded = torch.tril(actual_matrix) + torch.tril(actual_matrix, diagonal=-1)
        torch.testing.assert_close(folded, torch.tril(expected_matrix + expected_matrix.T - torch.diag(torch.diag(expected_matrix))))
    else:
        torch.testing.assert_close(actual_matrix, expected_matrix)


def test_multivariate_normal_covariance_gradient_finite_difference():
    mu = torch.tensor([0.2, -0.3])
    cov = torch.tensor(COV)
    x = torch.tensor([1.0, 0.4])
    grad = MultivariateNormal(mu, cov).log_pdf_param_grads(x)["covariance"]
    eps = 1e-6
    for i, j in [(0, 0), (1, 1), (0, 1)]:
        E = torch.zeros(2, 2)
        E[i, j] = E[j, i] = eps
        numeric = (MultivariateNormal(mu, cov + E).log_pdf(x) - MultivariateNormal(mu, cov - E).log_pdf(x)) / (2 * eps)
        directional = (grad * E / eps).sum()
        assert directional.item() == pytest.approx(numeric.item(), rel=1e-6)


@pytest.mark.parametrize("name", ["bernoulli", "bernoulli_logits", "binomial", "binomial_logits", "geometric",
                                  "geometric_logits", "discrete_uniform", "poisson", "negative_binomial",
                                  "negative_binomial_logits", "categorical", "categorical_logits", "multinomial",
                                  "zero_inflated_poisson"])
def test_discrete_probabilities(name):
    if name == "bernoulli":
        dist, expected_fn, x = Bernoulli(0.3), td.Bernoulli(0.3).log_prob, torch.arange(2.0)
    elif name == "bernoulli_logits":
        dist, expected_fn, x = Bernoulli(logits=-0.8), td.Bernoulli(logits=t(-0.8)).log_prob, torch.arange(2.0)
    elif name == "binomial":
        dist, expected_fn, x = Binomial(5, 0.3), td.Binomial(5, t(0.3)).log_prob, torch.arange(6.0)
    elif name == "binomial_logits":
        dist, expected_fn, x = Binomial(5, logits=0.4), td.Binomial(5, logits=t(0.4)).log_prob, torch.arange(6.0)
    elif name == "geometric":
        # BayesianDLL counts trials; PyTorch counts failures before the success.
        dist, x = Geometric(0.3), torch.arange(1.0, 20.0)
        expected_fn = lambda v: td.Geometric(t(0.3)).log_prob(v - 1)  # noqa: E731
    elif name == "geometric_logits":
        dist, x = Geometric(logits=0.3), torch.arange(1.0, 20.0)
        expected_fn = lambda v: td.Geometric(logits=t(0.3)).log_prob(v - 1)  # noqa: E731
    elif name == "poisson":
        dist, expected_fn, x = Poisson(2.3), td.Poisson(t(2.3)).log_prob, torch.arange(30.0)
    elif name == "negative_binomial":
        dist, expected_fn, x = NegativeBinomial(3.5, 0.4), td.NegativeBinomial(t(3.5), t(0.4)).log_prob, torch.arange(60.0)
    elif name == "negative_binomial_logits":
        dist, expected_fn, x = NegativeBinomial(3.5, logits=-0.3), td.NegativeBinomial(t(3.5), logits=t(-0.3)).log_prob, torch.arange(60.0)
    elif name == "categorical":
        dist, expected_fn, x = Categorical(t([0.2, 0.5, 0.3])), td.Categorical(t([0.2, 0.5, 0.3])).log_prob, torch.arange(3.0)
    elif name == "categorical_logits":
        dist, expected_fn, x = Categorical(logits=t([0.2, -1.5, 0.3])), td.Categorical(logits=t([0.2, -1.5, 0.3])).log_prob, torch.arange(3.0)
    elif name == "multinomial":
        dist = Multinomial(3, t([0.2, 0.5, 0.3]))
        x = torch.tensor([[a, b, 3.0 - a - b] for a in range(4) for b in range(4 - a)])
        expected_fn = td.Multinomial(3, t([0.2, 0.5, 0.3])).log_prob
    elif name == "zero_inflated_poisson":
        dist, x = ZeroInflated(Poisson(2.3), 0.25), torch.arange(30.0)
        base = td.Poisson(t(2.3)).log_prob(x).exp()
        expected_fn = lambda v: torch.log(0.75 * base + 0.25 * (v == 0))  # noqa: E731
    else:
        dist, x = DiscreteUniform(2, 5), torch.arange(2.0, 6.0)
        expected_fn = lambda v: torch.full_like(v, -math.log(4.0))  # noqa: E731
    expected = expected_fn(x)
    torch.testing.assert_close(dist.log_pdf(x), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dist.pdf(x), expected.exp(), atol=1e-6, rtol=1e-6)
    if not name.startswith("geometric") and name not in ("poisson", "negative_binomial", "negative_binomial_logits", "zero_inflated_poisson"):
        assert dist.pdf(x).sum().item() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(NotImplementedError, match="discrete"):
        dist.log_pdf_grad(x)


def test_logits_are_stable_for_extreme_values():
    assert Bernoulli(logits=t(200.0)).log_pdf(t(0.0)).item() == pytest.approx(-200.0)
    assert Bernoulli(logits=t(-200.0)).log_pdf(t(0.0)).item() == pytest.approx(0.0)
    assert Binomial(10, logits=t(-300.0)).log_pdf(t(10.0)).item() == pytest.approx(-3000.0)
    assert Geometric(logits=t(-100.0)).log_pdf(t(3.0)).item() == pytest.approx(-100.0, abs=1e-10)
    # probabilities on the boundary are exact, not clamped
    assert Bernoulli(t(1.0)).log_pdf(t(1.0)).item() == 0.0
    assert Bernoulli(t(0.0)).log_pdf(t(0.0)).item() == 0.0
    assert torch.isneginf(Bernoulli(t(0.0)).log_pdf(t(1.0)))
    with pytest.raises(ValueError, match="exactly one"):
        Bernoulli(0.3, logits=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        Categorical()


def test_binomial_keeps_float64():
    dist = Binomial(torch.tensor(7), torch.tensor(0.3, dtype=torch.float64))
    out = dist.log_pdf(torch.tensor([0.0, 3.0, 7.0], dtype=torch.float64))
    assert out.dtype == torch.float64
    expected = td.Binomial(7, torch.tensor(0.3, dtype=torch.float64)).log_prob(torch.tensor([0.0, 3.0, 7.0], dtype=torch.float64))
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_uniform_support_and_gradients():
    dist = Uniform(-2.0, 3.0)
    outside = torch.tensor([-2.1, 3.1])
    assert torch.isneginf(dist.log_pdf(outside)).all()
    assert (dist.pdf(outside) == 0).all()
    torch.testing.assert_close(dist.log_pdf_grad(torch.tensor([0.0])), torch.zeros(1))
    grads = dist.log_pdf_param_grads(torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(grads["low"], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(grads["high"], torch.tensor([-0.2, 0.0]))
    grads = DiscreteUniform(0, 3).log_pdf_param_grads(torch.tensor([1.0, 7.0]))
    torch.testing.assert_close(grads["low"], torch.tensor([0.25, 0.0]))
    assert torch.isneginf(DiscreteUniform(0, 3).log_pdf(torch.tensor([7.0])))


def test_distribution_batch_and_event_shapes():
    assert Normal(torch.zeros(3, 1), torch.ones(4)).batch_shape == (3, 4)
    for dist in (Dirichlet(torch.ones(2, 3)), MultivariateNormal(torch.zeros(2, 3), torch.eye(3)), Multinomial(4, torch.ones(2, 3))):
        assert dist.event_shape == (3,)
        assert dist.batch_shape == (2,)
    assert Categorical(torch.ones(2, 3)).batch_shape == (2,)
    assert LKJCholesky(4, torch.ones(2)).event_shape == (4, 4)
    mixture = Mixture([Normal(torch.zeros(3), 1.0), Normal(2.0, 1.0)], [0.4, 0.6])
    assert mixture.event_shape == ()
    assert mixture.batch_shape == (3,)


def test_mixture_weights_and_shared_parameter_gradients():
    mean = torch.tensor(0.4, requires_grad=True)
    weights = torch.tensor([0.3, 0.7], requires_grad=True)
    dist = Mixture([Normal(mean, 1.0), Normal(mean, 3.0)], weights)
    x = torch.tensor([-2.0, 0.1, 2.0])
    mean_grad, weight_grad = torch.autograd.grad(dist.log_pdf(x).sum(), (mean, weights))
    actual = dist.log_pdf_param_grads(x)
    torch.testing.assert_close(actual["mean"].sum(), mean_grad)
    torch.testing.assert_close(actual["weights"].sum(0), weight_grad)


def _assert_model_gradients_match_autograd(model):
    values = {name: p.unconstrained_value.detach().clone().requires_grad_() for name, p in model.params.items()}
    with model.temporarily_set_many(values):
        expected = torch.autograd.grad(model.model_log_prob(), tuple(values.values()))
        actual = model.joint_grad_log_prob()
    for name, grad in zip(values, expected):
        assert actual[name].shape == values[name].shape
        torch.testing.assert_close(actual[name], grad, msg=f"gradient of {name}")


def test_model_gradient_through_stick_breaking_dirichlet():
    with Model() as model:
        weights = RandomParameter("weights", Dirichlet(t([1.5, 2.0, 3.0])))
        ObservedParameter("data", Categorical(weights), t([0.0, 2.0, 2.0, 1.0, 2.0]))
    assert model.params["weights"].unconstrained_value.shape == (2,)
    model.params["weights"].set_unconstrained_value(t([0.4, -0.3]))
    _assert_model_gradients_match_autograd(model)


def test_model_gradient_through_lkj_cholesky():
    with Model() as model:
        L = RandomParameter("L", LKJCholesky(3, 2.0))
        ObservedParameter("data", MultivariateNormal(torch.zeros(3), scale_tril=L), t([[0.3, -0.2, 1.0], [1.5, 0.4, -0.7]]))
    assert model.params["L"].unconstrained_value.shape == (3,)
    torch.testing.assert_close(model.params["L"].constrained_value, torch.eye(3))
    model.params["L"].set_unconstrained_value(t([0.3, -0.5, 0.8]))
    _assert_model_gradients_match_autograd(model)


def test_model_gradient_mvn_with_random_mean_and_deterministic_covariance():
    with Model() as model:
        mean = RandomParameter("mean", Normal(torch.zeros(2), 2.0), shape=2)
        log_scale = RandomParameter("log_scale", Normal(0.0, 1.0))
        base = t([[1.0, 0.3], [0.3, 0.5]])
        covariance = DeterministicParameter(
            "covariance", lambda s: torch.exp(s) * base,
            lambda s: {"log_scale": torch.exp(s) * base}, [log_scale])
        ObservedParameter("data", MultivariateNormal(mean, covariance), t([[0.3, -0.2], [1.5, 0.4], [0.2, 0.1]]))
    model.params["mean"].set_unconstrained_value(t([0.2, -0.1]))
    model.params["log_scale"].set_unconstrained_value(t([0.3]))
    _assert_model_gradients_match_autograd(model)


def test_truncated_and_transformed_unconstrained_support():
    trunc = Truncated(Normal(0.0, 1.0), low=0.5)
    z = torch.linspace(-3, 3, 5)
    assert (trunc.transform.inverse(z) > 0.5).all()
    transformed = TransformedDistribution(Normal(0.0, 1.0), LogTransform())
    torch.testing.assert_close(transformed._log_prob_unconstrained(z), Normal(0.0, 1.0).log_pdf(z).sum())


# ------------------------------------------------------------ discrete supports
# (name, factory(params), params, torch reference(params, x), x inside, x outside)
_SUPPORT_CASES = [
    ("geometric", lambda p: Geometric(p), [0.2], lambda p, x: td.Geometric(p).log_prob(x - 1),
     [1.0, 2.0, 7.0], [0.0, -1.0, 1.5]),
    ("geometric_logits", lambda l: Geometric(logits=l), [-0.4], lambda l, x: td.Geometric(logits=l).log_prob(x - 1),
     [1.0, 2.0, 7.0], [0.0, -2.0, 2.5]),
    ("poisson", lambda r: Poisson(r), [2.3], lambda r, x: td.Poisson(r).log_prob(x),
     [0.0, 1.0, 6.0], [-1.0, 0.5, -3.0]),
    ("bernoulli", lambda p: Bernoulli(p), [0.3], lambda p, x: td.Bernoulli(p).log_prob(x),
     [0.0, 1.0], [-1.0, 2.0, 0.5]),
    ("bernoulli_logits", lambda l: Bernoulli(logits=l), [0.6], lambda l, x: td.Bernoulli(logits=l).log_prob(x),
     [0.0, 1.0], [-1.0, 2.0, 0.5]),
    ("binomial", lambda p: Binomial(5, p), [0.3], lambda p, x: td.Binomial(5, p).log_prob(x),
     [0.0, 2.0, 5.0], [-1.0, 6.0, 2.5]),
    ("binomial_logits", lambda l: Binomial(5, logits=l), [0.4], lambda l, x: td.Binomial(5, logits=l).log_prob(x),
     [0.0, 2.0, 5.0], [-1.0, 6.0, 2.5]),
    ("negative_binomial", lambda r, p: NegativeBinomial(r, p), [3.5, 0.4],
     lambda r, p, x: td.NegativeBinomial(r, p).log_prob(x), [0.0, 3.0, 9.0], [-1.0, 1.5, -4.0]),
    ("negative_binomial_logits", lambda r, l: NegativeBinomial(r, logits=l), [3.5, -0.3],
     lambda r, l, x: td.NegativeBinomial(r, logits=l).log_prob(x), [0.0, 3.0, 9.0], [-1.0, 1.5, -4.0]),
    ("categorical", lambda p: Categorical(p), [[0.2, 0.5, 0.3]], lambda p, x: td.Categorical(p).log_prob(x),
     [0.0, 1.0, 2.0], [-1.0, 3.0, 0.5]),
    ("categorical_logits", lambda l: Categorical(logits=l), [[0.2, -1.5, 0.3]],
     lambda l, x: td.Categorical(logits=l).log_prob(x), [0.0, 1.0, 2.0], [-1.0, 3.0, 1.5]),
    ("discrete_uniform", lambda: DiscreteUniform(2, 5), [], lambda x: torch.full_like(x, -math.log(4.0)),
     [2.0, 3.0, 5.0], [1.0, 6.0, 2.5]),
]


@pytest.mark.parametrize("name,factory,values,reference,inside,outside", _SUPPORT_CASES, ids=[c[0] for c in _SUPPORT_CASES])
def test_discrete_log_pdf_is_minus_infinity_outside_the_support(name, factory, values, reference, inside, outside):
    params = [torch.tensor(v, requires_grad=True) for v in values]
    dist = factory(*params)
    x_in, x_out = torch.tensor(inside), torch.tensor(outside)
    x = torch.cat([x_in, x_out])

    log_p = dist.log_pdf(x)
    with torch.no_grad():
        expected = reference(*params, x_in)
    torch.testing.assert_close(log_p[: len(inside)].detach(), expected)
    assert torch.isneginf(log_p[len(inside):]).all(), log_p
    pdf = dist.pdf(x).detach()
    assert (pdf[len(inside):] == 0).all() and (pdf[: len(inside)] > 0).all()

    if not params:
        return
    # Gradients are finite everywhere, match autograd inside the support and
    # are zero outside it.
    grads = dist.log_pdf_param_grads(x)
    keys = list(grads)  # in the order of the parameters
    expected_grads = torch.autograd.grad(log_p[: len(inside)].sum(), params)
    for key, param, expected_grad in zip(keys, params, expected_grads):
        grad = torch.as_tensor(grads[key]).broadcast_to((len(x), *param.shape))
        assert torch.isfinite(grad).all(), (key, grad)
        torch.testing.assert_close(grad[: len(inside)].sum(0), expected_grad)
        assert (grad[len(inside):] == 0).all(), (key, grad)


def test_geometric_pdf_at_zero_is_zero():
    assert Geometric(0.2).pdf(t(0.0)).item() == 0.0
    assert Geometric(logits=t(0.3)).pdf(t(0.0)).item() == 0.0
    assert torch.isneginf(Geometric(0.2).log_pdf(t(0.0)))


def test_multinomial_log_pdf_outside_the_support():
    probs = t([0.2, 0.5, 0.3], requires_grad=True)
    for kind, param in (("probs", probs), ("logits", torch.log(probs.detach()).requires_grad_(True))):
        dist = Multinomial(4, **{kind: param})
        inside = t([[1.0, 2.0, 1.0], [0.0, 4.0, 0.0]])
        outside = t([[-1.0, 3.0, 2.0], [5.0, 0.0, -1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 1.5, 2.0]])
        x = torch.cat([inside, outside])
        log_p = dist.log_pdf(x)
        torch.testing.assert_close(log_p[:2].detach(), td.Multinomial(4, probs.detach()).log_prob(inside))
        assert torch.isneginf(log_p[2:]).all(), log_p
        grads = dist.log_pdf_param_grads(x)[kind]
        assert torch.isfinite(grads).all()
        assert (grads[2:] == 0).all()
        expected, = torch.autograd.grad(log_p[:2].sum(), param)
        torch.testing.assert_close(grads[:2].sum(0), expected)
