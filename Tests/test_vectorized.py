"""Vectorized chains: batched model evaluation and the batched samplers.

A batched model (``Model.batch``) holds every latent value with a leading
chain dimension. Its log density and gradients must equal separate unbatched
evaluations of every chain to rounding error; the vectorized samplers must
target the same posterior as the process-per-chain samplers.
"""

import math
import warnings

import pytest
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import (
    Add, Affine, CholeskyCovariance, Concat, CovarianceFromCholesky, Exp, Index, Linear, MatMul, Multiply,
    Softmax, Softplus, Stack, Sum,
)
from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Categorical, Cauchy, Dirichlet, Exponential, Gamma, HalfCauchy, HalfNormal,
    InvGamma, LKJCholesky, Laplace, LogNormal, Mixture, MultivariateNormal, Normal, Poisson, StudentT,
    Truncated, Uniform, Wishart,
)


# ============================================================ model zoo
def normal_normal():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.5))
        ObservedParameter("y", Normal(mu, sigma), torch.tensor([0.3, 1.2, -0.4, 2.0]))
    return model


def beta_bernoulli():
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 3.0))
        ObservedParameter("y", Bernoulli(p), torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]))
    return model


def eight_schools():
    y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    with Model() as model:
        school = Data("school", torch.arange(8))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        with plate("schools", 8):
            eta = RandomParameter("eta", Normal(0.0, 1.0), shape=())
        theta = Affine("theta", eta, loc=mu, scale=tau)
        theta_obs = Index("theta_obs", theta, school)
        ObservedParameter("y", Normal(theta_obs, sigma), y)
    return model


def dirichlet_categorical():
    with Model() as model:
        weights = RandomParameter("weights", Dirichlet(torch.tensor([1.0, 2.0, 3.0])))
        ObservedParameter("y", Categorical(probs=weights), torch.tensor([0.0, 2.0, 2.0, 1.0, 2.0]))
    return model


def lkj_mvn():
    data = torch.tensor([[0.5, -0.2, 1.0], [1.3, 0.4, -0.3], [-0.7, 0.2, 0.1], [0.1, -1.1, 0.8]])
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0), shape=3)
        L_corr = RandomParameter("L_corr", LKJCholesky(3, eta=2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0), shape=3)
        scale_tril = CholeskyCovariance("scale_tril", L_corr, sigma)
        ObservedParameter("y", MultivariateNormal(mu, scale_tril=scale_tril), data)
    return model


def enumerated_mixture():
    y = torch.tensor([-2.1, -1.7, 0.2, 1.9, 2.4, 2.2])
    with Model() as model:
        means = RandomParameter("means", Normal(0.0, 3.0), shape=2)
        scale = RandomParameter("scale", HalfNormal(1.0))
        with plate("data", len(y)):
            z = RandomParameter("z", Bernoulli(0.4), shape=(), enumerate=True)
        mean = Index("mean", means, z)
        ObservedParameter("y", Normal(mean, scale), y)
    return model


def matrix_data_scalar_params():
    data = torch.tensor([[0.1, 0.5, -0.2], [1.0, 0.3, 0.4]])
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        log_sigma = RandomParameter("log_sigma", Normal(0.0, 0.5))
        sigma = Exp("sigma", log_sigma)
        ObservedParameter("y", Normal(mu, sigma), data)
    return model


def linear_regression():
    x = torch.tensor([[0.2, 1.0], [1.5, -0.3], [-0.4, 0.8], [0.9, 0.9], [0.0, -1.2]])
    y = torch.tensor([1.0, 0.4, 0.7, 1.9, -0.8])
    with Model() as model:
        X = Data("X", x)
        w = RandomParameter("w", Normal(0.0, 1.0), shape=2)
        b = RandomParameter("b", Normal(0.0, 1.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        mean = Linear("mean", X, w, b)
        ObservedParameter("y", Normal(mean, sigma), y)
    return model


def scalar_slope_regression():
    with Model() as model:
        x = Data("x", torch.tensor([0.0, 0.5, 1.0, 1.5]))
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(0.0, 1.0))
        mean = Linear("mean", x, slope=b, intercept=a)
        ObservedParameter("y", Normal(mean, 0.5), torch.tensor([0.1, 0.4, 1.2, 1.4]))
    return model


def user_lambda():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(0.0, 1.0))
        mean = DeterministicParameter(
            "mean", lambda a, b: a + b * x, lambda a, b: {"a": torch.ones_like(x), "b": x}, inputs=[a, b])
        ObservedParameter("y", Normal(mean, 0.3), torch.tensor([0.2, 1.1, 2.3, 2.8]))
    return model


def blocks_network():
    x = torch.tensor([[0.3, -0.2], [1.0, 0.5], [-0.6, 0.9], [0.2, 0.2]])
    labels = torch.tensor([0.0, 2.0, 1.0, 2.0])
    with Model() as model:
        X = Data("X", x)
        W = RandomParameter("W", Normal(0.0, 1.0), shape=(2, 3))
        c = RandomParameter("c", Normal(0.0, 1.0), shape=3)
        hidden = Softplus("hidden", Add("pre", MatMul("xw", X, W), c))
        logits = Multiply("logits", hidden, 1.5)
        ObservedParameter("labels", Categorical(logits=logits), labels)
        total = Sum("total", hidden, dim=1)
        stacked = Stack("stacked", c, c, dim=0)
        joined = Concat("joined", c, torch.tensor([0.5]), dim=0)
        prob = Softmax("prob", joined, dim=0)
        ObservedParameter("t", Normal(total, 2.0), torch.tensor([1.0, 2.0, 1.5, 0.7]))
        ObservedParameter("s", Normal(stacked, 1.0), torch.zeros(2, 3))
        ObservedParameter("p", Dirichlet(Affine("conc", prob, loc=0.1, scale=10.0)), torch.tensor([0.1, 0.2, 0.3, 0.4]))
    return model


def wishart_model():
    X = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    with Model() as model:
        L = RandomParameter("L", LKJCholesky(2, eta=1.5))
        scale = RandomParameter("scale", HalfNormal(1.0), shape=2)
        S = CovarianceFromCholesky("S", CholeskyCovariance("Ls", L, scale))
        ObservedParameter("X", Wishart(4.0, S), X)
        P = RandomParameter("P", Wishart(5.0, torch.eye(2)))
    return model


def mixture_distribution():
    with Model() as model:
        w = RandomParameter("w", Dirichlet(torch.ones(2)))
        loc = RandomParameter("loc", Normal(0.0, 3.0), shape=2)
        components = [Normal(Index("l0", loc, torch.tensor([0])), 1.0), Normal(Index("l1", loc, torch.tensor([1])), 1.0)]
        ObservedParameter("y", Mixture(components, w), torch.tensor([-1.5, -1.0, 2.0, 2.5, 1.7]))
    return model


def bounded_and_truncated():
    with Model() as model:
        upper = RandomParameter("upper", Gamma(2.0, 1.0))
        x = RandomParameter("x", Uniform(0.0, upper))
        loc = RandomParameter("loc", Laplace(0.0, 1.0))
        ObservedParameter("t", Truncated(Normal(loc, 1.0), low=0.0), torch.tensor([0.3, 1.2, 0.8]))
        rate = RandomParameter("rate", LogNormal(0.0, 1.0))
        ObservedParameter("counts", Poisson(rate), torch.tensor([1.0, 3.0, 2.0]))
        nu = RandomParameter("nu", InvGamma(3.0, 2.0))
        ObservedParameter("st", StudentT(nu, x, 1.0), torch.tensor([0.5, 0.1]))
        e = RandomParameter("e", Exponential(1.0))
        ObservedParameter("c", Cauchy(e, 1.0), torch.tensor([0.2, -0.1]))
    return model


def hierarchical_plates():
    y = torch.tensor([[0.2, 0.5, 0.1], [1.2, 0.9, 1.4]])
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 2.0))
        tau = RandomParameter("tau", HalfNormal(1.0))
        with plate("groups", 3):
            theta = RandomParameter("theta", Normal(mu, tau), shape=())
            with plate("obs", 2):
                ObservedParameter("y", Normal(theta, 0.5), y)
    return model


def binomial_logits_regression():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 1.0))
        b = RandomParameter("b", Normal(0.0, 1.0))
        logits = Add("logits", a, Multiply("bx", b, x))
        ObservedParameter("k", Binomial(10, logits=logits), torch.tensor([1.0, 3.0, 6.0, 9.0]))
    return model


def brute_force_enumeration():
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 1.0))
        z1 = RandomParameter("z1", Bernoulli(0.3), enumerate=True)
        z2 = RandomParameter("z2", Bernoulli(0.6), enumerate=True)
        both = Add("both", z1, z2)
        ObservedParameter("y", Normal(Add("m", mean, both), 1.0), torch.tensor([0.4, 1.3]))
    return model


def gibbs_index_model():
    y = torch.tensor([-1.0, 2.2, 1.8])
    with Model() as model:
        means = RandomParameter("means", Normal(0.0, 3.0), shape=2)
        with plate("data", 3):
            z = RandomParameter("z", Bernoulli(0.5), shape=())
        ObservedParameter("y", Normal(Index("mz", means, z), 1.0), y)
    return model


def mutable_data_model():
    with Model() as model:
        x = Data("x", torch.tensor([0.0, 1.0, 2.0]))
        slope = RandomParameter("slope", Normal(0.0, 1.0))
        with plate("obs", x):
            ObservedParameter("y", Normal(Multiply("m", slope, x), 1.0), Data("y_data", torch.tensor([0.1, 0.9, 2.2])))
    return model


ZOO = {
    "normal_normal": normal_normal,
    "beta_bernoulli": beta_bernoulli,
    "eight_schools": eight_schools,
    "dirichlet_categorical": dirichlet_categorical,
    "lkj_mvn": lkj_mvn,
    "enumerated_mixture": enumerated_mixture,
    "matrix_data_scalar_params": matrix_data_scalar_params,
    "linear_regression": linear_regression,
    "scalar_slope_regression": scalar_slope_regression,
    "user_lambda": user_lambda,
    "blocks_network": blocks_network,
    "wishart_model": wishart_model,
    "mixture_distribution": mixture_distribution,
    "bounded_and_truncated": bounded_and_truncated,
    "hierarchical_plates": hierarchical_plates,
    "binomial_logits_regression": binomial_logits_regression,
    "brute_force_enumeration": brute_force_enumeration,
    "gibbs_index_model": gibbs_index_model,
    "mutable_data_model": mutable_data_model,
}


def random_values(model, n_chains, generator):
    values = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value
        if parameter.distribution.transformed_state_space.is_continuous():
            values[name] = 0.6 * torch.randn((n_chains, *u.shape), generator=generator, dtype=u.dtype)
        else:
            support = parameter.distribution.state_space.values.to(u.dtype)
            index = torch.randint(0, len(support), (n_chains, *u.shape), generator=generator)
            values[name] = support[index]
    return values


def chain_by_chain(model, values, names):
    lps, grads = [], []
    for chain in range(next(iter(values.values())).shape[0]):
        for name, value in values.items():
            model.params[name].set_unconstrained_value(value[chain])
        lps.append(model.model_log_prob())
        grads.append(model.joint_grad_log_prob(names))
    return torch.stack([torch.as_tensor(lp) for lp in lps]), {n: torch.stack([g[n] for g in grads]) for n in names}


@pytest.mark.parametrize("name", list(ZOO))
def test_batched_log_prob_and_gradients_equal_chain_by_chain(name):
    model = ZOO[name]()
    n_chains = 5
    values = random_values(model, n_chains, torch.Generator().manual_seed(7))
    names = [n for n, p in model.params.items() if p.distribution.transformed_state_space.is_continuous()]
    expected_lp, expected_grads = chain_by_chain(model, values, names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(n_chains, values=values):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            lp = model.model_log_prob()
            grads = model.joint_grad_log_prob(names)
            for block in names:
                block_lp = model.block_log_prob([block])
                assert block_lp.shape == (n_chains,)
    assert lp.shape == (n_chains,)
    torch.testing.assert_close(lp, expected_lp, rtol=1e-12, atol=1e-12)
    for n in names:
        assert grads[n].shape == expected_grads[n].shape
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=1e-12, atol=1e-12)


# ============================================================ fallbacks and enumeration
def exact_gp_model():
    from BayesianDLL.GP import ExactGP, RBF
    x = torch.linspace(0.0, 1.0, 5)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Uniform(0.2, 1.0))
        variance = RandomParameter("variance", Uniform(0.25, 2.0))
        noise = RandomParameter("noise", Uniform(0.01, 0.2))
        ObservedParameter("y", ExactGP("f", inputs, RBF(lengthscale, variance), noise_variance=noise), torch.sin(6 * x))
    return model


def latent_gp_model():
    from BayesianDLL.GP import LatentGP, RBF
    x = torch.linspace(0.0, 1.0, 4)
    with Model() as model:
        inputs = Data("inputs", x)
        lengthscale = RandomParameter("lengthscale", Exponential(1.0))
        latent = LatentGP("f", inputs, RBF(lengthscale, 1.0))
        ObservedParameter("y", Normal(latent, 0.3), torch.sin(6 * x))
    return model


def matmul_lambda_model():
    x = torch.tensor([[0.2, 1.0], [1.5, -0.3], [-0.4, 0.8]])
    with Model() as model:
        w = RandomParameter("w", Normal(0.0, 1.0), shape=2)
        mean = DeterministicParameter("mean", lambda w: x @ w, vjp_func=lambda g, w: {"w": x.T @ g}, inputs=[w])
        ObservedParameter("y", Normal(mean, 1.0), torch.tensor([1.0, 0.0, 0.5]))
    return model


@pytest.mark.parametrize("factory, expect_warning", [
    (exact_gp_model, False), (latent_gp_model, False), (matmul_lambda_model, True),
])
def test_loop_fallback_matches_chain_by_chain(factory, expect_warning):
    """Nodes that cannot be batched are evaluated chain by chain, with identical results."""
    model = factory()
    n_chains = 3
    values = random_values(model, n_chains, torch.Generator().manual_seed(3))
    names = list(model.params)
    expected_lp, expected_grads = chain_by_chain(model, values, names)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with model.batched(n_chains, values=values):
            assert model._batch_mode == "loop"
            assert model.batch_fallback_reason
            lp = model.model_log_prob()
            grads = model.joint_grad_log_prob(names)
    assert any("chain by chain" in str(w.message) for w in caught) == expect_warning
    torch.testing.assert_close(lp, expected_lp, rtol=1e-12, atol=1e-12)
    for n in names:
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("factory", [enumerated_mixture, brute_force_enumeration])
def test_batched_enumerated_draws_have_per_chain_probabilities(factory):
    model = factory()
    model.prepare_enumeration()
    n_chains = 4
    values = random_values(model, n_chains, torch.Generator().manual_seed(1))
    with model.batched(n_chains, values=values):
        draws = model.sample_enumerated()
        for name, (value, probabilities) in draws.items():
            shape = model.enumerated_params[name].constrained_value.shape
            assert value.shape == shape and shape[0] == n_chains
            assert probabilities.shape == (*shape, 2)
    for chain in range(n_chains):
        for name, value in values.items():
            model.params[name].set_unconstrained_value(value[chain])
        for name, (_, probabilities) in model.sample_enumerated().items():
            torch.testing.assert_close(draws[name][1][chain], probabilities)
