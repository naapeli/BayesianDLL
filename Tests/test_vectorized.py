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
    Add, Affine, Concat, Exp, Index, Linear, MatMul, Multiply,
    Softmax, Softplus, Stack, Sum,
)
from BayesianDLL.Deterministic._blocks import CholeskyCovariance, CovarianceFromCholesky
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


def lkj_mvn_covariance():
    """MVN given a random covariance matrix (event dims of ``covariance_matrix``)."""
    data = torch.tensor([[0.5, -0.2, 1.0], [1.3, 0.4, -0.3], [-0.7, 0.2, 0.1], [0.1, -1.1, 0.8]])
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0), shape=3)
        L_corr = RandomParameter("L_corr", LKJCholesky(3, eta=2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.0), shape=3)
        covariance = CovarianceFromCholesky("covariance", CholeskyCovariance("scale_tril", L_corr, sigma))
        ObservedParameter("y", MultivariateNormal(mu, covariance), data)
    return model


def enumerated_mixture():
    y = torch.tensor([-2.1, -1.7, 0.2, 1.9, 2.4, 2.2])
    with Model() as model:
        means = RandomParameter("means", Normal(0.0, 3.0), shape=2)
        scale = RandomParameter("scale", HalfNormal(1.0))
        with plate("data", len(y)):
            z = RandomParameter("z", Bernoulli(0.4), shape=(), marginalize=True)
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
        mean = Linear("mean", x, coefficients=b, intercept=a)
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
        z1 = RandomParameter("z1", Bernoulli(0.3), marginalize=True)
        z2 = RandomParameter("z2", Bernoulli(0.6), marginalize=True)
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
    "lkj_mvn_covariance": lkj_mvn_covariance,
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


@pytest.mark.parametrize("factory, batchable", [
    (exact_gp_model, True), (latent_gp_model, True), (matmul_lambda_model, False),
])
def test_loop_fallback_matches_chain_by_chain(factory, batchable):
    """GP nodes are batched; nodes that cannot be batched are evaluated chain by chain. Same results either way."""
    model = factory()
    n_chains = 3
    values = random_values(model, n_chains, torch.Generator().manual_seed(3))
    names = list(model.params)
    expected_lp, expected_grads = chain_by_chain(model, values, names)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with model.batched(n_chains, values=values):
            assert model._batch_mode == ("tensor" if batchable else "loop")
            assert bool(model.batch_fallback_reason) != batchable
            lp = model.model_log_prob()
            grads = model.joint_grad_log_prob(names)
    assert any("chain by chain" in str(w.message) for w in caught) != batchable
    tolerance = 1e-10 if batchable else 1e-12
    torch.testing.assert_close(lp, expected_lp, rtol=tolerance, atol=tolerance)
    for n in names:
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=tolerance, atol=tolerance)


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


# ============================================================ vectorized sampling
def vectorized(model, n_samples=400, warmup=300, n_chains=8, seed=11, **kwargs):
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample(n_samples, warmup, n_chains=n_chains, model=model, progress_bar=False,
                      chain_method="vectorized", seed=seed, **kwargs)


def parallel(model, n_samples=400, warmup=300, n_chains=4, seed=11, **kwargs):
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample(n_samples, warmup, n_chains=n_chains, model=model, progress_bar=False,
                      chain_method="parallel", seed=seed, **kwargs)


def assert_moments(draws, mean, sd, tolerance=0.08):
    """Posterior mean within ``tolerance`` posterior sds; sd within 15 %."""
    draws = draws.reshape(-1, *draws.shape[2:]).to(torch.float64)
    mean = torch.as_tensor(mean, dtype=torch.float64)
    sd = torch.as_tensor(sd, dtype=torch.float64)
    assert torch.all((draws.mean(0) - mean).abs() <= tolerance * sd + 1e-3), (draws.mean(0), mean)
    assert torch.all((draws.std(0) / sd - 1).abs() < 0.15), (draws.std(0), sd)


def conjugate_normal():
    """Normal(0, sd 2) prior and unit-variance data: posterior N(1.6, 4/13)."""
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 2.0))
        ObservedParameter("data", Normal(mean, 1.0), torch.tensor([1.0, 2.0, 2.2]))
    return model


@pytest.mark.parametrize("sampler, kwargs", [
    ("nuts", {}), ("nuts", {"metric": "dense"}), ("slice", {}), ("metropolis", {}),
])
def test_vectorized_samplers_match_conjugate_normal(sampler, kwargs):
    with Model() as model:
        mean = RandomParameter("mean", Normal(torch.zeros(2), 2.0), shape=2, sampler=sampler)
        ObservedParameter("data", Normal(mean, 1.0), torch.tensor([[1.0, -1.0], [2.0, 0.0], [2.2, 0.4]]))
    n_samples = 1500 if sampler == "metropolis" else 400
    result = vectorized(model, n_samples=n_samples, n_chains=16, **kwargs)
    precision = 1 / 4 + 3
    expected_mean = torch.tensor([5.2, -0.6]) / precision
    assert_moments(result["mean"], expected_mean, torch.full((2,), precision ** -0.5),
                   tolerance=0.12 if sampler == "metropolis" else 0.08)
    assert result.sample_stats[0]["accept_stat"].shape == (16, n_samples)


def test_vectorized_matches_analytic_posteriors_through_transforms():
    # Beta(2, 3) prior, 4 successes in 6 trials: Beta(6, 5).
    result = vectorized(beta_bernoulli())
    a, b = 6.0, 5.0
    assert_moments(result["p"], a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    # Dirichlet(1, 2, 3) prior and category counts (1, 1, 3): Dirichlet(2, 3, 6).
    result = vectorized(dirichlet_categorical())
    alpha = torch.tensor([2.0, 3.0, 6.0])
    total = alpha.sum()
    assert_moments(result["weights"], alpha / total, (alpha * (total - alpha) / (total ** 2 * (total + 1))).sqrt())


def test_vectorized_lkj_prior_correlation_moments():
    # A correlation of LKJ(eta) in D dimensions is 2 * Beta(b, b) - 1 with b = eta - 1 + D / 2.
    with Model() as model:
        RandomParameter("L", LKJCholesky(3, eta=2.0))
    result = vectorized(model, n_samples=600, n_chains=16)
    b = 2.0 - 1.0 + 1.5
    correlation = result["L"][..., 1, 0]
    assert_moments(correlation, 0.0, math.sqrt(1 / (2 * b + 1)))


def test_vectorized_eight_schools_matches_parallel():
    vector = vectorized(eight_schools(), n_samples=250, warmup=250, n_chains=12, delta=0.85)
    reference = parallel(eight_schools(), n_samples=500, warmup=250, n_chains=4, delta=0.85)
    for name in ("mu", "tau", "theta"):
        a, b = vector[name].reshape(-1, *vector[name].shape[2:]), reference[name].reshape(-1, *reference[name].shape[2:])
        sd = b.std(0)
        assert torch.all((a.mean(0) - b.mean(0)).abs() < 0.25 * sd), name
        assert torch.all((a.std(0) / sd - 1).abs() < 0.25), name


def test_vectorized_two_block_gibbs_nuts_matches_exact_posterior():
    """Gibbs (z) + NUTS (means) blocks against the exact posterior by enumeration of z."""
    import itertools
    model = gibbs_index_model()
    result = vectorized(model, n_samples=600, warmup=200, n_chains=12)
    assert result.sample_stats.samplers == ["nuts", "gibbs"]
    y = torch.tensor([-1.0, 2.2, 1.8])
    log_weights, means = [], []
    for z in itertools.product([0, 1], repeat=3):
        z = torch.tensor(z)
        mean = torch.zeros(2)
        log_marginal = 3 * math.log(0.5)
        for k in range(2):
            members = y[z == k]
            n = len(members)
            precision = 1 / 9 + n
            mean[k] = members.sum() / precision
            # Marginal likelihood of the members under means[k] ~ N(0, 9), y ~ N(means[k], 1).
            covariance = torch.eye(n) + 9 * torch.ones(n, n)
            if n:
                log_marginal += float(MultivariateNormal(torch.zeros(n), covariance).log_pdf(members))
        log_weights.append(log_marginal)
        means.append(mean)
    weights = torch.softmax(torch.tensor(log_weights), 0)
    expected = (weights[:, None] * torch.stack(means)).sum(0)
    draws = result["means"].reshape(-1, 2)
    assert torch.all((draws.mean(0) - expected).abs() < 0.12), (draws.mean(0), expected)


def asymmetric_enumerated_mixture():
    y = torch.tensor([-2.1, -1.7, -2.5, 1.9, 2.4, 2.2, 1.5])
    with Model() as model:
        # A prior tight enough to rule out the label-switched mode.
        means = RandomParameter("means", Normal(torch.tensor([-2.0, 2.0]), 0.5), shape=2)
        with plate("data", len(y)):
            z = RandomParameter("z", Bernoulli(0.4), shape=(), marginalize=True)
        ObservedParameter("y", Normal(Index("mean", means, z), 0.7), y)
    return model


def test_vectorized_enumerated_mixture_matches_parallel():
    vector = vectorized(asymmetric_enumerated_mixture(), n_samples=300, warmup=200, n_chains=8)
    reference = parallel(asymmetric_enumerated_mixture(), n_samples=500, warmup=200, n_chains=4)
    for name in ("means", "z"):
        a = vector[name].reshape(-1, *vector[name].shape[2:])
        b = reference[name].reshape(-1, *reference[name].shape[2:])
        assert torch.all((a.mean(0) - b.mean(0)).abs() < 0.08), name
    assert vector.enumerated_probs["z"].shape == (8, 300, 7, 2)
    assert vector.log_likelihood["y"].shape == (8, 300, 7)


def test_vectorized_exact_gp_matches_parallel():
    vector = vectorized(exact_gp_model(), n_samples=200, warmup=100, n_chains=4)
    reference = parallel(exact_gp_model(), n_samples=200, warmup=100, n_chains=4)
    for name in ("lengthscale", "variance", "noise"):
        a, b = vector[name].reshape(-1), reference[name].reshape(-1)
        assert abs(float(a.mean() - b.mean())) < 0.3 * float(b.std()), name


def test_vectorized_with_mutable_data():
    model = mutable_data_model()
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    y = torch.tensor([0.2, 0.4, 1.1, 1.3, 2.1])
    model.data["x"].set_value(x)
    model.data["y_data"].set_value(y)
    result = vectorized(model, n_chains=8)
    precision = 1 + float((x ** 2).sum())
    assert_moments(result["slope"], float((x * y).sum()) / precision, precision ** -0.5)
    assert result.deterministic_trace["m"].shape == (8, 400, 5)


def test_vectorized_result_structure_and_downstream_tools():
    from BayesianDLL.Evaluation import loo, summary
    model = linear_regression()
    result = vectorized(model, n_samples=200, warmup=200, n_chains=6)
    assert result.trace["w"].shape == (6, 200, 2)
    assert result.deterministic_trace["mean"].shape == (6, 200, 5)
    assert result.log_likelihood["y"].shape == (6, 200, 5)
    assert len(result.divergences) == 6 and len(result.step_sizes) == 6 and len(result.acceptance_probabilities) == 6
    stats = result.sample_stats[0]
    assert set(stats) == {"step_size", "accept_stat", "tree_depth", "n_leapfrog", "diverging", "energy", "lp"}
    for value in stats.values():
        assert value.shape == (6, 200) and value.dtype == torch.float64
    assert all(t.device.type == "cpu" for t in result.trace.values())
    summary(result)
    loo(result)
    predictive = model.posterior_predicative(result, n_samples=10, samples_per_step=3, progress_bar=False)
    assert predictive["y"].shape == (10, 3, 5)


def test_vectorized_seeding_is_reproducible():
    first = vectorized(eight_schools(), n_samples=30, warmup=30, n_chains=4, seed=5)
    second = vectorized(eight_schools(), n_samples=30, warmup=30, n_chains=4, seed=5)
    other = vectorized(eight_schools(), n_samples=30, warmup=30, n_chains=4, seed=6)
    torch.testing.assert_close(first["eta"], second["eta"], rtol=0, atol=0)
    assert not torch.equal(first["eta"], other["eta"])
    # The chains differ from each other.
    assert not torch.equal(first["eta"][0], first["eta"][1])


def test_vectorized_explicit_blocks_and_default_chain_method():
    from BayesianDLL import sample
    model = normal_normal()
    result = vectorized(model, n_samples=50, warmup=50, n_chains=3, blocks=[["mu"], ["sigma"]])
    assert result.sample_stats.params == [["mu"], ["sigma"]]
    with pytest.raises(ValueError):
        sample(10, 10, n_chains=2, model=model, chain_method="threads", progress_bar=False)


# ============================================================ SMC
@pytest.mark.parametrize("kernel", ["metropolis", "nuts", "slice"])
def test_batched_smc_matches_particle_by_particle_smc(kernel):
    from BayesianDLL import sample_smc
    # Posterior N(1.6, 4/13) with log evidence of the conjugate model.
    y = torch.tensor([1.0, 2.0, 2.2])
    covariance = torch.eye(3) + 4.0 * torch.ones(3, 3)
    exact = float(MultivariateNormal(torch.zeros(3), covariance).log_pdf(y))
    results = [
        sample_smc(conjugate_normal(), n_particles=400, kernel=kernel, seed=4, progress_bar=False, vectorized=flag)
        for flag in (True, False)
    ]
    for result in results:
        assert result.log_marginal_likelihood == pytest.approx(exact, abs=0.25)
        assert_moments(result["mean"], 1.6, math.sqrt(4 / 13), tolerance=0.15)
    assert results[0]["mean"].shape == (1, 400, 1)


def test_batched_smc_with_a_model_evaluated_chain_by_chain():
    from BayesianDLL import sample_smc
    result = sample_smc(exact_gp_model(), n_particles=40, seed=1, progress_bar=False)
    assert result["lengthscale"].shape == (1, 40, 1)
    assert torch.isfinite(torch.as_tensor(result.log_marginal_likelihood))
