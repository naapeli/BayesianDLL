"""Exact marginalisation of finite-support discrete parameters (``marginalize=True``).

All Normal distributions use a unit scale, so the tests hold whether the
second argument of ``Normal`` is a variance or a standard deviation.
"""

import itertools
import math

import pytest
import torch

from BayesianDLL import (
    DeterministicParameter, Model, ObservedParameter, RandomParameter, condition, marginalize, plate,
)
from BayesianDLL.Distributions import Bernoulli, Beta, Binomial, Exponential, Mixture, Normal


Y3 = torch.tensor([-1.0, 0.3, 2.5])


def _mixture_model(marginalize, y=Y3):
    """p ~ Beta, mu0, mu1 ~ N, z_i ~ Bernoulli(p), y_i ~ N(mu_{z_i}, 1)."""
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 2.0))
        mu0 = RandomParameter("mu0", Normal(-1.0, 1.0))
        mu1 = RandomParameter("mu1", Normal(1.0, 1.0))
        with plate("n", len(y)):
            z = RandomParameter("z", Bernoulli(p), shape=(), marginalize=marginalize)
            mean = DeterministicParameter(
                "mean", lambda a, b, z: a + (b - a) * z,
                lambda a, b, z: {"mu0": 1 - z, "mu1": z}, [mu0, mu1, z],
            )
            ObservedParameter("y", Normal(mean, 1.0), y)
    return model


def _three_component_model(marginalize):
    """K = 3: z_i ~ Binomial(2, p), y_i ~ N(mus[z_i], 1)."""
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 2.0))
        mus = RandomParameter("mus", Normal(torch.tensor([-2.0, 0.0, 2.0]), 1.0), shape=3)
        with plate("n", 3):
            z = RandomParameter("z", Binomial(2, p), shape=(), marginalize=marginalize)

            def derivative(m, z):
                return {"mus": torch.nn.functional.one_hot(z.long(), 3).to(m.dtype)}

            mean = DeterministicParameter("mean", lambda m, z: m[z.long()], derivative, [mus, z])
            ObservedParameter("y", Normal(mean, 1.0), Y3)
    return model


def _randomise(models, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in models[0].params.items():
        value = torch.randn(parameter.unconstrained_value.shape, generator=generator, dtype=torch.float64)
        for model in models:
            model.params[name].set_unconstrained_value(value.clone())


def _brute_force_log_prob(model, names):
    """log sum over all joint assignments of the non-enumerated model's ``names``."""
    parameters = [model.params[name] for name in names]
    supports = [list(p.distribution.state_space) for p in parameters]
    shapes = [p.constrained_value.shape for p in parameters]
    per_element = [s for s, p in zip(supports, parameters) for _ in range(p.constrained_value.numel())]
    values = []
    for assignment in itertools.product(*per_element):
        offset = 0
        for parameter, shape in zip(parameters, shapes):
            n = parameter.constrained_value.numel()
            parameter.set_constrained_value(torch.tensor(assignment[offset:offset + n], dtype=torch.float64).reshape(shape))
            offset += n
        values.append(model.model_log_prob())
    return torch.logsumexp(torch.stack(values), dim=0)


def _finite_difference(model, eps=1e-6):
    grads = {}
    for name, parameter in model.params.items():
        u = parameter.unconstrained_value.clone()
        grad = torch.zeros_like(u)
        for i in range(u.numel()):
            step = torch.zeros_like(u).reshape(-1)
            step[i] = eps
            step = step.reshape(u.shape)
            with model.temporarily_set(name, u + step):
                upper = model.model_log_prob()
            with model.temporarily_set(name, u - step):
                lower = model.model_log_prob()
            grad.reshape(-1)[i] = (upper - lower) / (2 * eps)
        grads[name] = grad
    return grads


def _assert_gradient_matches_finite_differences(model):
    analytic = model.joint_grad_log_prob()
    numeric = _finite_difference(model)
    for name in model.params:
        torch.testing.assert_close(analytic[name].reshape(-1), numeric[name].reshape(-1), atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------ marginal value
@pytest.mark.parametrize("builder", [_mixture_model, _three_component_model], ids=["K=2", "K=3"])
def test_marginal_log_prob_equals_brute_force_sum(builder):
    enumerated, full = builder(True), builder(False)
    _randomise([enumerated, full])
    assert list(enumerated.params) == [n for n in full.params if n != "z"]
    assert [g.mode for g in enumerated.prepare_enumeration()] == ["factorised"]
    torch.testing.assert_close(enumerated.model_log_prob(), _brute_force_log_prob(full, ["z"]))
    # A sampling block of the continuous parameters sees the whole marginal.
    block = enumerated.block_log_prob(list(enumerated.params))
    torch.testing.assert_close(block, enumerated.model_log_prob())


@pytest.mark.parametrize("builder", [_mixture_model, _three_component_model], ids=["K=2", "K=3"])
def test_marginal_gradient_matches_finite_differences(builder):
    model = builder(True)
    _randomise([model], seed=3)
    _assert_gradient_matches_finite_differences(model)
    # Restricting the reverse pass to a block's nodes gives the same gradient.
    names = ["mu0", "mu1"] if "mu0" in model.params else ["mus"]
    _, _, nodes = model.block_terms(names)
    full = model.joint_grad_log_prob(names)
    restricted = model.joint_grad_log_prob(names, nodes=nodes)
    for name in names:
        torch.testing.assert_close(restricted[name], full[name])


def test_continuous_random_child_of_enumerated_parameter():
    # x_i ~ N(mu_{z_i}, 1) is latent: its own prior gradient is weighted per coordinate.
    def build(marginalize):
        with Model() as model:
            mu0 = RandomParameter("mu0", Normal(-1.0, 1.0))
            mu1 = RandomParameter("mu1", Normal(1.0, 1.0))
            rate = RandomParameter("rate", Exponential(1.0))
            with plate("n", 3):
                z = RandomParameter("z", Bernoulli(0.4), shape=(), marginalize=marginalize)
                mean = DeterministicParameter(
                    "mean", lambda a, b, z: a + (b - a) * z,
                    lambda a, b, z: {"mu0": 1 - z, "mu1": z}, [mu0, mu1, z],
                )
                x = RandomParameter("x", Normal(mean, 1.0), shape=())
                ObservedParameter("y", Exponential(rate), torch.tensor([0.5, 1.0, 2.0]))
                ObservedParameter("w", Normal(x, 1.0), Y3)
        return model

    enumerated, full = build(True), build(False)
    _randomise([enumerated, full], seed=5)
    torch.testing.assert_close(enumerated.model_log_prob(), _brute_force_log_prob(full, ["z"]))
    _assert_gradient_matches_finite_differences(enumerated)


# --------------------------------------------- non-factorising and brute force
def _cumulative_model(n, marginalize=True):
    # mean_i = mu + sum_{j <= i} z_j couples the coordinates of z.
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        with plate("n", n):
            z = RandomParameter("z", Bernoulli(0.3), shape=(), marginalize=marginalize)
            mean = DeterministicParameter(
                "mean", lambda mu, z: mu + torch.cumsum(z, 0),
                lambda mu, z: {"mu": torch.ones_like(z)}, [mu, z],
            )
            ObservedParameter("y", Normal(mean, 1.0), torch.linspace(-1.0, 2.0, n))
    return model


def test_non_factorising_group_falls_back_to_brute_force():
    model, full = _cumulative_model(3), _cumulative_model(3, marginalize=False)
    _randomise([model, full], seed=1)
    assert [g.mode for g in model.prepare_enumeration()] == ["brute_force"]
    torch.testing.assert_close(model.model_log_prob(), _brute_force_log_prob(full, ["z"]))
    _assert_gradient_matches_finite_differences(model)
    value, probabilities = model.sample_enumerated()["z"]
    assert value.shape == (3,) and probabilities.shape == (3, 2)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=torch.float64))


def test_non_factorising_group_too_large_is_refused():
    model = _cumulative_model(13)
    with pytest.raises(ValueError, match="does not factorise"):
        model.model_log_prob()


def test_interacting_enumerated_parameters_are_summed_jointly():
    def build(marginalize):
        with Model() as model:
            mu = RandomParameter("mu", Normal(0.0, 1.0))
            a = RandomParameter("a", Bernoulli(0.3), marginalize=marginalize)
            b = RandomParameter("b", Bernoulli(0.6), marginalize=marginalize)
            mean = DeterministicParameter(
                "mean", lambda mu, a, b: mu + 2 * a - b,
                lambda mu, a, b: {"mu": torch.ones_like(mu)}, [mu, a, b],
            )
            ObservedParameter("y", Normal(mean, 1.0), torch.tensor([0.5, 1.5]))
        return model

    model, full = build(True), build(False)
    _randomise([model, full], seed=2)
    groups = model.prepare_enumeration()
    assert len(groups) == 1 and groups[0].mode == "brute_force"
    torch.testing.assert_close(model.model_log_prob(), _brute_force_log_prob(full, ["a", "b"]))
    _assert_gradient_matches_finite_differences(model)


# ------------------------------------------------------------- API and errors
def test_only_finite_discrete_parameters_can_be_enumerated():
    with pytest.raises(ValueError, match="finite discrete support"):
        with Model():
            RandomParameter("x", Normal(0.0, 1.0), marginalize=True)
    with Model() as model:
        RandomParameter("x", Normal(0.0, 1.0))
    with pytest.raises(ValueError, match="finite discrete support"):
        marginalize(model, ["x"])


def test_marginalize_matches_enumerate_flag_and_leaves_original_unchanged():
    original, flagged = _mixture_model(False), _mixture_model(True)
    _randomise([flagged, original])
    marginal = marginalize(original, "z")
    assert "z" in original.params and "z" not in marginal.params and "z" in marginal.enumerated_params
    torch.testing.assert_close(marginal.model_log_prob(), flagged.model_log_prob())
    assert not original.enumerated_params and not original._enumeration_groups
    with pytest.raises(ValueError, match="enumerated"):
        marginal.sample(5, 5, n_chains=1, progress_bar=False, blocks=[["p", "mu0", "mu1"], ["z"]])


def test_conditioning_an_enumerated_parameter_makes_it_observed():
    model, full = _mixture_model(True), _mixture_model(False)
    _randomise([model, full])
    value = torch.tensor([0.0, 1.0, 1.0])
    conditioned = condition(model, z=value)
    assert "z" in conditioned.observed_params and not conditioned.enumerated_params
    full.params["z"].set_constrained_value(value)
    torch.testing.assert_close(conditioned.model_log_prob(), full.model_log_prob())


# -------------------------------------------------------- recovery of z
def test_recovered_probabilities_are_the_analytic_responsibilities():
    model = _mixture_model(True)
    _randomise([model], seed=4)
    p = model.params["p"].constrained_value
    mu0, mu1 = model.params["mu0"].constrained_value, model.params["mu1"].constrained_value
    log_one = torch.log(p) - 0.5 * (Y3 - mu1) ** 2
    log_zero = torch.log(1 - p) - 0.5 * (Y3 - mu0) ** 2
    responsibility = torch.sigmoid(log_one - log_zero)

    _, probabilities = model.sample_enumerated()["z"]
    torch.testing.assert_close(probabilities[..., 1], responsibility)

    torch.manual_seed(0)
    draws = torch.stack([model.sample_enumerated()["z"][0] for _ in range(4000)])
    torch.testing.assert_close(draws.mean(0), responsibility, atol=0.03, rtol=0)
    # Drawing z does not change the marginal density.
    before = model.model_log_prob()
    model.sample_enumerated()
    torch.testing.assert_close(model.model_log_prob(), before)


@pytest.mark.integration
def test_enumerated_mixture_matches_mixture_likelihood_with_nuts():
    torch.manual_seed(1)
    y = torch.cat([torch.randn(18) - 2.0, torch.randn(42) + 2.0])

    with Model() as enumerated:
        mu0 = RandomParameter("mu0", Normal(-2.0, 1.0))
        mu1 = RandomParameter("mu1", Normal(2.0, 1.0))
        with plate("n", len(y)):
            z = RandomParameter("z", Bernoulli(0.7), shape=(), marginalize=True)
            mean = DeterministicParameter(
                "mean", lambda a, b, z: a + (b - a) * z,
                lambda a, b, z: {"mu0": 1 - z, "mu1": z}, [mu0, mu1, z],
            )
            ObservedParameter("y", Normal(mean, 1.0), y)

    with Model() as mixture:
        mu0 = RandomParameter("mu0", Normal(-2.0, 1.0))
        mu1 = RandomParameter("mu1", Normal(2.0, 1.0))
        with plate("n", len(y)):
            ObservedParameter("y", Mixture([Normal(mu0, 1.0), Normal(mu1, 1.0)], torch.tensor([0.3, 0.7])), y)

    _randomise([enumerated, mixture], seed=7)
    torch.testing.assert_close(enumerated.model_log_prob(), mixture.model_log_prob())

    result = enumerated.sample(400, 200, n_chains=2, progress_bar=False, seed=3)
    reference = mixture.sample(400, 200, n_chains=2, progress_bar=False, seed=3)
    assert result.sample_stats.samplers == ["nuts"]
    for name, truth in (("mu0", -2.0), ("mu1", 2.0)):
        mean = result[name].mean().item()
        assert mean == pytest.approx(reference[name].mean().item(), abs=0.1)
        assert mean == pytest.approx(truth, abs=0.5)
        assert result[name].std().item() == pytest.approx(reference[name].std().item(), rel=0.25)

    assert result["z"].shape == (2, 400, len(y))
    assert result.enumerated_probs["z"].shape == (2, 400, len(y), 2)
    assert set(result["z"].unique().tolist()) <= {0.0, 1.0}
    # The observations drawn from the second component are assigned to it.
    responsibilities = result.enumerated_probs["z"][..., 1].mean((0, 1))
    assert responsibilities[:18].mean() < 0.15 and responsibilities[18:].mean() > 0.85
    torch.testing.assert_close(result["z"].mean((0, 1)), responsibilities, atol=0.12, rtol=0)
    # The deterministic node is recorded with the drawn assignments.
    expected_mean = result["mu0"] + (result["mu1"] - result["mu0"]) * result["z"]
    torch.testing.assert_close(result["mean"], expected_mean)
    thinned = result.thin(4)
    assert thinned.enumerated_probs["z"].shape == (2, 100, len(y), 2)


# ------------------------------------------------------- scalar (no plate)
def _scalar_model(marginalize):
    # z ~ Bernoulli(0.5), mu ~ N(0, 1), y = 2 ~ N(mu + 3 z, 1).
    with Model() as model:
        z = RandomParameter("z", Bernoulli(0.5), marginalize=marginalize)
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        mean = DeterministicParameter(
            "mean", lambda mu, z: mu + 3 * z,
            lambda mu, z: {"mu": torch.ones_like(mu), "z": 3 * torch.ones_like(z)}, [mu, z],
        )
        ObservedParameter("y", Normal(mean, 1.0), torch.tensor([2.0]))
    return model


def test_scalar_enumerated_parameter_value_and_gradient():
    model, full = _scalar_model(True), _scalar_model(False)
    _randomise([model, full], seed=6)
    assert model.enumerated_params["z"].constrained_value.shape == (1,)
    torch.testing.assert_close(model.model_log_prob(), _brute_force_log_prob(full, ["z"]))
    _assert_gradient_matches_finite_differences(model)


@pytest.mark.integration
def test_scalar_enumerated_parameter_sampling():
    # Marginally y | z ~ N(3 z, 2), so P(z = 1 | y) = 1 / (1 + exp(-0.75)).
    model = _scalar_model(True)
    result = model.sample(1500, 300, n_chains=2, progress_bar=False, seed=5)
    p_z = 1 / (1 + math.exp(-0.75))
    assert result["z"].shape == (2, 1500, 1)
    assert result["z"].mean().item() == pytest.approx(p_z, abs=0.05)
    assert result.enumerated_probs["z"][..., 1].mean().item() == pytest.approx(p_z, abs=0.03)
    expected_mu = p_z * (-0.5) + (1 - p_z) * 1.0
    assert result["mu"].mean().item() == pytest.approx(expected_mu, abs=0.1)


def test_random_parameter_marginalize_alias():
    from BayesianDLL.Distributions import Bernoulli
    with Model() as model:
        z = RandomParameter("z", Bernoulli(0.3), marginalize=True)
    assert z.marginalized is True
    assert z.enumerated is True
    assert "z" in model.enumerated_params
