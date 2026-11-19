"""Guides built from Gaussian groups, a default family and hand-written entries; guide moments."""

import importlib
import math

import pytest
import torch

from BayesianDLL import (
    Data, DeterministicParameter, Group, Guide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate,
)
from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Categorical, Dirichlet, DiscreteUniform, Exponential, Gamma, Geometric, HalfNormal,
    InvGamma, Laplace, LogNormal, Mixture, MultivariateNormal, NegativeBinomial, Normal, Poisson, StudentT, Uniform,
    Cauchy, Wishart,
)
from BayesianDLL.Variational import BBVI, elbo


def _regression(n=8, noise=0.3, seed=0, extra=True):
    """``y = a + b x`` with correlated ``a, b``; ``c`` (Normal) and ``sigma`` (HalfNormal) are extra latents."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.5, 1.5, n)
    y = 1.0 + 2.0 * x + noise * torch.randn(n, generator=generator)
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 3.0))
        b = RandomParameter("b", Normal(0.0, 3.0))
        xs = Data("x", x)
        mu = DeterministicParameter("mu", lambda a, b, x: a + b * x, lambda a, b, x: {"a": torch.ones_like(x), "b": x}, [a, b, xs])
        with plate("data", xs):
            ObservedParameter("y", Normal(mu, noise), y)
        if extra:
            c = RandomParameter("c", Normal(0.0, 1.0))
            sigma = RandomParameter("sigma", HalfNormal(1.0))
            ObservedParameter("z", Normal(c, sigma), torch.tensor([0.4, -0.3, 0.9]))
    return model, x, y


def _two_latent_model():
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 2.0))
        sigma = RandomParameter("sigma", HalfNormal(1.5))
        DeterministicParameter("twice", lambda m: 2 * m, lambda m: {"mean": 2 * torch.ones_like(m)}, [mean])
        with plate("data", 4):
            ObservedParameter("y", Normal(mean, sigma), torch.tensor([0.3, 1.2, 0.8, 1.9]))
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validation_errors():
    model, _, _ = _regression()
    with pytest.raises(ValueError, match=r"no distribution for the model latent\(s\) \['c', 'sigma'\]"):
        with Guide(model) as guide:
            Group(["a", "b"], family="full_rank")
        guide.validate()
    with pytest.raises(ValueError, match="already in the group 'group0'"):
        with Guide(model, default_family="mean_field"):
            Group(["a", "b"])
            Group(["b", "c"])
    with pytest.raises(ValueError, match="'nope' is not a latent"):
        with Guide(model, default_family="mean_field"):
            Group(["a", "nope"])
    with pytest.raises(ValueError, match="family must be"):
        with Guide(model, default_family="mean_field"):
            Group(["a"], family="diagonal")
    with pytest.raises(ValueError, match="default_family must be"):
        Guide(model, default_family="full")
    with pytest.raises(RuntimeError, match="inside a `with Guide"):
        Group(["a"])
    # A hand-written entry and a group for the same latent.
    with pytest.raises(ValueError, match="exactly one group or hand-written entry"):
        with Guide(model, default_family="mean_field") as guide:
            Group(["a", "sigma"])
            RandomParameter("sigma", LogNormal(VariationalParameter("m", 0.0), VariationalParameter("s", 0.5)))
        guide.validate()


def test_discrete_latents_are_rejected_by_groups_but_allowed_as_entries():
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 2.0))
        z = RandomParameter("z", Bernoulli(p))
        ObservedParameter("y", Normal(z, 1.0), torch.tensor([0.8]))
    with pytest.raises(ValueError, match=r"enumerat.*hand-written guide entry"):
        with Guide(model, default_family="mean_field"):
            Group([z])
    with pytest.raises(ValueError, match="enumerat"):
        Guide(model, default_family="mean_field").validate()
    with Guide(model, default_family="mean_field") as guide:
        RandomParameter("z", Bernoulli(VariationalParameter("q", torch.tensor([0.4]), min=1e-3, max=1 - 1e-3)))
    value, grads = elbo(model, guide, n_samples=4)
    assert set(grads) == {"default_p_loc", "default_p_scale", "z_q"}
    assert torch.isfinite(value)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def test_grouped_guide_gradient_matches_autograd():
    model, _, _ = _regression()
    with Guide(model, default_family="mean_field", init_scale=0.3) as guide:
        Group(["a", "b"], family="full_rank", name="ab")
        RandomParameter("sigma", LogNormal(VariationalParameter("loc", torch.tensor([-0.2])), VariationalParameter("scale", torch.tensor([0.4]))))
    table = guide.variational_parameters()
    assert set(table) == {"ab_loc", "ab_scale_tril", "default_c_loc", "default_c_scale", "sigma_loc", "sigma_scale"}
    table["ab_loc"].set_new_value(torch.tensor([0.8, 1.7]))
    table["ab_scale_tril"].set_new_value(torch.tensor([[0.3, 0.0], [-0.2, 0.25]]))
    table["default_c_loc"].set_new_value(torch.tensor([0.2]))
    for parameter in table.values():
        parameter.value.requires_grad_(True)
    value, grads = elbo(model, guide, n_samples=5)
    expected = torch.autograd.grad(value, [table[key].value for key in grads])
    for key, reference in zip(grads, expected):
        if key == "ab_scale_tril":
            reference = torch.tril(reference)
        torch.testing.assert_close(grads[key], reference, msg=key)


def _quadrature_elbo(a, b, counts):
    """Exact ELBO of the ``x`` part for a Gamma(a, b) guide, x ~ Exp(1), counts ~ Poisson(x)."""
    q = Gamma(torch.tensor([a]), torch.tensor([b]))
    grid = torch.linspace(1e-5, 25.0, 200001)
    log_q = q.log_pdf(grid)
    log_p = Exponential(1.0).log_pdf(grid) + Poisson(grid).log_pdf(counts.reshape(-1, 1)).sum(0)
    return torch.trapezoid(torch.exp(log_q) * (log_p - log_q), grid)


def test_score_function_entry_with_gaussian_group_is_unbiased():
    """The ELBO separates into an ``x`` part and a ``mu`` part; the score-function gradient of the Gamma entry
    (whose learning signal is the full ELBO sample, including the group's noise) matches quadrature."""
    counts = torch.tensor([2.0, 4.0, 3.0])
    with Model() as model:
        x = RandomParameter("x", Exponential(1.0))
        mu = RandomParameter("mu", Normal(0.0, 1.0))
        ObservedParameter("counts", Poisson(x), counts)
        ObservedParameter("y", Normal(mu, 1.0), torch.tensor([0.5, 1.5]))
    with Guide(model, default_family="mean_field") as guide:
        RandomParameter("x", Gamma(VariationalParameter("a", torch.tensor([3.0])), VariationalParameter("b", torch.tensor([1.5]))))
    torch.manual_seed(0)
    _, grads = elbo(model, guide, n_samples=6000)
    h = 1e-4
    da = (_quadrature_elbo(3.0 + h, 1.5, counts) - _quadrature_elbo(3.0 - h, 1.5, counts)) / (2 * h)
    db = (_quadrature_elbo(3.0, 1.5 + h, counts) - _quadrature_elbo(3.0, 1.5 - h, counts)) / (2 * h)
    assert grads["x_a"].item() == pytest.approx(da.item(), abs=0.15)
    assert grads["x_b"].item() == pytest.approx(db.item(), abs=0.3)
    # The group's pathwise gradient of the mu part: d/dloc E[log p] = sum(y - loc) - loc at scale 0.1.
    group = guide.groups[0]
    loc = group.loc.item()
    assert grads["default_mu_loc"].item() == pytest.approx(2.0 - 3 * loc, abs=0.05)


# ---------------------------------------------------------------------------
# Exact posterior recovery and equivalence with the removed automatic guides
# ---------------------------------------------------------------------------
def test_full_rank_group_recovers_the_correlated_block():
    model, x, y = _regression(extra=False)
    design = torch.stack([torch.ones_like(x), x], dim=1)
    precision = design.T @ design / 0.09 + torch.eye(2) / 9
    covariance = torch.linalg.inv(precision)
    mean = covariance @ design.T @ y / 0.09

    torch.manual_seed(0)
    grouped = Guide(model, default_family="full_rank")
    BBVI(model, grouped, n_samples=4, epochs=1000, lr=0.03, lrd=0.999, verbose=False)
    fitted = grouped.covariance()
    torch.testing.assert_close(fitted.mean, mean, atol=0.05, rtol=0)
    torch.testing.assert_close(fitted.matrix, covariance, atol=0.2 * covariance.abs().max().item(), rtol=0)

    torch.manual_seed(0)
    mean_field = Guide(model, default_family="mean_field")
    BBVI(model, mean_field, n_samples=4, epochs=1000, lr=0.03, lrd=0.999, verbose=False)
    diagonal = mean_field.covariance().matrix.diagonal()
    # Mean-field matches the conditional variances 1 / precision_ii, far below the marginal ones.
    torch.testing.assert_close(diagonal, 1 / precision.diagonal(), rtol=0.25, atol=0)
    assert (diagonal < 0.5 * covariance.diagonal()).all()
    assert mean_field.covariance().matrix[0, 1].item() == 0.0


# Reference values computed with the removed AutoNormal / AutoMultivariateNormal
# classes (same model, parameter values and seed).
_OLD_MEAN_FIELD = (-7.1495405749243774, {
    "mean_loc": [2.8531777268919156], "mean_scale": [1.5159328784513313],
    "sigma_loc": [0.3184454045549784], "sigma_scale": [1.9422920118822002],
})
_OLD_FULL_RANK = (-7.212718899032812, {
    "loc": [3.267323282451153, 0.8485996102542108],
    "scale_tril": [[1.1107869815825333, 0.0], [-0.8987165566403169, 2.000349446034971]],
})
_OLD_FULL_RANK_SAMPLE = [0.5766101625288188, 1.0810713364927451, 2.1204964952451353]


def test_default_family_reproduces_the_removed_automatic_guides():
    model = _two_latent_model()
    guide = Guide(model, default_family="mean_field", init_scale=0.4)
    group = guide.groups[0]
    group.locs["mean"].set_new_value(torch.tensor([0.7]))
    group.locs["sigma"].set_new_value(torch.tensor([-0.2]))
    group.scales["mean"].set_new_value(torch.tensor([0.3]))
    group.scales["sigma"].set_new_value(torch.tensor([0.5]))
    torch.manual_seed(123)
    value, grads = elbo(model, guide, n_samples=3)
    assert value.item() == pytest.approx(_OLD_MEAN_FIELD[0], rel=1e-12)
    assert set(grads) == set(_OLD_MEAN_FIELD[1])
    for key, reference in _OLD_MEAN_FIELD[1].items():
        torch.testing.assert_close(grads[key], torch.tensor(reference), rtol=1e-10, atol=1e-12)

    model = _two_latent_model()
    guide = Guide(model, default_family="full_rank", init_scale=0.4)
    group = guide.groups[0]
    group.loc_parameter.set_new_value(torch.tensor([0.7, -0.2]))
    group.scale_tril_parameter.set_new_value(torch.tensor([[0.3, 0.0], [0.1, 0.5]]))
    torch.manual_seed(123)
    value, grads = elbo(model, guide, n_samples=3)
    assert value.item() == pytest.approx(_OLD_FULL_RANK[0], rel=1e-12)
    assert set(grads) == set(_OLD_FULL_RANK[1])
    for key, reference in _OLD_FULL_RANK[1].items():
        torch.testing.assert_close(grads[key], torch.tensor(reference), rtol=1e-10, atol=1e-12)
    torch.manual_seed(5)
    torch.testing.assert_close(guide.sample(3)["sigma"].reshape(-1), torch.tensor(_OLD_FULL_RANK_SAMPLE))


# ---------------------------------------------------------------------------
# Mean and covariance
# ---------------------------------------------------------------------------
def test_unconstrained_moments_are_exact():
    model = _two_latent_model()
    guide = Guide(model, default_family="full_rank")
    group = guide.groups[0]
    group.loc_parameter.set_new_value(torch.tensor([0.7, -0.2]))
    L = torch.tensor([[0.3, 0.0], [0.1, 0.5]])
    group.scale_tril_parameter.set_new_value(L)
    result = guide.covariance()
    assert result.estimated == [] and result.n_samples is None
    torch.testing.assert_close(result.matrix, L @ L.T)
    torch.testing.assert_close(result.mean, torch.tensor([0.7, -0.2]))
    means = guide.mean()
    torch.testing.assert_close(means["sigma"], torch.tensor([-0.2]))
    torch.testing.assert_close(group.covariance().matrix, L @ L.T)

    guide = Guide(model, default_family="mean_field")
    guide.groups[0].scales["sigma"].set_new_value(torch.tensor([0.4]))
    torch.testing.assert_close(guide.covariance().matrix, torch.diag(torch.tensor([0.01, 0.16])))


@pytest.mark.parametrize("family", ["mean_field", "full_rank"])
def test_constrained_log_normal_moments_match_monte_carlo(family):
    model = _two_latent_model()  # sigma has a log transform, mean the identity
    guide = Guide(model, default_family=family)
    group = guide.groups[0]
    if family == "mean_field":
        group.locs["sigma"].set_new_value(torch.tensor([0.1]))
        group.scales["mean"].set_new_value(torch.tensor([0.6]))
        group.scales["sigma"].set_new_value(torch.tensor([0.5]))
    else:
        group.loc_parameter.set_new_value(torch.tensor([0.3, 0.1]))
        group.scale_tril_parameter.set_new_value(torch.tensor([[0.6, 0.0], [0.3, 0.4]]))
    result = guide.covariance("constrained")
    assert result.estimated == []
    z = group.sample_unconstrained(400000, torch.Generator().manual_seed(0))
    draws = torch.stack([z[:, 0], torch.exp(z[:, 1])], dim=1)
    torch.testing.assert_close(result.mean, draws.mean(0), rtol=0.01, atol=0.005)
    torch.testing.assert_close(result.matrix, torch.cov(draws.T), rtol=0.03, atol=0.003)
    if family == "full_rank":
        assert result.matrix[0, 1].abs() > 0.05


def test_monte_carlo_blocks_are_flagged_and_reproducible():
    with Model() as model:
        p = RandomParameter("p", Beta(2.0, 3.0))
        a = RandomParameter("a", Normal(0.0, 1.0))
        ObservedParameter("y", Normal(a, 1.0), torch.tensor([0.4]))
        ObservedParameter("k", Bernoulli(p), torch.tensor([1.0]))
    with Guide(model, default_family="mean_field") as guide:
        Group(["p", "a"], family="full_rank")
    group = guide.groups[0]
    group.loc_parameter.set_new_value(torch.tensor([0.2, -0.1]))
    group.scale_tril_parameter.set_new_value(torch.tensor([[0.5, 0.0], [0.4, 0.3]]))
    state = torch.random.get_rng_state()
    result = guide.covariance("constrained", n_samples=4000, seed=3)
    assert torch.equal(torch.random.get_rng_state(), state)  # the global generator is untouched
    assert result.estimated == ["p"] and result.exact == ["a"] and result.n_samples == 4000
    assert set(result.standard_error) == {"p"}
    # The identity block is exact even inside a group with an estimated latent.
    torch.testing.assert_close(result.block("a"), torch.tensor([[0.25]]))
    torch.testing.assert_close(result.mean[1:], torch.tensor([-0.1]))
    z = group.sample_unconstrained(400000, torch.Generator().manual_seed(1))
    p_draws = torch.sigmoid(z[:, 0])
    assert result.mean[0].item() == pytest.approx(p_draws.mean().item(), abs=4 * result.standard_error["p"].item())
    assert result.standard_error["p"].item() == pytest.approx(p_draws.std().item() / math.sqrt(4000), rel=0.1)
    cross = torch.cov(torch.stack([p_draws, z[:, 1]]))[0, 1]
    assert result.block("p", "a").item() == pytest.approx(cross.item(), abs=0.01)
    again = guide.covariance("constrained", n_samples=4000, seed=3)
    torch.testing.assert_close(again.matrix, result.matrix)


def test_cross_group_blocks_are_zero_and_entries_use_closed_forms():
    model, _, _ = _regression()
    with Guide(model, default_family="mean_field") as guide:
        Group(["a", "b"], family="full_rank")
        RandomParameter("sigma", LogNormal(VariationalParameter("m", torch.tensor([-0.2])), VariationalParameter("s", torch.tensor([0.4]))))
    for space in ("unconstrained", "constrained"):
        result = guide.covariance(space)
        assert result.names == ["a", "b", "c", "sigma"]
        for first, second in (("a", "c"), ("b", "sigma"), ("c", "sigma")):
            assert torch.count_nonzero(result.block(first, second)) == 0
    # A support-matched LogNormal entry is exact in the constrained space ...
    constrained = guide.covariance("constrained")
    assert constrained.estimated == []
    reference = LogNormal(torch.tensor([-0.2]), torch.tensor([0.4]))
    torch.testing.assert_close(constrained.block("sigma"), torch.diag(reference.variance()))
    torch.testing.assert_close(guide.mean("constrained")["sigma"], reference.mean())
    # ... and Monte Carlo in the unconstrained space, where it is a Normal(-0.2, 0.4).
    unconstrained = guide.covariance("unconstrained", n_samples=20000)
    assert unconstrained.estimated == ["sigma"]
    assert unconstrained.mean[-1].item() == pytest.approx(-0.2, abs=0.02)
    assert unconstrained.block("sigma").item() == pytest.approx(0.16, rel=0.05)
    # A real-line entry is exact in the unconstrained space.
    with Guide(model, default_family="mean_field") as guide:
        RandomParameter("sigma", Normal(VariationalParameter("m", torch.tensor([-0.2])), VariationalParameter("s", torch.tensor([0.4]))))
    result = guide.covariance()
    assert result.estimated == []
    assert result.block("sigma").item() == pytest.approx(0.16)


# ---------------------------------------------------------------------------
# Distribution moments
# ---------------------------------------------------------------------------
D = torch.distributions
_A = torch.tensor([0.7, 2.5])
_B = torch.tensor([1.3, 0.4])
_P = torch.tensor([0.2, 0.65])
_MOMENT_CASES = [
    (Normal(_A, _B), D.Normal(_A, _B)),
    (HalfNormal(_B), D.HalfNormal(_B)),
    (LogNormal(_A - 1, _B), D.LogNormal(_A - 1, _B)),
    (StudentT(torch.tensor([3.0, 5.0]), _A, _B), D.StudentT(torch.tensor([3.0, 5.0]), _A, _B)),
    (Laplace(_A, _B), D.Laplace(_A, _B)),
    (Exponential(_B), D.Exponential(_B)),
    (Gamma(_A, _B), D.Gamma(_A, _B)),
    (InvGamma(_A + 2, _B), D.InverseGamma(_A + 2, _B)),
    (Beta(_A, _B), D.Beta(_A, _B)),
    (Uniform(_B - 2, _A), D.Uniform(_B - 2, _A)),
    (Dirichlet(torch.tensor([0.5, 1.5, 3.0])), D.Dirichlet(torch.tensor([0.5, 1.5, 3.0]))),
    (MultivariateNormal(_A, torch.tensor([[1.0, 0.3], [0.3, 0.5]])), D.MultivariateNormal(_A, torch.tensor([[1.0, 0.3], [0.3, 0.5]]))),
    (Bernoulli(_P), D.Bernoulli(_P)),
    (Binomial(torch.tensor([5.0, 9.0]), _P), D.Binomial(torch.tensor([5.0, 9.0]), _P)),
    (Poisson(_A), D.Poisson(_A)),
    (NegativeBinomial(torch.tensor([3.0, 1.5]), _P), D.NegativeBinomial(torch.tensor([3.0, 1.5]), _P)),
    (Categorical(torch.tensor([0.2, 0.5, 0.3])), None),
    (Geometric(_P), None),
    (DiscreteUniform(torch.tensor([1.0, -2.0]), torch.tensor([4.0, 3.0])), None),
    (Wishart(torch.tensor(4.0), torch.tensor([[1.0, 0.3], [0.3, 0.5]])), D.Wishart(torch.tensor(4.0), torch.tensor([[1.0, 0.3], [0.3, 0.5]]))),
    (Mixture([Normal(-1.0, 0.5), Normal(2.0, 1.0)], torch.tensor([0.3, 0.7])), None),
]


@pytest.mark.parametrize("ours, reference", _MOMENT_CASES, ids=[type(case[0]).__name__ for case in _MOMENT_CASES])
def test_distribution_moments(ours, reference):
    if reference is None:
        if isinstance(ours, Categorical):
            p = torch.tensor([0.2, 0.5, 0.3])
            k = torch.arange(3.0)
            expected_mean, expected_variance = (p * k).sum(), (p * k ** 2).sum() - (p * k).sum() ** 2
        elif isinstance(ours, Geometric):
            # Number of trials: torch counts failures, so the mean is shifted by one.
            torch_geometric = D.Geometric(_P)
            expected_mean, expected_variance = torch_geometric.mean + 1, torch_geometric.variance
        elif isinstance(ours, DiscreteUniform):
            draws = torch.stack([torch.arange(1.0, 5.0).mean(), torch.arange(-2.0, 4.0).mean()])
            expected_mean = draws
            expected_variance = torch.stack([torch.arange(1.0, 5.0).var(unbiased=False), torch.arange(-2.0, 4.0).var(unbiased=False)])
        else:  # Mixture
            draws = ours.sample(400000)
            expected_mean, expected_variance = draws.mean(), draws.var()
            torch.testing.assert_close(ours.mean(), expected_mean, atol=0.02, rtol=0)
            torch.testing.assert_close(ours.variance(), expected_variance, atol=0, rtol=0.02)
            assert ours.mean().item() == pytest.approx(0.3 * -1.0 + 0.7 * 2.0)
            return
    else:
        expected_mean, expected_variance = reference.mean, reference.variance
    torch.testing.assert_close(ours.mean(), expected_mean.to(ours.mean().dtype))
    torch.testing.assert_close(ours.variance(), expected_variance.to(ours.variance().dtype))
    if isinstance(ours, MultivariateNormal):
        torch.testing.assert_close(ours.covariance(), reference.covariance_matrix)
    if isinstance(ours, Dirichlet):
        draws = reference.sample((400000,))
        torch.testing.assert_close(ours.covariance(), torch.cov(draws.T).to(ours.mean().dtype), atol=1e-3, rtol=0)


def test_moments_that_do_not_exist():
    t = StudentT(torch.tensor([0.8, 1.5, 3.0]))
    assert torch.isnan(t.mean()[0]) and t.mean()[1] == 0
    assert torch.isnan(t.variance()[0]) and torch.isinf(t.variance()[1]) and t.variance()[2] == pytest.approx(3.0)
    assert torch.isinf(InvGamma(torch.tensor([0.5, 1.5]), 1.0).variance()).all()
    with pytest.raises(ValueError, match="no mean"):
        Cauchy(0.0, 1.0).mean()
    from BayesianDLL.Distributions import Truncated
    with pytest.raises(NotImplementedError):
        Truncated(Normal(0.0, 1.0), low=0.0).mean()


# ---------------------------------------------------------------------------
# Sampling, minibatches and exact samplers
# ---------------------------------------------------------------------------
def test_sample_shapes_and_deterministics(monkeypatch):
    module = importlib.import_module("BayesianDLL.Samplers._sample")
    monkeypatch.setattr(module, "sample", lambda *args, **kwargs: pytest.fail("a guide must never run MCMC"))
    model, x, _ = _regression()
    with Guide(model, default_family="mean_field") as guide:
        Group(["a", "b"], family="full_rank")
        RandomParameter("sigma", LogNormal(VariationalParameter("m", torch.tensor([-0.2])), VariationalParameter("s", torch.tensor([0.4]))))
    trace = guide.sample(50, n_chains=3)
    assert trace["a"].shape == (3, 50, 1) and trace["sigma"].shape == (3, 50, 1)
    assert (trace["sigma"] > 0).all()
    assert trace["mu"].shape == (3, 50, len(x))
    torch.testing.assert_close(trace["mu"], trace["a"] + trace["b"] * x)


def test_entry_without_an_exact_sampler_raises():
    class NoSampler(Normal):
        has_rsample = False

        def sample(self, n_samples=1, _reparametrization_trick_grad=False):
            raise NotImplementedError

    model = _two_latent_model()
    with Guide(model, default_family="mean_field") as guide:
        RandomParameter("sigma", NoSampler(VariationalParameter("m", torch.tensor([0.0])), VariationalParameter("s", torch.tensor([1.0]))))
    with pytest.raises(NotImplementedError, match="'sigma' \\(NoSampler\\) has no exact sampler.*never with MCMC"):
        guide.sample(10)
    with pytest.raises(NotImplementedError, match="no exact sampler"):
        elbo(model, guide, n_samples=2)
    # Closed-form moments need no sampler; Monte Carlo ones do.
    assert guide.covariance().estimated == []
    with pytest.raises(NotImplementedError, match="no exact sampler"):
        guide.covariance("constrained")


def test_minibatch_bbvi_with_a_grouped_guide():
    model, _, _ = _regression(n=40, extra=True)
    with Guide(model, default_family="mean_field") as guide:
        Group(["a", "b"], family="full_rank")
    history = BBVI(model, guide, n_samples=2, epochs=30, lr=0.02, batch_size=10, verbose=False)
    assert len(history) == 30 and all(math.isfinite(value) for value in history)
    assert model.data["x"].value.shape == (40,)
    assert torch.isfinite(guide.groups[0].scale_tril).all()


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def test_guides_cannot_be_conditioned_but_models_can():
    from BayesianDLL import condition, marginalize

    model, _, _ = _regression()
    guide = Guide(model, default_family="mean_field")
    for attempt in (
        lambda: condition(guide, a=0.5), lambda: guide.condition(a=0.5),
        lambda: marginalize(guide, ["a"]), lambda: guide.marginalize(["a"]),
    ):
        with pytest.raises(TypeError, match=r"cannot be conditioned.*condition the MODEL.*needs? MCMC"):
            attempt()

    conditioned = condition(model, sigma=0.3)
    assert "sigma" not in conditioned.params
    guide = Guide(conditioned, default_family="mean_field")
    assert guide.groups[0].names == ["a", "b", "c"]
    trace = guide.sample(5)
    assert "sigma" not in trace.trace
    # The conditioned latent is observed: a group or an entry for it is an error.
    with pytest.raises(ValueError, match="observed in the model"):
        with Guide(conditioned, default_family="mean_field"):
            Group(["sigma"])
    with Guide(conditioned, default_family="mean_field") as guide:
        RandomParameter("sigma", LogNormal(VariationalParameter("m", 0.0), VariationalParameter("s", 0.5)))
    with pytest.raises(ValueError, match="observed in the model"):
        guide.validate()
