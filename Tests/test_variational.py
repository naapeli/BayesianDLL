import pytest
import torch

from BayesianDLL import Guide, Model, RandomParameter, VariationalParameter, find_MAP
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI, elbo


@pytest.mark.integration
@pytest.mark.parametrize("use_method", [False, True])
def test_map_recovers_conjugate_normal_mode(normal_model, use_method):
    before = normal_model.model_log_prob().item()
    if use_method:
        normal_model.find_MAP(lr=0.08, epochs=250, verbose=False)
    else:
        find_MAP(normal_model, lr=0.08, epochs=250, verbose=False)
    assert normal_model.params["mean"].constrained_value.item() == pytest.approx(1.6, abs=1e-4)
    assert normal_model.model_log_prob().item() > before


def test_map_history_and_reporting(normal_model, capsys):
    result = normal_model.find_MAP(epochs=5, callback_frequency=2)
    history = result.history
    assert len(history) == 3
    assert capsys.readouterr().out.count("Epoch:") == 3
    assert history[-1] > history[0]


def test_map_records_history_when_silent(normal_model, capsys):
    result = find_MAP(normal_model, epochs=5, callback_frequency=2, verbose=False)
    assert len(result.history) == 3
    # log_prob is the log density at the returned values, not the last recorded one.
    assert result.log_prob == normal_model.model_log_prob().item()
    assert capsys.readouterr().out == ""


def make_guide(model, mean_value=0.3, std_value=1.1, requires_grad=False):
    mean = torch.tensor([mean_value], requires_grad=requires_grad)
    std = torch.tensor([std_value], requires_grad=requires_grad)
    with Guide(model) as guide:
        RandomParameter("mean", Normal(VariationalParameter("loc", mean), VariationalParameter("std", std, min=0.1)))
    return guide, mean, std


def test_elbo_reparameterization_gradients_match_autograd(normal_model):
    guide, mean, std = make_guide(normal_model, requires_grad=True)
    model_before = normal_model.params["mean"].unconstrained_value.clone()
    guide_before = guide.params["mean"].unconstrained_value.clone()
    value, grads = elbo(normal_model, guide, n_samples=10)
    expected = torch.autograd.grad(value, (mean, std))
    assert value.ndim == 0
    for key, grad in zip(("mean_loc", "mean_std"), expected):
        torch.testing.assert_close(grads[key], grad)
    torch.testing.assert_close(normal_model.params["mean"].unconstrained_value, model_before)
    torch.testing.assert_close(guide.params["mean"].unconstrained_value, guide_before)


def test_elbo_preserves_vector_reparameterization_gradients():
    dimension = 3
    with Model() as model:
        RandomParameter(
            "z",
            Normal(torch.zeros(dimension), torch.ones(dimension)),
            shape=dimension,
        )

    mean = torch.tensor([0.3, -0.4, 0.8], requires_grad=True)
    std = torch.tensor([1.1, 0.8, 1.2], requires_grad=True)
    with Guide(model) as guide:
        RandomParameter(
            "z",
            Normal(
                VariationalParameter("loc", mean),
                VariationalParameter("std", std, min=0.1),
            ),
            shape=dimension,
        )

    value, grads = elbo(model, guide, n_samples=10)
    expected = torch.autograd.grad(value, (mean, std))
    torch.testing.assert_close(grads["z_loc"], expected[0])
    torch.testing.assert_close(grads["z_std"], expected[1])


def test_elbo_is_zero_for_identical_model_and_guide():
    with Model() as model:
        RandomParameter("mean", Normal(torch.tensor([0.3]), torch.tensor([1.1])))
    guide, _, _ = make_guide(model)
    value, _ = elbo(model, guide, n_samples=8)
    assert value.item() == pytest.approx(0, abs=1e-12)


@pytest.mark.integration
def test_bbvi_approaches_analytic_normal_posterior(normal_model):
    guide, _, _ = make_guide(normal_model, mean_value=-1.0, std_value=1.0)
    BBVI(normal_model, guide, epochs=200, n_samples=12, lr=0.05, verbose=False)
    dist = guide.params["mean"].distribution
    assert dist.mu.value.item() == pytest.approx(1.6, abs=0.18)
    assert dist.sigma.value.item() ** 2 == pytest.approx(4 / 13, abs=0.15)
    assert dist.sigma.value.item() > 0


def test_bbvi_reports_history(normal_model, capsys):
    guide, _, _ = make_guide(normal_model)
    history = BBVI(normal_model, guide, epochs=4, n_samples=2, callback_frequency=2)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
    assert capsys.readouterr().out.count("ELBO:") == 2


# ---------------------------------------------------------------------------
# Constrained-space guides, deterministic guides and REINFORCE
# ---------------------------------------------------------------------------
from BayesianDLL import DeterministicParameter, ObservedParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Beta, Exponential, Gamma, LogNormal, Poisson


def _exponential_model(counts=None):
    with Model() as model:
        x = RandomParameter("x", Exponential(1.0))
        if counts is not None:
            ObservedParameter("counts", Poisson(x), counts)
    return model


def test_lognormal_guide_describes_constrained_value_and_matches_autograd():
    model = _exponential_model(counts=torch.tensor([2.0, 4.0, 3.0]))
    mean = torch.tensor([0.4], requires_grad=True)
    std = torch.tensor([0.6], requires_grad=True)
    with Guide(model) as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", mean), VariationalParameter("s", std)))
    value, grads = elbo(model, guide, n_samples=7)
    expected = torch.autograd.grad(value, (mean, std))
    torch.testing.assert_close(grads["x_m"], expected[0])
    torch.testing.assert_close(grads["x_s"], expected[1])


def test_lognormal_guide_elbo_is_zero_for_the_exact_posterior():
    with Model() as model:
        RandomParameter("x", LogNormal(0.3, 0.7))
    with Guide(model) as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", torch.tensor([0.3])), VariationalParameter("s", torch.tensor([0.7]))))
    value, _ = elbo(model, guide, n_samples=5)
    assert value.item() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.integration
def test_bbvi_lognormal_guide_reaches_known_optimum():
    """For p = Exp(1) the best LogNormal(m, s) has s = 1 and m = -1/2."""
    model = _exponential_model()
    from BayesianDLL.Distributions import LogTransform
    with Guide(model) as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", torch.tensor([1.0])), VariationalParameter("s", torch.tensor([0.4]), transform=LogTransform())))
    BBVI(model, guide, n_samples=16, epochs=1500, lr=0.02, verbose=False)
    distribution = guide.params["x"].distribution
    assert distribution.mu.value.item() == pytest.approx(-0.5, abs=0.12)
    assert distribution.sigma.value.item() == pytest.approx(1.0, abs=0.12)


def test_deterministic_guide_gradient_matches_autograd():
    model = _exponential_model(counts=torch.tensor([1.0, 2.0]))
    mean = torch.tensor([0.1], requires_grad=True)
    std = torch.tensor([0.5], requires_grad=True)
    with Guide(model) as guide:
        u = RandomParameter("u", Normal(VariationalParameter("m", mean), VariationalParameter("s", std)))
        # x = exp(2 u): the Jacobian terms do not cancel.
        DeterministicParameter("x", lambda u: torch.exp(2 * u), lambda u: {"u": 2 * torch.exp(2 * u)}, [u])
    value, grads = elbo(model, guide, n_samples=6)
    expected = torch.autograd.grad(value, (mean, std))
    torch.testing.assert_close(grads["u_m"], expected[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(grads["u_s"], expected[1], rtol=1e-6, atol=1e-7)


def test_exp_deterministic_guide_equals_normal_guide_on_log_scale():
    model = _exponential_model(counts=torch.tensor([1.0, 2.0]))
    with Guide(model) as direct:
        RandomParameter("x", Normal(VariationalParameter("m", torch.tensor([0.2])), VariationalParameter("s", torch.tensor([0.3]))))
    with Guide(model) as indirect:
        log_x = RandomParameter("log_x", Normal(VariationalParameter("m", torch.tensor([0.2])), VariationalParameter("s", torch.tensor([0.3]))))
        Exp("x", log_x)
    torch.manual_seed(1)
    value_direct, grads_direct = elbo(model, direct, n_samples=4)
    torch.manual_seed(1)
    value_indirect, grads_indirect = elbo(model, indirect, n_samples=4)
    torch.testing.assert_close(value_direct, value_indirect)
    torch.testing.assert_close(grads_direct["x_m"], grads_indirect["log_x_m"], rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(grads_direct["x_s"], grads_indirect["log_x_s"], rtol=1e-6, atol=1e-8)


def _quadrature_elbo(a, b, counts):
    """Exact ELBO of a Gamma(a, b) guide for x ~ Exp(1), counts ~ Poisson(x)."""
    q = Gamma(torch.tensor([a]), torch.tensor([b]))
    grid = torch.linspace(1e-5, 25.0, 400001)
    log_q = q.log_pdf(grid)
    log_p = Exponential(1.0).log_pdf(grid) + Poisson(grid).log_pdf(counts.reshape(-1, 1)).sum(0)
    return torch.trapezoid(torch.exp(log_q) * (log_p - log_q), grid)


def test_reinforce_gradient_matches_exact_gradient_without_mcmc():
    import importlib
    module = importlib.import_module("BayesianDLL.Variational._ELBO")
    assert not hasattr(module, "sample")  # draws come from distribution.sample, not MCMC
    counts = torch.tensor([2.0, 4.0, 3.0])
    model = _exponential_model(counts=counts)
    with Guide(model) as guide:
        RandomParameter("x", Gamma(VariationalParameter("a", torch.tensor([3.0])), VariationalParameter("b", torch.tensor([1.5]))))
    value, grads = elbo(model, guide, n_samples=20000)
    h = 1e-4
    da = (_quadrature_elbo(3.0 + h, 1.5, counts) - _quadrature_elbo(3.0 - h, 1.5, counts)) / (2 * h)
    db = (_quadrature_elbo(3.0, 1.5 + h, counts) - _quadrature_elbo(3.0, 1.5 - h, counts)) / (2 * h)
    assert value.item() == pytest.approx(_quadrature_elbo(3.0, 1.5, counts).item(), abs=0.03)
    assert grads["x_a"].item() == pytest.approx(da.item(), abs=0.05)
    assert grads["x_b"].item() == pytest.approx(db.item(), abs=0.1)


def test_guide_support_must_match_model_support():
    model = _exponential_model()
    with Guide(model) as guide:
        RandomParameter("x", Beta(VariationalParameter("a", torch.tensor([2.0])), VariationalParameter("b", torch.tensor([2.0]))))
    with pytest.raises(ValueError, match="support"):
        elbo(model, guide)


def test_guide_must_cover_every_model_latent(normal_model):
    with Guide(normal_model) as guide:
        RandomParameter("other", Normal(VariationalParameter("loc", torch.zeros(1)), VariationalParameter("scale", torch.ones(1))))
    with pytest.raises(ValueError, match="no distribution"):
        elbo(normal_model, guide)


def test_bbvi_float32_variational_parameters_with_float64_model():
    """Python-number VariationalParameters are float32; the model is float64."""
    from BayesianDLL import Guide, Model, RandomParameter, VariationalParameter
    from BayesianDLL.Distributions import Mixture, Normal
    from BayesianDLL.Variational import BBVI

    torch.manual_seed(0)
    with Model() as model:
        RandomParameter("x", Mixture([Normal(-1, 0.5), Normal(2, 1.0)], [0.3, 0.7]))
    with Guide(model) as guide:
        mean = VariationalParameter("mean", 0)
        std = VariationalParameter("std", 2, min=1e-3)
        RandomParameter("x", Normal(mean, std))

    BBVI(model, guide, epochs=5, lr=1e-2, verbose=False)
    assert mean.value.dtype == torch.float32
    assert std.value.dtype == torch.float32


# ---------------------------------------------------------------------------
# Guide: sampling, validation and agreement with a mean-field Gaussian group
# ---------------------------------------------------------------------------
import math

from BayesianDLL.Distributions import Bernoulli
from BayesianDLL.Variational import BaseGuide


def test_guide_interface(normal_model):
    guide, _, _ = make_guide(normal_model)
    assert isinstance(guide, BaseGuide) and isinstance(guide, Model)
    assert isinstance(Guide(normal_model, default_family="mean_field"), BaseGuide)
    assert set(guide.variational_parameters()) == {"mean_loc", "mean_std"}


def test_guide_sample_lognormal_moments_without_mcmc(monkeypatch):
    import importlib
    module = importlib.import_module("BayesianDLL.Samplers._sample")
    monkeypatch.setattr(module, "sample", lambda *args, **kwargs: pytest.fail("Guide.sample must not run MCMC"))
    with Model() as model:
        x = RandomParameter("x", Exponential(1.0))
        DeterministicParameter("twice", lambda x: 2 * x, lambda x: {"x": 2 * torch.ones_like(x)}, [x])
    m, s = 0.3, 0.5
    with Guide(model) as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", torch.tensor([m])), VariationalParameter("s", torch.tensor([s]))))
    trace = guide.sample(20000, n_chains=2)
    assert trace["x"].shape == (2, 20000, 1)
    draws = trace["x"].reshape(-1)
    assert draws.mean().item() == pytest.approx(math.exp(m + s ** 2 / 2), rel=0.02)
    assert draws.var().item() == pytest.approx((math.exp(s ** 2) - 1) * math.exp(2 * m + s ** 2), rel=0.06)
    torch.testing.assert_close(trace["twice"], 2 * trace["x"])


def test_guide_sample_normal_moments(normal_model):
    guide, _, _ = make_guide(normal_model, mean_value=0.3, std_value=1.1)
    guide.validate(normal_model)
    draws = guide.sample(40000)["mean"].reshape(-1)
    assert draws.mean().item() == pytest.approx(0.3, abs=0.03)
    assert draws.std().item() == pytest.approx(1.1, rel=0.02)


def test_real_line_guide_samples_the_unconstrained_value():
    model = _exponential_model()
    with Guide(model) as guide:
        RandomParameter("x", Normal(VariationalParameter("m", torch.tensor([0.2])), VariationalParameter("s", torch.tensor([0.4]))))
    draws = guide.sample(40000)["x"].reshape(-1)
    assert (draws > 0).all()
    assert torch.log(draws).mean().item() == pytest.approx(0.2, abs=0.01)
    assert torch.log(draws).std().item() == pytest.approx(0.4, rel=0.02)


def test_guide_needs_a_model():
    with pytest.raises(TypeError, match="needs the model"):
        Guide()
    with pytest.raises(TypeError, match="needs the model"):
        Guide("mean_field")


def test_guide_must_be_mean_field(normal_model):
    with Guide(normal_model) as guide:
        loc = RandomParameter("loc", Normal(VariationalParameter("m", torch.zeros(1)), VariationalParameter("s", torch.ones(1))))
        RandomParameter("mean", Normal(loc, VariationalParameter("std", torch.ones(1))))
    with pytest.raises(NotImplementedError, match="mean-field"):
        guide.validate(normal_model)


def test_guide_rejects_enumerated_latents():
    with Model() as model:
        RandomParameter("z", Bernoulli(0.3), marginalize=True)
        RandomParameter("x", Normal(0.0, 1.0))
    with Guide(model) as guide:
        RandomParameter("z", Bernoulli(VariationalParameter("p", torch.tensor([0.5]))), marginalize=True)
        RandomParameter("x", Normal(VariationalParameter("m", torch.zeros(1)), VariationalParameter("s", torch.ones(1))))
    with pytest.raises(ValueError, match="enumerated"):
        guide.validate(model)
    with Guide(model) as guide:
        RandomParameter("z", Bernoulli(VariationalParameter("p", torch.tensor([0.5]))))
        RandomParameter("x", Normal(VariationalParameter("m", torch.zeros(1)), VariationalParameter("s", torch.ones(1))))
    with pytest.raises(ValueError, match="enumerated"):
        guide.validate(model)


def test_bbvi_rejects_objects_that_are_not_guides(normal_model):
    with Model() as not_a_guide:
        RandomParameter("mean", Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="guide must be"):
        BBVI(normal_model, not_a_guide, epochs=1, verbose=False)


@pytest.mark.integration
def test_bbvi_guide_and_mean_field_group_reach_the_same_optimum(normal_model):
    """For a conjugate Normal model both Normal families contain the exact posterior N(1.6, 4/13)."""
    torch.manual_seed(0)
    guide, _, _ = make_guide(normal_model, mean_value=-1.0, std_value=1.0)
    BBVI(normal_model, guide, epochs=1500, n_samples=16, lr=0.02, lrd=0.999, verbose=False)
    torch.manual_seed(0)
    auto = Guide(normal_model, default_family="mean_field", init_scale=1.0)
    BBVI(normal_model, auto, epochs=1500, n_samples=16, lr=0.02, lrd=0.999, verbose=False)
    distribution = guide.params["mean"].distribution
    hand_loc, hand_scale = distribution.mu.value.item(), distribution.sigma.value.item()
    auto_loc, auto_scale = auto.groups[0].locs["mean"].value.item(), auto.groups[0].scales["mean"].value.item()
    assert hand_loc == pytest.approx(1.6, abs=0.05)
    assert hand_scale == pytest.approx(math.sqrt(4 / 13), abs=0.05)
    assert hand_loc == pytest.approx(auto_loc, abs=0.05)
    assert hand_scale == pytest.approx(auto_scale, abs=0.05)


def test_guide_mixture_registers_component_variational_parameters():
    """A Mixture in a guide exposes its components' VariationalParameters and gets an ELBO gradient for them."""
    from BayesianDLL.Distributions import Mixture

    with Model() as model:
        RandomParameter("x", Normal(0.0, 1.0))
    m1 = VariationalParameter("m1", torch.tensor(-1.0, dtype=torch.float64))
    m2 = VariationalParameter("m2", torch.tensor(1.5, dtype=torch.float64))
    with Guide(model) as guide:
        RandomParameter("x", Mixture([Normal(m1, 1.0), Normal(m2, 1.0)], torch.tensor([0.4, 0.6], dtype=torch.float64)))
    assert set(guide.variational_parameters()) == {"x_m1", "x_m2"}
    assert not Mixture.has_rsample  # score-function path

    # Oracle: the exact ELBO by quadrature, differentiated by central differences.
    grid = torch.linspace(-15, 15, 30001, dtype=torch.float64)

    def exact_elbo(a, b):
        log_q = torch.logsumexp(torch.stack([
            math.log(0.4) - 0.5 * (grid - a) ** 2, math.log(0.6) - 0.5 * (grid - b) ** 2,
        ]), 0) - 0.5 * math.log(2 * math.pi)
        log_p = -0.5 * grid ** 2 - 0.5 * math.log(2 * math.pi)
        return torch.trapezoid(log_q.exp() * (log_p - log_q), grid)

    h = 1e-4
    expected = torch.stack([
        (exact_elbo(-1.0 + h, 1.5) - exact_elbo(-1.0 - h, 1.5)) / (2 * h),
        (exact_elbo(-1.0, 1.5 + h) - exact_elbo(-1.0, 1.5 - h)) / (2 * h),
    ])
    torch.manual_seed(0)
    _, grads = elbo(model, guide, n_samples=4000)
    actual = torch.stack([grads["x_m1"], grads["x_m2"]]).reshape(2)
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=0.1, rtol=0)
