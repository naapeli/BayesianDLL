import pytest
import torch

from BayesianDLL import MeanFieldGuide, Model, RandomParameter, VariationalParameter, find_MAP
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
    history = normal_model.find_MAP(epochs=5, callback_frequency=2)
    assert len(history) == 3
    assert capsys.readouterr().out.count("Epoch:") == 3
    assert history[-1] > history[0]


def make_guide(mean_value=0.3, std_value=1.1, requires_grad=False):
    mean = torch.tensor([mean_value], requires_grad=requires_grad)
    std = torch.tensor([std_value], requires_grad=requires_grad)
    with MeanFieldGuide() as guide:
        RandomParameter("mean", Normal(VariationalParameter("loc", mean), VariationalParameter("std", std, min=0.1)))
    return guide, mean, std


def test_elbo_reparameterization_gradients_match_autograd(normal_model):
    guide, mean, std = make_guide(requires_grad=True)
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
    with MeanFieldGuide() as guide:
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
    guide, _, _ = make_guide()
    value, _ = elbo(model, guide, n_samples=8)
    assert value.item() == pytest.approx(0, abs=1e-12)


@pytest.mark.integration
def test_bbvi_approaches_analytic_normal_posterior(normal_model):
    guide, _, _ = make_guide(mean_value=-1.0, std_value=1.0)
    BBVI(normal_model, guide, epochs=200, n_samples=12, lr=0.05, verbose=False)
    dist = guide.params["mean"].distribution
    assert dist.mu.value.item() == pytest.approx(1.6, abs=0.18)
    assert dist.sigma.value.item() ** 2 == pytest.approx(4 / 13, abs=0.15)
    assert dist.sigma.value.item() > 0


def test_bbvi_reports_history(normal_model, capsys):
    guide, _, _ = make_guide()
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
    with MeanFieldGuide() as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", mean), VariationalParameter("s", std)))
    value, grads = elbo(model, guide, n_samples=7)
    expected = torch.autograd.grad(value, (mean, std))
    torch.testing.assert_close(grads["x_m"], expected[0])
    torch.testing.assert_close(grads["x_s"], expected[1])


def test_lognormal_guide_elbo_is_zero_for_the_exact_posterior():
    with Model() as model:
        RandomParameter("x", LogNormal(0.3, 0.7))
    with MeanFieldGuide() as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", torch.tensor([0.3])), VariationalParameter("s", torch.tensor([0.7]))))
    value, _ = elbo(model, guide, n_samples=5)
    assert value.item() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.integration
def test_bbvi_lognormal_guide_reaches_known_optimum():
    """For p = Exp(1) the best LogNormal(m, s) has s = 1 and m = -1/2."""
    model = _exponential_model()
    with MeanFieldGuide() as guide:
        RandomParameter("x", LogNormal(VariationalParameter("m", torch.tensor([1.0])), VariationalParameter("s", torch.tensor([0.4]), log_scale=True)))
    BBVI(model, guide, n_samples=16, epochs=1500, lr=0.02, verbose=False)
    distribution = guide.params["x"].distribution
    assert distribution.mu.value.item() == pytest.approx(-0.5, abs=0.12)
    assert distribution.sigma.value.item() == pytest.approx(1.0, abs=0.12)


def test_deterministic_guide_gradient_matches_autograd():
    model = _exponential_model(counts=torch.tensor([1.0, 2.0]))
    mean = torch.tensor([0.1], requires_grad=True)
    std = torch.tensor([0.5], requires_grad=True)
    with MeanFieldGuide() as guide:
        u = RandomParameter("u", Normal(VariationalParameter("m", mean), VariationalParameter("s", std)))
        # x = exp(2 u): the Jacobian terms do not cancel.
        DeterministicParameter("x", lambda u: torch.exp(2 * u), lambda u: {"u": 2 * torch.exp(2 * u)}, [u])
    value, grads = elbo(model, guide, n_samples=6)
    expected = torch.autograd.grad(value, (mean, std))
    torch.testing.assert_close(grads["u_m"], expected[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(grads["u_s"], expected[1], rtol=1e-6, atol=1e-7)


def test_exp_deterministic_guide_equals_normal_guide_on_log_scale():
    model = _exponential_model(counts=torch.tensor([1.0, 2.0]))
    with MeanFieldGuide() as direct:
        RandomParameter("x", Normal(VariationalParameter("m", torch.tensor([0.2])), VariationalParameter("s", torch.tensor([0.3]))))
    with MeanFieldGuide() as indirect:
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
    with MeanFieldGuide() as guide:
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
    with MeanFieldGuide() as guide:
        RandomParameter("x", Beta(VariationalParameter("a", torch.tensor([2.0])), VariationalParameter("b", torch.tensor([2.0]))))
    with pytest.raises(ValueError, match="support"):
        elbo(model, guide)


def test_guide_must_cover_every_model_latent(normal_model):
    with MeanFieldGuide() as guide:
        RandomParameter("other", Normal(VariationalParameter("loc", torch.zeros(1)), VariationalParameter("scale", torch.ones(1))))
    with pytest.raises(ValueError, match="no distribution"):
        elbo(normal_model, guide)
