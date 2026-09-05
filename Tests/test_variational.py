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


def make_guide(mean_value=0.3, variance_value=1.2, requires_grad=False):
    mean = torch.tensor([mean_value], requires_grad=requires_grad)
    variance = torch.tensor([variance_value], requires_grad=requires_grad)
    with MeanFieldGuide() as guide:
        RandomParameter("mean", Normal(VariationalParameter("loc", mean), VariationalParameter("var", variance, min=0.01)))
    return guide, mean, variance


def test_elbo_reparameterization_gradients_match_autograd(normal_model):
    guide, mean, variance = make_guide(requires_grad=True)
    model_before = normal_model.params["mean"].unconstrained_value.clone()
    guide_before = guide.params["mean"].unconstrained_value.clone()
    value, grads = elbo(normal_model, guide, n_samples=10)
    expected = torch.autograd.grad(value, (mean, variance))
    assert value.ndim == 0
    for key, grad in zip(("mean_loc", "mean_var"), expected):
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
    variance = torch.tensor([1.2, 0.7, 1.5], requires_grad=True)
    with MeanFieldGuide() as guide:
        RandomParameter(
            "z",
            Normal(
                VariationalParameter("loc", mean),
                VariationalParameter("var", variance, min=0.01),
            ),
            shape=dimension,
        )

    value, grads = elbo(model, guide, n_samples=10)
    expected = torch.autograd.grad(value, (mean, variance))
    torch.testing.assert_close(grads["z_loc"], expected[0])
    torch.testing.assert_close(grads["z_var"], expected[1])


def test_elbo_is_zero_for_identical_model_and_guide():
    with Model() as model:
        RandomParameter("mean", Normal(torch.tensor([0.3]), torch.tensor([1.2])))
    guide, _, _ = make_guide()
    value, _ = elbo(model, guide, n_samples=8)
    assert value.item() == pytest.approx(0, abs=1e-12)


@pytest.mark.integration
def test_bbvi_approaches_analytic_normal_posterior(normal_model):
    guide, _, _ = make_guide(mean_value=-1.0, variance_value=1.0)
    BBVI(normal_model, guide, epochs=200, n_samples=12, lr=0.05, verbose=False)
    dist = guide.params["mean"].distribution
    assert dist.mu.value.item() == pytest.approx(1.6, abs=0.18)
    assert dist.variance.value.item() == pytest.approx(4 / 13, abs=0.15)
    assert dist.variance.value.item() > 0


def test_bbvi_reports_history(normal_model, capsys):
    guide, _, _ = make_guide()
    history = BBVI(normal_model, guide, epochs=4, n_samples=2, callback_frequency=2)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
    assert capsys.readouterr().out.count("ELBO:") == 2
