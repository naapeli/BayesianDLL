"""Low-rank plus diagonal guide groups and KL annealing in BBVI."""

import math

import pytest
import torch

from BayesianDLL import Group, Guide
from BayesianDLL.Variational import BBVI, elbo
from BayesianDLL.Variational import _ELBO

from test_guide_groups import _regression


def _set_random_low_rank(group, rank_scale=0.4):
    generator = torch.Generator().manual_seed(3)
    dim = group.layout.dim
    group.loc_parameter.set_new_value(torch.randn(dim, generator=generator) * 0.5 + 1.0)
    group.scale_parameter.set_new_value(0.2 + 0.3 * torch.rand(dim, generator=generator))
    group.cov_factor_parameter.set_new_value(rank_scale * torch.randn(dim, group.rank, generator=generator))


def test_low_rank_gradient_matches_autograd():
    """Reparameterised gradient of E[log p] plus the entropy 0.5 log det S, at fixed noise."""
    model, _, _ = _regression()
    guide = Guide(model, default_family="low_rank", rank=2)
    pieces = guide.validate(model)
    group = pieces[0]
    assert group.rank == 2
    _set_random_low_rank(group)
    table = guide.variational_parameters()
    assert set(table) == {"loc", "scale", "cov_factor"}
    for parameter in table.values():
        parameter.value.requires_grad_(True)
    torch.manual_seed(7)
    _, grads = _ELBO.estimate_elbo(model, pieces, 6)
    torch.manual_seed(7)
    group.draw(6)
    log_p = 0.0
    for i in range(6):
        values, jacobian = group.to_model(i)
        with model.temporarily_set_many(values):
            log_p = log_p + model._log_prob_of_terms(list(model.params.values()), list(model.observed_params.values())) + jacobian
    V = group.cov_factor
    entropy = 0.5 * torch.logdet(torch.diag(group.scale ** 2) + V @ V.T)
    expected = torch.autograd.grad(log_p / 6 + entropy, [table[key].value for key in grads])
    for key, reference in zip(grads, expected):
        torch.testing.assert_close(grads[key], reference, msg=key)


def test_low_rank_log_density_matches_dense_formula():
    model, _, _ = _regression()
    guide = Guide(model, default_family="low_rank", rank=2)
    guide.validate(model)
    group = guide.groups[0]
    _set_random_low_rank(group)
    covariance = group.moments("unconstrained").matrix
    V = group.cov_factor
    torch.testing.assert_close(covariance, torch.diag(group.scale ** 2) + V @ V.T)
    group.draw(5)
    reference = torch.distributions.MultivariateNormal(group.loc, covariance).log_prob(group._z)
    for i in range(5):
        torch.testing.assert_close(group.log_q(i), reference[i])


def test_low_rank_samples_have_the_stated_covariance():
    model, _, _ = _regression()
    guide = Guide(model, default_family="low_rank", rank=3)
    guide.validate(model)
    group = guide.groups[0]
    _set_random_low_rank(group)
    draws = group.sample_unconstrained(200000, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(draws.mean(0), group.loc, atol=0.02, rtol=0)
    torch.testing.assert_close(torch.cov(draws.T), group.moments("unconstrained").matrix, atol=0.03, rtol=0)


def test_low_rank_with_zero_factor_equals_mean_field():
    model, _, _ = _regression()
    low = Guide(model, default_family="low_rank", rank=1, init_scale=0.3)
    mean_field = Guide(model, default_family="mean_field", init_scale=0.3)
    low.validate(model)
    mean_field.validate(model)
    group = low.groups[0]
    group.cov_factor_parameter.set_new_value(torch.zeros_like(group.cov_factor))
    torch.manual_seed(5)
    value_low, grads_low = elbo(model, low, n_samples=4)
    torch.manual_seed(5)
    value_mf, grads_mf = elbo(model, mean_field, n_samples=4)
    torch.testing.assert_close(value_low, value_mf)
    names = group.names
    torch.testing.assert_close(grads_low["loc"], torch.cat([grads_mf[f"{n}_loc"].reshape(-1) for n in names]))
    torch.testing.assert_close(grads_low["scale"], torch.cat([grads_mf[f"{n}_scale"].reshape(-1) for n in names]))
    assert torch.count_nonzero(grads_low["cov_factor"]) > 0


def test_low_rank_recovers_a_correlated_gaussian_posterior():
    model, x, y = _regression(extra=False)
    design = torch.stack([torch.ones_like(x), x], dim=1)
    covariance = torch.linalg.inv(design.T @ design / 0.09 + torch.eye(2) / 9)
    mean = covariance @ design.T @ y / 0.09
    torch.manual_seed(0)
    guide = Guide(model, default_family="low_rank", rank=1)
    BBVI(model, guide, n_samples=4, epochs=1500, lr=0.03, lrd=0.999, verbose=False)
    moments = guide.moments()
    torch.testing.assert_close(moments.mean(), mean, atol=0.05, rtol=0)
    torch.testing.assert_close(moments.matrix, covariance, atol=0.2 * covariance.abs().max().item(), rtol=0)


def test_low_rank_group_options_and_validation():
    model, _, _ = _regression()
    with Guide(model, default_family="mean_field") as guide:
        Group(["a", "b"], family="low_rank", rank=1, name="ab")
    assert {"ab_loc", "ab_scale", "ab_cov_factor"} <= set(guide.variational_parameters())
    assert guide.moments().matrix.shape[0] == 4  # a, b, c, sigma
    assert "rank=1" in repr(guide.groups[0])
    with pytest.raises(ValueError, match="rank"):
        Guide(model, default_family="full_rank", rank=2)
    for bad in (0, 5, 1.5):
        broken = Guide(model, default_family="low_rank", rank=bad)
        with pytest.raises(ValueError, match="rank"):
            broken.validate(model)


def test_low_rank_minibatch_bbvi():
    model, _, _ = _regression(n=40)
    guide = Guide(model, default_family="low_rank", rank=2)
    history = BBVI(model, guide, n_samples=2, epochs=30, lr=0.02, batch_size=10, verbose=False)
    assert len(history) == 30 and all(math.isfinite(v) for v in history)


# ---------------------------------------------------------------------------
# KL annealing
def _fit(model, annealing, epochs=40, **kwargs):
    torch.manual_seed(2)
    guide = Guide(model, default_family="mean_field")
    history = BBVI(model, guide, n_samples=2, epochs=epochs, lr=0.02, verbose=False, kl_annealing=annealing, **kwargs)
    return history, guide.moments().mean()


@pytest.mark.parametrize("annealing", [0, lambda epoch: 1.0])
def test_unit_weight_equals_plain_bbvi(annealing):
    model, _, _ = _regression()
    history, mean = _fit(model, None)
    history_other, mean_other = _fit(model, annealing)
    assert history == history_other
    torch.testing.assert_close(mean, mean_other)


def test_annealed_gradient_is_the_weighted_objective_gradient():
    model, _, _ = _regression()
    guide = Guide(model, default_family="full_rank", init_scale=0.3)
    pieces = guide.validate(model)
    group = pieces[0]
    w = 0.3
    table = guide.variational_parameters()
    for parameter in table.values():
        parameter.value.requires_grad_(True)
    torch.manual_seed(4)
    _, grads = _ELBO.estimate_elbo(model, pieces, 4, kl_weight=w)
    # Reference by autograd for the same draws: E[log p(y|theta) + w (log p(theta) + jac - log q)].
    torch.manual_seed(4)
    group.draw(4)
    prior = list(model.params.values())
    likelihood = list(model.observed_params.values())
    total = 0.0
    for i in range(4):
        values, jacobian = group.to_model(i)
        with model.temporarily_set_many(values):
            log_prior = model._log_prob_of_terms(prior, [])
            log_lik = model._log_prob_of_terms([], likelihood)
        total = total + log_lik + w * (log_prior + jacobian - group.log_q(i))
    total = total / 4
    expected = torch.autograd.grad(total, [table[key].value for key in grads])
    for key, reference in zip(grads, expected):
        if key == "scale_tril":
            reference = torch.tril(reference)
        torch.testing.assert_close(grads[key], reference, msg=key)


def test_annealing_changes_the_path_but_reports_finite_elbo():
    model, _, _ = _regression()
    plain, mean_plain = _fit(model, None, epochs=10)
    annealed, mean_annealed = _fit(model, 10, epochs=10)
    assert annealed != plain
    assert not torch.allclose(mean_plain, mean_annealed)


def test_annealing_with_minibatches_and_validation():
    model, _, _ = _regression(n=40)
    history, _ = _fit(model, 10, epochs=20, batch_size=10)
    assert len(history) == 20 and all(math.isfinite(v) for v in history)
    for bad in (-1, 2.5, "x"):
        with pytest.raises(TypeError):
            _fit(model, bad, epochs=1)
    with pytest.raises(ValueError):
        _fit(model, lambda epoch: 1.5, epochs=1)


def test_annealing_with_low_rank_guide():
    model, _, _ = _regression()
    torch.manual_seed(0)
    guide = Guide(model, default_family="low_rank", rank=2)
    history = BBVI(model, guide, n_samples=2, epochs=30, lr=0.02, verbose=False, kl_annealing=15)
    assert all(math.isfinite(v) for v in history)
