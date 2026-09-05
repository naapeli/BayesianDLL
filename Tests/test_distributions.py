"""Density oracles and analytic gradients for every exported distribution."""

import pytest
import torch

from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Dirichlet, DiscreteUniform, Exponential,
    Geometric, HalfCauchy, InvGamma, Mixture, MultivariateNormal, Normal, Uniform,
)


def continuous_case(name):
    t = torch.tensor
    cases = {
        "normal": lambda: (Normal(0.4, 2.0), torch.distributions.Normal(0.4, 2.0**0.5), t([-1.0, 0.3, 2.0])),
        "multivariate_normal": lambda: (
            MultivariateNormal(t([0.2, -0.3]), t([[2.0, 0.4], [0.4, 1.0]])),
            torch.distributions.MultivariateNormal(t([0.2, -0.3]), t([[2.0, 0.4], [0.4, 1.0]])),
            t([[0.1, 0.5], [1.2, -0.4]])),
        "beta": lambda: (Beta(2.0, 3.0), torch.distributions.Beta(2.0, 3.0), t([0.1, 0.4, 0.8])),
        "exponential": lambda: (Exponential(1.3), torch.distributions.Exponential(1.3), t([0.1, 0.8, 2.0])),
        "uniform": lambda: (Uniform(-2.0, 3.0), torch.distributions.Uniform(-2.0, 3.0), t([-1.0, 0.2, 2.0])),
        "inverse_gamma": lambda: (InvGamma(2.0, 3.0), torch.distributions.InverseGamma(2.0, 3.0), t([0.3, 1.2, 3.0])),
        "half_cauchy": lambda: (HalfCauchy(1.7), torch.distributions.HalfCauchy(1.7), t([0.1, 1.0, 3.0])),
        "dirichlet": lambda: (Dirichlet(t([2.0, 3.0, 4.0])), torch.distributions.Dirichlet(t([2.0, 3.0, 4.0])), t([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])),
        "mixture": lambda: (
            Mixture([Normal(-1.0, 0.5), Normal(2.0, 2.0)], t([0.3, 0.7])),
            torch.distributions.MixtureSameFamily(torch.distributions.Categorical(t([0.3, 0.7])), torch.distributions.Normal(t([-1.0, 2.0]), t([0.5, 2.0]).sqrt())),
            t([-2.0, 0.5, 3.0])),
    }
    return cases[name]()


CONTINUOUS = ["normal", "multivariate_normal", "beta", "exponential", "uniform", "inverse_gamma", "half_cauchy", "dirichlet", "mixture"]


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
def test_unconstrained_gradients_match_autograd(name):
    dist, _, x = continuous_case(name)
    z = dist.transform.forward(x).detach().requires_grad_()
    logp = dist._log_prob_unconstrained(z)
    assert logp.ndim == 0
    expected, = torch.autograd.grad(logp, z)
    torch.testing.assert_close(dist._log_prob_grad_unconstrained(z), expected)


@pytest.mark.parametrize("factory,keys,values,x", [
    (Normal, ["mean", "variance"], [0.3, 1.7], [-0.5, 0.2, 1.5]),
    (Beta, ["a", "b"], [2.0, 3.0], [0.2, 0.4, 0.7]),
    (Exponential, ["rate"], [1.4], [0.2, 1.0, 3.0]),
    (InvGamma, ["alpha", "beta"], [2.0, 3.0], [0.2, 1.0, 3.0]),
    (HalfCauchy, ["scale"], [1.4], [0.2, 1.0, 3.0]),
    (Bernoulli, ["p"], [0.4], [0.0, 1.0, 1.0]),
    (Geometric, ["p"], [0.4], [1.0, 2.0, 4.0]),
    (lambda p: Binomial(5, p), ["p"], [0.4], [0.0, 2.0, 5.0]),
])
def test_parameter_gradients(factory, keys, values, x):
    params = [torch.tensor(v, requires_grad=True) for v in values]
    dist = factory(*params)
    x = torch.tensor(x)
    expected = torch.autograd.grad(dist.log_pdf(x).sum(), params)
    actual = dist.log_pdf_param_grads(x)
    for key, grad in zip(keys, expected):
        torch.testing.assert_close(actual[key].sum(), grad)


def test_dirichlet_parameter_gradient():
    alpha = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    dist = Dirichlet(alpha)
    x = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    expected, = torch.autograd.grad(dist.log_pdf(x).sum(), alpha)
    torch.testing.assert_close(dist.log_pdf_param_grads(x)["alpha"].sum(0), expected)


@pytest.mark.parametrize("name", ["bernoulli", "binomial", "geometric", "discrete_uniform"])
def test_discrete_probabilities(name):
    if name == "bernoulli":
        dist, ref, x = Bernoulli(0.3), torch.distributions.Bernoulli(0.3), torch.arange(2.0)
        expected = ref.log_prob(x)
    elif name == "binomial":
        dist, ref, x = Binomial(5, 0.3), torch.distributions.Binomial(5, 0.3), torch.arange(6.0)
        expected = ref.log_prob(x)
    elif name == "geometric":
        dist, x = Geometric(0.3), torch.arange(1.0, 20.0)
        # BayesianDLL counts trials; PyTorch counts failures before the success.
        expected = torch.distributions.Geometric(0.3).log_prob(x - 1)
    else:
        dist, x = DiscreteUniform(2, 5), torch.arange(2.0, 6.0)
        expected = torch.full_like(x, -torch.log(torch.tensor(4.0)))
    torch.testing.assert_close(dist.log_pdf(x), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dist.pdf(x), expected.exp(), atol=1e-6, rtol=1e-6)
    if name != "geometric":
        assert dist.pdf(x).sum().item() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(NotImplementedError, match="discrete"):
        dist.log_pdf_grad(x)


def test_uniform_support_and_unsupported_gradients():
    dist = Uniform(-2.0, 3.0)
    outside = torch.tensor([-2.1, 3.1])
    assert torch.isneginf(dist.log_pdf(outside)).all()
    assert (dist.pdf(outside) == 0).all()
    torch.testing.assert_close(dist.log_pdf_grad(torch.tensor([0.0])), torch.zeros(1))
    for uniform in (dist, DiscreteUniform(0, 3)):
        with pytest.raises(RuntimeError, match="not differentiable"):
            uniform.log_pdf_param_grads(torch.tensor([1.0]))
    with pytest.raises(NotImplementedError):
        MultivariateNormal(torch.zeros(2), torch.eye(2)).log_pdf_param_grads(torch.zeros(2))


@pytest.mark.parametrize("factory,key,value", [(lambda p: Normal(p, 2.0), "mean", 0.5), (Exponential, "rate", 1.5)])
def test_reparameterized_samples(factory, key, value):
    p = torch.tensor([value], requires_grad=True)
    samples, grads = factory(p).sample(12, _reparametrization_trick_grad=True)
    assert samples.shape == (12, 1)
    expected, = torch.autograd.grad(samples.sum(), p)
    torch.testing.assert_close(grads[key].sum(0), expected)


def test_normal_and_exponential_sample_moments():
    normal = Normal(torch.tensor([2.0]), 4.0).sample(20000)
    exponential = Exponential(torch.tensor([2.0])).sample(20000)
    assert normal.mean().item() == pytest.approx(2.0, abs=0.06)
    assert normal.var().item() == pytest.approx(4.0, abs=0.16)
    assert exponential.mean().item() == pytest.approx(0.5, abs=0.02)
    assert (exponential > 0).all()


def test_distribution_batch_and_event_shapes():
    assert Normal(torch.zeros(3, 1), torch.ones(4)).batch_shape == (3, 4)
    for dist in (Dirichlet(torch.ones(2, 3)), MultivariateNormal(torch.zeros(2, 3), torch.eye(3))):
        assert dist.event_shape == (3,)
        assert dist.batch_shape == (2,)
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
