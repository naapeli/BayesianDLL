"""Fisher information of the distributions' parameters (``log_pdf_param_curvature``)."""

import pytest
import torch

from BayesianDLL.Distributions import (
    Bernoulli, Beta, Binomial, Categorical, Cauchy, Exponential, Gamma, HalfNormal, Laplace, LogNormal, Multinomial,
    MultivariateNormal, Normal, Poisson, StudentT,
)


def _fisher_by_scores(distribution, n=400000):
    """``{(a, b): E[score_a score_b^T]}`` from samples."""
    samples = distribution.sample(n)
    scores = distribution.log_pdf_param_grads(samples)
    flat = {key: value.reshape(n, -1) for key, value in scores.items()}
    return {(a, b): flat[a].T @ flat[b] / n for a in flat for b in flat}


_FISHER_CASES = {
    "normal": lambda: Normal(torch.tensor(0.3), torch.tensor(1.7)),
    "lognormal": lambda: LogNormal(torch.tensor(0.3), torch.tensor(0.6)),
    "laplace": lambda: Laplace(torch.tensor(0.3), torch.tensor(1.7)),
    "cauchy": lambda: Cauchy(torch.tensor(0.3), torch.tensor(1.7)),
    "student_t": lambda: StudentT(torch.tensor(4.5), torch.tensor(0.3), torch.tensor(1.7)),
    "exponential": lambda: Exponential(torch.tensor(1.7)),
    "gamma": lambda: Gamma(torch.tensor(2.5), torch.tensor(1.7)),
    "beta": lambda: Beta(torch.tensor(2.5), torch.tensor(1.7)),
    "poisson": lambda: Poisson(torch.tensor(3.2)),
    "bernoulli_probs": lambda: Bernoulli(probs=torch.tensor(0.3)),
    "bernoulli_logits": lambda: Bernoulli(logits=torch.tensor(0.4)),
    "binomial_logits": lambda: Binomial(torch.tensor(7), logits=torch.tensor(0.4)),
    "binomial_probs": lambda: Binomial(torch.tensor(7), probs=torch.tensor(0.35)),
    "categorical_logits": lambda: Categorical(logits=torch.tensor([0.2, -0.5, 1.0])),
    "categorical_probs": lambda: Categorical(probs=torch.tensor([0.2, 0.5, 0.3])),
    "multinomial_logits": lambda: Multinomial(5, logits=torch.tensor([0.2, -0.5, 1.0])),
    "mvn_mean": lambda: MultivariateNormal(torch.tensor([0.1, -0.2]), covariance=torch.tensor([[1.0, 0.4], [0.4, 2.0]])),
}


@pytest.mark.parametrize("case", sorted(_FISHER_CASES))
def test_parameter_curvature_is_the_fisher_information(case):
    distribution = _FISHER_CASES[case]()
    curvature = distribution.log_pdf_param_curvature(distribution.sample(1))
    reference = _fisher_by_scores(distribution)
    scale = max(float(value.abs().max()) for value in reference.values())
    for (a, b), block in curvature.items():
        expected = reference[(a, b)]
        torch.testing.assert_close(block.reshape(expected.shape), expected, atol=0.03 * scale, rtol=0.05)
    # Pairs that are left out are zero (the covariance of the multivariate normal is left to finite differences).
    for (a, b), expected in reference.items():
        if (a, b) not in curvature and (b, a) not in curvature and case != "mvn_mean":
            assert float(expected.abs().max()) < 0.03 * scale


def test_unsupported_distributions_raise():
    with pytest.raises(NotImplementedError):
        HalfNormal(1.0).log_pdf_param_curvature(torch.tensor(1.0))
