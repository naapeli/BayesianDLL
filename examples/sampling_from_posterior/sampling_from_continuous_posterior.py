"""
Posterior of a Success Probability
==================================

Twenty coin flips ``y_i ~ Bernoulli(p)`` with a uniform ``Beta(1, 1)`` prior on
``p``. The prior is conjugate, so the posterior is exactly
``Beta(1 + successes, 1 + failures)``, which makes this a good check of the
sampler. NUTS moves on ``logit(p)``, where the posterior is unbounded and
close to Gaussian, and maps the draws back to ``(0, 1)``.

The figure compares the draws with the exact posterior and the prior, and
shows the posterior predictive distribution of the number of successes in
twenty new flips against the exact beta-binomial distribution.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Bernoulli, Beta
from BayesianDLL.Evaluation import summary


torch.manual_seed(0)

n_flips, p_true = 20, 0.3
flips = torch.bernoulli(torch.full((n_flips,), p_true, dtype=torch.float64))
a_prior, b_prior = 1.0, 1.0
successes = int(flips.sum())
print(f"{successes} successes in {n_flips} flips")

with Model() as model:
    p = RandomParameter("p", Beta(a_prior, b_prior))
    with plate("flips", n_flips):
        ObservedParameter("flips", Bernoulli(p), flips)

result = model.sample(2000, 500, n_chains=4, seed=0, delta=0.85, progress_bar=False, chain_method="vectorized")
print(summary(result).to_string())
a_post, b_post = a_prior + successes, b_prior + n_flips - successes
print(f"exact posterior mean {a_post / (a_post + b_post):.3f}")

predictive = model.posterior_predicative(result, n_samples=3000, samples_per_step=10, progress_bar=False)
new_successes = predictive["flips"].reshape(-1, n_flips).sum(1).long()


def log_beta_function(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


k = torch.arange(n_flips + 1, dtype=torch.float64)
log_binomial_coefficient = (
    torch.lgamma(torch.tensor(n_flips + 1.0, dtype=torch.float64))
    - torch.lgamma(k + 1)
    - torch.lgamma(n_flips - k + 1)
)
beta_binomial = torch.exp(
    log_binomial_coefficient
    + log_beta_function(k + a_post, n_flips - k + b_post)
    - log_beta_function(torch.tensor(float(a_post), dtype=torch.float64), torch.tensor(float(b_post), dtype=torch.float64))
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax = axes[0]
grid = torch.linspace(1e-3, 1 - 1e-3, 400, dtype=torch.float64)
ax.hist(result["p"].reshape(-1).numpy(), bins=50, density=True, alpha=0.5, label="NUTS draws")
ax.plot(grid, Beta(a_post, b_post).pdf(grid.unsqueeze(-1)).reshape(-1), color="black", label="exact posterior")
ax.plot(grid, Beta(a_prior, b_prior).pdf(grid.unsqueeze(-1)).reshape(-1), color="gray", linestyle=":", label="prior")
ax.axvline(p_true, color="C3", linestyle="--", label="true p")
ax.set_xlabel("p")
ax.set_ylabel("density")
ax.set_title("Posterior of the success probability")
ax.legend(frameon=False, fontsize="small")

ax = axes[1]
frequency = torch.bincount(new_successes, minlength=n_flips + 1).double() / len(new_successes)
ax.bar(k, frequency, alpha=0.5, label="posterior predictive draws")
ax.plot(k, beta_binomial, "ko", markersize=4, label="exact beta-binomial")
ax.axvline(successes, color="C3", linestyle="--", label="observed successes")
ax.set_xlabel(f"successes in {n_flips} new flips")
ax.set_ylabel("probability")
ax.set_title("Posterior predictive distribution")
ax.legend(frameon=False, fontsize="small")
plt.show()
