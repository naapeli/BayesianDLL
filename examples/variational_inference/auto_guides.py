"""
Automatic Guides
================

Fits a Bayesian linear regression with black-box variational inference using the
automatic guides ``AutoNormal`` (a mean-field Normal over every latent's
unconstrained value) and ``AutoMultivariateNormal`` (one full-rank Normal over all
unconstrained latents). Neither guide needs to be written by hand.

The full-rank guide captures the strong posterior correlation between the
intercept and the slope, which the mean-field guide cannot represent. Both guides
are sampled directly (no MCMC) with ``guide.sample``, which returns a trace in the
constrained space, including deterministic nodes, so ``summary`` and the posterior
plots work as for MCMC results.
"""

import torch
import matplotlib.pyplot as plt

from BayesianDLL import AutoMultivariateNormal, AutoNormal, Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation.Graphics import plot_posterior
from BayesianDLL.Variational import BBVI


torch.manual_seed(1)

N = 40
x = torch.linspace(2.0, 4.0, N, dtype=torch.float64)  # far from zero: intercept and slope correlate
y = 1.0 + 0.8 * x + 0.4 * torch.randn(N, dtype=torch.float64)

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    x_data = Data("x", x)
    mu = Linear("mu", x_data, slope=slope, intercept=intercept)
    with plate("data", x_data):
        ObservedParameter("y", Normal(mu, sigma), y)

# %%
# Mean-field and full-rank guides
# -------------------------------
mean_field = AutoNormal(model)
history_mean_field = BBVI(model, mean_field, n_samples=4, epochs=1000, lr=0.03, verbose=False)

full_rank = AutoMultivariateNormal(model)
history_full_rank = BBVI(model, full_rank, n_samples=4, epochs=1000, lr=0.03, verbose=False)

covariance = full_rank.covariance
correlation = covariance[0, 1] / (covariance[0, 0] * covariance[1, 1]).sqrt()
print(f"Posterior correlation of intercept and slope (full rank): {correlation.item():.3f}")

posterior_mean_field = mean_field.sample(1000)
posterior_full_rank = full_rank.sample(1000)
print(posterior_full_rank.summary())

# %%
# ELBO histories and joint posteriors
# -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
axes[0].plot(history_mean_field, label="AutoNormal")
axes[0].plot(history_full_rank, label="AutoMultivariateNormal")
axes[0].set_ylim(min(history_full_rank[200:]) - 10, max(history_full_rank) + 5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("ELBO")
axes[0].legend()

for trace, label in ((posterior_mean_field, "AutoNormal"), (posterior_full_rank, "AutoMultivariateNormal")):
    axes[1].scatter(trace["intercept"].flatten(), trace["slope"].flatten(), s=3, alpha=0.4, label=label)
axes[1].set_xlabel("intercept")
axes[1].set_ylabel("slope")
axes[1].legend()

plot_posterior(posterior_full_rank, parameters=["intercept", "slope", "sigma"])
plt.show()
