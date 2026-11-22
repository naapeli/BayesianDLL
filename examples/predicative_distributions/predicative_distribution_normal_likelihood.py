"""
Prior and Posterior Predictive: Normal Data
===========================================

Normal measurements are modelled with an unknown mean and standard deviation,
``y ~ Normal(mean, sigma)`` with ``mean ~ Normal(0, 10)`` and
``sigma ~ HalfNormal(5)``.

The **prior predictive** distribution shows what data the model expects
before seeing any: ``prior_predicative`` draws parameters from their
priors and then data from the likelihood, by ancestral sampling and without
MCMC. Weakly informative priors should give replicated data on a plausible
scale that comfortably covers the observations; the prior draws themselves
are returned in ``predictive.prior``.

The **posterior predictive** distribution uses posterior draws instead (from
NUTS here), and replicated data should then resemble the observations
closely. The figure puts both next to each other, as densities and as
distribution functions, against the data.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import Graphics, summary


torch.manual_seed(0)

n_obs = 1000
mean_true, sigma_true = 5.0, 3 ** 0.5
y = mean_true + sigma_true * torch.randn(n_obs, dtype=torch.float64)
print(f"data: mean {y.mean():.2f}, standard deviation {y.std():.2f}")

with Model() as model:
    mean = RandomParameter("mean", Normal(0.0, 10.0))
    sigma = RandomParameter("sigma", HalfNormal(5.0))
    with plate("observations", n_obs):
        ObservedParameter("y", Normal(mean, sigma), y)

prior_predictive = model.prior_predicative(30, samples_per_step=5, progress_bar=False)
print("\nprior draws of the parameters")
print(summary(prior_predictive.prior).to_string())

result = model.sample(1000, 500, n_chains=4, seed=0, delta=0.85, progress_bar=False)
print("\nposterior")
print(summary(result).to_string())
posterior_predictive = model.posterior_predicative(result, n_samples=30, samples_per_step=5, progress_bar=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for row, (label, predictive) in enumerate((("Prior", prior_predictive), ("Posterior", posterior_predictive))):
    for column, kind in enumerate(("pdf", "cdf")):
        ax = axes[row, column]
        Graphics.plot_predicative_distribution(predictive, y, kind=kind, ax=ax)
        ax.set_title(f"{label} predictive {'density' if kind == 'pdf' else 'distribution function'} vs data")
        ax.set_xlabel("y")
        ax.legend(loc="upper left" if kind == "pdf" else "lower right", fontsize="small", frameon=False)
plt.show()
