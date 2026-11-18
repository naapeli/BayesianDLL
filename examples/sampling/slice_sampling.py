"""
Slice Sampling
==============

The slice sampler (Neal, 2003) updates one coordinate at a time. It draws a
height under the density at the current point, steps an interval out until
both ends lie below that height, and then shrinks the interval until a
uniform draw from it lands above the height. Every update is accepted, no
gradient is needed, and the only setting, the initial interval width, adapts
to the posterior scale during warmup. Like NUTS it works on the unconstrained
scale of each parameter.

Three cases are shown:

* the rate of exponential waiting times with a ``Gamma(2, 1)`` prior, whose
  posterior ``Gamma(2 + n, 1 + sum(y))`` is known exactly; the same posterior
  is also sampled with NUTS for comparison;
* a ``Beta(2, 5)`` distribution, sampled through its logit;
* a Normal model where NUTS updates the mean and the slice sampler the
  standard deviation, in one Gibbs scheme over the two blocks.

The figure overlays the draws on the exact densities, shows the fitted Normal
density over the data, and the joint posterior of the last model.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate, sample
from BayesianDLL.Distributions import Beta, Exponential, Gamma, HalfNormal, Normal


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
options = {"n_chains": 4, "chain_method": "vectorized", "progress_bar": False}


# %%
# A Gamma posterior: slice sampler and NUTS
# -----------------------------------------
waits = torch.tensor([0.5, 1.5, 0.3, 0.9, 2.1])


def rate_model(sampler):
    with Model() as model:
        rate = RandomParameter("rate", Gamma(2.0, 1.0), sampler=sampler)
        with plate("waits", len(waits)):
            ObservedParameter("waits", Exponential(rate), waits)
    return model


slice_rate = sample(1000, 500, model=rate_model("slice"), seed=0, **options)
nuts_rate = sample(1000, 500, model=rate_model("nuts"), seed=0, delta=0.85, **options)
shape, rate = 2.0 + len(waits), 1.0 + waits.sum().item()
print(f"posterior mean of the rate: slice {slice_rate['rate'].mean():.3f}, NUTS {nuts_rate['rate'].mean():.3f}, "
      f"exact {shape / rate:.3f}")
stats = slice_rate.sample_stats[0]
print(f"slice sampler: adapted width {stats['step_size'].mean():.2f}, "
      f"log density evaluations per draw {stats['n_evals'].double().mean():.1f}")

# %%
# A bounded distribution
# ----------------------
with Model() as beta_model:
    RandomParameter("p", Beta(2.0, 5.0), sampler="slice")
beta_draws = sample(1000, 500, model=beta_model, seed=1, **options)

# %%
# NUTS and slice in one model
# ---------------------------
mu_true, sigma_true = 1.0, 0.5
data = mu_true + sigma_true * torch.randn(50)
with Model() as normal_model:
    mu = RandomParameter("mu", Normal(0.0, 5.0), delta=0.85)
    sigma = RandomParameter("sigma", HalfNormal(2.0), sampler="slice")
    with plate("data", len(data)):
        ObservedParameter("data", Normal(mu, sigma), data)
mixed = sample(1000, 500, model=normal_model, seed=2, **options)
print("sampler per block:", list(zip(mixed.sample_stats.params, mixed.sample_stats.samplers)))
print(f"divergent transitions: {sum(mixed.divergences)}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")

ax = axes[0, 0]
grid = torch.linspace(1e-3, 4, 300)
ax.hist(slice_rate["rate"].reshape(-1).numpy(), bins=50, density=True, alpha=0.5, label="slice")
ax.hist(nuts_rate["rate"].reshape(-1).numpy(), bins=50, density=True, histtype="step", linewidth=1.5, label="NUTS")
ax.plot(grid, Gamma(shape, rate).pdf(grid.unsqueeze(-1)).reshape(-1), color="black", label="exact")
ax.set_title("Posterior of an exponential rate")
ax.set_xlabel("rate")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize="small")

ax = axes[0, 1]
grid = torch.linspace(1e-3, 1 - 1e-3, 300)
ax.hist(beta_draws["p"].reshape(-1).numpy(), bins=50, density=True, alpha=0.5, label="slice")
ax.plot(grid, Beta(2.0, 5.0).pdf(grid.unsqueeze(-1)).reshape(-1), color="black", label="exact")
ax.set_title("Beta(2, 5)")
ax.set_xlabel("p")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize="small")

ax = axes[1, 0]
grid = torch.linspace(data.min() - 1, data.max() + 1, 300)
mu_draws, sigma_draws = mixed["mu"].reshape(-1, 1), mixed["sigma"].reshape(-1, 1)
densities = torch.exp(-0.5 * ((grid - mu_draws) / sigma_draws) ** 2) / (sigma_draws * (2 * torch.pi) ** 0.5)
ax.hist(data.numpy(), bins=15, density=True, color="lightgray", label="data")
ax.fill_between(grid, densities.quantile(0.05, 0), densities.quantile(0.95, 0), color="C0", alpha=0.3,
                linewidth=0, label="90% band of the fitted density")
ax.plot(grid, densities.mean(0), color="C0", label="posterior mean density")
ax.plot(grid, Normal(mu_true, sigma_true).pdf(grid.unsqueeze(-1)).reshape(-1), "--", color="C3", label="true density")
ax.set_title("Normal model fitted with NUTS + slice")
ax.set_xlabel("y")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize="small")

ax = axes[1, 1]
ax.scatter(mixed["mu"].reshape(-1), mixed["sigma"].reshape(-1), s=3, alpha=0.3)
ax.plot(mu_true, sigma_true, "r*", markersize=12, label="true values")
ax.set_xlabel("mu (NUTS)")
ax.set_ylabel("sigma (slice)")
ax.set_title("Joint posterior")
ax.legend(frameon=False, fontsize="small")
plt.show()
