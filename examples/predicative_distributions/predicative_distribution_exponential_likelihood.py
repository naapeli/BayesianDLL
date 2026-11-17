"""
Posterior Predictive Check: Exponential Waiting Times
=====================================================

A thousand waiting times are modelled as ``Exponential(rate)``. The rate is
positive, so the model places the prior on its logarithm,
``log_rate ~ Normal(0, 1.5)``, and maps it back with the deterministic block
``Exp``. On the log scale the posterior is close to Gaussian, which suits
NUTS, and the prior covers rates from about 0.05 to 20.

After sampling, ``posterior_predicative`` draws replicated datasets: for each
selected posterior draw the parameters are fixed and new waiting times are
drawn exactly from the likelihood. If the model fits, the replicated data
look like the observed data. The figures compare their densities and
distribution functions with the data, show the posterior of the rate against
the exact posterior (computed on a grid), and check the largest waiting time
with a posterior predictive p-value.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Exponential, Normal
from BayesianDLL.Evaluation import Graphics, ppc_pvalue, summary


torch.manual_seed(0)

n_obs = 1000
rate_true = 3.0
waits = torch.distributions.Exponential(rate_true).sample((n_obs,)).double()

with Model() as model:
    log_rate = RandomParameter("log_rate", Normal(0.0, 1.5))
    rate = Exp("rate", log_rate)
    with plate("waits", n_obs):
        ObservedParameter("waits", Exponential(rate), waits)

result = model.sample(1000, 500, n_chains=4, seed=0, delta=0.85, progress_bar=False)
print(summary(result, include_deterministic=True).to_string())
print(f"true rate {rate_true}, divergent transitions {sum(result.divergences)}")

predictive = model.posterior_predicative(result, n_samples=40, samples_per_step=5, progress_bar=False)
check = ppc_pvalue(predictive, waits, "max")
print(f"posterior predictive p-value of the largest waiting time: {check.p_value:.2f}")

# Exact posterior of the rate on a grid (prior on log_rate, change of variables to rate).
rate_grid = torch.linspace(2.4, 3.6, 400, dtype=torch.float64)
log_density = (torch.distributions.Normal(0.0, 1.5).log_prob(rate_grid.log()) - rate_grid.log()
               + n_obs * rate_grid.log() - rate_grid * waits.sum())
exact = torch.softmax(log_density, 0) / (rate_grid[1] - rate_grid[0])

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
Graphics.plot_predicative_distribution(predictive, waits, kind="pdf", ax=axes[0, 0])
axes[0, 0].set_title("Replicated densities vs data")
axes[0, 0].set_xlabel("waiting time")
Graphics.plot_predicative_distribution(predictive, waits, kind="cdf", ax=axes[0, 1])
axes[0, 1].set_title("Replicated distribution functions vs data")
axes[0, 1].set_xlabel("waiting time")
axes[0, 0].legend(loc="upper right", fontsize="small", frameon=False)
axes[0, 1].legend(loc="lower right", fontsize="small", frameon=False)

ax = axes[1, 0]
ax.hist(result["rate"].reshape(-1).numpy(), bins=50, density=True, alpha=0.5, label="NUTS draws")
ax.plot(rate_grid, exact, color="black", label="exact posterior")
ax.axvline(rate_true, color="C3", linestyle="--", label="true rate")
ax.set_xlabel("rate")
ax.set_ylabel("density")
ax.set_title("Posterior of the rate")
ax.legend(frameon=False, fontsize="small")

Graphics.plot_ppc_statistic(check, ax=axes[1, 1])
axes[1, 1].set_title(f"Largest waiting time: p = {check.p_value:.2f}")
axes[1, 1].set_xlabel("max of a replicated dataset")
plt.show()
