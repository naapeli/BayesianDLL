"""
NUTS With a Dense Mass Matrix
=============================

NUTS adapts a mass matrix (``metric``) during warmup. The default
``metric="diag"`` learns one scale per coordinate; ``metric="dense"`` learns
the full posterior covariance and therefore also removes linear correlations
between coordinates.

The test case is a straight-line fit ``y = a + b x`` with a covariate that is
away from zero (``x`` between 2 and 6) and a known noise level. Without
centring, the intercept and the slope are strongly anti-correlated a
posteriori (correlation about -0.96). With the conjugate Normal priors used
here the posterior is exactly Gaussian, so its density contours can be drawn
for reference.

A diagonal metric leaves a narrow diagonal ridge: NUTS needs a small step and
longer trajectories. The dense metric turns the ridge into a round target and
NUTS takes long steps with short trees. (Centring ``x``, as in the regression
examples, removes the correlation too; the dense metric helps when no such
reparameterisation is at hand.) The figures show the draws of both runs on the
exact contours and compare step size, tree depth and ESS per second.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Normal
from BayesianDLL.Evaluation import effective_sample_size


torch.manual_seed(3)

n_obs = 30
a_true, b_true, noise = 1.0, 0.5, 0.5
prior_sd = 10.0
x = torch.linspace(2.0, 6.0, n_obs, dtype=torch.float64)
y = a_true + b_true * x + noise * torch.randn(n_obs, dtype=torch.float64)
design = torch.stack([torch.ones_like(x), x], dim=1)

# Exact Gaussian posterior of (a, b).
precision = design.T @ design / noise ** 2 + torch.eye(2, dtype=torch.float64) / prior_sd ** 2
posterior_cov = torch.linalg.inv(precision)
posterior_mean = posterior_cov @ design.T @ y / noise ** 2
correlation = posterior_cov[0, 1] / (posterior_cov[0, 0] * posterior_cov[1, 1]).sqrt()
print(f"exact posterior correlation of intercept and slope: {correlation:.4f}")

with Model() as model:
    X = Data("design", design)
    coefficients = RandomParameter("coefficients", Normal(0.0, prior_sd), shape=2)
    mu = Linear("mu", X, coefficients=coefficients)
    with plate("observations", n_obs):
        ObservedParameter("y", Normal(mu, noise), y)

runs = {}
for metric in ("diag", "dense"):
    start = time.perf_counter()
    result = model.sample(1000, 500, n_chains=4, seed=1, metric=metric, delta=0.85, progress_bar=False,
                          chain_method="vectorized")
    seconds = time.perf_counter() - start
    ess = effective_sample_size({"c": result["coefficients"]})["c"].min().item()
    runs[metric] = {"result": result, "seconds": seconds, "ess": ess}
    stats = result.sample_stats
    print(f"{metric:5s}: step size {stats['step_size'].double().mean():.4f}, "
          f"mean tree depth {stats['tree_depth'].double().mean():.1f}, "
          f"leapfrog steps per draw {stats['n_leapfrog'].double().mean():.0f}, "
          f"min ESS {ess:.0f}, {seconds:.1f} s, ESS/s {ess / seconds:.0f}, "
          f"divergences {sum(result.divergences)}")

# %%
# Draws on the exact posterior contours
# -------------------------------------
sd = posterior_cov.diagonal().sqrt()
a_grid = torch.linspace(posterior_mean[0] - 4 * sd[0], posterior_mean[0] + 4 * sd[0], 200, dtype=torch.float64)
b_grid = torch.linspace(posterior_mean[1] - 4 * sd[1], posterior_mean[1] + 4 * sd[1], 200, dtype=torch.float64)
A, B = torch.meshgrid(a_grid, b_grid, indexing="ij")
offset = torch.stack([A, B], dim=-1) - posterior_mean
density = torch.exp(-0.5 * torch.einsum("...i,ij,...j->...", offset, precision, offset))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
for ax, metric in zip(axes[:2], ("diag", "dense")):
    draws = runs[metric]["result"]["coefficients"].reshape(-1, 2)
    ax.contour(A, B, density, levels=[0.011, 0.135, 0.607], colors="black", linewidths=1, zorder=3)
    ax.scatter(draws[:600, 0], draws[:600, 1], s=4, alpha=0.4, color="C0" if metric == "diag" else "C1")
    stats = runs[metric]["result"].sample_stats
    ax.set_title(f'metric="{metric}": step size {stats["step_size"].double().mean():.3f}, '
                 f'tree depth {stats["tree_depth"].double().mean():.1f}', fontsize="medium")
    ax.set_xlabel("intercept a")
    ax.set_ylabel("slope b")

ax = axes[2]
names = ["ESS per second", "leapfrog steps per draw"]
values = {m: [runs[m]["ess"] / runs[m]["seconds"],
              runs[m]["result"].sample_stats["n_leapfrog"].double().mean().item()] for m in runs}
positions = torch.arange(len(names), dtype=torch.float64)
ax.bar(positions - 0.2, values["diag"], width=0.4, label='metric="diag"')
ax.bar(positions + 0.2, values["dense"], width=0.4, label='metric="dense"')
ax.set_xticks(positions, names)
ax.set_yscale("log")
ax.set_title("Efficiency (log scale)")
ax.legend(frameon=False, fontsize="small")
fig.suptitle("Exact posterior contours (1, 2 and 3 standard deviations) with NUTS draws")
plt.show()
