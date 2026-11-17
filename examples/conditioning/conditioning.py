"""
Conditioning a Model
====================

``model.condition(...)`` (or ``condition(model, {...})``) returns a copy of a
model in which some nodes take fixed values. The original model is left
unchanged, so one model definition can be reused for several questions:

* **new data**: replacing the values of ``Data`` nodes fits the same model to
  another dataset;
* **what-if**: fixing a parameter, here the intercept to 0, gives the
  posterior of the remaining parameters under that assumption;
* **sensitivity**: fixing the noise level to a small and to a large value
  shows how much the slope estimate depends on it.

The model is a straight line with a ``Normal`` likelihood. The covariate is
centred at zero, which keeps the intercept and slope nearly uncorrelated, and
the noise has a weakly informative ``HalfNormal`` prior. The figures show the
fitted lines with 95% bands for every conditioned copy over its data, together
with the true lines, and the slope posteriors under the different assumptions.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, condition, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import summary


torch.manual_seed(42)

n_obs = 30
a_true, b_true, sigma_true = 1.5, 2.0, 0.5
x_obs = torch.linspace(-2.0, 2.0, n_obs, dtype=torch.float64)
y_obs = a_true + b_true * x_obs + sigma_true * torch.randn(n_obs, dtype=torch.float64)

with Model() as model:
    x = Data("x", x_obs)
    y = Data("y", y_obs)
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(2.0))
    mu = Linear("mu", x, slope=slope, intercept=intercept)
    with plate("observations", x):
        ObservedParameter("y_obs", Normal(mu, sigma), y)

# All chains run in one process (``chain_method="vectorized"``), which avoids
# starting worker processes for each of the five fits.
options = {"n_samples": 800, "warmup_length": 400, "n_chains": 4, "delta": 0.85, "chain_method": "vectorized",
           "progress_bar": False}


def fit(conditioned_model, label, seed):
    result = conditioned_model.sample(seed=seed, **options)
    names = [name for name in ("intercept", "slope", "sigma") if name in result.trace]
    print(f"\n{label} ({sum(result.divergences)} divergences)")
    print(summary({name: result[name] for name in names}).to_string())
    return result


# %%
# The model as defined
# --------------------
original = fit(model, "original data", seed=0)

# %%
# The same model on another dataset
# ---------------------------------
a_new, b_new = -0.75, -1.25
x_new = torch.linspace(-3.0, 3.0, 18, dtype=torch.float64)
y_new = a_new + b_new * x_new + sigma_true * torch.randn(18, dtype=torch.float64)
new_data = fit(model.condition(x=x_new, y=y_new), "conditioned on a new dataset", seed=1)

# %%
# What if the line went through the origin?
# -----------------------------------------
no_intercept = fit(model.condition(intercept=0.0), "intercept fixed to 0", seed=2)

# %%
# Sensitivity to the noise level
# ------------------------------
low_noise = fit(condition(model, {"sigma": 0.1}), "sigma fixed to 0.1", seed=3)
high_noise = fit(condition(model, {"sigma": 2.0}), "sigma fixed to 2", seed=4)


def line_band(result, x_grid, intercept_value=None):
    a = result["intercept"].reshape(-1, 1) if intercept_value is None else intercept_value
    lines = a + result["slope"].reshape(-1, 1) * x_grid
    return lines.mean(0), lines.quantile(0.025, 0), lines.quantile(0.975, 0)


fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
x_grid = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)

ax = axes[0, 0]
for result, xs, ys, a, b, color, label in (
    (original, x_obs, y_obs, a_true, b_true, "C0", "original data"),
    (new_data, x_new, y_new, a_new, b_new, "C4", "new data"),
):
    mean, low, high = line_band(result, x_grid)
    ax.scatter(xs, ys, s=18, color=color, alpha=0.7, label=label)
    ax.fill_between(x_grid, low, high, color=color, alpha=0.25, linewidth=0)
    ax.plot(x_grid, mean, color=color)
    ax.plot(x_grid, a + b * x_grid, color=color, linestyle="--", linewidth=1)
ax.set_title("Same model, two datasets (dashed: true lines)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=False, fontsize="small")

ax = axes[0, 1]
ax.scatter(x_obs, y_obs, s=18, color="gray", alpha=0.7, label="original data")
for result, intercept_value, color, label in ((original, None, "C0", "intercept inferred"),
                                              (no_intercept, 0.0, "C3", "intercept fixed to 0")):
    mean, low, high = line_band(result, x_grid, intercept_value)
    ax.fill_between(x_grid, low, high, color=color, alpha=0.25, linewidth=0)
    ax.plot(x_grid, mean, color=color, label=label)
ax.plot(x_grid, a_true + b_true * x_grid, color="black", linestyle="--", linewidth=1, label="true line")
ax.set_title("What-if: line forced through the origin")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=False, fontsize="small")

ax = axes[1, 0]
ax.scatter(x_obs, y_obs, s=18, color="gray", alpha=0.7, label="original data")
for result, color, label in ((low_noise, "C2", "sigma = 0.1"), (original, "C0", "sigma inferred"),
                             (high_noise, "C1", "sigma = 2")):
    mean, low, high = line_band(result, x_grid)
    ax.fill_between(x_grid, low, high, color=color, alpha=0.25, linewidth=0, label=f"95% band, {label}")
ax.set_title("Sensitivity: bands of the mean line")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=False, fontsize="small")

ax = axes[1, 1]
for result, color, label in ((low_noise, "C2", "sigma = 0.1"), (original, "C0", "sigma inferred"),
                             (high_noise, "C1", "sigma = 2"), (no_intercept, "C3", "intercept = 0")):
    ax.hist(result["slope"].reshape(-1).numpy(), bins=50, density=True, histtype="step", linewidth=1.5,
            color=color, label=label)
ax.axvline(b_true, color="black", linestyle="--", label="true slope")
ax.set_title("Posterior of the slope under each assumption")
ax.set_xlabel("slope")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize="small")
plt.show()
