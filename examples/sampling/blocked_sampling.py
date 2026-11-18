"""
Blocked Sampling With Several Samplers
======================================

A regression line is fitted to data containing a few gross outliers. Each
observation has a discrete indicator ``z_i``: ``z_i = 1`` marks an outlier,
whose noise is ten times larger. The model has four kinds of unknowns, and
each one is updated by a sampler that suits it:

* the intercept and slope, continuous and correlated, jointly with NUTS;
* the noise scale ``sigma`` with the slice sampler;
* the outlier fraction ``w`` with random-walk Metropolis;
* the 18 indicators ``z`` with the exact Gibbs sampler (finite support).

The samplers are chosen per parameter with ``sampler=`` and grouped into
blocks with ``sample(..., blocks=[...])``. Every iteration updates the blocks
in turn, each conditional on the current values of the others (a Gibbs scheme
over blocks). ``result.sample_stats`` reports which sampler ran each block.
The same model without ``sampler=`` arguments and without ``blocks`` is
sampled for comparison: NUTS then takes all continuous parameters in one block
and Gibbs takes ``z``. Both schemes must give the same posterior.

The covariate is standardised and the per-observation noise scale is built
from deterministic blocks (``Affine`` and ``Multiply``). The figures show the
fitted line with each point coloured by its posterior outlier probability, and
the marginal posteriors from both schemes overlaid.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Affine, Linear, Multiply
from BayesianDLL.Distributions import Bernoulli, Beta, HalfNormal, Normal


torch.manual_seed(11)

n_obs = 18
a_true, b_true, sigma_true = 1.0, 2.0, 0.5
x = torch.linspace(-2.0, 3.0, n_obs, dtype=torch.float64)
y = a_true + b_true * x + sigma_true * torch.randn(n_obs, dtype=torch.float64)
outliers = torch.tensor([12, 15, 17])
y[outliers] -= torch.tensor([6.0, 7.0, 6.0], dtype=torch.float64)
x_mean, x_sd = x.mean(), x.std()
outlier_inflation = 10.0


def robust_model(samplers):
    """``samplers`` maps parameter names to a requested sampler ("auto" if missing)."""
    def choice(name):
        return samplers.get(name, "auto")

    with Model() as model:
        z_cov = Data("x_std", (x - x_mean) / x_sd)
        # Sampler settings given to a parameter apply to its block only; a
        # setting passed to sample() reaches every block whose sampler knows it
        # (``delta`` is NUTS-only; Metropolis has ``target_accept_rate``).
        alpha = RandomParameter("alpha", Normal(0.0, 5.0), sampler=choice("alpha"), delta=0.85)
        beta = RandomParameter("beta", Normal(0.0, 5.0), sampler=choice("beta"), delta=0.85)
        sigma = RandomParameter("sigma", HalfNormal(2.0), sampler=choice("sigma"))
        # A one-dimensional random walk mixes best at about 44% acceptance.
        w_settings = {"target_accept_rate": 0.44} if choice("w") == "metropolis" else {}
        w = RandomParameter("w", Beta(1.0, 9.0), sampler=choice("w"), **w_settings)
        mu = Linear("mu", z_cov, coefficients=beta, intercept=alpha)
        with plate("observations", n_obs):
            z = RandomParameter("z", Bernoulli(w), shape=(), sampler=choice("z"))
            factor = Affine("factor", z, loc=1.0, scale=outlier_inflation - 1.0)
            scale = Multiply("scale", factor, sigma)
            ObservedParameter("y", Normal(mu, scale), y)
    return model


# %%
# Hand-picked samplers and blocks
# -------------------------------
custom_model = robust_model({"sigma": "slice", "w": "metropolis", "z": "gibbs"})
custom = custom_model.sample(600, 300, n_chains=4, seed=5, chain_method="vectorized", progress_bar=False,
                             blocks=[["alpha", "beta"], "sigma", "w", "z"])

# %%
# Automatic choice
# ----------------
automatic = robust_model({}).sample(600, 300, n_chains=4, seed=6, chain_method="vectorized", progress_bar=False)

for label, result in (("hand-picked", custom), ("automatic", automatic)):
    print(f"\n{label} blocks:")
    for params, sampler, stats in zip(result.sample_stats.params, result.sample_stats.samplers, result.sample_stats):
        accept = stats["accept_stat"].double().mean().item()
        print(f"  {sampler:10s} {str(params):40s} mean acceptance statistic {accept:.2f}")
    print(f"  divergent transitions: {sum(result.divergences)}")

# Slope and intercept per unit of the original covariate.
for label, result in (("hand-picked", custom), ("automatic", automatic)):
    slope = result["beta"] / x_sd
    print(f"{label:12s}: slope {slope.mean():.2f} +- {slope.std():.2f}, sigma {result['sigma'].mean():.2f}, "
          f"w {result['w'].mean():.3f}")
print(f"true values : slope {b_true}, sigma {sigma_true}, outlier fraction {len(outliers) / n_obs:.3f}")

# %%
# Figures
# -------
outlier_probability = custom["z"].double().reshape(-1, n_obs).mean(0)
x_grid = torch.linspace(x.min(), x.max(), 100, dtype=torch.float64)
lines = (custom["alpha"].reshape(-1, 1)
         + custom["beta"].reshape(-1, 1) * ((x_grid - x_mean) / x_sd))
design = torch.stack([torch.ones_like(x), x], dim=1)
least_squares = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze()

fig = plt.figure(figsize=(13, 8), layout="constrained")
grid = fig.add_gridspec(2, 3)
ax = fig.add_subplot(grid[0, :])
ax.fill_between(x_grid, lines.quantile(0.025, 0), lines.quantile(0.975, 0), color="C0", alpha=0.3, linewidth=0,
                label="95% band of the robust fit")
ax.plot(x_grid, lines.mean(0), color="C0", label="robust fit (posterior mean)")
ax.plot(x_grid, a_true + b_true * x_grid, color="black", linestyle="--", label="true line")
ax.plot(x_grid, least_squares[0] + least_squares[1] * x_grid, color="C3", linestyle=":", label="least squares")
points = ax.scatter(x, y, c=outlier_probability, cmap="coolwarm", vmin=0, vmax=1, edgecolors="black", zorder=3)
fig.colorbar(points, ax=ax, label="P(outlier | data)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Robust regression with discrete outlier indicators")
ax.legend(frameon=False, loc="upper left")

for k, (name, label) in enumerate((("beta", "standardised slope"), ("sigma", "noise scale sigma"),
                                   ("w", "outlier fraction w"))):
    ax = fig.add_subplot(grid[1, k])
    for result, title in ((custom, "NUTS + slice + Metropolis + Gibbs"), (automatic, "automatic (NUTS + Gibbs)")):
        ax.hist(result[name].reshape(-1).numpy(), bins=40, density=True, histtype="step", linewidth=1.5, label=title)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior of {label}", fontsize="medium")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
plt.show()
