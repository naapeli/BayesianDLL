r"""
Ordinal Regression
==================

A satisfaction score with four ordered levels (0 to 3) depends on two
covariates. The levels have no numeric distance, so we model a latent logistic
variable that falls between increasing cutpoints
:math:`c_1 < c_2 < c_3`:

.. math::

    P(y = k) = \sigma(c_{k+1} - \eta) - \sigma(c_k - \eta), \qquad
    \eta = \beta_1 x_1 + \beta_2 x_2,

with :math:`c_0 = -\infty` and :math:`c_4 = \infty`. ``OrderedLogistic(eta,
cuts)`` is this likelihood. The cutpoints get an ``Ordered`` prior, which
samples them on an unconstrained scale and keeps them sorted. There is no
intercept: it would only shift the cutpoints.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Normal, Ordered, OrderedLogistic

torch.manual_seed(3)

n_obs = 600
beta_true = torch.tensor([1.1, -0.8], dtype=torch.float64)
cuts_true = torch.tensor([-1.2, 0.4, 1.8], dtype=torch.float64)
x1 = torch.randn(n_obs, dtype=torch.float64)
x2 = torch.bernoulli(torch.full((n_obs,), 0.5, dtype=torch.float64))
X = torch.stack([x1, x2], dim=1)
y = OrderedLogistic(X @ beta_true, cuts_true).sample(1)[0]
print("category counts:", torch.bincount(y.long()).tolist())

# %%
# The model
# ---------
# Weak ``Normal`` priors go on the coefficients, and the cutpoints are the
# sorted values of three draws from a wide ``Normal``.

with Model() as model:
    inputs = Data("inputs", X)
    beta = RandomParameter("beta", Normal(0.0, 2.0), shape=2)
    cuts = RandomParameter("cuts", Ordered(Normal(0.0, 3.0), 3), shape=3)
    eta = Linear("eta", inputs, coefficients=beta)
    with plate("observations", inputs):
        ObservedParameter("y", OrderedLogistic(eta, cuts), y)

start = time.perf_counter()
result = model.sample(400, 300, n_chains=4, seed=1, progress_bar=False, chain_method="vectorized")
elapsed = time.perf_counter() - start
summary = result.summary()
print(summary.to_string())
print(f"true beta = {beta_true.tolist()}, true cutpoints = {cuts_true.tolist()}")
print(f"sampling: {elapsed:.1f} s, divergences: {sum(result.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

# %%
# Recovered cutpoints and effects
# -------------------------------
# The posterior intervals cover the true values. The coefficients are log odds
# ratios: the odds of a higher category, whatever the cutpoint, are multiplied
# by :math:`e^{\beta_j}` per unit of :math:`x_j`.

beta_draws = result["beta"].reshape(-1, 2)
cut_draws = result["cuts"].reshape(-1, 3)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for ax, draws, truth, names, title in (
    (axes[0], cut_draws, cuts_true, ["$c_1$", "$c_2$", "$c_3$"], "Cutpoints"),
    (axes[1], beta_draws, beta_true, [r"$\beta_1$ (continuous)", r"$\beta_2$ (binary)"], "Coefficients"),
):
    lo, med, hi = torch.quantile(draws, torch.tensor([0.03, 0.5, 0.97], dtype=draws.dtype), dim=0)
    pos = torch.arange(draws.shape[1])
    ax.errorbar(pos, med, yerr=torch.stack([med - lo, hi - med]), fmt="o", color="C0", capsize=4, label="posterior median, 94% interval")
    ax.scatter(pos, truth, marker="x", color="C3", zorder=3, label="true value")
    ax.set_xticks(pos, names)
    ax.set_title(title)
axes[0].legend(frameon=False, fontsize="small")
plt.show()

# %%
# Predicted category probabilities
# --------------------------------
# For each posterior draw, ``OrderedLogistic.probs()`` gives the probabilities
# of the four categories. Here they are shown against :math:`x_1` for both
# values of the binary covariate, together with the observed frequencies in
# bins of :math:`x_1`.

grid = torch.linspace(-2.5, 2.5, 100, dtype=torch.float64)
edges = torch.linspace(-2.5, 2.5, 6, dtype=torch.float64)
centres = 0.5 * (edges[1:] + edges[:-1])
bin_index = torch.bucketize(x1, edges[1:-1])

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
for ax, group in zip(axes, (0.0, 1.0)):
    eta_grid = beta_draws[:, :1] * grid + beta_draws[:, 1:] * group  # (draws, grid)
    probs = OrderedLogistic(eta_grid, cut_draws.unsqueeze(1)).probs()  # (draws, grid, 4)
    mean = probs.mean(0)
    for k in range(4):
        ax.fill_between(grid, probs[..., k].quantile(0.03, 0), probs[..., k].quantile(0.97, 0), color=f"C{k}", alpha=0.25, linewidth=0)
        ax.plot(grid, mean[:, k], color=f"C{k}", label=f"P(y = {k})")
        observed = torch.stack([(y[(bin_index == b) & (x2 == group)] == k).double().mean() for b in range(len(centres))])
        ax.scatter(centres, observed, color=f"C{k}", edgecolors="black", zorder=3)
    ax.set(xlabel="$x_1$", title=f"$x_2 = {int(group)}$")
axes[0].set_ylabel("probability")
axes[0].legend(frameon=False, fontsize="small", loc="center left")
plt.show()
