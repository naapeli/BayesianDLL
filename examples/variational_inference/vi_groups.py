"""
Grouped Variational Guides
==========================

A guide does not have to treat every latent the same way. ``Guide(model,
default_family=...)`` collects every latent into one Gaussian, but inside a
``with Guide(...)`` block the latents can be split into independent pieces:
a ``Group`` with its own family for latents that depend on each other, and
hand-written entries for latents that deserve a different distribution. The
rest goes into the default group.

The model is a regression with group-level offsets,

.. math::

    \\mu \\sim N(0, 5), \\quad \\beta \\sim N(0, 2), \\quad
    \\tau, \\sigma \\sim \\text{HalfNormal}(1), \\quad
    u_j \\sim N(0, \\tau), \\quad
    y_{ij} \\sim N(\\mu + u_j + \\beta x_{ij}, \\sigma),

with 15 groups of 12 observations. Every group sees covariates around
``x = 8``, far from zero, so the data fix the line near ``x = 8`` much better
than its intercept: the global intercept ``mu`` and the slope ``beta`` are
almost perfectly negatively correlated a posteriori. The 15 offsets ``u_j``
are each pinned down by their own group's data and are nearly independent of
everything else, so a mean-field family suits them.

Four approximations are compared:

* **all mean-field**: ``Guide(model, default_family="mean_field")``;
* **all full-rank**: ``Guide(model, default_family="full_rank")``, one
  multivariate Normal over all 19 unconstrained coordinates;
* **grouped**: a full-rank ``Group`` for ``(mu, beta)``, a hand-written
  ``LogNormal`` entry for ``sigma`` and a mean-field default group for the
  rest;
* **NUTS** as the reference posterior.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Group, Guide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate
from BayesianDLL.Deterministic import Add, Index, Linear
from BayesianDLL.Distributions import HalfNormal, LogNormal, LogTransform, Normal
from BayesianDLL.Variational import BBVI

torch.manual_seed(0)

n_groups, per_group = 15, 12
true_mu, true_beta, true_tau, true_sigma = 1.0, 0.5, 0.7, 1.0
group = torch.arange(n_groups).repeat_interleave(per_group)
x = 8.0 + torch.randn(len(group), dtype=torch.float64)
true_u = true_tau * torch.randn(n_groups, dtype=torch.float64)
y = true_mu + true_u[group] + true_beta * x + true_sigma * torch.randn(len(group), dtype=torch.float64)

with Model() as model:
    group_data = Data("group", group)
    x_data = Data("x", x)
    mu = RandomParameter("mu", Normal(0.0, 5.0))
    beta = RandomParameter("beta", Normal(0.0, 2.0))
    tau = RandomParameter("tau", HalfNormal(1.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    u = RandomParameter("u", Normal(0.0, tau), shape=(n_groups,))
    with plate("observations", x_data):
        intercept = Add("intercept", mu, Index("u_obs", u, group_data))
        mean = Linear("mean", x_data, coefficients=beta, intercept=intercept)
        ObservedParameter("y", Normal(mean, sigma), y)

# %%
# Three guides
# ------------
# The grouped guide is written inside a ``with Guide(...)`` block. ``Group``
# registers itself with that guide, just as a ``RandomParameter`` registers
# with the active model. The hand-written ``LogNormal`` describes ``sigma``
# itself (its support matches), not ``log sigma``. Every latent the block does
# not mention goes into the mean-field default group. Groups and entries are
# independent, so the joint guide is their product.

with Guide(model, default_family="mean_field") as grouped:
    Group(["mu", "beta"], family="full_rank", name="line")
    RandomParameter("sigma", LogNormal(VariationalParameter("sigma_loc", torch.tensor([0.0], dtype=torch.float64)),
                                       VariationalParameter("sigma_scale", torch.tensor([0.2], dtype=torch.float64), transform=LogTransform())))

guides = {
    "all mean-field": Guide(model, default_family="mean_field"),
    "all full-rank": Guide(model, default_family="full_rank"),
    "grouped": grouped,
}


def n_variational_parameters(guide):
    """Free numbers the optimiser adjusts (a d x d scale_tril has d (d + 1) / 2)."""
    total = 0
    for parameter in guide.variational_parameters().values():
        d = parameter.value.shape[-1]
        total += d * (d + 1) // 2 if parameter.value.ndim == 2 else parameter.value.numel()
    return total


histories, traces, seconds = {}, {}, {}
for label, guide in guides.items():
    torch.manual_seed(1)
    start = time.perf_counter()
    histories[label] = BBVI(model, guide, n_samples=2, epochs=1500, lr=0.03, lrd=0.999, callback_frequency=5, verbose=False)
    seconds[label] = time.perf_counter() - start
    traces[label] = guide.sample(1500)

start = time.perf_counter()
traces["NUTS"] = model.sample(n_samples=300, warmup_length=300, n_chains=4, progress_bar=False, seed=0,
                              metric="dense", chain_method="vectorized")
seconds["NUTS"] = time.perf_counter() - start
print(f"NUTS: {sum(traces['NUTS'].divergences)} divergences")

print(f"{'method':>15} {'time [s]':>9} {'parameters':>11} {'final ELBO':>11} {'corr(mu, beta)':>15}")
for label, trace in traces.items():
    pair = torch.stack([trace["mu"].flatten(), trace["beta"].flatten()])
    parameters = n_variational_parameters(guides[label]) if label in guides else "-"
    final = f"{sum(histories[label][-40:]) / 40:.1f}" if label in histories else "-"
    print(f"{label:>15} {seconds[label]:9.1f} {parameters:>11} {final:>11} {torch.corrcoef(pair)[0, 1].item():+15.3f}")

# %%
# Optimisation
# ------------
# The ELBO curves are averaged over 40 records. The mean-field and grouped
# guides converge within a few hundred steps, and the mean-field guide levels
# off about 2 nats lower: that gap is the KL divergence it pays for ignoring
# the ``(mu, beta)`` correlation. The all-full-rank guide can represent the
# same correlation, but it has to learn a 19 x 19 Cholesky factor and reaches
# the optimum much later.

colors = {"all mean-field": "C1", "all full-rank": "C2", "grouped": "C0", "NUTS": "black"}
fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for label, history in histories.items():
    smoothed = torch.tensor(history).unfold(0, 40, 1).mean(-1)
    ax.plot(5 * torch.arange(40, len(history) + 1), smoothed, color=colors[label], label=label)
best = max(sum(h[-40:]) / 40 for h in histories.values())
ax.set_ylim(best - 15, best + 4)
ax.set_xlabel("epoch")
ax.set_ylabel("ELBO (moving average)")
ax.set_title("ELBO during BBVI")
ax.legend()
plt.show()

# %%
# The correlated pair and the marginals
# -------------------------------------
# NUTS draws of ``(mu, beta)`` lie on a thin diagonal ridge. The full-rank
# and grouped guides follow it. The mean-field guide is an axis-aligned
# ellipse. To stay on the ridge it must shrink both marginals, so its
# intervals for ``mu`` and ``beta`` are several times too narrow. The
# hyperparameters ``tau`` and ``sigma`` and the offsets are nearly
# independent of the pair, and all three guides describe them well.

angles = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
circle = torch.stack([angles.cos(), angles.sin()]) * 2.448  # radius of the 95% region in 2-D
fig = plt.figure(figsize=(13, 7.5), layout="constrained")
grid = fig.add_gridspec(2, 3)
ax = fig.add_subplot(grid[:, 0])
nuts_pair = torch.stack([traces["NUTS"]["mu"].flatten(), traces["NUTS"]["beta"].flatten()])
ax.scatter(nuts_pair[0], nuts_pair[1], s=3, alpha=0.15, color="black", label="NUTS draws")
for label in guides:
    pair = torch.stack([traces[label]["mu"].flatten(), traces[label]["beta"].flatten()])
    ellipse = pair.mean(1, keepdim=True) + torch.linalg.cholesky(torch.cov(pair)) @ circle
    ax.plot(ellipse[0], ellipse[1], color=colors[label], lw=2, label=f"{label} (95%)")
ax.set_xlabel("mu")
ax.set_ylabel("beta")
ax.set_title("Joint posterior of mu and beta")
ax.legend(fontsize="small")
for position, name in zip([(0, 1), (0, 2), (1, 1), (1, 2)], ["mu", "beta", "tau", "sigma"]):
    ax = fig.add_subplot(grid[position])
    for label, trace in traces.items():
        ax.hist(trace[name].flatten(), bins=50, density=True, histtype="step", lw=1.6, color=colors[label], label=label)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Marginal posterior of {name}")
ax.legend(fontsize="small")
plt.show()

# %%
# Cost
# ----
# The full-rank guide optimises a 19 x 19 Cholesky factor, 190 numbers for
# the covariance alone, and every step costs ``O(d^2)``. The grouped guide
# spends a full covariance only on the two latents that need one. Here the
# model's gradient dominates the cost of a BBVI step, so the run times are
# similar; the number of parameters, and with it the cost of a full-rank
# step, grows quadratically with the number of latents in the group.

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
labels = list(traces)
axes[0].bar(labels, [seconds[label] for label in labels], color=[colors[label] for label in labels])
axes[0].set_ylabel("seconds")
axes[0].set_title("Run time (BBVI 1500 steps; NUTS 4 x 600 iterations)")
axes[1].bar(list(guides), [n_variational_parameters(guide) for guide in guides.values()], color=[colors[label] for label in guides])
axes[1].set_ylabel("count")
axes[1].set_title("Variational parameters")
for ax in axes:
    ax.tick_params(axis="x", labelsize="small")
plt.show()

# %%
# Mean and covariance of a guide
# ------------------------------
# ``guide.moments(space)`` returns a :class:`~BayesianDLL.Variational.GuideMoments`
# with the mean and the block-diagonal covariance of the guide (pieces of a
# guide are independent). Specific latents or groups can be accessed directly:
#
# * ``m.mean("mu")``, ``m.variance("mu")``, ``m.covariance("mu")``, ``m.correlation("mu")``
# * ``m.mean(group="line")``, ``m.covariance(group="line")``, ``m.correlation(group="line")``
# * ``m.mean()``, ``m.variance()``, ``m.covariance()``, ``m.correlation()`` (full tensors)
#
# Each block is exact where a closed form exists and a Monte Carlo estimate
# otherwise (listed in ``estimated``, with ``standard_error`` = sd / sqrt(n)).
# In the **unconstrained** space every Gaussian block is exact, but the
# ``LogNormal`` entry describes ``sigma`` itself, so the moments of
# ``log sigma`` are estimated from independent draws. In the **constrained**
# space ``tau`` (log scale in the mean-field group) gets the exact log-normal
# moments and the ``LogNormal`` entry its closed form, so nothing is sampled.
# These are moments of the approximation ``q``, not of the posterior.
#
# Constrained moments are not the transformed unconstrained ones:
# ``exp(E[log tau])`` is the median of the log-normal, smaller than ``E[tau]``
# by the factor ``exp(-s^2 / 2)``. Use the unconstrained moments to inspect the
# optimisation and the constrained ones to report results.

for space in ("unconstrained", "constrained"):
    m = grouped.moments(space)
    print(f"{space:>13}: exact {m.exact}, Monte Carlo {m.estimated}")

unconstrained = grouped.moments("unconstrained")
constrained = grouped.moments("constrained")

line_mean = constrained.mean(group="line")
line_cov = constrained.covariance(group="line")
line_corr = constrained.correlation(group="line")
corr_mu_beta = line_corr[0, 1]

print(f"line group mean [mu, beta]: {line_mean.tolist()}")
print(f"corr(mu, beta) inside line group: {corr_mu_beta.item():+.3f}")
print(f"tau: exp(E[log tau]) = {unconstrained.mean('tau').exp().item():.4f}, "
      f"E[tau] = {constrained.mean('tau').item():.4f}, "
      f"Monte Carlo from guide.sample: {traces['grouped']['tau'].mean().item():.4f}")
print(f"sigma: E[sigma] = {constrained.mean('sigma').item():.4f} (closed form of the LogNormal entry), "
      f"E[log sigma] = {unconstrained.mean('sigma').item():.4f} +- {unconstrained.standard_error['sigma'].item():.4f} (Monte Carlo)")
