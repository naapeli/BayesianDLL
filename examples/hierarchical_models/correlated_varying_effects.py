"""
Correlated Varying Intercepts and Slopes
========================================

Twelve groups each have their own regression line, and a group's intercept
and slope are drawn together from a bivariate Normal population, so they can
be correlated:

.. math::

    (\\alpha_j, \\beta_j) \\sim N(\\mu, \\Sigma), \\quad
    \\Sigma = \\operatorname{diag}(\\tau) \\, R \\, \\operatorname{diag}(\\tau), \\quad
    y_i \\sim N(\\alpha_{g[i]} + \\beta_{g[i]} x_i, \\sigma_y).

The correlation matrix ``R`` gets an LKJ prior on its Cholesky factor,
``L_R ~ LKJCholesky(2, 2)``, which mildly favours weak correlations, and the
group scales ``tau`` get ``HalfNormal(1.5)`` priors. The
``CholeskyCovariance`` block combines them into the Cholesky factor
``L = diag(tau) L_R`` of ``Sigma``, which ``MultivariateNormal`` takes as
``scale_tril`` without ever forming or factorising ``Sigma``.

**Parameterisation.** Each group has twelve observations with little noise,
so the data pin down every group's line, and the posterior of the group
effects has no funnel. The effects are therefore sampled directly
(centered), as one ``(12, 2)`` ``RandomParameter`` with the multivariate
Normal prior. Compared with the non-centered form ``mu + L z``, this removes
the posterior dependence between the standardised effects and ``L`` that
made NUTS hit long trajectories: the centered model needs no more than a
few dozen leapfrog steps per iteration, has no divergences, and gives
R-hat below 1.01. The covariate is standardised (mean 0, sd 1), so
intercepts and slopes are nearly independent within a group.

``Index`` picks every observation's ``(alpha, beta)`` row, and ``Multiply``
and ``Sum`` form ``alpha + beta x`` with the design rows ``(1, x)``.

The figures show the posterior of the correlation ``rho``, the estimated
group effects against the true ones with the population ellipse, and every
group's data with its fitted line and 90% band.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import CholeskyCovariance, Index, Multiply, Sum
from BayesianDLL.Distributions import HalfNormal, LKJCholesky, MultivariateNormal, Normal


torch.manual_seed(11)

n_groups, per_group = 12, 12
true_mu = torch.tensor([1.0, 0.5], dtype=torch.float64)
true_tau = torch.tensor([1.0, 0.7], dtype=torch.float64)
true_rho = -0.7
true_R = torch.tensor([[1.0, true_rho], [true_rho, 1.0]], dtype=torch.float64)
true_cov = torch.diag(true_tau) @ true_R @ torch.diag(true_tau)
true_effects = true_mu + torch.randn(n_groups, 2, dtype=torch.float64) @ torch.linalg.cholesky(true_cov).T

group = torch.arange(n_groups).repeat_interleave(per_group)
x = torch.randn(n_groups * per_group, dtype=torch.float64)
y = true_effects[group, 0] + true_effects[group, 1] * x + 0.4 * torch.randn(len(x), dtype=torch.float64)

with Model() as model:
    group_data = Data("group", group)
    design = Data("design", torch.stack([torch.ones_like(x), x], dim=-1))  # rows (1, x_i)

    mu = RandomParameter("mu", Normal(0.0, 2.0), shape=(2,))
    tau = RandomParameter("tau", HalfNormal(1.5), shape=(2,))
    L_corr = RandomParameter("L_corr", LKJCholesky(2, 2.0))
    L = CholeskyCovariance("L", L_corr, tau)
    effects = RandomParameter("effects", MultivariateNormal(mu, scale_tril=L), shape=(n_groups, 2))  # (alpha_j, beta_j)
    sigma_y = RandomParameter("sigma_y", HalfNormal(1.0))

    with plate("observations", len(x)):
        effects_obs = Index("effects_obs", effects, group_data, dim=0)  # (N, 2)
        mean = Sum("mean", Multiply("terms", effects_obs, design), dim=-1)  # alpha + beta x
        ObservedParameter("y", Normal(mean, sigma_y), y)

trace = model.sample(n_samples=500, warmup_length=300, n_chains=4, progress_bar=False, seed=0, delta=0.9,
                     chain_method="vectorized")
summary = trace.summary()
print(summary.loc[[name for name in summary.index if not name.startswith("effects")]])
print(f"divergent transitions: {sum(trace.divergences)}, largest R-hat: {summary['r_hat'].max():.3f}, "
      f"smallest bulk ESS: {summary['ess_bulk'].min():.0f}")

# The correlation is R[1, 0] = (L_R L_R^T)[1, 0] = L_R[1, 0], since L_R[0, 0] = 1.
rho = trace["L_corr"][..., 1, 0].flatten()
print(f"posterior mean of rho: {rho.mean().item():.2f} (true value {true_rho})")

# %%
# Correlation, group effects and fitted lines
# -------------------------------------------
# With only twelve groups the correlation is loosely determined: the
# posterior leans towards a negative value but still allows values near zero. The estimated effects lie close
# to the true ones; the dashed ellipse is the 90% region of the population
# distribution at the posterior means of ``mu``, ``tau`` and ``rho``.

effect_draws = trace["effects"].flatten(0, 1)  # (S, J, 2)
effects_mean = effect_draws.mean(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
axes[0].hist(rho, bins=40, density=True, color="C0", alpha=0.7, label="posterior")
axes[0].axvline(true_rho, color="black", linestyle="--", label="true value")
axes[0].set_xlabel(r"correlation $\rho$ of intercepts and slopes")
axes[0].set_ylabel("density")
axes[0].set_title("Correlation")
axes[0].legend(fontsize="small")

angles = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
mean_R = torch.tensor([[1.0, rho.mean()], [rho.mean(), 1.0]], dtype=torch.float64)
mean_tau = trace["tau"].flatten(0, 1).mean(0)
population_cov = torch.diag(mean_tau) @ mean_R @ torch.diag(mean_tau)
ellipse = trace["mu"].flatten(0, 1).mean(0).unsqueeze(-1) + 2.146 * torch.linalg.cholesky(population_cov) @ torch.stack([angles.cos(), angles.sin()])
axes[1].plot(ellipse[0], ellipse[1], color="grey", linestyle="--", label="population (90%)")
axes[1].scatter(true_effects[:, 0], true_effects[:, 1], color="black", marker="x", label="true")
axes[1].scatter(effects_mean[:, 0], effects_mean[:, 1], color="C1", label="posterior mean")
for j in range(n_groups):
    axes[1].plot([true_effects[j, 0], effects_mean[j, 0]], [true_effects[j, 1], effects_mean[j, 1]], color="grey", lw=0.8)
axes[1].set_xlabel(r"intercept $\alpha_j$")
axes[1].set_ylabel(r"slope $\beta_j$")
axes[1].set_title("Group effects")
axes[1].legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
plt.show()

# %%
# The lines are ``alpha_j + beta_j x`` for every posterior draw; the band is
# their 90% interval and the dashed line the true group line.

grid = torch.linspace(-3.0, 3.0, 60, dtype=torch.float64)
lines = effect_draws[..., :1] + effect_draws[..., 1:] * grid  # (S, J, G)
lower, upper = lines.quantile(torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)

fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True, sharey=True, layout="constrained")
for j, ax in enumerate(axes.flat):
    mask = group == j
    ax.fill_between(grid, lower[j], upper[j], color="C0", alpha=0.3)
    ax.plot(grid, lines[:, j].mean(0), color="C0")
    ax.plot(grid, true_effects[j, 0] + true_effects[j, 1] * grid, color="black", linestyle="--", lw=1)
    ax.scatter(x[mask], y[mask], color="black", s=10)
    ax.set_title(f"group {j}", fontsize="small")
for ax in axes[-1]:
    ax.set_xlabel("x (standardised)")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
handles = [plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.3), plt.Line2D([], [], color="C0"),
           plt.Line2D([], [], color="black", linestyle="--"), plt.Line2D([], [], color="black", marker="o", linestyle="")]
fig.legend(handles, ["90% band", "posterior mean line", "true line", "data"], loc="outside lower center", ncol=4, fontsize="small")
fig.suptitle("Fitted line of every group")
plt.show()
