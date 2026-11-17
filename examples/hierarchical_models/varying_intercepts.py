"""
Partial Pooling with Varying Intercepts
=======================================

Eight groups share one slope but each has its own intercept. The groups are
deliberately unbalanced: some contribute dozens of observations, the last two
only two and one. The model is

.. math::

    \\mu_\\alpha \\sim N(0, 2), \\quad \\sigma_\\alpha \\sim \\text{HalfNormal}(1.5), \\quad
    \\alpha_j \\sim N(\\mu_\\alpha, \\sigma_\\alpha), \\\\
    \\beta \\sim N(0, 1), \\quad \\sigma \\sim \\text{HalfNormal}(1), \\quad
    y_i \\sim N(\\alpha_{g[i]} + \\beta x_i, \\sigma).

Because the intercepts share the population distribution
``N(mu_alpha, sigma_alpha)``, each group borrows strength from the others:
a group with plenty of data keeps close to its own least-squares intercept,
while a group with one or two points is pulled ("shrunk") towards
``mu_alpha``. This is partial pooling, between no pooling (one free intercept
per group) and complete pooling (one intercept for all).

The intercepts are sampled directly (the **centered** form,
``alpha ~ Normal(mu_alpha, sigma_alpha)``). The eight schools example shows
that this form fails when every group is weakly informed. Here most groups
have many observations, which pin their intercepts down and keep the
funnel away. With the same settings the centered model samples without
divergences in about half the time of the non-centered form
(``Affine(z, loc=mu_alpha, scale=sigma_alpha)``) and with nearly three times
its smallest effective sample size. The covariate is
standardised, so the intercepts are the group means at the average ``x``
and are nearly uncorrelated with the slope. ``Index`` picks every
observation's intercept and ``Linear`` adds the slope term.

The figures show

1. every group's data with the posterior mean line and a 90% band for the
   *mean* ``alpha_j + beta x`` (computed from the parameter draws on a grid),
   with the true line dashed;
2. the shrinkage of the intercepts: the no-pooling estimate, the posterior
   mean with a 90% interval and the true value for every group;
3. a prediction for a group that was never observed, next to a well observed
   group.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Index, Linear
from BayesianDLL.Distributions import HalfNormal, Normal


torch.manual_seed(3)

group_sizes = torch.tensor([30, 25, 20, 15, 10, 6, 2, 1])
n_groups = len(group_sizes)
true_mu_alpha, true_sigma_alpha = 1.0, 0.8
true_alpha = true_mu_alpha + true_sigma_alpha * torch.randn(n_groups, dtype=torch.float64)
true_beta, true_sigma = 0.8, 1.0

group = torch.arange(n_groups).repeat_interleave(group_sizes)
x_raw = 5.0 + 2.0 * torch.randn(len(group), dtype=torch.float64)
x_mean, x_std = x_raw.mean(), x_raw.std()
x = (x_raw - x_mean) / x_std  # the model works with the standardised covariate
y = true_alpha[group] + true_beta * x + true_sigma * torch.randn(len(group), dtype=torch.float64)

with Model() as model:
    group_data = Data("group", group)
    x_data = Data("x", x)

    mu_alpha = RandomParameter("mu_alpha", Normal(0.0, 2.0))
    sigma_alpha = RandomParameter("sigma_alpha", HalfNormal(1.5))
    alpha = RandomParameter("alpha", Normal(mu_alpha, sigma_alpha), shape=(n_groups,))
    beta = RandomParameter("beta", Normal(0.0, 1.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))

    with plate("observations", x_data):
        mean = Linear("mean", x_data, slope=beta, intercept=Index("alpha_obs", alpha, group_data))
        ObservedParameter("y", Normal(mean, sigma), y)

trace = model.sample(n_samples=500, warmup_length=500, n_chains=4, progress_bar=False, seed=0, delta=0.9,
                     chain_method="vectorized")
summary = trace.summary()
print(summary)
print(f"divergent transitions: {sum(trace.divergences)}, largest R-hat: {summary['r_hat'].max():.3f}")

alpha_draws = trace["alpha"].flatten(0, 1)  # (S, J)
beta_draws = trace["beta"].flatten(0, 1)  # (S, 1)
mu_draws = trace["mu_alpha"].flatten(0, 1)  # (S, 1)
tau_draws = trace["sigma_alpha"].flatten(0, 1)  # (S, 1)
sigma_draws = trace["sigma"].flatten(0, 1)  # (S, 1)
probabilities = torch.tensor([0.05, 0.95], dtype=torch.float64)

# %%
# Fitted lines for every group
# ----------------------------
# The band is the 90% interval of the regression line ``alpha_j + beta x``
# over the posterior draws, so it reflects uncertainty about the parameters
# only. It is wide for the groups with one or two observations, whose lines
# rely mostly on the population distribution.

grid = torch.linspace(-2.5, 2.5, 100, dtype=torch.float64)
grid_raw = x_mean + x_std * grid
lines = alpha_draws.unsqueeze(-1) + beta_draws.unsqueeze(-1) * grid  # (S, J, G)
line_mean = lines.mean(0)
line_lower, line_upper = lines.quantile(probabilities, dim=0)

n_cols = 4
fig, axes = plt.subplots(2, n_cols, figsize=(14, 6.5), sharex=True, sharey=True, layout="constrained")
for j, ax in enumerate(axes.flat):
    mask = group == j
    ax.fill_between(grid_raw, line_lower[j], line_upper[j], color=f"C{j}", alpha=0.25, label="90% band of the mean")
    ax.plot(grid_raw, line_mean[j], color=f"C{j}", label="posterior mean line")
    ax.plot(grid_raw, true_alpha[j] + true_beta * grid, color="black", linestyle="--", lw=1, label="true line")
    ax.scatter(x_raw[mask], y[mask], color=f"C{j}", edgecolor="black", s=22, zorder=3, label="data")
    ax.set_title(f"group {j} (n = {int(group_sizes[j])})")
for ax in axes[-1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle("Varying-intercept fit: posterior mean lines with 90% bands")
plt.show()

# %%
# Shrinkage of the intercepts
# ---------------------------
# The no-pooling estimate of a group is its mean residual
# ``mean(y_i - beta_hat x_i)`` with the posterior mean slope ``beta_hat``.
# The posterior means lie between the no-pooling estimates and the
# population mean ``mu_alpha``. The arrows show the pull: it is large for the
# small groups 5 and 6, whose few observations say little about their
# intercept, and almost absent for the large groups 0-3. The single point of
# group 7 happens to lie close to the population mean, so its estimate barely
# moves, but its interval is the widest: it is set mostly by ``sigma_alpha``.

beta_hat = beta_draws.mean()
no_pooling = torch.stack([(y[group == j] - beta_hat * x[group == j]).mean() for j in range(n_groups)])
alpha_mean = alpha_draws.mean(0)
alpha_lower, alpha_upper = alpha_draws.quantile(probabilities, dim=0)
positions = torch.arange(n_groups, dtype=torch.float64)

fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
ax.axhline(mu_draws.mean(), color="grey", linestyle=":", label=r"posterior mean of $\mu_\alpha$")
ax.errorbar(positions, alpha_mean, yerr=torch.stack([alpha_mean - alpha_lower, alpha_upper - alpha_mean]),
            fmt="o", color="C0", capsize=4, label="partial pooling (mean, 90% interval)")
ax.scatter(positions - 0.18, no_pooling, marker="s", color="C1", label="no pooling (group mean residual)")
ax.scatter(positions + 0.18, true_alpha, marker="x", color="black", s=50, label="true intercept")
for j in range(n_groups):
    ax.annotate("", xy=(j, alpha_mean[j]), xytext=(j - 0.18, no_pooling[j]),
                arrowprops=dict(arrowstyle="->", color="C1", alpha=0.6))
ax.set_xticks(positions, [f"{j}\nn={int(n)}" for j, n in enumerate(group_sizes)])
ax.set_xlabel("group")
ax.set_ylabel(r"intercept $\alpha_j$")
ax.set_title("Shrinkage towards the population mean")
ax.legend(fontsize=8)
plt.show()

# %%
# Predicting an unseen group
# --------------------------
# A new group has no data, so its intercept is drawn from the fitted
# population distribution: for every posterior draw,
# ``alpha_new ~ N(mu_alpha, sigma_alpha)``. Its line ``alpha_new + beta x``
# varies by the full between-group spread ``sigma_alpha`` and also carries
# the uncertainty in ``mu_alpha`` and ``sigma_alpha``, so its band is much
# wider than that of an observed group. Adding ``N(0, sigma)`` noise to the
# lines gives the predictive interval of a single new observation.

generator = torch.Generator().manual_seed(1)
alpha_new = mu_draws + tau_draws * torch.randn(mu_draws.shape, dtype=torch.float64, generator=generator)
new_lines = alpha_new + beta_draws * grid  # (S, G)
observed = 0


def add_noise(mean_lines, replicates=20):
    """Draws of a new observation: ``replicates`` noise draws per posterior draw (smooth quantiles)."""
    noise = torch.randn(replicates, *mean_lines.shape, dtype=torch.float64, generator=generator)
    return (mean_lines + sigma_draws * noise).flatten(0, 1)


new_observations = add_noise(new_lines)
observed_observations = add_noise(lines[:, observed])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
panels = (
    (axes[0], f"observed group {observed}", lines[:, observed], observed_observations, group == observed),
    (axes[1], "new, unobserved group", new_lines, new_observations, None),
)
for ax, title, mean_lines, single, mask in panels:
    lower, upper = single.quantile(probabilities, dim=0)
    ax.fill_between(grid_raw, lower, upper, color="C2", alpha=0.15, label="90% interval of a new observation")
    lower, upper = mean_lines.quantile(probabilities, dim=0)
    ax.fill_between(grid_raw, lower, upper, color="C2", alpha=0.4, label="90% band of the mean line")
    ax.plot(grid_raw, mean_lines.mean(0), color="C2", label="posterior mean line")
    if mask is not None:
        ax.scatter(x_raw[mask], y[mask], color="black", s=14, label="data")
    else:
        ax.scatter(x_raw, y, color="grey", s=8, alpha=0.4, label="data of all groups")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("y")
plt.show()
