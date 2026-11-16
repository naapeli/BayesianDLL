"""
Varying-Intercept Regression
============================

Observations belong to ``J`` groups, and every group gets its own intercept
drawn from a common population distribution:

.. math::

    \\alpha_j \\sim N(\\mu_\\alpha, \\sigma_\\alpha), \\quad
    y_i \\sim N(\\alpha_{g[i]} + \\beta x_i, \\sigma).

``Index`` looks up the intercept of every observation, ``alpha[group]``. Every
group has about twenty observations, so the intercepts are well identified and
the centered parameterisation samples efficiently (with only a few noisy
observations per group, use the non-centered form of the eight schools
example).

The group labels and the covariate are ``Data`` nodes, so after sampling they
are replaced with ``Data.set_value`` to compute the posterior predictive distribution of a new
data set without rebuilding the model.

The model reserves one extra intercept slot without any observations. Its
posterior equals the population distribution given the hyperparameters, so it
serves as the intercept of a *new* group in the predictions.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, Index, Multiply
from BayesianDLL.Distributions import HalfNormal, Normal


torch.manual_seed(3)

n_groups = 6
n_observations = 120
true_alpha = 1.0 + 1.5 * torch.randn(n_groups, dtype=torch.float64)
true_beta, true_sigma = 0.8, 0.5
group = torch.randint(0, n_groups, (n_observations,))
x = torch.randn(n_observations, dtype=torch.float64)
y = true_alpha[group] + true_beta * x + true_sigma * torch.randn(n_observations, dtype=torch.float64)

new_group = n_groups  # index of the reserved slot without observations

with Model() as model:
    group_data = Data("group", group)
    x_data = Data("x", x)
    y_data = Data("y_values", y)

    mu_alpha = RandomParameter("mu_alpha", Normal(0.0, 3.0))
    sigma_alpha = RandomParameter("sigma_alpha", HalfNormal(2.0))
    alpha = RandomParameter("alpha", Normal(mu_alpha, sigma_alpha), shape=(n_groups + 1,))
    beta = RandomParameter("beta", Normal(0.0, 2.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))

    with plate("observations", y_data):
        mean = Add("mean", Index("alpha_obs", alpha, group_data), Multiply("beta_x", beta, x_data))
        ObservedParameter("y", Normal(mean, sigma), y_data)

trace = model.sample(n_samples=500, warmup_length=500, n_chains=2, progress_bar=False, seed=0)
print(trace.summary())
print("divergences:", sum(trace.divergences))

alpha_draws = trace["alpha"].flatten(0, 1)
print("true intercepts:     ", [round(v, 2) for v in true_alpha.tolist()])
print("posterior means:     ", [round(v, 2) for v in alpha_draws[:, :n_groups].mean(0).tolist()])

# %%
# Posterior predictive for a new data set
# ---------------------------------------
# New covariate values for group 0 and for the unobserved new group. Replacing
# the ``Data`` values changes the number of observations and the plate size;
# the observed values only fix the shape of the generated data.

x_new = torch.linspace(-2.5, 2.5, 25, dtype=torch.float64)
groups_new = torch.cat([torch.zeros(25, dtype=torch.long), torch.full((25,), new_group)])
group_data.set_value(groups_new)
x_data.set_value(torch.cat([x_new, x_new]))
y_data.set_value(torch.zeros(50, dtype=torch.float64))

predictive = model.posterior_predicative(trace, n_samples=200, samples_per_step=5, warmup_per_sample=20, progress_bar=False)
draws = predictive["y"].flatten(0, 1)  # (draws, 50)
lower, median, upper = draws.quantile(torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

fig, ax = plt.subplots(figsize=(9, 5))
mask = group == 0
ax.scatter(x[mask], y[mask], color="C0", s=18, label="observed, group 0")
for offset, label, color in ((0, "group 0", "C0"), (25, "new group", "C1")):
    sl = slice(offset, offset + 25)
    ax.plot(x_new, median[sl], color=color, label=f"predictive median, {label}")
    ax.fill_between(x_new, lower[sl], upper[sl], color=color, alpha=0.2, label=f"90% interval, {label}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Posterior predictive for an observed and a new group")
ax.legend(fontsize=8)
plt.show()
