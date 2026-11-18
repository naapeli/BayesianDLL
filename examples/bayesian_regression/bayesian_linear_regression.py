"""
Bayesian Linear Regression
==========================

A straight line ``y = a + b x + noise`` is fitted to twelve noisy points and
checked against three held-out points.

The covariate is centred before it enters the model: the line is written as
``alpha + beta * (x - x_mean)``, so ``alpha`` is the expected response at the
centre of the data. With an uncentred covariate the intercept and slope are
strongly correlated a posteriori, which forces NUTS into small steps; after
centring they are nearly independent. The intercept at ``x = 0`` is recovered
with the deterministic block ``Affine`` and appears in the trace as ``a``.

The figures show the model graph, the marginal posteriors with their traces,
and the fitted line with bands for the mean and for new observations, drawn
from the posterior predictive distribution, over the training points, the
held-out points and the true line.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Affine, Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import Graphics, summary


torch.manual_seed(7)

n_points = 15
a_true, b_true, sigma_true = 1.0, 2.5, 0.7
x = torch.linspace(0.0, 1.0, n_points, dtype=torch.float64)
y = a_true + b_true * x + sigma_true * torch.randn(n_points, dtype=torch.float64)

order = torch.randperm(n_points)
train, test = order[:12], order[12:]
x_train, y_train = x[train], y[train]
x_test, y_test = x[test], y[test]
x_center = x_train.mean()

# %%
# The model
# ---------
# ``Linear`` builds the mean ``alpha + beta * x_centred``; the priors are
# weakly informative on the scale of the data.
with Model() as model:
    x_centred = Data("x_centred", x_train - x_center)
    alpha = RandomParameter("alpha", Normal(0.0, 5.0))
    beta = RandomParameter("beta", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(2.0))

    mu = Linear("mu", x_centred, slope=beta, intercept=alpha)
    Affine("a", beta, loc=alpha, scale=-x_center)  # intercept at x = 0

    with plate("observations", x_centred):
        ObservedParameter("y", Normal(mu, sigma), y_train)

Graphics.plot_model(model)

result = model.sample(1000, 500, n_chains=4, seed=1, delta=0.85, progress_bar=False)
print(summary({name: result[name] for name in ("alpha", "beta", "sigma", "a")}).to_string())
print(f"true values: a = {a_true}, b = {b_true}, sigma = {sigma_true}")
print("divergent transitions:", sum(result.divergences))

# %%
# Predictions
# -----------
# Replacing the covariate with ``Data.set_value`` resizes the plate, so the
# same model draws replicated responses at new inputs.
x_centred.set_value(x_test - x_center)
held_out = model.posterior_predicative(result, n_samples=1000, samples_per_step=4, progress_bar=False)["y"]
held_out = held_out.reshape(-1, len(x_test))
inside = (held_out.quantile(0.025, 0) <= y_test) & (y_test <= held_out.quantile(0.975, 0))
print(f"held-out points inside their 95% predictive interval: {int(inside.sum())} of {len(x_test)}")

x_grid = torch.linspace(x.min(), x.max(), 100, dtype=torch.float64)
x_centred.set_value(x_grid - x_center)
new_y = model.posterior_predicative(result, n_samples=1000, samples_per_step=4, progress_bar=False)["y"]
new_y = new_y.reshape(-1, len(x_grid))
line = result["alpha"].reshape(-1, 1) + result["beta"].reshape(-1, 1) * (x_grid - x_center)

Graphics.plot_posterior(result)

fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
ax.fill_between(x_grid, new_y.quantile(0.025, 0), new_y.quantile(0.975, 0), color="C0", alpha=0.15,
                linewidth=0, label="95% interval, new observation")
ax.fill_between(x_grid, line.quantile(0.025, 0), line.quantile(0.975, 0), color="C0", alpha=0.35,
                linewidth=0, label="95% interval, mean")
ax.plot(x_grid, line.mean(0), color="black", label="posterior mean line")
ax.plot(x_grid, a_true + b_true * x_grid, color="C3", linestyle="--", label="true line")
ax.scatter(x_train, y_train, color="C0", edgecolors="white", zorder=3, label="training points")
ax.scatter(x_test, y_test, color="C1", marker="D", edgecolors="white", zorder=3, label="held-out points")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fitted line and predictive bands")
fig.legend(loc="outside right upper", frameon=False, fontsize="small")
plt.show()
