"""
Automatic Relevance Determination
=================================

Fits an exact Gaussian process with one length-scale per input feature (ARD)
to data with three inputs, only two of which affect the response. NUTS
samples the length-scales, and the posterior assigns a very long length-scale
to the irrelevant feature, effectively switching it off. The posterior
predictive mean is then shown along each feature.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP, exact_gp_predictive


torch.manual_seed(0)

n, n_features = 60, 3
x = torch.rand(n, n_features, dtype=torch.float64) * 4.0 - 2.0
# Feature 0 varies quickly, feature 1 slowly and feature 2 is pure noise.
f_true = torch.sin(2.0 * x[:, 0]) + 0.5 * x[:, 1]
y = f_true + 0.1 * torch.randn(n, dtype=torch.float64)

# %%
# The model
# ---------
# ``log_lengthscale`` has ``shape=3``, so the RBF kernel uses
# ``k(x, x') = v exp(-1/2 sum_d (x_d - x'_d)^2 / l_d^2)`` with an individual
# ``l_d`` for each feature. Its gradient w.r.t. every ``l_d`` is derived by
# hand and flows through the ``Exp`` transform to ``log_lengthscale``.

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.5), shape=n_features)
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_noise = RandomParameter("log_noise", Normal(-3.0, 1.0))
    lengthscale = Exp("lengthscale", log_lengthscale)
    f = ExactGP(
        "f",
        inputs,
        RBF(lengthscale=lengthscale, variance=Exp("variance", log_variance)),
        noise_variance=Exp("noise", log_noise),
    )
    ObservedParameter("y", f, y)

trace = model.sample(300, 300, n_chains=2, start_point_variance=0.1, seed=1)
print(trace.summary(include_deterministic=True))

# %%
# Posterior length-scales
# -----------------------
# The irrelevant third feature gets a length-scale far beyond the data range.

lengthscale_draws = trace["lengthscale"].reshape(-1, n_features)
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.boxplot([lengthscale_draws[:, d].log10().numpy() for d in range(n_features)], showfliers=False)
ax.set_xticks(range(1, n_features + 1), [f"feature {d}" for d in range(n_features)])
ax.set(ylabel="log10 length-scale", title="ARD length-scale posterior")
plt.show()

# %%
# Predictions along each feature
# ------------------------------
# Each panel varies one feature over its range while the other two are held at
# zero. The irrelevant feature produces a flat prediction.

grid = torch.linspace(-2.0, 2.0, 50, dtype=torch.float64)
thinned = trace.thin(10)
fig, axes = plt.subplots(1, n_features, figsize=(12, 3.5), sharey=True, layout="constrained")
for d, ax in enumerate(axes):
    new_inputs = torch.zeros(grid.numel(), n_features, dtype=torch.float64)
    new_inputs[:, d] = grid
    predictions = exact_gp_predictive(f, thinned, y, new_inputs, n_samples=5).reshape(-1, grid.numel())
    lower, median, upper = torch.quantile(predictions, torch.tensor([0.05, 0.5, 0.95], dtype=predictions.dtype), dim=0)
    truth = torch.sin(2.0 * new_inputs[:, 0]) + 0.5 * new_inputs[:, 1]
    ax.fill_between(grid, lower, upper, alpha=0.25, color="tab:blue", label="90% interval")
    ax.plot(grid, median, color="tab:blue", label="Posterior median")
    ax.plot(grid, truth, "k--", label="True function")
    ax.set(title=f"Varying feature {d}", xlabel=f"x[{d}]")
axes[0].set_ylabel("f")
axes[0].legend()
plt.show()
