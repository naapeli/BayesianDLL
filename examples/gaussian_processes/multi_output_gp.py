"""
Multi-Output GPs with Coregionalisation
=======================================

Three related signals are measured at different times: the first two densely,
the third (a sensor that failed early) only on the left half of the
interval. Modelling the outputs jointly lets the dense outputs inform the
missing part of the third.

The intrinsic coregionalisation model (ICM) writes the covariance between
output ``i`` at ``x`` and output ``j`` at ``x'`` as

``k((x, i), (x', j)) = B[i, j] k(x, x')``,  ``B = W Wᵀ + diag(κ)``,

with a shared kernel ``k`` and a learned ``3 x 3`` matrix ``B``. The observations
of all outputs are stacked into one vector, and every input row holds the
feature ``x`` and the integer index of the output in the last column.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, IntrinsicCoregionalization, Matern52, exact_gp_predictive

torch.manual_seed(0)
dtype = torch.float64

def signals(x):
    base = torch.sin(1.5 * x) + 0.3 * x
    return torch.stack([base, 0.8 * base + 0.3 * torch.cos(3.0 * x), -0.6 * base])


# %%
# Data
# ----
# Outputs 0 and 1 are observed at 20 points, output 2 only at the first 8.

x = torch.linspace(0.0, 5.0, 20, dtype=dtype)
truth = signals(x)
observed = [20, 20, 8]
rows, values = [], []
for output, count in enumerate(observed):
    rows.append(torch.stack([x[:count], torch.full((count,), float(output), dtype=dtype)], dim=-1))
    values.append(truth[output, :count] + 0.1 * torch.randn(count, dtype=dtype))
inputs, y = torch.cat(rows), torch.cat(values)

# %%
# The model
# ---------
# ``W`` has rank 1 (the outputs are close to multiples of one latent signal)
# and ``κ`` gives every output its own independent variance.

with Model() as model:
    data = Data("inputs", inputs)
    W = RandomParameter("W", Normal(0.0, 1.0), shape=(3, 1), initial_value=torch.tensor([[0.8], [0.6], [-0.4]], dtype=dtype))
    kappa = Exp("kappa", RandomParameter("log_kappa", Normal(-3.0, 1.0), shape=3))
    lengthscale = Exp("lengthscale", RandomParameter("log_lengthscale", Normal(0.0, 0.5)))
    noise = Exp("noise", RandomParameter("log_noise", Normal(-3.0, 1.0)))
    kernel = IntrinsicCoregionalization(Matern52(lengthscale, 1.0), W, kappa)
    f = ExactGP("f", data, kernel, noise_variance=noise)
    ObservedParameter("y", f, y)

trace = model.sample(100, 100, n_chains=1, max_depth=4, seed=1, progress_bar=False)
print(f"divergences: {sum(trace.divergences)}")

# %%
# Predicting the missing part of output 2
# ---------------------------------------
# New inputs are rows ``(x, 2)``. The coregionalisation matrix transfers what
# is learned about the shared signal from outputs 0 and 1.

grid = torch.linspace(0.0, 5.0, 100, dtype=dtype)
new = torch.stack([grid, torch.full_like(grid, 2.0)], dim=-1)
draws = exact_gp_predictive(f, trace, y, new, n_samples=1).reshape(-1, grid.numel())
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=dtype), dim=0)

fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.3, label="90% band, output 2")
ax.plot(grid, median, color="tab:blue", label="posterior median")
ax.plot(x, truth[2], "k--", linewidth=1, label="truth")
ax.scatter(x[:8], y[-8:], color="k", s=14, label="observations")
ax.set(xlabel="x", title="Output 2 observed only on the left, predicted through the coregionalisation")
ax.legend(fontsize="small")
plt.show()

unobserved = grid > x[7]
error = (median - signals(grid)[2])[unobserved].pow(2).mean().sqrt()
print(f"RMSE of the median of output 2 where it was not observed: {error:.3f}")
