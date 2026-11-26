"""
Multi-Output GPs with Coregionalisation
=======================================

Three related signals are measured at different times: the first two densely,
the third (a sensor that failed early) only on the left part of the
interval. Modelling the outputs jointly lets the dense outputs inform the
missing part of the third.

The intrinsic coregionalisation model (ICM) writes the covariance between
output ``i`` at ``x`` and output ``j`` at ``x'`` as

``k((x, i), (x', j)) = B[i, j] k(x, x')``,  ``B = W Wᵀ + diag(κ)``,

with a shared kernel ``k`` and a learned ``3 x 3`` matrix ``B``. Every output
then has the same length-scale. The linear model of coregionalisation (LMC)
sums several such terms, each with its own latent kernel and its own matrix,

``k((x, i), (x', j)) = Σ_q B_q[i, j] k_q(x, x')``,

so that the outputs can mix a smooth and a fast latent process in different
proportions. The observations of all outputs are stacked into one vector, and
every input row holds the feature ``x`` and the integer index of the output in
the last column. Both models are fitted by type-II MAP.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, IntrinsicCoregionalization, LinearCoregionalization, Matern52

torch.manual_seed(0)
dtype = torch.float64


def signals(x):
    smooth = 1.5 * torch.sin(0.8 * x) + 0.3 * x
    fast = 0.8 * torch.sin(6.0 * x)
    return torch.stack([smooth + fast, 0.8 * smooth - 0.5 * fast, -0.6 * smooth - 0.7 * fast])


# %%
# Data
# ----
# All three outputs are mixtures of a slow trend and a fast oscillation (period
# about one unit), with different weights and signs.
# Outputs 0 and 1 are observed at 40 points, output 2 only at the first 16
# (``x < 2``).

x = torch.linspace(0.0, 5.0, 40, dtype=dtype)
truth = signals(x)
observed = [40, 40, 16]
rows, values = [], []
for output, count in enumerate(observed):
    rows.append(torch.stack([x[:count], torch.full((count,), float(output), dtype=dtype)], dim=-1))
    values.append(truth[output, :count] + 0.1 * torch.randn(count, dtype=dtype))
inputs, y = torch.cat(rows), torch.cat(values)

# %%
# The models
# ----------
# The ICM has rank-1 ``W`` (the outputs are multiples of one latent signal)
# and ``κ`` gives every output its own independent variance. The LMC has two
# latent processes, each with a rank-1 ``W_q``, ``κ_q`` and a length-scale
# of its own. The initial mixing weights only break the symmetry between the
# two processes.


def build(kind):
    with Model() as model:
        data = Data("inputs", inputs)
        noise = Exp("noise", RandomParameter("log_noise", Normal(-3.0, 1.0)))
        if kind == "ICM":
            W = RandomParameter("W", Normal(0.0, 1.0), shape=(3, 1), initial_value=torch.tensor([[0.8], [0.6], [-0.4]], dtype=dtype))
            kappa = Exp("kappa", RandomParameter("log_kappa", Normal(-3.0, 1.0), shape=3))
            lengthscale = Exp("lengthscale", RandomParameter("log_lengthscale", Normal(0.0, 0.5)))
            kernel = IntrinsicCoregionalization(Matern52(lengthscale, 1.0), W, kappa)
        else:
            weights = [
                RandomParameter(f"W{q}", Normal(0.0, 1.0), shape=(3, 1), initial_value=torch.tensor(start, dtype=dtype))
                for q, start in enumerate([[[0.8], [0.6], [-0.4]], [[0.5], [-0.5], [0.3]]])
            ]
            kappas = [Exp(f"kappa{q}", RandomParameter(f"log_kappa{q}", Normal(-3.0, 1.0), shape=3)) for q in range(2)]
            lengthscales = [Exp(f"lengthscale{q}", RandomParameter(f"log_lengthscale{q}", Normal(mean, 0.5))) for q, mean in enumerate([0.7, -1.2])]
            kernel = LinearCoregionalization([Matern52(l, 1.0) for l in lengthscales], weights, kappas)
        f = ExactGP("f", data, kernel, noise_variance=noise)
        ObservedParameter("y", f, y)
    return model, f


grid = torch.linspace(0.0, 5.0, 150, dtype=dtype)
new = torch.stack([grid, torch.full_like(grid, 2.0)], dim=-1)
unobserved = grid > x[15]
results = {}
for kind in ("ICM", "LMC"):
    model, f = build(kind)
    map_result = model.find_MAP(lr=0.05, epochs=500, verbose=False)
    log_density = f.log_pdf(y).item()
    # Output 2 at new inputs ``(x, 2)``: the coregionalisation transfers what is
    # learned about the shared signals from outputs 0 and 1.
    model.data["inputs"].set_value(new)
    draws = model.predict(map_result.mode(), samples_per_step=300)["f"].reshape(-1, grid.numel())
    model.data["inputs"].set_value(inputs)
    band = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=dtype), dim=0)
    error = (band[1] - signals(grid)[2])[unobserved].pow(2).mean().sqrt()
    results[kind] = band
    print(f"{kind}: log marginal likelihood {log_density:.1f}, RMSE of the median of output 2 where it was not observed: {error:.3f}")
    if kind == "LMC":
        print("fitted length-scales:", ", ".join(f"{model.deterministic_params[f'lengthscale{q}'].constrained_value.item():.2f}" for q in range(2)))

# %%
# Predicting the missing part of output 2
# ---------------------------------------
# A single length-scale cannot describe both components: the ICM settles on a
# long one and smooths over the fast oscillation, even where output 2 is
# observed, with a much lower marginal likelihood. The LMC gives each latent
# process its own length-scale and recovers both components in the missing
# part of output 2.

fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained", sharey=True)
for ax, (kind, (lower, median, upper)) in zip(axes, results.items()):
    ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.3, label="90% band, output 2")
    ax.plot(grid, median, color="tab:blue", label="posterior median")
    ax.plot(grid, signals(grid)[2], "k--", linewidth=1, label="truth")
    ax.scatter(x[:16], y[-16:], color="k", s=14, label="observations")
    ax.set(xlabel="x", title=f"{kind}: output 2 observed only for x < 2")
axes[0].legend(fontsize="small")
plt.show()
