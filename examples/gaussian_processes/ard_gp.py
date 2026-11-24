"""
Finding Relevant Inputs with ARD
================================

A response depends on three measured inputs, but in truth only two of them
matter: ``f(x) = sin(2 x_0) + 0.5 x_1``, and ``x_2`` is unrelated. Can the
model discover this by itself?

An RBF kernel with *automatic relevance determination* (ARD) gives every
input its own length-scale ``l_d``:
``k(x, x') = v exp(-½ Σ_d (x_d - x'_d)² / l_d²)``. A short ``l_d`` lets the
function change quickly along input ``d``; a very long one makes the
function flat in that direction, which effectively removes the input. The
three log-length-scales are one vector parameter (``shape=3``) mapped through
``Exp``; their gradients are hand-derived inside the kernel.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP


torch.manual_seed(0)

n, n_inputs = 60, 3


def true_function(x):
    return torch.sin(2.0 * x[:, 0]) + 0.5 * x[:, 1]


x = torch.rand(n, n_inputs, dtype=torch.float64) * 4.0 - 2.0
y = true_function(x) + 0.1 * torch.randn(n, dtype=torch.float64)

# %%
# The model
# ---------
# A wide prior on the log-length-scales (standard deviation 1.5, i.e. a
# factor of about 20 either way) leaves room for both short and very long
# length-scales, so the data decide which inputs matter.

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.5), shape=n_inputs)
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_noise = RandomParameter("log_noise", Normal(-3.0, 1.0))
    f = ExactGP(
        "f",
        inputs,
        RBF(lengthscale=Exp("lengthscale", log_lengthscale), variance=Exp("variance", log_variance)),
        noise_variance=Exp("noise", log_noise),
    )
    ObservedParameter("y", f, y)

start = time.perf_counter()
trace = model.sample(300, 300, n_chains=2, max_depth=6, seed=1, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary()
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

# %%
# Relevance and partial dependence
# --------------------------------
# Left: the posterior length-scale of each input on a log axis; the
# irrelevant input is pushed far beyond the data range (width 4). Right:
# slices through the fitted function, varying one input while the other two
# are held at zero, compared with the true function along the same slice.

lengthscales = trace["lengthscale"].reshape(-1, n_inputs)
grid = torch.linspace(-2.0, 2.0, 60, dtype=torch.float64)
thinned = trace.thin(10)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
axes[0].boxplot([lengthscales[:, d].log10().numpy() for d in range(n_inputs)], showfliers=False)
axes[0].set_xticks(range(1, n_inputs + 1), [f"x{d}" for d in range(n_inputs)])
axes[0].axhline(torch.tensor(4.0).log10(), color="0.5", linestyle=":")
axes[0].set(ylabel="log10 length-scale", title="Posterior length-scales")

for d in range(n_inputs):
    ax = axes[d + 1]
    points = torch.zeros(grid.numel(), n_inputs, dtype=torch.float64)
    points[:, d] = grid
    inputs.set_value(points)
    draws = model.posterior_predicative(thinned, samples_per_step=5, progress_bar=False)["f"].reshape(-1, grid.numel())
    lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)
    ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.25, label="90% band")
    ax.plot(grid, median, color="tab:blue", label="median")
    ax.plot(grid, true_function(points), "k--", label="truth")
    ax.set(xlabel=f"x{d} (others = 0)", title=f"Slice along x{d}")
    if d > 0:
        ax.sharey(axes[1])
axes[1].set_ylabel("f")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize="small")
plt.show()
