"""
Two Component Gaussian Mixture
==============================

The data come from a mixture of two normal distributions,

.. math::
    y_i \\sim w_0 \\, \\mathcal{N}(\\mu_0, \\sigma_0^2) + w_1 \\, \\mathcal{N}(\\mu_1, \\sigma_1^2),

and we infer the weights, means and standard deviations with NUTS. A mixture
likelihood is symmetric under relabelling of its components, so every
posterior mode appears twice. Chains that settle on different labellings
disagree, and R-hat flags the run although every chain is fine on its own.

Mixtures also have degenerate secondary modes: both components sit on the
same cluster while one of them widens to cover the other cluster too. A chain
that falls into such a mode during warmup adapts its step size there and stays.

Three choices make all chains sample the same, single mode:

* **Ordered means.** ``Ordered(Normal(0, 5), 2)`` is a vector with
  ``mu_0 < mu_1`` whose elements have the ``Normal(0, 5)`` prior. The sampler
  works in the unconstrained coordinates ``(mu_0, log(mu_1 - mu_0))``, so the
  constraint is never violated and the mirrored mode is gone.
* **Log-scale standard deviations.** ``log sigma_k ~ Normal(0, 1)`` and the
  ``Exp`` block maps them back. The sampler moves in ``log sigma``, where the
  posterior is close to normal, and the prior keeps the scales moderate.
* **Start at the posterior mode.** ``find_MAP`` finds the mode and the chains
  start close to it (``start_point_variance=0.1`` in the unconstrained space).
  With the default jitter of one unconstrained unit, a start point now and
  then lands in the basin of a degenerate mode, and that chain never joins the
  others.

Component means and standard deviations are taken from the vectors with
``Index`` blocks.
"""

import math

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Deterministic import Exp, Index
from BayesianDLL.Distributions import Dirichlet, Mixture, Normal, Ordered


# %%
# Data
# ----
# 200 draws from a mixture of a narrow cluster at -1 and a wider one at 2.

torch.manual_seed(0)

K = 2
n = 200
true_weights = torch.tensor([0.6, 0.4], dtype=torch.float64)
true_means = torch.tensor([-1.0, 2.0], dtype=torch.float64)
true_stds = torch.tensor([0.5, 1.0], dtype=torch.float64)

labels = torch.multinomial(true_weights, n, replacement=True)
data = true_means[labels] + true_stds[labels] * torch.randn(n, dtype=torch.float64)


# %%
# Model
# -----

with Model() as model:
    weights = RandomParameter("weights", Dirichlet(torch.full((K,), 2.0, dtype=torch.float64)))
    means = RandomParameter("means", Ordered(Normal(0.0, 5.0), K))
    log_stds = RandomParameter("log_stds", Normal(0.0, 1.0), shape=K)
    stds = Exp("stds", log_stds)

    components = [
        Normal(Index(f"mean_{k}", means, [k]), Index(f"std_{k}", stds, [k]))
        for k in range(K)
    ]
    with plate("data", n):
        ObservedParameter("y", Mixture(components, weights), data)


# %%
# Sampling
# --------
# Optimise to the posterior mode first, then start four chains next to it.

model.find_MAP(lr=5e-2, epochs=400, verbose=False)
print("Mode:", {name: [round(v, 3) for v in p.constrained_value.flatten().tolist()] for name, p in model.params.items()})

result = model.sample(500, 400, n_chains=4, start_point_variance=0.1, seed=1, progress_bar=False)
print(result.summary())
print("Divergences per chain:", result.divergences)


# %%
# Fitted density
# --------------
# The mixture density for every posterior draw gives the posterior mean density
# and a 90 % credible band. The components are drawn with their posterior mean
# weights, means and standard deviations.

def normal_pdf(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


w = result["weights"].flatten(0, 1)  # (draws, K)
mu = result["means"].flatten(0, 1)
sigma = result["stds"].flatten(0, 1)

grid = torch.linspace(data.min().item() - 1, data.max().item() + 1, 400, dtype=torch.float64)
component_pdfs = w[:, None, :] * normal_pdf(grid[None, :, None], mu[:, None, :], sigma[:, None, :])  # (draws, grid, K)
mixture_pdf = component_pdfs.sum(-1)
lower, upper = torch.quantile(mixture_pdf, torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
true_pdf = (true_weights * normal_pdf(grid[:, None], true_means, true_stds)).sum(-1)

colors = ["tab:blue", "tab:orange"]
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.hist(data.numpy(), bins=35, density=True, color="0.85", edgecolor="white", label="Data")
ax.fill_between(grid.numpy(), lower.numpy(), upper.numpy(), color="tab:green", alpha=0.3, label="90 % band")
ax.plot(grid.numpy(), mixture_pdf.mean(0).numpy(), color="tab:green", lw=2, label="Posterior mean")
for k in range(K):
    ax.plot(grid.numpy(), component_pdfs[..., k].mean(0).numpy(), color=colors[k], lw=1.5, ls="--", label=f"Component {k}")
ax.plot(grid.numpy(), true_pdf.numpy(), color="black", lw=1.5, ls=":", label="True density")
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title("Data and fitted mixture")
ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


# %%
# Agreement between chains
# ------------------------
# Trace and posterior density of every parameter, one colour per chain. The
# chains overlap, and the ordering keeps mean 0 below mean 1 in all of them.

n_chains = result["means"].shape[0]
truths = {"means": true_means, "stds": true_stds, "weights": true_weights}
rows = [(name, k) for name in ("means", "stds", "weights") for k in range(K)]
fig, axes = plt.subplots(len(rows), 2, figsize=(11, 11), constrained_layout=True,
                         gridspec_kw={"width_ratios": [2, 1]})
for row, (name, k) in enumerate(rows):
    label = f"{name[:-1]} {k}"
    for c in range(n_chains):
        draws = result[name][c, :, k].numpy()
        axes[row, 0].plot(draws, lw=0.5, alpha=0.8, color=f"C{c}", label=f"Chain {c}")
        axes[row, 1].hist(draws, bins=30, density=True, histtype="step", lw=1.2, color=f"C{c}")
    axes[row, 0].axhline(truths[name][k].item(), color="black", ls=":", lw=1.5, label="True value")
    axes[row, 1].axvline(truths[name][k].item(), color="black", ls=":", lw=1.5)
    axes[row, 0].set_ylabel(label)
    axes[row, 1].set_ylabel("Density")
    if row > 0:
        axes[row, 0].sharex(axes[0, 0])
    axes[row, 0].tick_params(axis="x", labelbottom=row == len(rows) - 1)
    axes[row, 1].set_xlabel(label)
axes[-1, 0].set_xlabel("Draw")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=n_chains + 1, fontsize="small")

plt.show()
