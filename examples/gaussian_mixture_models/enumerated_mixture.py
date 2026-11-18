"""
Mixture with Enumerated Assignments
===================================

A mixture model can be written with an explicit component label ``z_i`` for
every observation:

.. math::
    z_i \\sim \\mathrm{Bernoulli}(w), \\qquad
    y_i \\mid z_i \\sim \\mathcal{N}(\\mu_{z_i}, \\sigma_{z_i}^2).

The labels are discrete, so NUTS cannot sample them. Declaring them with
``enumerate=True`` sums them out of the joint density exactly: each ``z_i``
only touches ``y_i`` (they live in a plate), so the sum over all ``2 ** n``
label vectors factorises into ``n`` sums of two terms. NUTS samples ``w``,
``mu`` and ``sigma`` from the marginal posterior, and after every draw the
library records the probability of each label and a draw of ``z``.

The component parameters of each observation are picked out of the vectors
``mu`` and ``sigma`` with ``Index`` blocks indexed by ``z``. The means are an
``Ordered`` vector (``mu_0 < mu_1``), which removes the mirrored mode in which
the two labels swap roles, so all chains describe the same labelling.
"""

import math

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Deterministic import Exp, Index
from BayesianDLL.Distributions import Bernoulli, Beta, Normal, Ordered
from BayesianDLL.Evaluation import summary


# %%
# Data
# ----
# 150 observations; about a third belong to the second component.

torch.manual_seed(0)

n = 150
true_weight = 0.35
true_means = torch.tensor([-2.0, 2.5], dtype=torch.float64)
true_stds = torch.tensor([1.0, 0.8], dtype=torch.float64)
true_labels = (torch.rand(n, dtype=torch.float64) < true_weight).long()
data = true_means[true_labels] + true_stds[true_labels] * torch.randn(n, dtype=torch.float64)


# %%
# Model
# -----

with Model() as model:
    weight = RandomParameter("weight", Beta(2.0, 2.0))  # P(z_i = 1)
    means = RandomParameter("means", Ordered(Normal(0.0, 5.0), 2))
    log_stds = RandomParameter("log_stds", Normal(0.0, 1.0), shape=2)
    stds = Exp("stds", log_stds)

    with plate("data", n):
        z = RandomParameter("z", Bernoulli(weight), shape=(), enumerate=True)
        ObservedParameter("y", Normal(Index("mean", means, z), Index("std", stds, z)), data)


# %%
# Sampling
# --------
# The chains start next to the posterior mode of the continuous parameters.

model.find_MAP(lr=5e-2, epochs=300, verbose=False)
result = model.sample(400, 300, n_chains=4, start_point_variance=0.1, seed=0, progress_bar=False)
print(summary({name: result[name] for name in ("weight", "means", "stds")}))
print("Divergences per chain:", result.divergences)

# P(z_i = 1 | data): the label probabilities averaged over the posterior draws.
responsibility = result.enumerated_probs["z"][..., 1].mean(dim=(0, 1))
accuracy = ((responsibility > 0.5).long() == true_labels).double().mean()
print(f"Observations assigned to their true component: {accuracy.item():.1%}")


# %%
# Fit and assignments
# -------------------
# Left: the data with the mixture density of every draw summarised by its mean
# and a 90 % credible band, the two weighted components and the density the
# data were drawn from. Right: the label probability of each observation
# against its value, with the true label as the marker.

def normal_pdf(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


w = result["weight"].flatten(0, 1)  # (draws, 1)
w = torch.cat([1 - w, w], dim=-1)  # weights of components 0 and 1
mu = result["means"].flatten(0, 1)
sigma = result["stds"].flatten(0, 1)

grid = torch.linspace(data.min().item() - 1, data.max().item() + 1, 400, dtype=torch.float64)
component_pdfs = w[:, None, :] * normal_pdf(grid[None, :, None], mu[:, None, :], sigma[:, None, :])
mixture_pdf = component_pdfs.sum(-1)
lower, upper = torch.quantile(mixture_pdf, torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
true_w = torch.tensor([1 - true_weight, true_weight], dtype=torch.float64)
true_pdf = (true_w * normal_pdf(grid[:, None], true_means, true_stds)).sum(-1)

colors = ["tab:blue", "tab:orange"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
ax = axes[0]
ax.hist(data.numpy(), bins=30, density=True, color="0.85", edgecolor="white", label="Data")
ax.fill_between(grid.numpy(), lower.numpy(), upper.numpy(), color="tab:green", alpha=0.3, label="90 % band")
ax.plot(grid.numpy(), mixture_pdf.mean(0).numpy(), color="tab:green", lw=2, label="Posterior mean")
for k in range(2):
    ax.plot(grid.numpy(), component_pdfs[..., k].mean(0).numpy(), color=colors[k], ls="--", label=f"Component {k}")
ax.plot(grid.numpy(), true_pdf.numpy(), color="black", ls=":", lw=1.5, label="True density")
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title("Data and fitted mixture")
ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

ax = axes[1]
for label, marker in [(0, "o"), (1, "^")]:
    mask = true_labels == label
    points = ax.scatter(data[mask].numpy(), responsibility[mask].numpy(), c=responsibility[mask].numpy(),
                        cmap="coolwarm", vmin=0, vmax=1, marker=marker, s=22, edgecolor="0.3", linewidth=0.3,
                        label=f"True label {label}")
fig.colorbar(points, ax=ax, label="P(z = 1 | data)")
ax.set_xlabel("y")
ax.set_ylabel("P(z = 1 | data)")
ax.set_ylim(-0.08, 1.08)
ax.set_title(f"Label probabilities ({accuracy.item():.0%} assigned correctly)")
ax.legend(fontsize="small", loc="center left")


# %%
# Agreement between chains
# ------------------------
# Trace and posterior density of the parameters, one colour per chain.

n_chains = result["means"].shape[0]
rows = [("means", 0, true_means[0]), ("means", 1, true_means[1]),
        ("stds", 0, true_stds[0]), ("stds", 1, true_stds[1]), ("weight", 0, torch.tensor(true_weight))]
fig, axes = plt.subplots(len(rows), 2, figsize=(11, 10), constrained_layout=True,
                         gridspec_kw={"width_ratios": [2, 1]})
for row, (name, k, truth) in enumerate(rows):
    label = "weight" if name == "weight" else f"{name[:-1]} {k}"
    for c in range(n_chains):
        draws = result[name][c, :, k].numpy()
        axes[row, 0].plot(draws, lw=0.5, alpha=0.8, color=f"C{c}", label=f"Chain {c}")
        axes[row, 1].hist(draws, bins=30, density=True, histtype="step", lw=1.2, color=f"C{c}")
    axes[row, 0].axhline(truth.item(), color="black", ls=":", lw=1.5, label="True value")
    axes[row, 1].axvline(truth.item(), color="black", ls=":", lw=1.5)
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
