"""
Softmax Regression
==================

Multinomial logistic regression for ``K = 3`` classes and two features:

.. math::

    \\eta_i = x_i W + b, \\quad y_i \\sim \\text{Categorical}(\\operatorname{softmax}(\\eta_i)).

The logits are built with ``MatMul`` and ``Add`` and passed directly to
``Categorical(logits=...)``, which evaluates the softmax in a numerically
stable way. Adding the same value to the logits of every class does not change
the probabilities, so the logits of the last (reference) class are fixed to
zero with ``Concat``; this removes the non-identified direction from the
posterior. The posterior mean class probabilities are shown on a grid.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, Concat, MatMul
from BayesianDLL.Distributions import Categorical, Normal


torch.manual_seed(2)

n_per_class, n_classes = 50, 3
centres = torch.tensor([[-1.5, -1.0], [1.5, -0.5], [0.0, 1.5]], dtype=torch.float64)
features = torch.cat([centre + 0.9 * torch.randn(n_per_class, 2, dtype=torch.float64) for centre in centres])
labels = torch.arange(n_classes).repeat_interleave(n_per_class).to(torch.float64)
n_observations = len(labels)

with Model() as model:
    X = Data("X", features)  # (N, 2)
    W = RandomParameter("W", Normal(0.0, 2.0), shape=(2, n_classes - 1))
    b = RandomParameter("b", Normal(0.0, 2.0), shape=(n_classes - 1,))

    with plate("observations", n_observations):
        free_logits = Add("free_logits", MatMul("XW", X, W), b)  # (N, K - 1)
        logits = Concat("logits", free_logits, torch.zeros(n_observations, 1, dtype=torch.float64), dim=-1)
        ObservedParameter("y", Categorical(logits=logits), labels)

trace = model.sample(n_samples=500, warmup_length=500, n_chains=2, progress_bar=False, seed=0)
print(trace.summary())
print("divergences:", sum(trace.divergences))

# %%
# Posterior mean class probabilities
# ----------------------------------

grid_1d = torch.linspace(-4.0, 4.0, 120, dtype=torch.float64)
g1, g2 = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (G, 2)

W_draws = trace["W"].flatten(0, 1)  # (S, 2, K - 1)
b_draws = trace["b"].flatten(0, 1)  # (S, K - 1)
grid_logits = grid @ W_draws + b_draws.unsqueeze(1)  # (S, G, K - 1)
grid_logits = torch.cat([grid_logits, torch.zeros_like(grid_logits[..., :1])], dim=-1)
probabilities = torch.softmax(grid_logits, dim=-1).mean(0)  # (G, K)

train_logits = features @ W_draws + b_draws.unsqueeze(1)
train_logits = torch.cat([train_logits, torch.zeros_like(train_logits[..., :1])], dim=-1)
train_accuracy = (torch.softmax(train_logits, -1).mean(0).argmax(-1) == labels.long()).double().mean()
print(f"training accuracy of the posterior mean probabilities: {train_accuracy.item():.3f}")

colors = torch.tensor([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]], dtype=torch.float64)
image = (probabilities @ colors).reshape(*g1.shape, 3)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(image.clamp(0, 1), origin="lower", extent=(-4, 4, -4, 4), alpha=0.5)
ax.contour(g1, g2, probabilities.max(-1).values.reshape(g1.shape), levels=[0.5, 0.8], colors="black", linewidths=0.6)
for k in range(n_classes):
    mask = labels == k
    ax.scatter(features[mask, 0], features[mask, 1], color=colors[k].tolist(), edgecolor="black", s=20, label=f"class {k}")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.set_title("Posterior mean class probabilities")
ax.legend()
plt.show()
