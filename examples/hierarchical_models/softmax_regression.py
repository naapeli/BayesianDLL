"""
Softmax Regression for Three Classes
====================================

Points in the plane belong to one of ``K = 3`` classes. Multinomial logistic
regression gives every class a linear score (logit) and turns the scores
into class probabilities with the softmax:

.. math::

    \\eta_i = x_i W + b, \\quad y_i \\sim \\text{Categorical}(\\operatorname{softmax}(\\eta_i)),
    \\quad W_{dk}, b_k \\sim N(0, 2).

**Identifiability.** Adding the same number to all ``K`` logits leaves the
probabilities unchanged, so with ``K`` free score functions the posterior
would have a flat direction. The logits of the last class are therefore
fixed at zero: ``W`` and ``b`` have only ``K - 1`` columns, and ``Concat``
appends a column of zeros. The other classes' coefficients are then log-odds
against the reference class.

**Scaling.** The two features are standardised, so ``N(0, 2)`` is a weakly
informative prior for log-odds per standard deviation, and the coefficients
are on comparable scales, which suits the diagonal mass matrix of NUTS.

The logits come from ``MatMul`` and ``Add`` and go straight into
``Categorical(logits=...)``, which evaluates the log-softmax stably.

The figure shows the data with the posterior mean class probabilities as a
colour mix and the decision boundaries (where two classes are equally
likely), next to the posterior standard deviation of the probability of the
predicted class, which is largest along the boundaries and away from the
data.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, Concat, MatMul
from BayesianDLL.Distributions import Categorical, Normal


torch.manual_seed(2)

n_per_class, n_classes = 50, 3
centres = torch.tensor([[-1.5, -1.0], [1.5, -0.5], [0.0, 1.5]], dtype=torch.float64)
raw_features = torch.cat([centre + 0.9 * torch.randn(n_per_class, 2, dtype=torch.float64) for centre in centres])
labels = torch.arange(n_classes).repeat_interleave(n_per_class)
n_observations = len(labels)
feature_mean, feature_std = raw_features.mean(0), raw_features.std(0)
features = (raw_features - feature_mean) / feature_std

with Model() as model:
    X = Data("X", features)  # (N, 2)
    W = RandomParameter("W", Normal(0.0, 2.0), shape=(2, n_classes - 1))
    b = RandomParameter("b", Normal(0.0, 2.0), shape=(n_classes - 1,))

    with plate("observations", n_observations):
        free_logits = Add("free_logits", MatMul("XW", X, W), b)  # (N, K - 1)
        logits = Concat("logits", free_logits, torch.zeros(n_observations, 1, dtype=torch.float64), dim=-1)  # (N, K)
        ObservedParameter("y", Categorical(logits=logits), labels.double())

trace = model.sample(n_samples=500, warmup_length=500, n_chains=4, progress_bar=False, seed=0, delta=0.9,
                     chain_method="vectorized")
summary = trace.summary()
print(summary)
print(f"divergent transitions: {sum(trace.divergences)}, largest R-hat: {summary['r_hat'].max():.3f}")

# %%
# Class probabilities and their uncertainty
# -----------------------------------------
# Every posterior draw of ``(W, b)`` gives a probability map; the figures
# summarise the maps of all draws on a grid in the original feature units.

W_draws = trace["W"].flatten(0, 1)  # (S, 2, K - 1)
b_draws = trace["b"].flatten(0, 1)  # (S, K - 1)


def class_probabilities(points):
    """Posterior draws of the class probabilities at ``points`` (original units), shape (S, P, K)."""
    standardised = (points - feature_mean) / feature_std
    free = standardised @ W_draws + b_draws.unsqueeze(1)
    return torch.softmax(torch.cat([free, torch.zeros_like(free[..., :1])], dim=-1), dim=-1)


train_probabilities = class_probabilities(raw_features).mean(0)
accuracy = (train_probabilities.argmax(-1) == labels).double().mean()
print(f"training accuracy of the posterior mean probabilities: {accuracy.item():.3f}")

grid_1d = torch.linspace(-4.0, 4.0, 120, dtype=torch.float64)
g1, g2 = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([g1.flatten(), g2.flatten()], dim=-1)  # (G, 2)
probabilities = class_probabilities(grid)  # (S, G, K)
mean_probabilities = probabilities.mean(0)  # (G, K)
predicted = mean_probabilities.argmax(-1)
uncertainty = probabilities.gather(-1, predicted.expand(probabilities.shape[0], -1).unsqueeze(-1)).squeeze(-1).std(0)

colors = torch.tensor([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17]], dtype=torch.float64)
image = (mean_probabilities @ colors).reshape(*g1.shape, 3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
axes[0].imshow(image.clamp(0, 1), origin="lower", extent=(-4, 4, -4, 4), alpha=0.5)
for k in range(n_classes):
    # Boundary of class k: where its probability equals the largest of the others.
    others = torch.cat([mean_probabilities[:, :k], mean_probabilities[:, k + 1:]], dim=-1).max(-1).values
    axes[0].contour(g1, g2, (mean_probabilities[:, k] - others).reshape(g1.shape), levels=[0.0], colors="black", linewidths=1.2)
axes[0].set_title("Posterior mean class probabilities\n(colour mix) and decision boundaries")

panel = axes[1].contourf(g1, g2, uncertainty.reshape(g1.shape), levels=15, cmap="Greys")
fig.colorbar(panel, ax=axes[1], label="posterior sd of P(predicted class)")
axes[1].set_title("Uncertainty of the predicted class probability")

for ax in axes:
    for k in range(n_classes):
        mask = labels == k
        ax.scatter(raw_features[mask, 0], raw_features[mask, 1], color=colors[k].tolist(), edgecolor="black", s=18, label=f"class {k}")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="outside lower center", ncol=3, fontsize="small")
plt.show()
