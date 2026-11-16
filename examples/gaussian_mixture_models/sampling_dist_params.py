"""
Gaussian Mixture Model
======================

Infers the mixture weights, means and standard deviations of a two component Gaussian mixture.
A ``DeterministicParameter`` keeps the component means ordered to avoid label
switching.
"""

import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Dirichlet, Normal, Mixture, Exponential, HalfCauchy
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, plate
from BayesianDLL.Evaluation.Graphics import plot_posterior, plot_model


torch.manual_seed(0)

K = 2
n = 200
true_weights = [0.6, 0.4]
data = torch.cat([
    torch.normal(-1, 0.5, size=(int(true_weights[0] * n),), dtype=torch.float64),
    torch.normal(2, 1, size=(int(true_weights[1] * n),), dtype=torch.float64)
])

with Model() as model:
    alpha = 1.5 * torch.ones(K, dtype=torch.float64)
    weights = RandomParameter("weights", Dirichlet(alpha), shape=K)

    mean1 = RandomParameter("mean1", Normal(0, 3))
    delta_mean = RandomParameter("delta_mean", HalfCauchy(1))
    mean2 = DeterministicParameter(
        "mean2",
        lambda m1, dm: m1 + dm,
        lambda m1, dm: {"mean1": torch.ones_like(m1), "delta_mean": torch.ones_like(dm)},
        [mean1, delta_mean],
    )
    means = [mean1, mean2]
    stds = [RandomParameter("std" + str(i + 1), Exponential(1.0)) for i in range(K)]

    components = [Normal(mu, std) for mu, std in zip(means, stds)]

    with plate("data", n):
        likelihood = ObservedParameter("likelihood", Mixture(components, weights), data)

    plot_model(model)

    # make sampling start close to the maximum a posteriori
    plt.figure()
    plt.plot(model.find_MAP(lr=1e-2, epochs=600, callback_frequency=25))
    print(weights.constrained_value)
    print([mean.constrained_value for mean in means])
    print([std.constrained_value for std in stds])
    plt.show()

    result = model.sample(500, 200, max_depth=4)
    result = result.thin(2)

plt.figure()
for k in range(K):
    plt.hist(result["weights"].flatten(0, 1)[:, k].numpy(), bins=30, density=True, alpha=0.6, label=f"Weight {k}")
plt.xlabel("Weight value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(result["mean1"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Mean 1")
plt.hist(result["mean2"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Mean 2")
plt.xlabel("Means")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(result["std1"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Std 1")
plt.hist(result["std2"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Std 2")
plt.xlabel("Standard deviations")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plot_posterior(result)

plt.show()
