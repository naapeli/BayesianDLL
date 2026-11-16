"""
Mixture Model with Enumerated Assignments
=========================================

Fits a two component Gaussian mixture written with an explicit assignment
``z_i`` for every observation. The assignments are created with
``enumerate=True``, so they are summed out of the model exactly: NUTS samples
the mixture weight and the component means from the marginal posterior, and a
posterior draw of every ``z_i`` (together with its probability, the
responsibility of the second component) is recovered after each draw.

The marginalisation exploits the plate: every ``z_i`` only affects ``y_i``, so
the sum over all ``2 ** n`` assignments factorises into ``n`` sums with two
terms each.
"""

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, plate
from BayesianDLL.Distributions import Bernoulli, Beta, Normal


torch.manual_seed(0)

n = 150
true_weight = 0.35
true_means = (-2.0, 2.5)
assignments = (torch.rand(n) < true_weight).double()
data = torch.where(assignments == 1, true_means[1], true_means[0]) + torch.randn(n, dtype=torch.float64)

with Model() as model:
    weight = RandomParameter("weight", Beta(2.0, 2.0))
    # Slightly different prior means break the symmetry between the components.
    mean0 = RandomParameter("mean0", Normal(-1.0, 1.0))
    mean1 = RandomParameter("mean1", Normal(1.0, 1.0))

    with plate("data", n):
        # z_i = 1 assigns observation i to the second component.
        z = RandomParameter("z", Bernoulli(weight), shape=(), enumerate=True)
        mean = DeterministicParameter(
            "mean",
            lambda m0, m1, z: m0 + (m1 - m0) * z,
            lambda m0, m1, z: {"mean0": 1 - z, "mean1": z},
            [mean0, mean1, z],
        )
        ObservedParameter("y", Normal(mean, 1.0), data)

result = model.sample(500, 300, n_chains=2, seed=0)
print(result.summary())

plt.figure()
plt.hist(result["mean0"].flatten().numpy(), bins=30, density=True, alpha=0.6, label="Mean 0")
plt.hist(result["mean1"].flatten().numpy(), bins=30, density=True, alpha=0.6, label="Mean 1")
for value in true_means:
    plt.axvline(value, color="black", linestyle="--")
plt.xlabel("Component mean")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

# Posterior probability that each observation belongs to the second component,
# averaged over the draws of the continuous parameters.
responsibility = result.enumerated_probs["z"][..., 1].mean(dim=(0, 1))
plt.figure()
plt.scatter(data.numpy(), responsibility.numpy(), c=assignments.numpy(), cmap="coolwarm", s=12)
plt.xlabel("Observation")
plt.ylabel("P(z = 1 | data)")
plt.title("Recovered assignments (colour: true component)")
plt.tight_layout()

accuracy = ((responsibility > 0.5).double() == assignments).double().mean()
print(f"Fraction of correctly assigned observations: {accuracy.item():.3f}")

plt.show()
