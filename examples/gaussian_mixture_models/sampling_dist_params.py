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

    mean1 = RandomParameter("mean1", Normal(0, 10))
    delta_mean = RandomParameter("delta_mean", HalfCauchy(1))
    mean2 = DeterministicParameter(
        "mean2",
        lambda m1, dm: m1 + dm,
        lambda m1, dm: {"mean1": torch.ones_like(m1), "delta_mean": torch.ones_like(dm)},
        [mean1, delta_mean],
    )
    means = [mean1, mean2]
    variances = [RandomParameter("variance" + str(i + 1), Exponential(0.5)) for i in range(K)]

    components = [Normal(mu, cov) for mu, cov in zip(means, variances)]

    with plate("data", n):
        likelihood = ObservedParameter("likelihood", Mixture(components, weights), data)

    plot_model(model)

    # make sampling start close to the maximum a posteriori
    plt.figure()
    plt.plot(model.find_MAP(lr=1e-2, epochs=600, callback_frequency=25))
    print(weights.constrained_value)
    print([mean.constrained_value for mean in means])
    print([variance.constrained_value for variance in variances])
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
plt.hist(result["variance1"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Variance 1")
plt.hist(result["variance2"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Variance 2")
plt.xlabel("Variances")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plot_posterior(result)

plt.show()
