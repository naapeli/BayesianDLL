import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Dirichlet, Normal, Mixture, Exponential
from BayesianDLL import Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Evaluation.Graphics import plot_posterior, plot_model


torch.manual_seed(0)

K = 2
n = 200
true_weights = [0.6, 0.4]
data = torch.cat([
    torch.normal(-1, 0.5, size=(int(true_weights[0] * n),), dtype=torch.float64),
    torch.normal(2, 1, size=(int(true_weights[1] * n),), dtype=torch.float64)
])

sampler_params = {"min_step_size": 1e-6, "max_step_size": 10, "delta": 0.6, "gamma": 0.5}
weight_sampler_params = {"min_step_size": 1e-6, "max_step_size": 10, "delta": 0.6, "gamma": 0.5, "max_depth": 4}

with Model() as model:
    alpha = 1.5 * torch.ones(K, dtype=torch.float64)
    weights = RandomParameter("weights", Dirichlet(alpha), shape=K, **weight_sampler_params)

    means = [RandomParameter("mean" + str(i + 1), Normal(0, 10), **sampler_params) for i in range(K)]
    variances = [RandomParameter("variance" + str(i + 1), Exponential(0.5), **sampler_params) for i in range(K)]

    components = [Normal(mu, cov) for mu, cov in zip(means, variances)]

    with plate("data", n):
        likelihood = ObservedParameter("likelihood", Mixture(components, weights), data)

    plt.figure()
    plot_model(model)

    # make sampling start close to the maximum a posteriori
    plt.figure()
    plt.plot(model.find_MAP(lr=1e-2, epochs=600, callback_frequency=25))
    print(weights.constrained_value)
    print([mean.constrained_value for mean in means])
    print([variance.constrained_value for variance in variances])
    plt.show()

    result = model.sample(400, 100, start_point_variance=1e-3)  # decrease the starting point variance as different chains may swap the order of the two groups otherwise
    samples = result.trace

plt.figure()
for k in range(K):
    plt.hist(samples["weights"].flatten(0, 1)[:, k].numpy(), bins=30, density=True, alpha=0.6, label=f"Weight {k}")
plt.xlabel("Weight value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(samples["mean1"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Mean 1")
plt.hist(samples["mean2"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Mean 2")
plt.xlabel("Means")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(samples["variance1"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Variance 1")
plt.hist(samples["variance2"].flatten(0, 1).numpy(), bins=30, density=True, alpha=0.6, label="Variance 2")
plt.xlabel("Variances")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 8))
plot_posterior(samples)

plt.show()
