import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Dirichlet, Normal, Mixture, Exponential
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample


torch.manual_seed(0)

K = 2
n = 200
true_weights = torch.tensor([0.6, 0.4])
data = torch.cat([
    torch.normal(-1, 0.5, size=(int(true_weights[0] * n),), dtype=torch.float64),
    torch.normal(2, 1, size=(int(true_weights[1] * n),), dtype=torch.float64)
])

sampler_params = {"min_step_size": 1e-6, "max_step_size": 10, "delta": 0.6, "gamma": 0.5}  # mainly decrease the minimum step size for more accurate, but slower sampling. Also increase gamma for faster step size adaptation during warmup
weight_sampler_params = {"min_step_size": 1e-10, "max_step_size": 10, "delta": 0.9, "gamma": 5}

with Model() as model:
    alpha = 1.5 * torch.ones(K, dtype=torch.float64)
    theta_init = torch.ones(K, dtype=torch.float64)
    theta_init[0], theta_init[1] = 2, 3  # make the initial point not uniform to stabilize the sampling
    theta_init = torch.softmax(theta_init, dim=0)
    weights = RandomParameter("weights", Dirichlet(alpha), theta_init, **weight_sampler_params)

    means = [RandomParameter("mean" + str(i + 1), Normal(0, 10), torch.randn(size=(1,), dtype=torch.float64), **sampler_params) for i in range(K)]
    variances = [RandomParameter("variance" + str(i + 1), Exponential(0.5), torch.rand(size=(1,), dtype=torch.float64) + 1, **sampler_params) for i in range(K)]
    # variances = [0.5 ** 2 for _ in range(K)]

    components = [Normal(mu, cov) for mu, cov in zip(means, variances)]

    likelihood = ObservedParameter("likelihood", Mixture(components, weights), data.unsqueeze(1))

    samples = sample(2000, 2000)

plt.figure()
for k in range(K):
    plt.hist(samples["weights"][:, k].numpy(), bins=30, density=True, alpha=0.6, label=f"Weight {k}")
plt.xlabel("Weight value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(samples["mean1"].numpy(), bins=30, density=True, alpha=0.6, label="Mean 1")
plt.hist(samples["mean2"].numpy(), bins=30, density=True, alpha=0.6, label="Mean 2")
plt.xlabel("Means")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.figure()
plt.hist(samples["variance1"].numpy(), bins=30, density=True, alpha=0.6, label="Variance 1")
plt.hist(samples["variance2"].numpy(), bins=30, density=True, alpha=0.6, label="Variance 2")
plt.xlabel("Variances")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.show()
