import torch
import matplotlib.pyplot as plt
from BayesianDLL.Distributions import Dirichlet, MultivariateNormal, Mixture
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample


torch.manual_seed(0)

data = torch.cat([
    torch.normal(-2, 0.5, size=(30,), dtype=torch.float64),
    torch.normal(2, 0.5, size=(20,), dtype=torch.float64)
])

with Model() as model:
    K = 2
    weights = RandomParameter("weights", Dirichlet(torch.ones(K)), torch.tensor([0.5], dtype=torch.float64), sampler="metropolis")

    means = [torch.tensor([-2.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)]
    covs = [torch.eye(1, dtype=torch.float64) * 0.5 ** 2, torch.eye(1, dtype=torch.float64) * 0.5 ** 2]
    components = [MultivariateNormal(mu, cov) for mu, cov in zip(means, covs)]
    mixture = Mixture(components, weights)

    likelihood = ObservedParameter("likelihood", mixture, data.unsqueeze(1))

    samples = sample(10000, 500)["weights"]

for k in range(K):
    plt.hist(samples[:, k].numpy(), bins=30, density=True, alpha=0.6, label=f"Weight {k}")
plt.xlabel("Weight value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
