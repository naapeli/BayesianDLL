import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, sample_prior_predicative, sample_posterior_predicative


torch.manual_seed(0)
N = 1000
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N, 1), dtype=torch.float64)
print(data.mean(), data.var())

n_samples = 10000
with Model() as joint_posterior_model:
    mean = RandomParameter("mean", Normal(mu0, tau), torch.tensor([0], dtype=torch.float64), sampler="auto", gamma=5)
    variance = RandomParameter("variance", InvGamma(a, b), torch.tensor([10], dtype=torch.float64), sampler="auto", gamma=5)
    likelihood = ObservedParameter("likelihood", Normal(mean, variance), data)
    prior_predicative = sample_prior_predicative(n_samples, 1000)
    posterior_predicative = sample_posterior_predicative(n_samples, 1000)

with Model() as joint_posterior_model:
    mean = RandomParameter("mean", Normal(mu0, tau), torch.tensor([0], dtype=torch.float64), sampler="auto", gamma=5)
    variance = RandomParameter("variance", InvGamma(a, b), torch.tensor([10], dtype=torch.float64), sampler="auto", gamma=5)
    prior_samples = sample(n_samples, 1000)

with Model() as joint_posterior_model:
    mean = RandomParameter("mean", Normal(mu0, tau), torch.tensor([0], dtype=torch.float64), sampler="auto", gamma=5)
    variance = RandomParameter("variance", InvGamma(a, b), torch.tensor([10], dtype=torch.float64), sampler="auto", gamma=5)
    likelihood = ObservedParameter("likelihood", Normal(mean, variance), data)
    posterior_samples = sample(n_samples, 1000)

plt.figure()
plt.subplot(1, 2, 1)
mu_samples = prior_samples["mean"]
var_samples = prior_samples["variance"]
y_prior_pred = torch.normal(mu_samples, var_samples.sqrt())
plt.hist(prior_predicative["likelihood"], bins=n_samples // 100, density=True, alpha=0.5, color='red', edgecolor='gray', label="BayesianDLL")
plt.hist(y_prior_pred.numpy(), bins=n_samples // 100, density=True, alpha=0.5, color='blue', edgecolor='gray', label="True")
plt.legend()
plt.title("Prior Predictive Distribution")
plt.xlabel("y")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 2, 2)
mu_samples = posterior_samples["mean"]
var_samples = posterior_samples["variance"]
y_posterior_pred = torch.normal(mu_samples, var_samples.sqrt())
plt.hist(posterior_predicative["likelihood"], bins=n_samples // 100, density=True, alpha=0.5, color='red', edgecolor='gray', label="BayesianDLL")
plt.hist(y_posterior_pred.numpy(), bins=n_samples // 100, density=True, alpha=0.5, color='blue', edgecolor='gray', label="True")
plt.legend()
plt.title("Posterior Predictive Distribution")
plt.xlabel("y")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

plt.show()
