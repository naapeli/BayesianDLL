import torch
import matplotlib.pyplot as plt

from BayesianDLL.Samplers import Metropolis
from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL.Distributions._state_space import ContinuousReal


torch.manual_seed(7)
N = 100
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())
prior_mean = Normal(mu0, tau)
prior_variance = InvGamma(a, b)
def log_posterior(theta):
    mean, variance = theta[0], theta[1]
    prior = prior_mean._log_prob_unconstrained(mean) + prior_variance._log_prob_unconstrained(variance)
    likelihood = Normal(prior_mean.transform.inverse(mean), prior_variance.transform.inverse(variance)).log_pdf(data).sum()
    return prior + likelihood

def inverse_transformation(theta):
    mean, std = theta[:, 0], theta[:, 1]
    return torch.stack([prior_mean.transform.inverse(mean), prior_variance.transform.inverse(std)], dim=-1)

sampler = Metropolis(log_posterior, ContinuousReal())
samples = inverse_transformation(sampler.sample(50000, torch.tensor([0, 0], dtype=torch.float64), 500))
mean_samples = samples[:, 0]
variance_samples = samples[:, 1]


plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.hist(mean_samples.numpy(), bins=50, density=True)
plt.title("Posterior of μ")

plt.subplot(2, 2, 2)
plt.hist(variance_samples.numpy(), bins=50, density=True)
plt.title("Posterior of σ²")

plt.subplot(2, 2, 3)
plt.plot(mean_samples)
print("Posterior mean: ", mean_samples.mean().item())
plt.plot(torch.ones_like(mean_samples) * mean_samples.mean().item())
plt.title("mean")

plt.subplot(2, 2, 4)
plt.plot(variance_samples)
print("Posterior variance: ", variance_samples.mean().item())
plt.plot(torch.ones_like(variance_samples) * variance_samples.mean().item())
plt.title("variance")

plt.tight_layout()
plt.savefig("Tests/sampling_from_joint_posterior/posteriors_with_Metropolis.png")
plt.show()
