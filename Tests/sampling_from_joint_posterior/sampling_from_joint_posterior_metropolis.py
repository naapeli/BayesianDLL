import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample


torch.manual_seed(7)
N = 100
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())

with Model() as model:
    prior_mean = RandomParameter("mean", Normal(mu0, tau), torch.zeros(1, dtype=torch.float64), sampler="metropolis")
    prior_variance = RandomParameter("variance", InvGamma(a, b), torch.ones(1, dtype=torch.float64), sampler="metropolis")

    likelihood = ObservedParameter("likelihood", Normal(prior_mean, prior_variance), data)
    samples = sample(20000, 500)


mean_samples, variance_samples = samples["mean"], samples["variance"]

colors = {
    "posterior": "blue",
    "true": "orange",
    "data": "green"
}

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(mean_samples.numpy(), bins=50, density=True, color="lightgray")
plt.axvline(mean_samples.mean().item(), color=colors["posterior"], label="posterior mean")
plt.axvline(true_mean, color=colors["true"], label="true mean")
plt.axvline(data.mean(), color=colors["data"], label="data mean")
plt.legend(loc="upper right")
plt.title("Posterior of Mean")

plt.subplot(2, 2, 2)
plt.hist(variance_samples.numpy(), bins=50, density=True, color="lightgray")
plt.axvline(variance_samples.mean().item(), color=colors["posterior"], label="posterior variance")
plt.axvline(true_variance, color=colors["true"], label="true variance")
plt.axvline(data.var(), color=colors["data"], label="data variance")
plt.legend(loc="upper right")
plt.title("Posterior of Variance")

plt.subplot(2, 2, 3)
plt.plot(mean_samples, color="lightgray", label="samples")
print("Posterior mean: ", mean_samples.mean().item())
plt.plot(torch.ones_like(mean_samples) * mean_samples.mean().item(), color=colors["posterior"], label="posterior mean")
plt.plot(torch.ones_like(mean_samples) * true_mean, color=colors["true"], label="true mean")
plt.plot(torch.ones_like(mean_samples) * data.mean(), color=colors["data"], label="data mean")
plt.legend(loc="upper right")
plt.title("Trace of Mean")

plt.subplot(2, 2, 4)
plt.plot(variance_samples, color="lightgray", label="samples")
print("Posterior variance: ", variance_samples.mean().item())
plt.plot(torch.ones_like(mean_samples) * variance_samples.mean().item(), color=colors["posterior"], label="posterior variance")
plt.plot(torch.ones_like(mean_samples) * true_variance, color=colors["true"], label="true variance")
plt.plot(torch.ones_like(mean_samples) * data.var(), color=colors["data"], label="data variance")
plt.legend(loc="upper right")
plt.title("Trace of Variance")
plt.savefig("Tests/sampling_from_joint_posterior/posteriors_with_Metropolis.png")
plt.tight_layout()
plt.show()
