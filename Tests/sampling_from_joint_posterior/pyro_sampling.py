import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

torch.manual_seed(7)
N = 100
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())

# Pyro model
def model(data):
    sigma2 = pyro.sample("sigma2", dist.InverseGamma(a, b))
    mu = pyro.sample("mu", dist.Normal(mu0, tau ** 0.5))
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mu, sigma2.sqrt()), obs=data)

# Run NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=20000, warmup_steps=500)
mcmc.run(data)

# Extract posterior samples
posterior_samples = mcmc.get_samples()
mean_samples = posterior_samples["mu"]
variance_samples = posterior_samples["sigma2"]


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
plt.savefig("Tests/sampling_from_joint_posterior/posteriors_with_pyro.png")
plt.tight_layout()
plt.show()
