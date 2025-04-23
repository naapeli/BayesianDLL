import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

torch.manual_seed(7)
N = 500
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
    # sigma2 = pyro.sample("sigma2", dist.Uniform(0, 10))
    mu = pyro.sample("mu", dist.Normal(mu0, tau ** 0.5))
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mu, sigma2.sqrt()), obs=data)

# Run NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=500)
mcmc.run(data)

# Extract posterior samples
posterior_samples = mcmc.get_samples()
posterior_mu = posterior_samples["mu"]
posterior_sigma2 = posterior_samples["sigma2"]

# Posterior summaries
print(f"Posterior mean of mu:     {posterior_mu.mean().item():.2f}")
print(f"Posterior mean of sigma²: {posterior_sigma2.mean().item():.2f}")

plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.hist(posterior_mu.numpy(), bins=50, density=True)
plt.title("Posterior of μ")

plt.subplot(2, 2, 2)
plt.hist(posterior_sigma2.numpy(), bins=50, density=True)
plt.title("Posterior of σ²")

plt.subplot(2, 2, 3)
plt.plot(posterior_mu.numpy())
plt.title("Posterior of μ")

plt.subplot(2, 2, 4)
plt.plot(posterior_sigma2.numpy())
plt.title("Posterior of σ²")

plt.tight_layout()
plt.savefig("Tests/sampling_from_joint_posterior/posteriors_with_pyro.png")
plt.show()
