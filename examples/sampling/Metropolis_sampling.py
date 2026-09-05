import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Bernoulli, Binomial, Geometric, Exponential, Beta
from BayesianDLL import Model, RandomParameter, sample


torch.manual_seed(0)

plt.figure(figsize=(8, 8))

n = 10000
bins = 30


plt.subplot(3, 3, 1)
distribution = Bernoulli(0.1)
with Model() as model:
    RandomParameter("bernoulli", distribution, torch.ones(1), sampler="metropolis")
    samples = sample(n, 1000, n_chains=1)["bernoulli"]
counts = torch.tensor([torch.sum(samples == s) for s in distribution.state_space])
frequencies = counts / counts.sum()
x = torch.tensor([[0, 1]])
plt.bar(x.squeeze(), frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel("State")
plt.ylabel("Estimated Probability")
plt.title("Bernoulli")
plt.legend()


plt.subplot(3, 3, 2)
k = 10
distribution = Binomial(k, 0.2)
with Model() as model:
    RandomParameter("binomial", distribution, torch.ones(1), sampler="metropolis")
    samples = sample(n, 1000, n_chains=1)["binomial"]
counts = torch.tensor([torch.sum(samples == s) for s in distribution.state_space])
frequencies = counts / counts.sum()
x = torch.arange(0, k + 1).unsqueeze(-1)
plt.bar(x.squeeze(), frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel("State")
plt.ylabel("Estimated Probability")
plt.title("Binomial")
plt.legend()


plt.subplot(3, 3, 3)
distribution = Geometric(0.2)
with Model() as model:
    RandomParameter("geometric", distribution, torch.ones(1), sampler="metropolis")
    samples = sample(n, 1000, n_chains=1)["geometric"]
x = torch.arange(samples.min().int().item(), samples.max().int().item()).unsqueeze(-1)
counts = torch.tensor([torch.sum(samples == s) for s in x])
frequencies = counts / counts.sum()
plt.bar(x.squeeze(), frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel("State")
plt.ylabel("Estimated Probability")
plt.title("Geometric")
plt.legend()


plt.subplot(3, 3, 4)
distribution = Exponential(0.3)
with Model() as model:
    RandomParameter("exponential", distribution, torch.ones(1, dtype=torch.float64), sampler="metropolis")
    samples = sample(n, 1000, n_chains=1)["exponential"].squeeze()
plt.hist(samples.numpy(), bins=bins, density=True, label="Estimated")
x = torch.linspace(1e-10, 20, 100).unsqueeze(-1)
plt.plot(x, distribution.pdf(x), label="True")
plt.xlabel("State")
plt.ylabel("Estimated Probability")
plt.title("Exponential")
plt.legend()


plt.subplot(3, 3, 5)
distribution = Beta(1, 1)
with Model() as model:
    RandomParameter("beta", distribution, 0.5 * torch.ones(1, dtype=torch.float64), sampler="metropolis", delta=0.6)
    samples = sample(n, 1000, n_chains=1)["beta"].squeeze()
plt.hist(samples.numpy(), bins=bins, density=True, label="Estimated")
x = torch.linspace(0, 1, 100).unsqueeze(-1)
plt.plot(x, distribution.pdf(x), label="True")
plt.xlabel("State")
plt.ylabel("Estimated Probability")
plt.title("Beta")
plt.legend()

plt.tight_layout()
plt.show()
