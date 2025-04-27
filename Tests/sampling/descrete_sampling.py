import torch
import matplotlib.pyplot as plt

# from BayesianDLL.Samplers.Metropolis import metropolis_hastings_discrete
from BayesianDLL.Samplers import Metropolis
from BayesianDLL.Distributions import Bernoulli, Binomial, Geometric
from BayesianDLL.Distributions._state_space import DiscreteRange, DiscretePositive


torch.manual_seed(0)

def target_pmf(theta):
    return distribution.pdf(theta)

plt.figure(figsize=(8, 8))

n = 20000


plt.subplot(3, 3, 1)
state_space = DiscreteRange(0, 1)
distribution = Bernoulli(0.1)
sampler = Metropolis(distribution.log_pdf, state_space)
samples = sampler.sample(n, torch.ones(1), 1000)
# samples = metropolis_hastings_discrete(target_pmf, torch.ones(1), num_samples=n, state_space=state_space)
counts = torch.tensor([torch.sum(samples == s) for s in state_space])
frequencies = counts / counts.sum()
x = torch.tensor([0, 1])
plt.bar(x, frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel('State')
plt.ylabel('Estimated Probability')
plt.title("Bernoulli")
plt.legend()


plt.subplot(3, 3, 2)
k = 10
state_space = DiscreteRange(0, k)
distribution = Binomial(k, 0.2)
sampler = Metropolis(distribution.log_pdf, state_space)
samples = sampler.sample(n, torch.ones(1), 1000)
# samples = metropolis_hastings_discrete(target_pmf, torch.ones(1), num_samples=n, state_space=state_space)
counts = torch.tensor([torch.sum(samples == s) for s in state_space])
frequencies = counts / counts.sum()
x = torch.arange(0, k + 1)
plt.bar(x, frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel('State')
plt.ylabel('Estimated Probability')
plt.title("Binomial")
plt.legend()


plt.subplot(3, 3, 3)
state_space = DiscretePositive()
distribution = Geometric(0.2)
sampler = Metropolis(distribution.log_pdf, state_space)
samples = sampler.sample(n, torch.ones(1), 1000)
# samples = metropolis_hastings_discrete(target_pmf, torch.ones(1), num_samples=n, state_space=state_space)
x = torch.arange(samples.min().int().item(), samples.max().int().item())
counts = torch.tensor([torch.sum(samples == s) for s in x])
frequencies = counts / counts.sum()
plt.bar(x, frequencies.numpy(), label="Estimated")
plt.scatter(x, distribution.pdf(x), label="True")
plt.xlabel('State')
plt.ylabel('Estimated Probability')
plt.title("Geometric")
plt.legend()



plt.tight_layout()
plt.show()
