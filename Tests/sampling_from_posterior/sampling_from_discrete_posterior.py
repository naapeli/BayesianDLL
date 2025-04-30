import torch
import matplotlib.pyplot as plt
from collections import Counter

from BayesianDLL.Samplers import Metropolis
from BayesianDLL.Distributions import Binomial, DiscreteUniform, Geometric


torch.manual_seed(2)
N = 5  # change this larger to make the posterior more concentrated on max_val
max_val = 3
data = torch.randint(0, max_val + 1, size=(N,))
p = 0.5
prior = DiscreteUniform(0, 10)
def log_posterior(theta):
    prior_prob = prior.log_pdf(theta)
    likelihood = Binomial(theta.item(), p).log_pdf(data).sum()
    return prior_prob + likelihood

sampler = Metropolis(log_posterior, prior.state_space)
samples = sampler.sample(10000, torch.tensor([1], dtype=torch.float64), 500)

samples = samples.squeeze().int()
counts = Counter(samples.tolist())
x_vals = sorted(counts.keys())
frequencies = [counts[x] / len(samples) for x in x_vals]
plt.bar(x_vals, frequencies, alpha=0.5, label="posterior")
plt.bar(prior.state_space.values, prior.pdf(prior.state_space.values.float()), alpha=0.5, label="prior")
plt.tight_layout()
plt.legend()
plt.show()
