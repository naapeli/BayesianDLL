import torch
import matplotlib.pyplot as plt
from collections import Counter

from BayesianDLL.Distributions import Binomial, DiscreteUniform
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample


torch.manual_seed(2)
N = 5  # change this larger to make the posterior more concentrated on max_val
max_val = 3
data = torch.randint(0, max_val + 1, size=(N,))
p = 0.5
with Model() as model:
    prior = RandomParameter("prior", DiscreteUniform(0, 10), torch.tensor([1], dtype=torch.float64))
    likelihood = ObservedParameter("likelihood", Binomial(prior, p), data)
    samples = sample(10000, 500)["prior"]

samples = samples.squeeze().int()
counts = Counter(samples.tolist())
x_vals = sorted(counts.keys())
frequencies = [counts[x] / len(samples) for x in x_vals]
plt.bar(x_vals, frequencies, alpha=0.5, label="posterior")
plt.bar(prior.distribution.state_space.values, prior.distribution.pdf(prior.distribution.state_space.values.float()), alpha=0.5, label="prior")
plt.tight_layout()
plt.legend()
plt.show()
