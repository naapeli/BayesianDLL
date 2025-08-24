import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample
from BayesianDLL.Evaluation.Graphics import plot_posterior


torch.manual_seed(7)
N = 100
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N, 1))
print(data.mean(), data.var())

with Model() as model:
    prior_mean = RandomParameter("mean", Normal(mu0, tau), torch.zeros(1, dtype=torch.float64), sampler="metropolis")
    prior_variance = RandomParameter("variance", InvGamma(a, b), torch.ones(1, dtype=torch.float64), sampler="metropolis")

    likelihood = ObservedParameter("likelihood", Normal(prior_mean, prior_variance), data)
    samples = sample(5000, 1000, n_chains=2)

plot_posterior(samples, method="kde")
plt.show()
