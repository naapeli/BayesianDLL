"""
Joint Posterior With Metropolis
===============================

Infers the mean and standard deviation of normally distributed data with the Metropolis sampler.
The effective sample size and Gelman-Rubin statistic are printed and the posterior is
plotted.
"""

import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfNormal
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, plate
from BayesianDLL.Evaluation.Graphics import plot_posterior, plot_model
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


torch.manual_seed(0)
N = 100
mu0 = 0
tau = 3  # prior standard deviation of the mean
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())

with Model() as model:
    prior_mean = RandomParameter("mean", Normal(mu0, tau), torch.zeros(1, dtype=torch.float64), sampler="metropolis", delta=0.5)
    prior_sigma = RandomParameter("sigma", HalfNormal(3.0), torch.ones(1, dtype=torch.float64), sampler="metropolis", delta=0.5)

    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(prior_mean, prior_sigma), data)
    plot_model(model)
    samples = sample(2000, 500, n_chains=2)

ess = effective_sample_size(samples)
print(ess)
print(gelman_rubin(samples))

plot_posterior(samples, method="kde")
plt.show()
