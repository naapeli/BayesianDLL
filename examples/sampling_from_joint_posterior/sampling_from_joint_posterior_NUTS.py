import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, plate
from BayesianDLL.Evaluation.Graphics import plot_posterior, plot_model
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


torch.manual_seed(0)
N = 100
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())

with Model() as model:
    prior_mean = RandomParameter("mean", Normal(mu0, tau), torch.zeros(1, dtype=torch.float64), sampler="nuts")
    prior_variance = RandomParameter("variance", InvGamma(a, b), torch.ones(1, dtype=torch.float64), sampler="nuts")

    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(prior_mean, prior_variance), data)
    plot_model(model)
    samples = sample(500, 500, n_chains=2)

ess = effective_sample_size(samples)
print(ess)
print(gelman_rubin(samples))

plot_posterior(samples, method="kde")
plt.show()
