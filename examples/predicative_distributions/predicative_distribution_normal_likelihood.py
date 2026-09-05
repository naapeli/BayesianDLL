import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample_prior_predicative, sample_posterior_predicative, plate
from BayesianDLL.Evaluation.Graphics import plot_predicative_distribution


torch.manual_seed(0)
N = 1000
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,), dtype=torch.float64)
print(data.mean(), data.var())

n_samples = 100
with Model() as model:
    mean = RandomParameter("mean", Normal(mu0, tau))
    variance = RandomParameter("variance", InvGamma(a, b))
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(mean, variance), data)
    predicative_distribution = model.sample_prior_predicative(n_samples, 5000, samples_per_step=10)
    # predicative_distribution = model.sample_posterior_predicative(n_samples, 1000, samples_per_step=10)

_, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
plot_predicative_distribution(predicative_distribution, data, kind="pdf", ax=axes[0])
plot_predicative_distribution(predicative_distribution, data, kind="cdf", ax=axes[1])
plt.show()
