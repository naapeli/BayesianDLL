import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample_prior_predicative, sample_posterior_predicative
from BayesianDLL.Evaluation.Graphics import plot_predicative_distribution


torch.manual_seed(0)
N = 1000
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N, 1), dtype=torch.float64)
print(data.mean(), data.var())

n_samples = 100
with Model() as model:
    mean = RandomParameter("mean", Normal(mu0, tau))
    variance = RandomParameter("variance", InvGamma(a, b))
    likelihood = ObservedParameter("likelihood", Normal(mean, variance), data)
    predicative_distribution = model.sample_prior_predicative(n_samples, 5000, samples_per_step=1000)
    # predicative_distribution = model.sample_posterior_predicative(n_samples, 1000, samples_per_step=1000)

plt.subplot(1, 2, 1)
plot_predicative_distribution(predicative_distribution, data, kind="pdf")
plt.subplot(1, 2, 2)
plot_predicative_distribution(predicative_distribution, data, kind="cdf")
plt.show()
