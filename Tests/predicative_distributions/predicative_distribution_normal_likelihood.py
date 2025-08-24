import torch

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
with Model() as joint_posterior_model:
    mean = RandomParameter("mean", Normal(mu0, tau), torch.tensor([0], dtype=torch.float64), sampler="auto", gamma=5)
    variance = RandomParameter("variance", InvGamma(a, b), torch.tensor([10], dtype=torch.float64), sampler="auto", gamma=5)
    likelihood = ObservedParameter("likelihood", Normal(mean, variance), data)
    # prior_predicative = sample_prior_predicative(n_samples, 1000, samples_per_step=1000)
    posterior_predicative = sample_posterior_predicative(n_samples, 1000, samples_per_step=1000)

plot_predicative_distribution(posterior_predicative, data, kind="pdf")
plot_predicative_distribution(posterior_predicative, data, kind="cdf")
