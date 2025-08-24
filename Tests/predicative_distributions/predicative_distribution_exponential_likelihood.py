import torch

from BayesianDLL.Distributions import HalfCauchy, Exponential
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample_prior_predicative, sample_posterior_predicative
from BayesianDLL.Evaluation.Graphics import plot_predicative_distribution


torch.manual_seed(0)
N = 1000
true_rate = 3
data = -torch.log(torch.rand(size=(N, 1), dtype=torch.float64)) / true_rate

prior_scale = 1

n_samples = 20
with Model() as joint_posterior_model:
    samples_params = {"min_step_size": 1e-6, "gamma": 0.5}
    rate = RandomParameter("rate", HalfCauchy(prior_scale), torch.tensor([1], dtype=torch.float64), **samples_params)
    likelihood = ObservedParameter("likelihood", Exponential(rate), data)
    # prior_predicative = sample_prior_predicative(n_samples, 1000, samples_per_step=1000)
    posterior_predicative = sample_posterior_predicative(n_samples, 1000, samples_per_step=1000, warmup_per_sample=100)

plot_predicative_distribution(posterior_predicative, data, kind="pdf")
plot_predicative_distribution(posterior_predicative, data, kind="cdf")
