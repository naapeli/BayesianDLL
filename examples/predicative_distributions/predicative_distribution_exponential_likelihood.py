import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import HalfCauchy, Exponential
from BayesianDLL import Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Evaluation.Graphics import plot_predicative_distribution


torch.manual_seed(0)
N = 1000
true_rate = 3
data = -torch.log(torch.rand(size=(N,), dtype=torch.float64)) / true_rate

prior_scale = 1

n_samples = 20
with Model() as model:
    rate = RandomParameter("rate", HalfCauchy(prior_scale))
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Exponential(rate), data)
    # predicative_distribution = model.sample_prior_predicative(n_samples, 5000, samples_per_step=10)
    predicative_distribution = model.sample_posterior_predicative(n_samples, 1000, samples_per_step=10, warmup_per_sample=100)

_, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
plot_predicative_distribution(predicative_distribution, data, kind="pdf", ax=axes[0])
plot_predicative_distribution(predicative_distribution, data, kind="cdf", ax=axes[1])
plt.show()
