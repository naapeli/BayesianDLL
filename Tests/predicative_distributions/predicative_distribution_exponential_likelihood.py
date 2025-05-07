import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import HalfCauchy, Exponential
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, sample_prior_predicative, sample_posterior_predicative


torch.manual_seed(0)
N = 1000
true_rate = 3
data = -torch.log(torch.rand(size=(N,), dtype=torch.float64)) / true_rate

prior_scale = 1

n_samples = 10000
with Model() as joint_posterior_model:
    rate = RandomParameter("rate", HalfCauchy(prior_scale), torch.tensor([1], dtype=torch.float64), sampler="auto", gamma=5)
    likelihood = ObservedParameter("likelihood", Exponential(rate), data)
    prior_predicative = sample_prior_predicative(n_samples, 1000)
    posterior_predicative = sample_posterior_predicative(n_samples, 1000)

with Model() as joint_posterior_model:
    rate = RandomParameter("rate", HalfCauchy(prior_scale), torch.tensor([1], dtype=torch.float64), sampler="auto", gamma=5)
    prior_samples = sample(n_samples, 1000)

with Model() as joint_posterior_model:
    rate = RandomParameter("rate", HalfCauchy(prior_scale), torch.tensor([1], dtype=torch.float64), sampler="auto", gamma=5)
    likelihood = ObservedParameter("likelihood", Exponential(rate), data)
    posterior_samples = sample(n_samples, 1000)

plt.figure()
plt.subplot(1, 2, 1)
plt.xscale("log")
plt.yscale("log")
xmin = prior_samples["rate"].min()
xmax = prior_samples["rate"].max()
bin_edges = torch.logspace(torch.log10(xmin), torch.log10(xmax), n_samples // 100)
rate_samples = prior_samples["rate"].squeeze()
y_prior_pred = -torch.log(torch.rand(size=(n_samples,), dtype=torch.float64)) / rate_samples
plt.hist(prior_predicative["likelihood"], bins=bin_edges, density=True, alpha=0.5, color='red', edgecolor='gray', label="BayesianDLL")
plt.hist(y_prior_pred.numpy(), bins=bin_edges, density=True, alpha=0.5, color='blue', edgecolor='gray', label="True")
plt.legend()
plt.title("Prior Predictive Distribution")
plt.xlabel("y")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

plt.subplot(1, 2, 2)
rate_samples = posterior_samples["rate"].squeeze()
y_posterior_pred = -torch.log(torch.rand(size=(n_samples,), dtype=torch.float64)) / rate_samples
plt.hist(posterior_predicative["likelihood"], bins=n_samples // 100, density=True, alpha=0.5, color='red', edgecolor='gray', label="BayesianDLL")
plt.hist(y_posterior_pred.numpy(), bins=n_samples // 100, density=True, alpha=0.5, color='blue', edgecolor='gray', label="True")
plt.legend()
plt.title("Posterior Predictive Distribution")
plt.xlabel("y")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

plt.show()
