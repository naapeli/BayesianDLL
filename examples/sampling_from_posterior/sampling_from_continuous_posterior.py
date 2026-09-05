import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Beta, Bernoulli, Binomial
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, plate
from BayesianDLL.Evaluation.Graphics import plot_posterior


torch.manual_seed(0)
N = 5
max_val = 1  # choose 1 for bernoulli prior and something else for binomial prior
data = torch.randint(0, max_val + 1, size=(N,))
a, b = 1, 1
with Model() as model:
    prior = RandomParameter("prior", Beta(a, b))  # , sampler="metropolis"

    likelihood_distribution = Bernoulli(prior) if max_val == 1 else Binomial(max_val, prior)
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", likelihood_distribution, data)
    samples = sample(1000, 500, n_chains=4, start_point_variance=0)

axes = plot_posterior(samples, method="kde")
x = torch.linspace(0, 1, 1000)
axes[0, 0].plot(x, Beta(a + data.sum(), b + (max_val - data).sum()).pdf(x), c="black", label="True posterior")
axes[0, 0].set_xlim(0, 1)
axes[0, 0].legend()
plt.show()
