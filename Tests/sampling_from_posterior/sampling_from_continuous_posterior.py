import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Beta, Bernoulli, Binomial
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample
from BayesianDLL.Evaluation.Graphics import plot_posterior


torch.manual_seed(0)
N = 5
max_val = 1  # choose 1 for bernoulli prior and something else for binomial prior
data = torch.randint(0, max_val + 1, size=(N,)).unsqueeze(1)
a, b = 1, 1
with Model() as model:
    prior = RandomParameter("prior", Beta(a, b), torch.tensor(0.5, dtype=torch.float64))

    likelihood_distribution = Bernoulli(prior) if max_val == 1 else Binomial(max_val, prior)
    likelihood = ObservedParameter("likelihood", likelihood_distribution, data)
    samples = sample(1000, 500, n_chains=4)

plot_posterior(samples, method="kde")
x = torch.linspace(0, 1, 1000).unsqueeze(1)
plt.subplot(1, 2, 1)
plt.plot(x, Beta(a + data.sum(), b + (max_val - data).sum()).pdf(x), c="black", label="True posterior")
plt.xlim(0, 1)
plt.legend()
plt.show()
