import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Beta, Bernoulli, Binomial
from BayesianDLL import Model, RandomParameter, ObservedParameter, sample


torch.manual_seed(0)
N = 5
max_val = 1  # choose 1 for bernoulli prior and something else for binomial prior
data = torch.randint(0, max_val + 1, size=(N,))
a, b = 1, 1
with Model() as model:
    prior = RandomParameter("prior", Beta(a, b), torch.tensor([0.5], dtype=torch.float64))

    likelihood_distribution = Bernoulli(prior) if max_val == 1 else Binomial(max_val, prior)
    likelihood = ObservedParameter("likelihood", likelihood_distribution, data)
    samples = sample(10000, 500)["prior"]

plt.hist(samples.numpy(), bins=30, density=True, label="Estimated posterior")
x = torch.linspace(0, 1, 1000)
plt.plot(x, Beta(a + data.sum(), b + (max_val - data).sum()).pdf(x), label="True posterior")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
