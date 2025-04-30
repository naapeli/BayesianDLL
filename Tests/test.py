import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, Uniform, Binomial
from BayesianDLL.Model import Model
from BayesianDLL.Samplers import sample
from BayesianDLL.parameters import RandomParameter


with Model() as normal_model:
    RandomParameter("x", Uniform(-4, -2), torch.tensor([3], dtype=torch.float64))
    RandomParameter("y", Normal(0, 1), torch.tensor([0], dtype=torch.float64))
    RandomParameter("z", Binomial(5, 0.3), torch.tensor([2], dtype=torch.float64))

samples = sample(normal_model, 10000, 1000)

plt.subplot(1, 2, 1)
plt.hist(samples["x"], alpha=0.5, density=True, bins=10, label="Uniform")
plt.hist(samples["y"], alpha=0.5, density=True, bins=30, label="Normal")
counts = torch.tensor([torch.sum(samples["z"] == state) for state in normal_model.params["z"].distribution.state_space])
frequencies = counts / counts.sum()
plt.bar(normal_model.params["z"].distribution.state_space.values, frequencies, alpha=0.5, label="Binomial")
x = torch.linspace(-5, 5, 1000)
plt.plot(x, Normal(0, 1).pdf(x))
plt.plot(x, Uniform(-4, -2).pdf(x))
plt.scatter(normal_model.params["z"].distribution.state_space.values, Binomial(5, 0.3).pdf(normal_model.params["z"].distribution.state_space.values))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(samples["x"], label="Uniform")
plt.plot(samples["y"], label="Normal")
plt.plot(samples["z"], label="Binomial")
plt.legend()
plt.show()
