import torch
import matplotlib.pyplot as plt

from BayesianDLL.distributions import Normal, Beta
from BayesianDLL.model import Model, Parameter, PosteriorWrapper
from BayesianDLL.sampler import NUTS


with Model() as model:
    mu = Parameter("mu", Normal(0, 1), init=torch.tensor(0.0))
    sigma = Parameter("sigma", Normal(1, 0.5), init=torch.tensor(1.0))
    x = Parameter("x", Normal(mu, sigma), observed=torch.tensor([1.0, 1.5, 0.8]))

posterior = PosteriorWrapper(model)
nuts = NUTS(posterior)

theta_init = torch.tensor([0.0, 1.0])
samples = nuts.sample(num_samples=100, theta_init=theta_init)
plt.plot(samples[:, 0].numpy())
plt.plot(samples[:, 1].numpy())
plt.show()
