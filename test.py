import torch
import matplotlib.pyplot as plt

from distributions import Normal, Beta
from model import Model, Parameter, PosteriorWrapper
from sampler import NUTS


sampler = NUTS(Normal(0, 1))
theta_init = torch.tensor([0.1]).float()
samples = sampler.sample(10000, theta_init, 1000)
plt.hist(samples.numpy(), bins=30, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000)
y = Normal(0, 1).pdf(x)
plt.plot(x, y)

sampler = NUTS(Normal(5, 3))
theta_init = torch.tensor([0.1]).float()
samples = sampler.sample(10000, theta_init, 1000)
plt.hist(samples.numpy(), bins=30, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000)
y = Normal(5, 3).pdf(x)
plt.plot(x, y)

plt.show()


# with Model() as model:
#     mu = Parameter("mu", Normal(0, 1), init=torch.tensor(0.0))
#     sigma = Parameter("sigma", Normal(1, 0.5), init=torch.tensor(1.0))
#     x = Parameter("x", Normal(mu, sigma), observed=torch.tensor([1.0, 1.5, 0.8]))

# posterior = PosteriorWrapper(model)
# nuts = NUTS(posterior)

# theta_init = torch.tensor([0.0, 1.0])
# samples = nuts.sample(num_samples=100, theta_init=theta_init)
# plt.plot(samples[:, 0].numpy())
# plt.plot(samples[:, 1].numpy())
# plt.show()
