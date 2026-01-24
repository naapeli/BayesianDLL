import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, MeanFieldGuide, RandomParameter, VariationalParameter
from BayesianDLL.Distributions import Normal, Mixture
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)
means = [-1, 2]
variances = [0.5 ** 2, 1 ** 2]
components = [Normal(mu, var) for mu, var in zip(means, variances)]
weights = [0.3, 0.7]
distribution = Mixture(components, weights)
theta_init = torch.tensor(0, dtype=torch.float64)
with Model() as model:
    RandomParameter("mixture", distribution, theta_init, sampler="nuts")

with MeanFieldGuide() as guide:
    mean = VariationalParameter("mean", 0)
    variance = VariationalParameter("variance", 1)
    RandomParameter("mixture", Normal(mean, variance), initial_value=torch.zeros(1))

history = BBVI(model, guide, n_samples=1000, epochs=140, callback_frequency=20)

plt.figure()
plt.plot(history)

plt.figure()
x = torch.linspace(-3, 5, 100).unsqueeze(1)
plt.plot(x, model.params["mixture"].distribution.pdf(x))
plt.plot(x, guide.params["mixture"].distribution.pdf(x))

plt.show()
