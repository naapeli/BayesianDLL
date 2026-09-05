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
with Model() as model:
    RandomParameter("mixture", distribution, sampler="nuts")

with MeanFieldGuide() as guide:
    mean = VariationalParameter("mean", 10)
    variance = VariationalParameter("variance", 10)
    RandomParameter("mixture", Normal(mean, variance))

history = BBVI(model, guide, epochs=3000, callback_frequency=20, lr=1e-2)

plt.figure()
plt.plot(history)

plt.figure()
x = torch.linspace(-3, 5, 100).unsqueeze(1)
plt.plot(x, model.params["mixture"].distribution.pdf(x))
plt.plot(x, guide.params["mixture"].distribution.pdf(x))

plt.show()
