import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, InvGamma
from BayesianDLL import Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Evaluation.Graphics import plot_model


torch.manual_seed(0)
N = 100
mu0 = 0
tau = 10
a_val = 2
b_val = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))

with Model() as model:
    a = RandomParameter("a", Normal(0, 1), torch.randn(1, dtype=torch.float64), sampler="nuts")
    b = RandomParameter("b", Normal(a, 1), torch.randn(1, dtype=torch.float64), sampler="nuts")
    c = RandomParameter("c", Normal(b, tau), torch.randn(1, dtype=torch.float64), sampler="nuts")
    d = RandomParameter("d", InvGamma(a, b), torch.ones(1, dtype=torch.float64), sampler="nuts")

    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(c, d), data)
    plot_model(model)


h = 1e-8
print((model.log_prob("a", a.unconstrained_value + h) - model.log_prob("a", a.unconstrained_value)) / h)
print(model.grad_log_prob("a", a.unconstrained_value))
print((model.log_prob("b", b.unconstrained_value + h) - model.log_prob("b", b.unconstrained_value)) / h)
print(model.grad_log_prob("b", b.unconstrained_value))
print((model.log_prob("c", c.unconstrained_value + h) - model.log_prob("c", c.unconstrained_value)) / h)
print(model.grad_log_prob("c", c.unconstrained_value))
print((model.log_prob("d", d.unconstrained_value + h) - model.log_prob("d", d.unconstrained_value)) / h)
print(model.grad_log_prob("d", d.unconstrained_value))
plt.show()
