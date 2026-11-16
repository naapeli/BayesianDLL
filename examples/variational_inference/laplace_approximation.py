"""
Laplace Approximation
=====================

Approximates the posterior of a Beta-Bernoulli model with ``laplace_approximation``.
The maximum a posteriori (MAP) estimate is found in the unconstrained (logit)
space, the Hessian of the log joint is computed by finite differences of the
hand-written gradient, and the posterior is approximated by a Gaussian with
covariance ``(-H)^-1``.

The approximation is compared with the exact Beta posterior, and its estimate of
the log marginal likelihood with the exact value.
"""

import math

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, ObservedParameter, RandomParameter, laplace_approximation, plate
from BayesianDLL.Distributions import Bernoulli, Beta


torch.manual_seed(0)

a, b = 2.0, 2.0
data = (torch.rand(30, dtype=torch.float64) < 0.7).double()
k, n = int(data.sum()), len(data)

with Model() as model:
    p = RandomParameter("p", Beta(a, b))
    with plate("data", n):
        ObservedParameter("y", Bernoulli(p), data)

approximation = laplace_approximation(model, map_epochs=300, lr=0.05)
print(approximation)
print("MAP estimate:", {name: value.item() for name, value in approximation.map_estimate().items()})

exact_log_evidence = (
    math.lgamma(a + k) + math.lgamma(b + n - k) - math.lgamma(a + b + n)
    - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
)
print(f"Laplace log marginal likelihood: {approximation.log_marginal_likelihood.item():.4f}")
print(f"Exact log marginal likelihood:   {exact_log_evidence:.4f}")

# %%
# Approximate and exact posterior
# -------------------------------
trace = approximation.sample(5000)
print(trace.summary())

grid = torch.linspace(0.3, 0.98, 400, dtype=torch.float64)
exact_density = Beta(a + k, b + n - k).pdf(grid)

plt.figure(figsize=(8, 4.5))
plt.hist(trace["p"].flatten().numpy(), bins=60, density=True, alpha=0.5, label="Laplace approximation (draws)")
plt.plot(grid, exact_density, color="black", label="Exact Beta posterior")
plt.xlabel("p")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.show()
