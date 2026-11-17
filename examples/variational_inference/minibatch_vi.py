"""
Minibatch Variational Inference
===============================

Fits a Bayesian linear regression on 20 000 observations with stochastic
(minibatch) variational inference. Every epoch of ``BBVI(..., batch_size=B)`` uses
a random subset of ``B`` observations along the plate ``"observations"``: the
covariates (``Data`` whose batch dimension is the plate dimension) and the
observed values are sliced consistently, and the likelihood is scaled by ``N / B``
so that the ELBO estimate stays unbiased. The plate size must come from ``Data``
(``plate("observations", x_data)``) for the observations to be subsampled.

The result is compared with the exact posterior of the model.
"""

import time

import torch
import matplotlib.pyplot as plt

from BayesianDLL import AutoNormal, Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)

N, batch_size, noise = 20_000, 200, 1.0
x = torch.randn(N, dtype=torch.float64)
y = 0.5 + 1.5 * x + noise * torch.randn(N, dtype=torch.float64)

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    x_data = Data("x", x)
    mu = Linear("mu", x_data, slope=slope, intercept=intercept)
    with plate("observations", x_data):
        ObservedParameter("y", Normal(mu, noise), y)

guide = AutoNormal(model)
start = time.perf_counter()
history = BBVI(model, guide, n_samples=4, epochs=1500, lr=0.01, lrd=0.999, batch_size=batch_size, callback_frequency=10, verbose=False)
print(f"Minibatch BBVI took {time.perf_counter() - start:.1f} s")

# Exact posterior of the coefficients (known noise, Normal priors).
design = torch.stack([torch.ones_like(x), x], dim=1)
precision = design.T @ design / noise ** 2 + torch.eye(2, dtype=x.dtype) / 25
exact_mean = torch.linalg.solve(precision, design.T @ y / noise ** 2)
print("Variational mean:", guide.loc.tolist(), "exact:", exact_mean.tolist())
print("Variational std: ", guide.scale.tolist(), "mean-field optimum:", precision.diagonal().rsqrt().tolist())

# %%
# The minibatch ELBO estimates are noisy, but unbiased.
plt.figure(figsize=(8, 4.5))
plt.plot(range(10, 10 * len(history) + 1, 10), history)
plt.ylim(min(history[20:]) - 50, max(history) + 10)
plt.xlabel("Epoch")
plt.ylabel("Minibatch ELBO estimate")
plt.tight_layout()
plt.show()
