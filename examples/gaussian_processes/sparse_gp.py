"""
Sparse Gaussian Process
=======================

Fits a Gaussian process to 2000 noisy observations with 20 inducing points.
``SparseGP`` evaluates Titsias' collapsed variational bound (VFE) in
``O(n m²)`` time, so NUTS can sample the kernel hyperparameters and the noise
variance without ever forming a 2000 x 2000 covariance matrix. The approximate
likelihoods of the VFE, DTC and FITC methods are compared, and the posterior
predictive distribution is plotted together with the inducing inputs.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import Matern52, SparseGP, sparse_gp_predictive


torch.manual_seed(0)

n, m = 2000, 20
x = torch.rand(n, dtype=torch.float64) * 10.0
y = torch.sin(x) + 0.3 * torch.cos(3.0 * x) + 0.2 * torch.randn(n, dtype=torch.float64)
# Fixed inducing inputs spread evenly over the data range.
inducing_points = torch.linspace(0.0, 10.0, m, dtype=torch.float64)

# %%
# The model
# ---------
# The kernel hyperparameters and the noise variance are model parameters; the
# gradient of the bound with respect to each of them is derived by hand.

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
    kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
    f = SparseGP(
        "f",
        inputs,
        kernel,
        inducing_points=inducing_points,
        noise_variance=Exp("noise", log_noise),
        method="vfe",
    )
    ObservedParameter("y", f, y)

start = time.perf_counter()
model.joint_grad_log_prob()
print(f"One log-density gradient with n={n}, m={m}: {1000 * (time.perf_counter() - start):.1f} ms")

trace = model.sample(400, 400, n_chains=2, start_point_variance=0.1, max_depth=5, seed=2)
print(trace.summary(include_deterministic=True))

# %%
# Comparing the approximations
# ----------------------------
# At the posterior mean hyperparameters, VFE is a lower bound on the exact log
# marginal likelihood, DTC drops the trace penalty, and FITC corrects the
# diagonal of the approximate covariance.

posterior_means = {name: trace[name].reshape(-1, 1).mean(0) for name in ("lengthscale", "variance", "noise")}
fixed_kernel = Matern52(posterior_means["lengthscale"], posterior_means["variance"])
with Model():
    comparison_inputs = Data("comparison_inputs", x)
    for method in ("vfe", "dtc", "fitc"):
        approximation = SparseGP(
            method, comparison_inputs, fixed_kernel, inducing_points=inducing_points,
            noise_variance=posterior_means["noise"], method=method,
        )
        print(f"{method.upper():>4} log likelihood: {approximation.log_pdf(y).item():.2f}")

# %%
# Posterior predictive distribution
# ---------------------------------

new_inputs = torch.linspace(-0.5, 10.5, 200, dtype=torch.float64)
predictions = sparse_gp_predictive(f, trace.thin(20), y, new_inputs, n_samples=10)
predictions = predictions.reshape(-1, new_inputs.numel())
lower, median, upper = torch.quantile(
    predictions, torch.tensor([0.05, 0.5, 0.95], dtype=predictions.dtype), dim=0
)

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.scatter(x, y, s=2, color="0.6", alpha=0.5, label="Observations")
ax.fill_between(new_inputs, lower, upper, color="tab:blue", alpha=0.3, label="Latent 90% interval")
ax.plot(new_inputs, median, color="tab:blue", label="Posterior median")
ax.plot(new_inputs, torch.sin(new_inputs) + 0.3 * torch.cos(3.0 * new_inputs), "k--", label="True function")
ax.plot(inducing_points, torch.full_like(inducing_points, y.min() - 0.3), "|", color="tab:red", markersize=12, label="Inducing inputs")
ax.set(title=f"Sparse GP (VFE) with n={n} and m={m}", xlabel="x", ylabel="y")
ax.legend(loc="upper right")
plt.show()
