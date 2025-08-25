import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, sample


torch.manual_seed(7)

# Generate synthetic data
N = 500
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).double().unsqueeze(1)
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N, 1))

with Model() as linear_model:
    # Priors
    prior_intercept = RandomParameter("intercept", Normal(0, 20), torch.tensor([0], dtype=torch.float64), sampler="auto")
    prior_slope = RandomParameter("slope", Normal(0, 20), torch.tensor([0], dtype=torch.float64), sampler="auto")
    prior_sigma = RandomParameter("sigma", HalfCauchy(10), torch.tensor([1], dtype=torch.float64), sampler="auto")

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)
    samples = sample(1000, 1000, n_chains=1)

intercept_samples = samples["intercept"].squeeze()
slope_samples = samples["slope"].squeeze()
sigma_samples = samples["sigma"].squeeze()

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.hist(intercept_samples.numpy(), bins=50, density=True)
plt.title("Posterior of Intercept")

plt.subplot(2, 2, 2)
plt.hist(slope_samples.numpy(), bins=50, density=True)
plt.title("Posterior of Slope")

plt.subplot(2, 2, 3)
plt.hist(sigma_samples.numpy(), bins=50, density=True)
plt.title("Posterior of Sigma")

plt.subplot(2, 2, 4)
plt.plot(intercept_samples, label="intercept")
plt.plot(slope_samples, label="slope")
plt.plot(sigma_samples, label="sigma")
plt.legend()
plt.title("Trace Plots")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/linear_trace_plots_and_posteriors.png")


x = x.squeeze()
y_preds = slope_samples[:, None] * x[None, :] + intercept_samples[:, None]
y_mean = y_preds.mean(dim=0)
y_lower = y_preds.quantile(0.025, dim=0)
y_upper = y_preds.quantile(0.975, dim=0)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)
plt.plot(x, y_mean, label="Posterior mean", color="black")
plt.fill_between(x.numpy(), y_lower.numpy(), y_upper.numpy(), color='gray', alpha=0.4, label="95% CI")
plt.title("Posterior Predictive with 95% Credible Interval")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/linear_fit.png")

plt.show()
