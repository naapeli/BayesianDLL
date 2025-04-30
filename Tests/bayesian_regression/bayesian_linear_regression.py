import torch
import matplotlib.pyplot as plt

from BayesianDLL.Samplers import NUTS
from BayesianDLL.Distributions import Normal, HalfCauchy


torch.manual_seed(7)

# Generate synthetic data
N = 500
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5  # NOTE: Stability of the sampler depends a lot on the original variance. If this is too large, the sampler might get stuck and one needs to use less data points.
x = torch.linspace(0, 1, N)
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N,))

# Priors
prior_intercept = Normal(0, 20)
prior_slope = Normal(0, 20)
prior_sigma = HalfCauchy(10)

def log_posterior(theta):
    intercept, slope, sigma_unconstrained = theta[0], theta[1], theta[2]
    sigma = prior_sigma.transform.inverse(sigma_unconstrained)
    
    prior = (
        prior_intercept._log_prob_unconstrained(intercept) +
        prior_slope._log_prob_unconstrained(slope) +
        prior_sigma._log_prob_unconstrained(sigma_unconstrained)
    )

    mu = prior_intercept.transform.inverse(intercept) + prior_slope.transform.inverse(slope) * x
    likelihood = Normal(mu, sigma).log_pdf(y).sum()
    
    return prior + likelihood

def log_posterior_derivative(theta):
    intercept, slope, sigma_unconstrained = theta[0], theta[1], theta[2]
    intercept_real = prior_intercept.transform.inverse(intercept)
    slope_real = prior_slope.transform.inverse(slope)
    sigma = prior_sigma.transform.inverse(sigma_unconstrained)

    mu = intercept_real + slope_real * x
    normal_dist = Normal(mu, sigma)
    grads = normal_dist.log_pdf_param_grads(y)
    
    d_intercept = grads["mean"].sum() * prior_intercept.transform.derivative(intercept)
    d_slope = (grads["mean"] * x).sum() * prior_slope.transform.derivative(slope)
    d_sigma = grads["variance"].sum() * prior_sigma.transform.derivative(sigma_unconstrained)

    grad = torch.tensor([
        prior_intercept._log_prob_grad_unconstrained(intercept) + d_intercept.item(),
        prior_slope._log_prob_grad_unconstrained(slope) + d_slope.item(),
        prior_sigma._log_prob_grad_unconstrained(sigma_unconstrained) + d_sigma.item()
    ], dtype=theta.dtype)

    return grad

def inverse_transformation(theta):
    intercept = prior_intercept.transform.inverse(theta[:, 0])
    slope = prior_slope.transform.inverse(theta[:, 1])
    sigma = prior_sigma.transform.inverse(theta[:, 2])
    return torch.stack([intercept, slope, sigma], dim=-1)

initial_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
sampler = NUTS(log_posterior, log_posterior_derivative, inverse_transformation)
samples, _, _ = sampler.sample(3000, initial_point, 500)

intercept_samples = samples[:, 0]
slope_samples = samples[:, 1]
sigma_samples = samples[:, 2]

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
plt.title("Posterior of σ")

plt.subplot(2, 2, 4)
plt.plot(intercept_samples, label="intercept")
plt.plot(slope_samples, label="slope")
plt.plot(sigma_samples, label="sigma")
plt.legend()
plt.title("Trace Plots")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/linear_trace_plots_and_posteriors.png")


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
