import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy, MultivariateNormal
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, sample


torch.manual_seed(7)

N = 100
true_coeffs, xmin, xmax = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111], -5, 5
true_variance = 0.1
x = torch.linspace(xmin, xmax, N, dtype=torch.float64)
X = torch.stack([x ** i for i in range(len(true_coeffs))], dim=1)
y = sum(c * x ** i for i, c in enumerate(true_coeffs)) + torch.normal(0, true_variance ** 0.5, size=(N,))

# Degree of the polynomial to be fitted
D = 5
x_scaled = (x - xmin) / (xmax - xmin) * 2 - 1  # Scaling to [-1, 1] for Chebyshev
phi_x = torch.stack([torch.cos(i * torch.acos(x_scaled)) for i in range(D + 1)], dim=1).to(torch.float64)  # Chebysev basis functions

with Model() as polynomial_model:
    # Priors
    prior_mean = torch.zeros(D + 1, dtype=torch.float64)
    prior_cov = torch.eye(D + 1, dtype=torch.float64)
    prior_coeffs = RandomParameter("coeffs", MultivariateNormal(prior_mean, prior_cov), torch.zeros_like(prior_mean, dtype=torch.float64), sampler="auto", gamma=5)  # critical to pass gamma=5 as the sampler is otherwise extremely slow
    prior_sigma = RandomParameter("sigma", HalfCauchy(10), torch.ones(1, dtype=torch.float64), sampler="auto", gamma=5)  # critical to pass gamma=5 as the sampler is otherwise extremely slow

    mu = DeterministicParameter("mu", lambda coeffs, phi_x: phi_x @ coeffs, lambda coeffs, phi_x: {"coeffs": phi_x}, [prior_coeffs, phi_x])
    
    likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)
    samples = sample(10000, 1000)


coeff_samples = samples["coeffs"]
coeff_means = coeff_samples.mean(dim=0)
variance_samples = samples["sigma"]
print(f"Posterior coefficient estimates: {coeff_means}")
print(f"Posterior variance estimate: {variance_samples.mean()}")

plt.figure(figsize=(12, 6))
for i, cs in enumerate(coeff_samples.T):
    plt.subplot(2, (D + 2) // 2 + 1, i + 1)
    plt.hist(cs.numpy(), bins=50, density=True)
    plt.title(f"Posterior of $\\theta_{i}$")
plt.subplot(2, (D + 2) // 2 + 1, D + 2)
plt.hist(variance_samples.numpy(), bins=50, density=True)
plt.title(f"Posterior of $\\sigma$")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_posteriors.png")

plt.figure(figsize=(12, 6))
for i, cs in enumerate(coeff_samples.T):
    plt.subplot(2, (D + 2) // 2 + 1, i + 1)
    plt.plot(cs.numpy())
    plt.title(f"Posterior of $\\theta_{i}$")
plt.subplot(2, (D + 2) // 2 + 1, D + 2)
plt.plot(variance_samples.numpy())
plt.title(f"Posterior of $\\sigma$")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_trace.png")

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)

x = torch.linspace(xmin, xmax, 100)
x_scaled = (x - xmin) / (xmax - xmin) * 2 - 1  # Scaling to [-1, 1] for Chebyshev
phi_x = torch.stack([torch.cos(i * torch.acos(x_scaled)) for i in range(D + 1)], dim=1).to(torch.float64)
y_preds = samples["coeffs"] @ phi_x.T
y_mean = y_preds.mean(dim=0)
y_lower = y_preds.quantile(0.025, dim=0)
y_upper = y_preds.quantile(0.975, dim=0)

plt.plot(x, y_mean, label="Posterior mean", color="black")
plt.fill_between(x.numpy(), y_lower.numpy(), y_upper.numpy(), color='gray', alpha=0.4, label="95% CI")
plt.title("Posterior Predictive with 95% Credible Interval")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_fit.png")

plt.show()
