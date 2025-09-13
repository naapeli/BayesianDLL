import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy, MultivariateNormal
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, sample, posterior_predicative
from BayesianDLL import Evaluation


torch.manual_seed(7)

N = 100
true_coeffs, xmin, xmax = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111], -5, 5
true_variance = 0.1
x = torch.linspace(xmin, xmax, N, dtype=torch.float64)
X = torch.stack([x ** i for i in range(len(true_coeffs))], dim=1)
y = sum(c * x ** i for i, c in enumerate(true_coeffs)).unsqueeze(1) + torch.normal(0, true_variance ** 0.5, size=(N, 1))
y = y.to(torch.float64)


# Degree of the polynomial to be fitted
D = 5
x_scaled = (x - xmin) / (xmax - xmin) * 2 - 1  # Scaling to [-1, 1] for Chebyshev
phi_x = torch.stack([torch.cos(i * torch.acos(x_scaled)) for i in range(D + 1)], dim=1).to(torch.float64)  # Chebysev basis functions

with Model() as polynomial_model:
    # Priors
    prior_mean = torch.zeros(D + 1, dtype=torch.float64)
    prior_cov = torch.eye(D + 1, dtype=torch.float64)
    prior_coeffs = RandomParameter("coeffs", MultivariateNormal(prior_mean, prior_cov), torch.randn_like(prior_mean, dtype=torch.float64), min_step_size=1e-1)
    prior_sigma = RandomParameter("sigma", HalfCauchy(10), torch.ones(1, dtype=torch.float64), min_step_size=1e-1)

    mu = DeterministicParameter("mu", lambda coeffs: phi_x @ coeffs.T, lambda coeffs: {"coeffs": phi_x}, [prior_coeffs])
    
    likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)
    samples = sample(2000, 1000)
    posterior_predicative_distribution = posterior_predicative(samples, n_samples=1, samples_per_step=500, warmup_per_sample=100)

# print(Evaluation.gelman_rubin(samples, method="classical"))
# print(Evaluation.gelman_rubin(samples, method="split"))
# print(Evaluation.gelman_rubin(samples, method="rank"))
plt.figure(figsize=(12, 8))
Evaluation.Graphics.plot_posterior(samples)
Evaluation.Graphics.plot_predicative_distribution(posterior_predicative_distribution, y, kind="pdf", method="hist")
Evaluation.Graphics.plot_predicative_distribution(posterior_predicative_distribution, y, kind="cdf", method="hist")

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)

x = torch.linspace(xmin, xmax, 100)
x_scaled = (x - xmin) / (xmax - xmin) * 2 - 1  # Scaling to [-1, 1] for Chebyshev
phi_x = torch.stack([torch.cos(i * torch.acos(x_scaled)) for i in range(D + 1)], dim=1).to(torch.float64)
y_preds = (samples["coeffs"] @ phi_x.T).reshape(-1, len(x))  # (n_chains * n_samples, n_features)
y_mean = y_preds.mean(dim=0)
y_lower = y_preds.quantile(0.025, dim=0)
y_upper = y_preds.quantile(0.975, dim=0)
std = samples["sigma"].mean().sqrt()

plt.plot(x, y_mean, label="Posterior mean", color="black")
plt.fill_between(x.numpy(), y_lower.numpy(), y_upper.numpy(), color="blue", alpha=0.2, label="95% CI for mean")
plt.fill_between(x.numpy(), y_mean - 1.96 * std, y_mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI for data points")
plt.title("Posterior predictive distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_fit.png")

plt.show()
