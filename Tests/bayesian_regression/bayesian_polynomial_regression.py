import torch
import matplotlib.pyplot as plt

from BayesianDLL.Samplers import NUTS
from BayesianDLL.Distributions import MultivariateNormal, HalfCauchy, Normal

torch.manual_seed(7)

N = 100
true_coeffs, xmin, xmax = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111], -5, 5
# true_coeffs, xmin, xmax = [0, 1, -3, 2], 0, 1
true_variance = 0.1  # NOTE: Stability of the sampler depends a lot on the original variance. If this is too large, the sampler might get stuck and one needs to use less data points.
x = torch.linspace(xmin, xmax, N)
X = torch.stack([x ** i for i in range(len(true_coeffs))], dim=1)
y = sum(c * x ** i for i, c in enumerate(true_coeffs)) + torch.normal(0, true_variance ** 0.5, size=(N,))

# Degree of the polynomial to be fitted
D = 5
phi_x = torch.stack([x ** i for i in range(D + 1)], dim=1).to(torch.float64)

prior_variance = 20
prior_mean = torch.zeros(D + 1, dtype=torch.float64)
prior_cov = prior_variance * torch.eye(D + 1, dtype=torch.float64)
prior_coeffs = MultivariateNormal(prior_mean, prior_cov)
prior_variance = HalfCauchy(10)

def log_posterior(theta):
    coeffs_unconstrained = theta[:D + 1]
    variance_unconstrained = theta[D + 1]

    coeffs = prior_coeffs.transform.inverse(coeffs_unconstrained)
    variance = prior_variance.transform.inverse(variance_unconstrained)

    prior = prior_coeffs._log_prob_unconstrained(coeffs_unconstrained)
    prior += prior_variance._log_prob_unconstrained(variance_unconstrained)

    mu = phi_x @ coeffs
    likelihood = Normal(mu, variance).log_pdf(y).sum()

    return prior + likelihood

def log_posterior_derivative(theta):
    coeffs_unconstrained = theta[:D + 1]
    variance_unconstrained = theta[D + 1]

    coeffs = prior_coeffs.transform.inverse(coeffs_unconstrained)
    variance = prior_variance.transform.inverse(variance_unconstrained)

    mu = phi_x @ coeffs

    normal_dist = Normal(mu, variance)
    grads = normal_dist.log_pdf_param_grads(y)

    grad_coeffs = phi_x.T @ grads["mean"] * prior_coeffs.transform.derivative(coeffs_unconstrained)
    grad_coeffs += prior_coeffs._log_prob_grad_unconstrained(coeffs_unconstrained)

    d_variance = grads["variance"].sum() * prior_variance.transform.derivative(variance_unconstrained)
    grad_variance = prior_variance._log_prob_grad_unconstrained(variance_unconstrained) + d_variance.item()

    return torch.cat([grad_coeffs, torch.tensor([grad_variance], dtype=theta.dtype)])

def inverse_transformation(theta):
    coeffs = prior_coeffs.transform.inverse(theta[:, :D + 1])
    variance = prior_variance.transform.inverse(theta[:, D + 1])
    return torch.cat([coeffs, variance.unsqueeze(-1)], dim=-1)

initial_point = torch.zeros(D + 2, dtype=torch.float64)
sampler = NUTS(log_posterior, log_posterior_derivative, inverse_transformation)
samples, _, _ = sampler.sample(1000, initial_point, 100)

coeff_samples = [samples[:, i] for i in range(D + 1)]
coeff_means = [coeff.mean() for coeff in coeff_samples]
variance_samples = samples[:, D + 1]
print(f"Posterior coefficient estimates: {coeff_means}")
print(f"Posterior variance estimate: {variance_samples.mean()}")

plt.figure(figsize=(12, 6))
for i, cs in enumerate(coeff_samples):
    plt.subplot(2, (D + 2) // 2 + 1, i + 1)
    plt.hist(cs.numpy(), bins=50, density=True)
    plt.title(f"Posterior of θ{i}")
plt.subplot(2, (D + 2) // 2 + 1, D + 2)
plt.hist(variance_samples.numpy(), bins=50, density=True)
plt.title("Posterior of σ")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_posteriors.png")

plt.figure(figsize=(12, 6))
for i, cs in enumerate(coeff_samples):
    plt.subplot(2, (D + 2) // 2 + 1, i + 1)
    plt.plot(cs.numpy())
    plt.title(f"Posterior of θ{i}")
plt.subplot(2, (D + 2) // 2 + 1, D + 2)
plt.plot(variance_samples.numpy())
plt.title("Posterior of σ")
plt.tight_layout()
plt.savefig("Tests/bayesian_regression/polynomial_trace_plots.png")

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)

x = torch.linspace(xmin, xmax, 100)
phi_x = torch.stack([x ** i for i in range(D + 1)], dim=1).to(torch.float64)
y_preds = torch.stack([phi_x @ torch.stack([samples[:, i] for i in range(D + 1)], dim=1)[j] for j in range(samples.shape[0])])
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
