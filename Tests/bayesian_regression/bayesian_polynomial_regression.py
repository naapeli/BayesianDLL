import torch
import matplotlib.pyplot as plt

from BayesianDLL.sampler import NUTS
from BayesianDLL.distributions import Normal, HalfCauchy

torch.manual_seed(7)

N = 100
# true_coeffs = [0.0, 3488.378906, 0.0, -855.187500, 0.0, 107.675000, 0.0, -5.857143, 0.0, 0.111111]
true_coeffs = [1, 2]
true_sigma = 1.0
xmin, xmax = 0, 1
x = torch.linspace(xmin, xmax, N)
X = torch.stack([x**i for i in range(len(true_coeffs))], dim=1)
y = sum(c * x**i for i, c in enumerate(true_coeffs)) + torch.normal(0, true_sigma, size=(N,))

# Degree of the polynomial to be fitted
D = 1

# Priors
prior_coeffs = [Normal(0, 10) for _ in range(D + 1)]
prior_sigma = HalfCauchy(10)

def log_posterior(theta):
    coeffs_unconstrained = theta[:D+1]
    sigma_unconstrained = theta[D+1]

    coeffs = [prior_coeffs[i].transform.inverse(coeffs_unconstrained[i]) for i in range(D+1)]
    sigma = prior_sigma.transform.inverse(sigma_unconstrained)

    prior = sum(prior_coeffs[i]._log_prob_unconstrained(coeffs_unconstrained[i]) for i in range(D+1))
    prior = prior + prior_sigma._log_prob_unconstrained(sigma_unconstrained)

    phi_x = torch.stack([x**i for i in range(D+1)], dim=1).to(theta.dtype)
    mu = phi_x @ torch.tensor(coeffs, dtype=theta.dtype)
    likelihood = Normal(mu, sigma).log_pdf(y).sum()

    return prior + likelihood

def log_posterior_derivative(theta):
    coeffs_unconstrained = theta[:D+1]
    sigma_unconstrained = theta[D+1]

    coeffs = [prior_coeffs[i].transform.inverse(coeffs_unconstrained[i]) for i in range(D+1)]
    sigma = prior_sigma.transform.inverse(sigma_unconstrained)

    phi_x = torch.stack([x**i for i in range(D+1)], dim=1).to(theta.dtype)
    mu = phi_x @ torch.tensor(coeffs, dtype=theta.dtype)

    normal_dist = Normal(mu, sigma)
    grads = normal_dist.log_pdf_param_grads(y)

    grad_coeffs = []
    for i in range(D+1):
        deriv = (grads["mean"] * x**i).sum()
        grad = deriv * prior_coeffs[i].transform.derivative(coeffs_unconstrained[i])
        grad += prior_coeffs[i]._log_prob_grad_unconstrained(coeffs_unconstrained[i])
        grad_coeffs.append(grad.item())

    d_sigma = grads["variance"].sum() * prior_sigma.transform.derivative(sigma_unconstrained)
    grad_sigma = prior_sigma._log_prob_grad_unconstrained(sigma_unconstrained) + d_sigma.item()

    return torch.tensor(grad_coeffs + [grad_sigma], dtype=theta.dtype)

def test_gradient(theta):
    theta = theta.detach().clone().requires_grad_(True)

    # Autograd-computed gradient
    autograd_logp = log_posterior(theta)
    autograd_grad = torch.autograd.grad(autograd_logp, theta)[0]

    # Manually implemented gradient
    manual_grad = log_posterior_derivative(theta.detach())

    # Compare both gradients
    difference = torch.norm(autograd_grad - manual_grad)
    print(f"L2 norm of gradient difference: {difference.item():.6e}")

    # Optional: print both gradients side-by-side
    print("\nManual gradient:")
    print(manual_grad)
    print("\nAutograd gradient:")
    print(autograd_grad)

    return difference.item()

# for _ in range(10):
#     # theta must be of shape (D+2,), matching your implementation
#     theta_test = torch.randn(D + 2, dtype=torch.float64)
#     difference = test_gradient(theta_test)

#     # Check if the difference is below a reasonable threshold (e.g., 1e-6)
#     assert difference < 1e-6, f"Gradient check failed! {difference}"

def grad(theta):
    theta = theta.detach().clone().requires_grad_(True)
    log_posterior(theta).backward()
    return theta.grad


def inverse_transformation(theta):
    coeffs = [prior_coeffs[i].transform.inverse(theta[:, i]) for i in range(D+1)]
    sigma = prior_sigma.transform.inverse(theta[:, D+1])
    return torch.stack(coeffs + [sigma], dim=-1)

initial_point = torch.zeros(D + 2, dtype=torch.float64)
# sampler = NUTS(log_posterior, log_posterior_derivative, inverse_transformation)
sampler = NUTS(log_posterior, grad, inverse_transformation)
samples, _, _ = sampler.sample(1000, initial_point, 0)
print(samples)

coeff_samples = [samples[:, i] for i in range(D+1)]
sigma_samples = samples[:, D+1]

plt.figure(figsize=(12, 6))
for i, cs in enumerate(coeff_samples):
    plt.subplot(2, (D + 2) // 2 + 1, i + 1)
    plt.hist(cs.numpy(), bins=50, density=True)
    plt.title(f"Posterior of θ{i}")
plt.subplot(2, (D + 2) // 2 + 1, D + 2)
plt.hist(sigma_samples.numpy(), bins=50, density=True)
plt.title("Posterior of σ")
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)

x = torch.linspace(xmin, xmax, 100)
phi_x = torch.stack([x**i for i in range(D+1)], dim=1).to(torch.float64)
y_preds = torch.stack([phi_x @ torch.stack([samples[:, i] for i in range(D+1)], dim=1)[j] for j in range(samples.shape[0])])
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

plt.savefig("Tests/bayesian_regression/polynomial.png")
plt.show()
