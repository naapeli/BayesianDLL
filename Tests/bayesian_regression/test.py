import torch
import matplotlib.pyplot as plt

from BayesianDLL.sampler import NUTS
from BayesianDLL.distributions import MultivariateNormal, HalfCauchy, Normal

torch.manual_seed(7)

N = 5  # One can not use many more data points as the implementation is very unstable (the sampler gets stuck for some reason)
# true_coeffs = [0.0, 3488.378906, 0.0, -855.187500, 0.0, 107.675000, 0.0, -5.857143, 0.0, 0.111111]
true_coeffs = [0, 1, -3, 2]
true_variance = 1.0
xmin, xmax = 0, 1
x = torch.linspace(xmin, xmax, N)
X = torch.stack([x**i for i in range(len(true_coeffs))], dim=1)
y = sum(c * x**i for i, c in enumerate(true_coeffs)) + torch.normal(0, true_variance ** 0.5, size=(N,))

# Degree of the polynomial to be fitted
D = 3

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

    # print("Prior coeffs: ", prior_coeffs._log_prob_unconstrained(coeffs_unconstrained))
    # print("Prior variance: ", prior_variance._log_prob_unconstrained(variance_unconstrained))

    phi_x = torch.stack([x**i for i in range(D + 1)], dim=1).to(theta.dtype)
    mu = phi_x @ coeffs
    likelihood = Normal(mu, variance).log_pdf(y).sum()
    # print("Likelihood: ", likelihood)

    return prior + likelihood

def log_posterior_derivative(theta):
    coeffs_unconstrained = theta[:D + 1]
    variance_unconstrained = theta[D + 1]

    coeffs = prior_coeffs.transform.inverse(coeffs_unconstrained)
    variance = prior_variance.transform.inverse(variance_unconstrained)

    phi_x = torch.stack([x**i for i in range(D + 1)], dim=1).to(theta.dtype)
    mu = phi_x @ coeffs

    normal_dist = Normal(mu, variance)
    grads = normal_dist.log_pdf_param_grads(y)

    grad_coeffs = phi_x.T @ grads["mean"]
    grad_coeffs *= prior_coeffs.transform.derivative(coeffs_unconstrained)
    grad_coeffs += prior_coeffs._log_prob_grad_unconstrained(coeffs_unconstrained)

    d_variance = grads["variance"].sum() * prior_variance.transform.derivative(variance_unconstrained)
    grad_variance = prior_variance._log_prob_grad_unconstrained(variance_unconstrained) + d_variance.item()

    return torch.cat([grad_coeffs, torch.tensor([grad_variance], dtype=theta.dtype)])

def inverse_transformation(theta):
    coeffs = prior_coeffs.transform.inverse(theta[:, :D + 1])
    variance = prior_variance.transform.inverse(theta[:, D + 1])
    return torch.cat([coeffs, variance.unsqueeze(-1)], dim=-1)





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
    # print("\nManual gradient:")
    # print(manual_grad)
    # print("\nAutograd gradient:")
    # print(autograd_grad)

    return difference.item()

# for _ in range(10):
#     # theta must be of shape (D+2,), matching your implementation
#     theta_test = torch.randn(D + 2, dtype=torch.float64)
#     difference = test_gradient(theta_test)

#     # Check if the difference is below a reasonable threshold (e.g., 1e-6)
#     assert difference < 1e-6, f"Gradient check failed! {difference}"


import pyro
import pyro.distributions as dist
from pyro.poutine import trace
# Define Pyro model
def model(x, y=None):
    coeffs = pyro.sample("coeffs", dist.MultivariateNormal(torch.zeros(D + 1), 10.0 * torch.eye(D + 1)))
    variance = pyro.sample("variance", dist.HalfCauchy(scale=10.0))
    phi_x = torch.stack([x**i for i in range(D + 1)], dim=1)
    mu = phi_x @ coeffs
    with pyro.plate("data", len(x)):
        pyro.sample("obs", dist.Normal(mu, variance.sqrt()), obs=y)

# Create trace of model execution to access log joint
def compute_log_posterior(theta):
    coeffs = theta[:D + 1]
    variance = theta[D + 1]

    def conditioned_model():
        pyro.sample("coeffs", dist.MultivariateNormal(torch.zeros(D + 1), 10.0 * torch.eye(D + 1)), obs=coeffs)
        pyro.sample("variance", dist.HalfCauchy(scale=10.0), obs=variance)
        phi_x = torch.stack([x**i for i in range(D + 1)], dim=1).to(theta.dtype)
        mu = phi_x @ coeffs
        with pyro.plate("data", len(x)):
            pyro.sample("obs", dist.Normal(mu, variance.sqrt()), obs=y)

    tr = trace(conditioned_model).get_trace()
    return tr.log_prob_sum()

# Test
# for _ in range(10):
#     theta_test = 1 + torch.randn((1, D + 2))  # [coeff_0, coeff_1, variance]
#     phi_x = torch.stack([x**i for i in range(D + 1)], dim=1).to(theta_test.dtype)
#     # log_post = compute_log_posterior(phi_x, y, inverse_transformation(theta_test).squeeze())
#     # print("Pyro log posterior:", log_post.item())
#     # print("My log posterior", log_posterior(theta_test.squeeze()).item())

#     log_posterior(theta_test.squeeze())
#     coeffs = inverse_transformation(theta_test).squeeze()[:D + 1]
#     variance = inverse_transformation(theta_test).squeeze()[D + 1]
#     print("Pyro prior coeffs:", dist.MultivariateNormal(torch.zeros(D + 1), 10.0 * torch.eye(D + 1)).log_prob(coeffs))
#     print("Pyro prior variance:", dist.HalfCauchy(10).log_prob(variance))
#     print("Pyro likelihood:", dist.Normal(phi_x @ coeffs, variance.sqrt()).log_prob(y).sum())


# raise NotImplementedError()

def grad(theta):
    theta = theta.detach().clone().requires_grad_(True)
    log_posterior(theta).backward()
    return theta.grad

initial_point = torch.zeros(D + 2, dtype=torch.float64)
sampler = NUTS(log_posterior, log_posterior_derivative, inverse_transformation)
# sampler = NUTS(log_posterior, grad, inverse_transformation)
samples, _, _ = sampler.sample(3000, initial_point, 500)

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

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)

x = torch.linspace(xmin, xmax, 100)
phi_x = torch.stack([x**i for i in range(D + 1)], dim=1).to(torch.float64)
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

plt.savefig("Tests/bayesian_regression/polynomial.png")
plt.show()
