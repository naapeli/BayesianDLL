import torch
import matplotlib.pyplot as plt

from BayesianDLL.sampler import NUTS
from BayesianDLL.distributions import Normal, InvGamma


torch.manual_seed(7)
N = 100  # Due to numerical instability, one cannot use much larger N (If one use torch.float32 in the initial point for the sampler, N can only be about 5)
mu0 = 0
tau = 10
a = 2
b = 2
true_mean, true_variance = 5, 3
data = torch.normal(mean=true_mean, std=true_variance ** 0.5, size=(N,))
print(data.mean(), data.var())
prior_mean = Normal(mu0, tau)
prior_variance = InvGamma(a, b)
def log_posterior(theta):
    mean, variance = theta[0], theta[1]
    prior = prior_mean._log_prob_unconstrained(mean) + prior_variance._log_prob_unconstrained(variance)
    likelihood = Normal(prior_mean.transform.inverse(mean), prior_variance.transform.inverse(variance)).log_pdf(data).sum()
    return prior + likelihood

def log_posterior_derivative(theta):
    mean, variance = theta[0], theta[1]
    likelihood_derivatives = Normal(prior_mean.transform.inverse(mean), prior_variance.transform.inverse(variance)).log_pdf_param_grads(data)
    likelihood_derivative_mean, likelihood_derivative_variance = likelihood_derivatives["mean"].sum(), likelihood_derivatives["variance"].sum()
    mean_derivative = prior_mean._log_prob_grad_unconstrained(mean) + likelihood_derivative_mean * prior_mean.transform.derivative(mean)
    std_derivative = prior_variance._log_prob_grad_unconstrained(variance) + likelihood_derivative_variance * prior_variance.transform.derivative(variance)
    return torch.tensor([mean_derivative.item(), std_derivative.item()], dtype=theta.dtype)

def test_log_posterior_derivative():
    for point in [torch.normal(0, 10, size=(2,)) for _ in range(10)]:
        theta = torch.tensor(point, dtype=torch.float64, requires_grad=True)

        lp = log_posterior(theta)
        lp.backward()
        autograd_grad = theta.grad.clone()

        theta = theta.detach()
        manual_grad = log_posterior_derivative(theta)

        print(autograd_grad, manual_grad)
        assert torch.allclose(autograd_grad, manual_grad, atol=1e-5), f"Autograd: {autograd_grad}, Manual: {manual_grad}"

# test_log_posterior_derivative()

def inverse_transformation(theta):
    mean, std = theta[:, 0], theta[:, 1]
    return torch.stack([prior_mean.transform.inverse(mean), prior_variance.transform.inverse(std)], dim=-1)

sampler = NUTS(log_posterior, log_posterior_derivative, inverse_transformation)
samples, _, _ = sampler.sample(2000, torch.tensor([0, 0], dtype=torch.float64), 500)
mean_samples = samples[:, 0]
variance_samples = samples[:, 1]


plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.hist(mean_samples.numpy(), bins=50, density=True)
plt.title("Posterior of μ")

plt.subplot(2, 2, 2)
plt.hist(variance_samples.numpy(), bins=50, density=True)
plt.title("Posterior of σ²")

plt.subplot(2, 2, 3)
plt.plot(mean_samples)
print("Posterior mean: ", mean_samples.mean().item())
plt.plot(torch.ones_like(mean_samples) * mean_samples.mean().item())
plt.title("mean")

plt.subplot(2, 2, 4)
plt.plot(variance_samples)
print("Posterior variance: ", variance_samples.mean().item())
plt.plot(torch.ones_like(variance_samples) * variance_samples.mean().item())
plt.title("variance")

plt.tight_layout()
plt.savefig("Tests/sampling_from_joint_posterior/posteriors.png")
plt.show()
