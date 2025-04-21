import torch
import matplotlib.pyplot as plt

from BayesianDLL.distributions import Normal, Beta, Exponential
from BayesianDLL.sampler import NUTS


# ================== DISTRIBUTIONS IN UNRESTRICTED SPACEES ==================
plt.figure()
plt.subplot(1, 3, 1)
a, b = 2, 2
distribution = Beta(a, b)
x = torch.linspace(-5, 5, 100)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()

plt.subplot(1, 3, 2)
distribution = Normal(0, 1)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()

plt.subplot(1, 3, 3)
distribution = Exponential(0.3)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()



# ================== SAMPLING ==================
n = 10000
bins = 30

plt.figure()
plt.subplot(1, 3, 1)
sampler = NUTS(Normal(0, 1))
theta_init = torch.tensor([0.1]).float()
samples, lnprob, _ = sampler.sample(n, theta_init, 1000)
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000)
y = Normal(0, 1).pdf(x)
plt.plot(x, y)

sampler = NUTS(Normal(5, 3))
theta_init = torch.tensor([0.1]).float()
samples, _, _ = sampler.sample(n, theta_init, 1000)
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000)
y = Normal(5, 3).pdf(x)
plt.plot(x, y)
plt.xlim(-5, 15)

plt.subplot(1, 3, 2)
distribution = Beta(2, 5)
sampler = NUTS(distribution)
theta_init = torch.tensor([0.0]).float()
samples, _, _ = sampler.sample(n, theta_init, 1000)
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0, 1, 100)
y = distribution.pdf(x)
plt.plot(x, y)

distribution = Beta(0.5, 0.5)
sampler = NUTS(distribution)
theta_init = torch.tensor([0.0]).float()
samples, _, _ = sampler.sample(n, theta_init, 1000)
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0.01, 0.99, 100)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(0, 1)

plt.subplot(1, 3, 3)
distribution = Exponential(0.3)
sampler = NUTS(distribution)
theta_init = torch.tensor([10]).float()
samples, _, _ = sampler.sample(n, theta_init, 1000)
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0, 20, 100)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(0, 20)

plt.show()
