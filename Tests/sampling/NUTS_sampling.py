import torch
import matplotlib.pyplot as plt
import numpy as np

from BayesianDLL.Distributions import Normal, Beta, Exponential, Uniform, InvGamma, HalfCauchy, Dirichlet, Mixture
from BayesianDLL import Model, RandomParameter, sample


# ================== DISTRIBUTIONS IN UNRESTRICTED SPACES ==================
plt.figure(figsize=(6, 6))
plt.subplot(3, 3, 1)
distribution = Beta(2, 2)
x = torch.linspace(-5, 5, 100).unsqueeze(1)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Beta")

plt.subplot(3, 3, 2)
distribution = Normal(0, 1)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Normal")

plt.subplot(3, 3, 3)
distribution = Exponential(0.3)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Exponential")

plt.subplot(3, 3, 4)
distribution = Uniform(2, 5)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Uniform")

plt.subplot(3, 3, 5)
distribution = InvGamma(2, 2)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Inverse gamma")

plt.subplot(3, 3, 6)
distribution = HalfCauchy(2)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Half Cauchy")

plt.subplot(3, 3, 7)
means = [-1, 2]
variances = [0.5 ** 2, 1 ** 2]
components = [Normal(mu, var) for mu, var in zip(means, variances)]
weights = [0.3, 0.7]
distribution = Mixture(components, weights)
plt.plot(x.numpy(), distribution._log_prob_unconstrained(x).numpy(), label='log_pdf')
plt.plot(x.numpy(), distribution._log_prob_grad_unconstrained(x).numpy(), label='log_pdf_grad')
plt.legend()
plt.title("Mixture of Gaussians")

plt.tight_layout()
plt.savefig("Tests/sampling/pdfs.png")


# # ================== SAMPLING ==================
n = 10000
bins = 30

plt.figure(figsize=(6, 6))

plt.subplot(3, 3, 1)
distribution = Normal(0, 1)
theta_init = torch.tensor(0.1, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.title("Normal")

distribution = Normal(5, 3)
theta_init = torch.tensor(0.1, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(-10, 20, 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(-5, 15)

plt.subplot(3, 3, 2)
distribution = Beta(2, 5)
theta_init = torch.tensor(0.5, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0, 1, 100).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.title("Beta")

distribution = Beta(0.5, 0.5)
theta_init = torch.tensor(0.5, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts", gamma=5, delta=0.9)
    samples = sample(n, 2000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0.01, 0.99, 100).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(0, 1)

plt.subplot(3, 3, 3)
distribution = Exponential(0.3)
theta_init = torch.tensor(2, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0, 20, 100).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(0, 20)
plt.title("Exponential")

plt.subplot(3, 3, 4)
distribution = Uniform(2, 5)
theta_init = torch.tensor(3, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(0, 7, 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(0, 7)
plt.title("Uniform")

plt.subplot(3, 3, 5)
distribution = InvGamma(3, 6)
theta_init = torch.tensor(1, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.xscale("log")
plt.yscale("log")
xmin = samples.min().item()
xmax = samples.max().item()
bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
plt.hist(samples.numpy(), bins=bin_edges, alpha=0.5, density=True)
x = torch.logspace(torch.log10(torch.tensor(xmin)), torch.log10(torch.tensor(xmax)), 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x.numpy(), y.numpy())
plt.title("Inverse gamma")

plt.subplot(3, 3, 6)
distribution = HalfCauchy(2)
theta_init = torch.tensor(1, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.xscale("log")
plt.yscale("log")
xmin = samples.min().item()
xmax = samples.max().item()
bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
plt.hist(samples.numpy(), bins=bin_edges, alpha=0.5, density=True)
x = torch.logspace(torch.log10(torch.tensor(xmin)), torch.log10(torch.tensor(xmax)), 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x.numpy(), y.numpy())
plt.title("Half Cauchy")

plt.subplot(3, 3, 7)
d = 3
# alpha = 5 * torch.ones(d, dtype=torch.float64)
alpha = torch.tensor([2, 4, 8], dtype=torch.float64)
distribution = Dirichlet(alpha)
theta_init = torch.ones(d, dtype=torch.float64)
theta_init[0], theta_init[1] = 2, -10  # make the initial point in an area with little probability mass to stabilize the sampling
theta_init = torch.softmax(theta_init, dim=0)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts", gamma=5)
    samples = sample(max(n // 20, 100), 2000)["sample"]
def project_to_2d(points):
    v1 = torch.tensor([0.0, 0.0])
    v2 = torch.tensor([1.0, 0.0])
    v3 = torch.tensor([0.5, 3**0.5 / 2])
    V = torch.stack([v1, v2, v3]).to(points.dtype)
    return points @ V
samples_2d = project_to_2d(samples)
def shade_pdf():
    resolution = 200
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 2])
    V = np.stack([v1, v2, v3], axis=0)
    grid = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            x = i / resolution
            y = j / resolution
            z = k / resolution
            grid.append([x, y, z])
    grid = np.array(grid)
    projected = grid @ V
    simplex_points = torch.tensor(grid, dtype=torch.float64)
    density = distribution.pdf(simplex_points).squeeze().numpy()
    return projected, density
projected, density = shade_pdf()
plt.tripcolor(projected[:, 0], projected[:, 1], density, shading="gouraud", cmap="Blues")
plt.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.7, s=1)
plt.title("Dirichlet")
triangle_vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 3**0.5 / 2], [0.0, 0.0]])
plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], '-', lw=2, c="black")

plt.subplot(3, 3, 8)
means = [-1, 2]
variances = [0.5 ** 2, 1 ** 2]
components = [Normal(mu, var) for mu, var in zip(means, variances)]
weights = [0.3, 0.7]
distribution = Mixture(components, weights)
theta_init = torch.tensor(0, dtype=torch.float64)
with Model() as model:
    RandomParameter("sample", distribution, theta_init, sampler="nuts")
    samples = sample(n, 1000)["sample"]
plt.hist(samples.numpy(), bins=bins, alpha=0.5, density=True)
x = torch.linspace(-5, 5, 1000).unsqueeze(1)
y = distribution.pdf(x)
plt.plot(x, y)
plt.xlim(-5, 5)
plt.title("Mixture of Gaussians")

plt.tight_layout()
plt.savefig("Tests/sampling/distributions.png")
plt.show()
