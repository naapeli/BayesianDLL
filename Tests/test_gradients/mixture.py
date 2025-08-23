import torch

from BayesianDLL.Distributions import Mixture, Normal, MultivariateNormal, Exponential


weights = torch.tensor([0.2, 0.8])
# distribution = Mixture([Normal(0, 1), Normal(2, 4)], weights)
# distribution = Mixture([Exponential(0.3), Exponential(0.2), Exponential(0.5)], torch.tensor([0.2, 0.3, 0.5]))
n = 5
cov1 = torch.randn((n, n), dtype=torch.float64)
cov1 = (cov1.T @ cov1) / 2
cov2 = torch.randn((n, n), dtype=torch.float64)
cov2 = (cov2.T @ cov2) / 2
mu1 = torch.randn(n, dtype=torch.float64)
mu2 = torch.randn(n, dtype=torch.float64)
distribution = Mixture([MultivariateNormal(mu1, cov1)], weights)  # , MultivariateNormal(mu2, cov2)

x = torch.randn(size=(3, n), dtype=torch.float64).requires_grad_(True)
print(distribution.pdf(x).log())
result = distribution.log_pdf(x)
print(result)
result.sum().backward()
print(x.grad, x.grad.shape)
print(distribution.log_pdf_grad(x), distribution.log_pdf_grad(x).shape)


distribution.weights.requires_grad = True
mu1.requires_grad = True
cov1.requires_grad = True
x = torch.randn(size=(3, n), dtype=torch.float64)
result = distribution.log_pdf(x)
result.sum().backward()
print(distribution.weights.grad, distribution.weights.grad.shape)
print(distribution.log_pdf_param_grads(x)["weights"], distribution.log_pdf_param_grads(x)["weights"].shape)
print(mu1.grad, mu1.grad.shape)
print(distribution.log_pdf_param_grads(x)["mean"], distribution.log_pdf_param_grads(x)["mean"].shape)
print(cov1.grad, cov1.grad.shape)
print(distribution.log_pdf_param_grads(x)["variance"], distribution.log_pdf_param_grads(x)["weights"].shape)
