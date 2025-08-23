import torch

from BayesianDLL.Distributions import Mixture, Normal


weights = torch.tensor([1], dtype=torch.float32)
mean = torch.tensor(1, dtype=torch.float32)
variance = torch.tensor(1, dtype=torch.float32)
distribution = Mixture([Normal(mean, variance)], weights)

x = torch.randn(size=(3, 1), dtype=torch.float64).requires_grad_(True)
print(distribution.pdf(x).log())
result = distribution.log_pdf(x)
print(result)
result.sum().backward()
print(x.grad, x.grad.shape)
print(distribution.log_pdf_grad(x), distribution.log_pdf_grad(x).shape)


distribution.weights.requires_grad = True
mean.requires_grad = True
variance.requires_grad = True
x = torch.randn(size=(3, 1), dtype=torch.float64)
result = distribution.log_pdf(x)
result.sum().backward()
print(distribution.weights.grad, distribution.weights.grad.shape)
print(distribution.log_pdf_param_grads(x)["weights"], distribution.log_pdf_param_grads(x)["weights"].shape, distribution.log_pdf_param_grads(x)["weights"].sum(dim=0))
print(mean.grad, mean.grad.shape)
print(distribution.log_pdf_param_grads(x)["mean"], distribution.log_pdf_param_grads(x)["mean"].shape, distribution.log_pdf_param_grads(x)["mean"].sum(dim=0))
print(variance.grad, variance.grad.shape)
print(distribution.log_pdf_param_grads(x)["variance"], distribution.log_pdf_param_grads(x)["variance"].shape, distribution.log_pdf_param_grads(x)["variance"].sum(dim=0))
