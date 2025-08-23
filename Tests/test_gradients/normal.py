import torch

from BayesianDLL.Distributions import Normal


mu, var = torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)
distribution = Normal(mu, var)

x = torch.randn(size=(3, 1), dtype=torch.float64).requires_grad_(True)
print(distribution.pdf(x).log())
result = distribution.log_pdf(x)
print(result)
result.sum().backward()
print(x.grad, x.grad.shape)
print(distribution.log_pdf_grad(x), distribution.log_pdf_grad(x).shape)


distribution.mu.requires_grad = True
distribution.variance.requires_grad = True
x = torch.randn(size=(3, 1), dtype=torch.float64)
result = distribution.log_pdf(x)
result.sum().backward()
print(distribution.mu.grad)
print(distribution.log_pdf_param_grads(x)["mean"], distribution.log_pdf_param_grads(x)["mean"].shape)
print(distribution.variance.grad)
print(distribution.log_pdf_param_grads(x)["variance"], distribution.log_pdf_param_grads(x)["variance"].shape)
