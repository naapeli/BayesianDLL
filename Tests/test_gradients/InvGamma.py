import torch

from BayesianDLL.Distributions import InvGamma


a, b = torch.tensor(1, dtype=torch.float32), torch.tensor(2, dtype=torch.float32)
distribution = InvGamma(a, b)

x = (0.5 + torch.rand(size=(3, 1), dtype=torch.float64)).requires_grad_(True)
print(distribution.pdf(x).log())
result = distribution.log_pdf(x)
print(result)
result.sum().backward()
print(x.grad, x.grad.shape)
print(distribution.log_pdf_grad(x), distribution.log_pdf_grad(x).shape)


a.requires_grad = True
b.requires_grad = True
x = 0.5 + torch.rand(size=(3, 1), dtype=torch.float64)
result = distribution.log_pdf(x)
result.sum().backward()
print(a.grad)
print(distribution.log_pdf_param_grads(x)["alpha"].sum(), distribution.log_pdf_param_grads(x)["alpha"].shape)
print(b.grad)
print(distribution.log_pdf_param_grads(x)["beta"].sum(), distribution.log_pdf_param_grads(x)["beta"].shape)
