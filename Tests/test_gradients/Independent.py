import torch

from BayesianDLL.Distributions import Normal, Independent



mu = torch.randn((10, 1))
mu.requires_grad = True
distribution = Independent(Normal(mu, 1), dims=0)
x = torch.randn(size=(1, 1))
x.requires_grad = True
result = distribution._log_prob_unconstrained(x)
result.backward()
print(x.grad)
print(distribution._log_prob_grad_unconstrained(x))
print(mu.grad)
print(distribution.log_pdf_param_grads(x)["mean"])
