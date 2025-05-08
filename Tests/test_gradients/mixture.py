import torch

from BayesianDLL.Distributions import Mixture, Normal, MultivariateNormal, Exponential


weights = torch.tensor([0.2, 0.8])
# distribution = Mixture([Normal(0, 1), Normal(2, 4)], weights)
# distribution = Mixture([Exponential(0.3), Exponential(0.2), Exponential(0.5)], torch.tensor([0.2, 0.3, 0.5]))
n = 5
cov1 = torch.randn((n, n), dtype=torch.float64)
cov2 = torch.randn((n, n), dtype=torch.float64)
distribution = Mixture([MultivariateNormal(torch.randn(n, dtype=torch.float64), (cov1.T @ cov1) / 2),
                        MultivariateNormal(torch.randn(n, dtype=torch.float64), (cov2.T @ cov2) / 2)
                        ], weights)

x = torch.ones((1, n), dtype=torch.float64).requires_grad_(True)
print(distribution.pdf(x).log())
result = distribution.log_pdf(x)
print(result)
result.backward()
print(x.grad)
print(distribution.log_pdf_grad(x))


distribution.weights.requires_grad = True
x = torch.ones((1, n), dtype=torch.float64)
result = distribution.log_pdf(x)
result.backward()
print(distribution.weights.grad)
print(distribution.log_pdf_param_grads(x)["weights"])
