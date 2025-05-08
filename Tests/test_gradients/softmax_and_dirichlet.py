import torch

from BayesianDLL.Distributions import Dirichlet, InvGamma
from BayesianDLL.Distributions._transforms import SoftMaxTransform


n = 5
alpha = torch.rand(5)
distribution = Dirichlet(alpha)

y = torch.randn((1, n)).requires_grad_(True)
x = torch.softmax(y, dim=1)
x.retain_grad()

result = distribution.log_pdf(x)
result.backward()
print(result, distribution.pdf(x).log())
print(x.grad, distribution.log_pdf_grad(x))

y = torch.randn((1, n)).requires_grad_(True)
result = distribution._log_prob_unconstrained(y)
print(result)
result.backward()
print(y.grad, distribution._log_prob_grad_unconstrained(y))

y = torch.randn((1, n))
x = torch.softmax(y, dim=1)
transform = SoftMaxTransform()
print(transform.inverse(transform.forward(x)), x)


x = torch.randn((1, 1)).requires_grad_(True)
distribution = InvGamma(2, 3)
result = distribution._log_prob_unconstrained(x)
print(result.shape)
result.backward()
print(x.grad, distribution._log_prob_grad_unconstrained(x))
