import torch
import matplotlib.pyplot as plt

from BayesianDLL.sampler import NUTS
from BayesianDLL.distributions import Beta, Bernoulli, Binomial
from BayesianDLL._transforms import LogitTransform


torch.manual_seed(0)
N = 5  # This needs to be small as pdf is not very numerically stable (sampling works well)
max_val = 1  # choose 1 for bernoulli prior and something else for binomial prior
data = torch.randint(0, max_val + 1, size=(N,))
a, b = 1, 1
def log_posterior(theta):
    if max_val == 1: return Beta(a, b)._log_prob_unconstrained(theta) + Bernoulli(LogitTransform().inverse(theta)).log_pdf(data).sum()
    return Beta(a, b)._log_prob_unconstrained(theta) + Binomial(max_val, LogitTransform().inverse(theta)).log_pdf(data).sum()

def log_posterior_derivative(theta):
    transform = LogitTransform()
    if max_val == 1: return Beta(a, b)._log_prob_grad_unconstrained(theta) + Bernoulli(transform.inverse(theta)).log_pdf_param_grads(data)["p"].sum() * transform.derivative(theta)
    return Beta(a, b)._log_prob_grad_unconstrained(theta) + Binomial(max_val, transform.inverse(theta)).log_pdf_param_grads(data)["p"].sum() * transform.derivative(theta)


sampler = NUTS(log_posterior, log_posterior_derivative, LogitTransform().inverse)
samples, _, _ = sampler.sample(10000, torch.tensor([2], dtype=torch.float64), 500)
plt.hist(samples.numpy(), bins=30, density=True)
x = torch.linspace(0, 1, 1000)
print(a + data.sum(), b + (max_val - data).sum())
plt.plot(x, Beta(a + data.sum(), b + (max_val - data).sum()).pdf(x))  # true posterior for both cases
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
