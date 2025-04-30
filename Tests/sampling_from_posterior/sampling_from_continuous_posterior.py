import torch
import matplotlib.pyplot as plt

from BayesianDLL.Samplers import NUTS
from BayesianDLL.Distributions import Beta, Bernoulli, Binomial
from BayesianDLL.Distributions._transforms import LogitTransform


torch.manual_seed(0)
N = 5  # This needs to be small as pdf is not very numerically stable (sampling works well)
max_val = 1  # choose 1 for bernoulli prior and something else for binomial prior
data = torch.randint(0, max_val + 1, size=(N,))
a, b = 1, 1
prior = Beta(a, b)
def log_posterior(theta):
    constrained_theta = prior.transform.inverse(theta)
    likelihood = Bernoulli(constrained_theta) if max_val == 1 else Binomial(max_val, constrained_theta)
    return prior._log_prob_unconstrained(theta) + likelihood.log_pdf(data).sum()

def log_posterior_derivative(theta):
    constrained_theta = prior.transform.inverse(theta)
    prior_derivative = prior._log_prob_grad_unconstrained(theta)
    likelihood = Bernoulli(constrained_theta) if max_val == 1 else Binomial(max_val, constrained_theta)
    likelihood_derivative = likelihood.log_pdf_param_grads(data)["p"].sum() * prior.transform.derivative(theta)
    return prior_derivative + likelihood_derivative

sampler = NUTS(log_posterior, log_posterior_derivative, LogitTransform().inverse)
theta = torch.tensor([2], dtype=torch.float64)
n = 10500
samples = torch.empty(size=(n,))
for i in range(n):
    theta = sampler.step(theta)
    samples[i] = theta
samples = LogitTransform().inverse(samples[500:])
# samples, _, _ = sampler.sample(10000, torch.tensor([2], dtype=torch.float64), 500)

plt.hist(samples.numpy(), bins=30, density=True)
x = torch.linspace(0, 1, 1000)
print(a + data.sum(), b + (max_val - data).sum())
plt.plot(x, Beta(a + data.sum(), b + (max_val - data).sum()).pdf(x))  # true posterior for both cases
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
