import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, MeanFieldGuide, RandomParameter, VariationalParameter, DeterministicParameter, ObservedParameter
from BayesianDLL.Distributions import Normal, Mixture
from BayesianDLL.Variational import elbo


torch.manual_seed(0)
means = [-1, 2]
variances = [0.5 ** 2, 1 ** 2]
components = [Normal(mu, var) for mu, var in zip(means, variances)]
weights = [0.3, 0.7]
distribution = Mixture(components, weights)
theta_init = torch.tensor(0, dtype=torch.float64)
with Model() as model:
    RandomParameter("mixture", distribution, theta_init, sampler="nuts")

with MeanFieldGuide() as guide:
    meantensor = torch.full((1,), 10, dtype=torch.float32, requires_grad=True)
    mean = VariationalParameter("mean", meantensor)
    variancetensor = torch.full((1,), 1, dtype=torch.float32, requires_grad=True)
    variance = VariationalParameter("variance", variancetensor)
    RandomParameter("mixture", Normal(mean, variance), initial_value=torch.zeros(1))


loss, grads = elbo(model, guide, n_samples=1)  # with 100000 samples, almost equal
print(loss)
loss.backward()
print(meantensor.grad, variancetensor.grad)
print(grads)






torch.manual_seed(7)
torch.set_default_dtype(torch.float64)

# Generate synthetic data
N = 500  # 10
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).unsqueeze(1)
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N, 1))

with Model() as model:
    # Priors
    prior_intercept = RandomParameter("intercept", Normal(0, 20), torch.tensor(0), sampler="auto", delta=0.4)
    prior_slope = RandomParameter("slope", Normal(0, 20), torch.tensor(0), sampler="auto", delta=0.4)
    # prior_sigma = RandomParameter("sigma", HalfCauchy(10), torch.tensor(1), sampler="auto")
    prior_sigma = 0.5

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)

with MeanFieldGuide() as guide:
    interceptmeantensor = torch.full((1,), 0, requires_grad=True, dtype=torch.float64)
    interceptmean = VariationalParameter("mean1", interceptmeantensor)
    interceptvariancetensor = torch.full((1,), 3, requires_grad=True, dtype=torch.float64)
    interceptvariance = VariationalParameter("variance1", interceptvariancetensor, min=1e-8)
    RandomParameter("intercept", Normal(interceptmean, interceptvariance), torch.zeros(1))
    slopemeantensor = torch.full((1,), 0, requires_grad=True, dtype=torch.float64)
    slopemean = VariationalParameter("mean2", slopemeantensor)
    slopevariancetensor = torch.full((1,), 3, requires_grad=True, dtype=torch.float64)
    slopevariance = VariationalParameter("variance2", slopevariancetensor, min=1e-8)
    RandomParameter("slope", Normal(slopemean, slopevariance), torch.zeros(1))


print("=====================")
loss, grads = elbo(model, guide, n_samples=1)
print(loss)
loss.backward()
print(interceptmeantensor.grad, interceptvariancetensor.grad, slopemeantensor.grad, slopevariancetensor.grad)
print(grads.values())
