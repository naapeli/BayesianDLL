Bayesian Deep Learning Library documentation
============================================

BayesianDLL is a Python library for Bayesian machine learning built on top of
PyTorch. It lets you write probabilistic models with probability distributions
and graphical models, and fit them with Markov chain Monte Carlo (NUTS and
Metropolis) or variational inference.

The library is intended as an educational project. It aims for clarity and ease
of understanding rather than raw performance, so for production work libraries
such as PyMC, Pyro or NumPyro are a better fit.

Library Structure
-----------------

Models are written inside a ``with Model():`` block using the core building
blocks found at the top level of the package (``Model``, ``RandomParameter``,
``ObservedParameter``, ``DeterministicParameter``, ``Data``, ``plate``, ...).
The rest of the library is split into the following packages:

- **Distributions:** Probability distributions, their state spaces and the transforms to unconstrained space.
- **Deterministic:** Deterministic nodes such as ``Linear``, ``Exp``, ``Log`` and ``Sigmoid`` for building model graphs.
- **Samplers:** MCMC samplers (NUTS and Metropolis), sampling utilities and prior / posterior predictive sampling.
- **Variational:** Maximum a posteriori estimation and black-box variational inference.
- **GP:** Gaussian process kernels and latent and exact Gaussian processes.
- **Evaluation:** Convergence diagnostics (R-hat, effective sample size, MCSE), posterior summaries and plotting.

Installation
------------

BayesianDLL requires Python >= 3.13 and `uv <https://docs.astral.sh/uv/>`_.
From the project root:

.. code-block:: sh

   uv sync --dev

A minimal example
-----------------

.. code-block:: python

   import torch
   from BayesianDLL import Model, RandomParameter, ObservedParameter, sample, plate
   from BayesianDLL.Distributions import Normal, InvGamma
   from BayesianDLL.Evaluation import summary

   data = torch.normal(5.0, 3.0 ** 0.5, size=(100,), dtype=torch.float64)

   with Model() as model:
       mean = RandomParameter("mean", Normal(0, 10))
       variance = RandomParameter("variance", InvGamma(2, 2))
       with plate("data", len(data)):
           ObservedParameter("likelihood", Normal(mean, variance), data)
       samples = sample(1000, 500, n_chains=2)

   print(summary(samples))

For detailed documentation or more example scripts, refer to the sections below.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   api/BayesianDLL

Examples
--------

.. toctree::
   :maxdepth: 2

   auto_examples/index
