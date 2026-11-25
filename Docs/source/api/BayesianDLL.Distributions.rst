Distributions
=============

BayesianDLL probability distributions define the prior and likelihood components of probabilistic models.
Every distribution provides methods to evaluate log densities (and their hand-written analytical gradients),
to sample prior or posterior values ancestrally, and to map between constrained support and unconstrained spaces.

Base Distribution
-----------------

.. currentmodule:: BayesianDLL.Distributions

.. autoclass:: Distribution
   :members:
   :show-inheritance:

Subpackages and Categories
--------------------------

.. toctree::
   :maxdepth: 2

   distributions/continuous
   distributions/discrete
   distributions/multivariate
   distributions/compound
   distributions/timeseries
   distributions/transforms
