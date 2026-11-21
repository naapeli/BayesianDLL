BayesianDLL
===========

The core modelling building blocks are importable directly from ``BayesianDLL``.
``Guide`` and ``Group`` (also importable from here) are documented in
:class:`BayesianDLL.Variational.Guide` and :class:`BayesianDLL.Variational.Group`.

.. automodule:: BayesianDLL
   :members: Model, condition, marginalize, Data, RandomParameter, ObservedParameter, DeterministicParameter, VariationalParameter, plate, sample, sample_prior_predicative, sample_posterior_predicative, laplace_approximation, find_MAP
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   BayesianDLL.Distributions
   BayesianDLL.Deterministic
   BayesianDLL.Samplers
   BayesianDLL.Variational
   BayesianDLL.GP
   BayesianDLL.Evaluation
