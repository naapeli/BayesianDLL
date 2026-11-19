BayesianDLL
===========

The core modelling building blocks are importable directly from ``BayesianDLL``.
``Guide`` (also importable from here) is documented with the other guides in
:class:`BayesianDLL.Variational.Guide`.

.. automodule:: BayesianDLL
   :members: Model, condition, Data, RandomParameter, ObservedParameter, DeterministicParameter, VariationalParameter, plate
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   BayesianDLL.Distributions
   BayesianDLL.Deterministic
   BayesianDLL.Samplers
   BayesianDLL.Variational
   BayesianDLL.GP
   BayesianDLL.Evaluation
