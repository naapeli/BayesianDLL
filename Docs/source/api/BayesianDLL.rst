BayesianDLL
===========

The core probabilistic programming building blocks are importable directly from the top-level ``BayesianDLL`` package.
These include the model context manager, plate indexing, parameter node types, and high-level sampling, prediction, and transformation functions.

Model and Context
-----------------

.. currentmodule:: BayesianDLL

.. autoclass:: Model
   :members:
   :show-inheritance:

.. autoclass:: plate
   :show-inheritance:

.. autoclass:: Data
   :members:
   :show-inheritance:

Parameter Nodes
---------------

.. currentmodule:: BayesianDLL

.. autoclass:: RandomParameter
   :members:
   :show-inheritance:

.. autoclass:: ObservedParameter
   :members:
   :show-inheritance:

.. autoclass:: DeterministicParameter
   :members:
   :show-inheritance:

.. autoclass:: VariationalParameter
   :members:
   :show-inheritance:

Inference and Model Operations
------------------------------

.. currentmodule:: BayesianDLL

.. autofunction:: condition

.. autofunction:: marginalize

.. autofunction:: predict

.. autofunction:: point_trace

.. autofunction:: prior_predicative

.. autofunction:: posterior_predicative

.. autofunction:: sample
   :no-index:

.. autofunction:: find_MAP
   :no-index:

.. autoclass:: MAPResult
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: laplace_approximation
   :no-index:

Packages
--------

.. toctree::
   :maxdepth: 2

   BayesianDLL.Distributions
   BayesianDLL.Deterministic
   BayesianDLL.Samplers
   BayesianDLL.Variational
   BayesianDLL.GP
   BayesianDLL.Evaluation
