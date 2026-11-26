Time Series
===========

Distributions over whole series, usable as priors on latent series and as likelihoods of observed
series, and state-space models whose hidden states are integrated out exactly. The special cases of
SARIMAX (MA, ARMA, ARIMA, seasonal ARIMA) are written with that one class, see its docstring.
:class:`~BayesianDLL.Deterministic.StationaryAR` and :class:`~BayesianDLL.Deterministic.InvertibleMA`,
which parameterise stationary AR and invertible MA coefficients, are documented under
:doc:`Deterministic </api/BayesianDLL.Deterministic>`.

.. currentmodule:: BayesianDLL.TimeSeries

.. autoclass:: GaussianRandomWalk
   :members:
   :show-inheritance:

.. autoclass:: AR
   :members:
   :show-inheritance:

.. autoclass:: StateSpace
   :members:
   :show-inheritance:

.. autoclass:: SARIMAX
   :members:
   :show-inheritance:
