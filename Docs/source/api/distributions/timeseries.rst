Time Series
===========

Distributions over whole series, usable as priors on latent series and as likelihoods of observed
series, and state-space models whose hidden states are integrated out exactly.

.. currentmodule:: BayesianDLL.TimeSeries

.. autoclass:: GaussianRandomWalk
   :members:
   :show-inheritance:

.. autoclass:: AutoRegressive
   :members:
   :show-inheritance:

.. autofunction:: is_stationary

.. autoclass:: LinearGaussianStateSpace
   :members:
   :show-inheritance:

.. autoclass:: SARIMAX
   :members:
   :show-inheritance:

.. autoclass:: SARIMA
   :show-inheritance:

.. autoclass:: ARIMA
   :show-inheritance:

.. autoclass:: ARMA
   :show-inheritance:

.. autoclass:: MovingAverage
   :show-inheritance:

.. autoclass:: StationaryAR
   :show-inheritance:

.. autoclass:: InvertibleMA
   :show-inheritance:
