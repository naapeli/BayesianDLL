Time Series
===========

Distributions over whole series (random walk, autoregression), the linear
Gaussian state-space model with an exact Kalman-filter likelihood, and SARIMAX
(MA, ARMA, ARIMA and seasonal ARIMA models) built on it. The hidden states are integrated
out exactly and every gradient is hand-written, so NUTS runs on them directly.
