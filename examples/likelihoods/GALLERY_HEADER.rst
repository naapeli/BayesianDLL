Likelihoods
===========

Observation models beyond the Gaussian, Bernoulli and Poisson: ordinal
outcomes (``OrderedLogistic``), measurements that are only known to lie
beyond a limit (``Censored``), counts with a separate zero process
(``Hurdle``) and counts that vary more than a Binomial, Multinomial or
Poisson allows (``BetaBinomial`` and ``DirichletMultinomial``). Every density
and gradient is hand-written, so NUTS samples them directly.
