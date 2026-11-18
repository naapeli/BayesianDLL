Examples
========

Short, self-contained scripts that show how to build Bayesian models with
BayesianDLL, sample or approximate their posteriors, and check the results.
Every script simulates its own data with a fixed seed, so its numbers and
figures can be reproduced, and most of them compare the fit with the true
parameters or an exact answer.

Run an example from the project root as a module, for example::

    uv run python -m examples.bayesian_regression.logistic_regression

Set ``MPLBACKEND=Agg`` to run the scripts without opening figure windows.
