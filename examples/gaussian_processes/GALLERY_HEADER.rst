Gaussian Processes
==================

A Gaussian process is a prior over functions: any finite set of function
values is jointly Gaussian, with a covariance given by a kernel that encodes
smoothness, periodicity or the relevance of each input. These examples fit
GP models with hand-derived gradients and sample their hyperparameters with
NUTS:

* ``ExactGP`` integrates the function out for Gaussian noise, and
  ``LatentGP`` samples it in whitened form for any likelihood
  (e.g. classification).
* ``SparseGP`` scales to many observations with inducing inputs (VFE, DTC,
  FITC), which can themselves be learned.
* ``KroneckerGP`` and ``ToeplitzGP`` stay exact but exploit grid and
  regular-spacing structure to avoid the cubic cost of a dense Cholesky
  factorisation.
