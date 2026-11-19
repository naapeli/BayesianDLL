Variational Inference
=====================

Optimisation-based alternatives to MCMC: the maximum a posteriori estimate
(``find_MAP``), the Laplace approximation, and black-box variational
inference (``BBVI``) with guides built from Gaussian groups
(``Guide(model, default_family=...)``, ``Group``) and entries written by
hand, including minibatch
training and Gaussian-process models. Wherever an exact posterior or a NUTS
run is cheap, the examples overlay it, so you can see what each
approximation captures and what it misses.
