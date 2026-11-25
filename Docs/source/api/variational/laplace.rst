Laplace Approximation
=====================

Gaussian approximation of the posterior in unconstrained space using the curvature (negative Hessian) at the MAP mode.

.. currentmodule:: BayesianDLL.Variational

.. autofunction:: laplace_approximation

.. autoclass:: LaplaceApproximation
   :members:
   :show-inheritance:

Curvature
---------

``hessian="fd"`` differentiates the hand-written gradient by central differences. The result is the exact
Hessian, but it is indefinite away from a proper mode, which makes Newton steps and the Gaussian fail for
neural networks.

``hessian="ggn"`` uses the generalised Gauss-Newton matrix instead, which is positive semi-definite and works for
every model. Each log-density term :math:`\log p(v \mid \eta(\theta))` contributes

.. math::

    J^\top F J, \qquad J = \frac{\partial \eta}{\partial \theta},

where :math:`\theta` are the unconstrained latents, :math:`\eta` the arguments of the distribution (any node:
latents, or deterministic nodes such as a network output) and :math:`F` the Fisher information of the
distribution with respect to its arguments (``Distribution.log_pdf_param_curvature``; implemented for
``Normal``, ``LogNormal``, ``StudentT``, ``Laplace``, ``Cauchy``, ``Exponential``, ``Gamma``, ``Beta``,
``Poisson``, ``Bernoulli``, ``Binomial``, ``Categorical``, ``Multinomial`` and the mean of
``MultivariateNormal``). The Jacobians come from the hand-written VJPs with one reverse pass per element of an
argument node.

* **Priors and transforms.** The prior terms, including the :math:`\log|\det J|` of the transforms, contribute
  their exact negative Hessian (central differences of the prior gradient; priors with constant arguments are
  differentiated one by one, hierarchical ones jointly). If that block is not positive semi-definite, as for a
  latent scale of a hierarchical prior, it is projected onto the positive semi-definite matrices.
* **Likelihoods without curvature** (any other distribution, a ``MultivariateNormal`` with a latent
  covariance, GP likelihoods) and the summed-out terms of enumerated latents contribute the positive
  semi-definite part of their exact negative Hessian, by finite differences of the gradient of that term only.
* For Gaussian likelihoods of fixed scale with arguments linear in the latents the matrix is the exact
  Hessian, so the Laplace approximation of Bayesian linear regression is exact. For canonical links (Poisson
  with a log rate, Bernoulli and Binomial with logits) it is also the exact Hessian; otherwise it replaces the
  observed information by the expected one.

The mode is refined by Levenberg-Marquardt steps with this matrix (damped Gauss-Newton; a step is accepted
only if the log joint does not decrease). Gauss-Newton converges linearly, so a few hundred cheap Adam steps
followed by ``newton_steps=100`` or more is a good setting for networks.

Subset (last-layer) Laplace
---------------------------

``laplace_approximation(model, params=["w5", "b5"], hessian="ggn")`` approximates only the chosen latents: the
others stay at the values the optimiser found, the Hessian and the Gaussian cover the subset, and the result is
the Gaussian approximation of the conditional posterior of the subset. Draws leave the other latents at their
values. For a network, choosing the last layer gives *last-layer Laplace* with a small Hessian; it is cheap, but
its uncertainty away from the data is much smaller than with all weights.

Linearised predictive
---------------------

``LaplaceApproximation.predictive`` pushes the Gaussian through the first-order expansion of the model outputs
:math:`f(\theta) \approx f(\theta^*) + J(\theta - \theta^*)` instead of evaluating the nonlinear model at sampled
parameters, which is poor for networks when the Gaussian is wide. The outputs are the deterministic arguments of
the likelihoods that depend on ``Data`` (override with ``outputs=[...]``). The Jacobian is evaluated at the
current data, so new inputs set with ``Data.set_value`` are predicted at; other nodes (a noise scale) are
evaluated at the draws. The result is a :class:`~BayesianDLL.Samplers.PredicativeResult`, exact for models
that are linear in the latents. The likelihood needs a plate that follows the inputs, ``plate("obs", x)``
with ``x`` the ``Data``.
