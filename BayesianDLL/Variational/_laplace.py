"""Laplace approximation of the posterior in the unconstrained space."""

import math
from warnings import warn

import torch

from ._MAP import find_MAP
from ._groups import LatentLayout
from ._guide import trace_from_unconstrained
from ._ggn import GaussNewtonCurvature, NodeJacobians, finite_difference_columns


def _log_joint(model, layout, flat):
    with model.temporarily_set_many(layout.unpack(flat)):
        return torch.as_tensor(model.model_log_prob()).reshape(())


def _gradient(model, layout, flat):
    with model.temporarily_set_many(layout.unpack(flat)):
        return layout.pack(model.joint_grad_log_prob(layout.names))


def finite_difference_hessian(model, layout, flat, step=None):
    """Hessian of the unconstrained log joint by central differences of the hand-written gradient.

    Column ``j`` is ``(g(x + h_j e_j) - g(x - h_j e_j)) / (2 h_j)`` with
    ``h_j = step * max(1, |x_j|)``; the result is symmetrised.
    """
    return finite_difference_columns(lambda x: _gradient(model, layout, x), flat, step)


def ggn_hessian(model, layout, flat, fd_step=None):
    """Generalised Gauss-Newton approximation of the Hessian of the unconstrained log joint (negative semi-definite).

    See :class:`~BayesianDLL.Variational._ggn.GaussNewtonCurvature`.
    """
    with model.temporarily_set_many(layout.unpack(flat)):
        return -GaussNewtonCurvature(model, layout, fd_step).matrix


def _stable_cholesky(matrix, jitter):
    """Cholesky factor of ``matrix``, adding diagonal jitter when it is not positive definite.

    Returns ``(L, added_jitter)``.
    """
    L, info = torch.linalg.cholesky_ex(matrix)
    if int(info) == 0:
        return L, 0.0
    scale = float(torch.diagonal(matrix).abs().mean()) or 1.0
    eigenvalues = torch.linalg.eigvalsh(matrix)
    added = max(jitter * scale, -float(eigenvalues.min()) + jitter * scale)
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    for _ in range(30):
        L, info = torch.linalg.cholesky_ex(matrix + added * identity)
        if int(info) == 0:
            return L, added
        added *= 10
    raise RuntimeError("Could not compute a Cholesky factor of the negative Hessian.")


class LaplaceApproximation:
    r"""Gaussian approximation :math:`\mathcal{N}(\mu, \Sigma)` of the posterior in unconstrained space.

    Approximates the posterior distribution by a multivariate normal centered at the
    maximum a posteriori (MAP) estimate :math:`\eta^*` with covariance equal to the
    inverse of the negative Hessian of the unconstrained log-joint density. With
    ``params`` (a subset of the latents) the Gaussian covers only those latents,
    conditional on the others, which stay at their values at the mode.

    Parameters
    ----------
    model : Model
        The probabilistic model.
    layout : LatentLayout
        Layout of the latents in the flat unconstrained vector.
    mean : Tensor
        Mode :math:`\eta^*` in unconstrained space.
    hessian : Tensor
        Hessian matrix :math:`H` of the unconstrained log-joint density at the mode
        (the generalised Gauss-Newton approximation for ``method="ggn"``).
    log_joint : Tensor or float
        Unconstrained log-joint density :math:`\log p(\eta^*, y)` evaluated at the mode.
    jitter : float
        Diagonal jitter added to ensure positive definiteness of :math:`-H`.
    method : {"fd", "ggn"}, default "fd"
        How ``hessian`` was computed.

    Attributes
    ----------
    mean : Tensor
        The posterior mode in unconstrained space.
    cov : Tensor
        Covariance matrix :math:`\Sigma = (-H)^{-1}` (with the jitter added to :math:`-H`).
    scale_tril : Tensor
        Lower Cholesky factor :math:`L` of the covariance matrix :math:`\Sigma = L L^T`.
    hessian : Tensor
        The Hessian matrix at the mode.
    jitter : float
        Jitter added to the negative Hessian.
    log_joint_at_mode : float or Tensor
        Log-joint density at the mode :math:`\log p(\eta^*, y)`.
    log_marginal_likelihood : Tensor
        Laplace approximation of the log marginal likelihood (evidence),
        :math:`\log p(\eta^*, y) + \frac{d}{2} \log(2\pi) - \frac{1}{2} \log\det(-H)` with
        :math:`d` the number of unconstrained values.
    names : list of str
        Names of the latent parameters in order.
    method : str
        ``"fd"`` or ``"ggn"``.
    """

    def __init__(self, model, layout, mean, hessian, log_joint, jitter, method="fd"):
        self.model = model
        self.layout = layout
        self.names = list(layout.names)
        self.mean = mean
        self.hessian = hessian
        self.method = method
        negative = -hessian
        L, added = _stable_cholesky(negative, jitter)
        if added > 0:
            warn(
                "The negative Hessian at the MAP is not positive definite "
                f"(jitter {added:.3g} added); the Laplace approximation may be poor or the MAP not converged."
            )
        self.jitter = added
        self.cov = torch.cholesky_inverse(L)
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.scale_tril = torch.linalg.cholesky(self.cov)
        log_det = 2 * torch.log(torch.diagonal(L)).sum()
        self.log_joint_at_mode = log_joint
        self.log_marginal_likelihood = log_joint + 0.5 * layout.dim * math.log(2 * math.pi) - 0.5 * log_det

    def map_estimate(self):
        """Constrained values ``{name: value}`` at the unconstrained mode (not the constrained-space mode)."""
        return self.layout.to_constrained(self.model, self.mean)

    def std(self):
        """Posterior standard deviations in the unconstrained space, per latent."""
        return self.layout.unpack(torch.sqrt(torch.diagonal(self.cov)))

    def sample_unconstrained(self, n):
        """``n`` draws ``(n, d)`` from ``N(mean, cov)``."""
        eps = torch.randn(n, self.layout.dim, dtype=self.mean.dtype, device=self.mean.device)
        return self.mean + eps @ self.scale_tril.T

    def sample(self, n_samples=1000, n_chains=1):
        """Independent draws as a :class:`SamplingResult` in the constrained space (deterministics included)."""
        draws = self.sample_unconstrained(n_chains * n_samples).reshape(n_chains, n_samples, self.layout.dim)
        return trace_from_unconstrained(self.model, self.layout, draws)

    def linearised_outputs(self):
        """Default outputs of :meth:`predictive`: the deterministic nodes that feed a likelihood and depend on ``Data``."""
        model = self.model
        jacobians = NodeJacobians(model, self.layout)
        found = []
        for observed in model.observed_params.values():
            for name in observed.distribution.parameters:
                if name in model.deterministic_params and name not in found and (jacobians.ancestors(name) & set(model.data)):
                    found.append(name)
        return found

    def predictive(self, n_samples=100, samples_per_step=1, linearise=True, outputs=None, progress_bar=False):
        r"""Posterior predictive draws, optionally through a linearisation of the model outputs.

        With ``linearise=False`` this evaluates the model at ``n_samples`` draws of the
        latents (like ``model.posterior_predicative(approximation.sample(n_samples))``).
        With ``linearise=True`` every output node :math:`f(\theta)` is replaced by
        its first-order expansion at the mode,
        :math:`f(\theta) \approx f(\theta^*) + J (\theta - \theta^*)`, where
        :math:`J` is the Jacobian w.r.t. the unconstrained latents (one reverse
        pass per output element through the hand-written VJPs, at the current
        ``Data``, so new inputs set with ``Data.set_value`` are predicted at),
        so that the output is Gaussian with covariance :math:`J \Sigma J^\top`
        (the *linearised* or GLM predictive). The remaining nodes, such as a
        noise scale, are evaluated at the draws of :math:`\theta`. For models
        that are linear in the latents the result is exact; for neural networks
        it avoids the poor predictions of the nonlinear network at sampled
        weights when the Gaussian is wide.

        Parameters
        ----------
        n_samples : int, default 100
            Number of draws of the latents.
        samples_per_step : int, default 1
            Observation draws per draw of the latents.
        linearise : bool, default True
            Linearise the output nodes.
        outputs : sequence of str, optional
            Names of the deterministic nodes to linearise. Default: the
            deterministic arguments of the likelihoods that depend on ``Data``
            (the model function, not scale parameters); none if there are none.
        progress_bar : bool, default False
            Whether to show a progress bar.

        Returns
        -------
        PredicativeResult
            Draws of shape ``(n_samples, samples_per_step, *predictive_shape)``; the
            (linearised) values of the deterministic nodes are in ``deterministic``.
        """
        from ..Samplers._sample import _ancestral_predictive
        model, layout = self.model, self.layout
        draws = self.sample_unconstrained(n_samples)
        names = [] if not linearise else (self.linearised_outputs() if outputs is None else list(outputs))
        for name in names:
            if name not in model.deterministic_params:
                raise ValueError(f"'{name}' is not a deterministic node of the model.")
        unconstrained = layout.unpack(draws)
        others = {name: parameter.unconstrained_value for name, parameter in {**model.params, **model.enumerated_params}.items()
                  if name not in unconstrained}
        linear = {}
        with model.temporarily_set_many(layout.unpack(self.mean)):
            jacobians = NodeJacobians(model, layout)
            for name in names:
                value = jacobians.node_value(name)
                moved = (draws - self.mean) @ jacobians.jacobian(name).T
                linear[name] = (value.reshape(1, -1) + moved).reshape(n_samples, *value.shape)

            def parameter_values(i):
                for name, values in linear.items():
                    model.deterministic_params[name]._value_override = values[i]
                return {**others, **{name: value[i] for name, value in unconstrained.items()}}

            try:
                return _ancestral_predictive(model, n_samples, samples_per_step, parameter_values, progress_bar)
            finally:
                for name in linear:
                    model.deterministic_params[name]._value_override = None

    def __repr__(self):
        return f"LaplaceApproximation(params={self.names}, log_marginal_likelihood={float(self.log_marginal_likelihood):.4f})"


def _newton(model, layout, x, steps, tolerance, fd_step, verbose):
    """Newton steps with the finite-difference Hessian and a backtracking line search."""
    for step in range(steps):
        gradient = _gradient(model, layout, x)
        if float(gradient.abs().max()) < tolerance:
            break
        H = finite_difference_hessian(model, layout, x, fd_step)
        L, info = torch.linalg.cholesky_ex(-H)
        if int(info) != 0:
            break
        direction = torch.cholesky_solve(gradient.unsqueeze(-1), L).squeeze(-1)
        current = _log_joint(model, layout, x)
        t = 1.0
        while t > 1e-6:
            candidate = x + t * direction
            value = _log_joint(model, layout, candidate)
            if torch.isfinite(value) and value >= current:
                break
            t *= 0.5
        else:
            break
        x = candidate
        if verbose:
            print(f"Newton step {step + 1} - Log_prob: {float(value):.6f}")
        if float((t * direction).abs().max()) < tolerance:
            break
    return x


def _gauss_newton(model, layout, x, steps, tolerance, damping, fd_step, verbose):
    r"""Levenberg-Marquardt steps :math:`(A + \lambda s I)\,\delta = g` with the Gauss-Newton curvature ``A``.

    ``s`` is the mean diagonal of ``A``. A step is accepted when it does not
    decrease the log joint (``lambda`` is divided by 3), otherwise ``lambda``
    is multiplied by 4 and the step is recomputed with the same curvature.
    """
    current = _log_joint(model, layout, x)
    damping = max(damping, 1e-12)
    identity = torch.eye(layout.dim, dtype=layout.dtype, device=layout.device)
    for step in range(steps):
        gradient = _gradient(model, layout, x)
        if float(gradient.abs().max()) < tolerance:
            break
        with model.temporarily_set_many(layout.unpack(x)):
            curvature = GaussNewtonCurvature(model, layout, fd_step).matrix
        scale = float(torch.diagonal(curvature).mean()) or 1.0
        while damping < 1e16:
            L, info = torch.linalg.cholesky_ex(curvature + damping * scale * identity)
            if int(info) == 0:
                delta = torch.cholesky_solve(gradient.unsqueeze(-1), L).squeeze(-1)
                value = _log_joint(model, layout, x + delta)
                if torch.isfinite(value) and value >= current:
                    break
            damping *= 4
        else:
            break
        x = x + delta
        improvement = float(value - current)
        current = value
        damping = max(damping / 3, 1e-12)
        if verbose:
            print(f"Gauss-Newton step {step + 1} - Log_prob: {float(value):.6f}, damping: {damping:.2e}")
        if float(delta.abs().max()) < tolerance or improvement <= 1e-14 * max(1.0, abs(float(value))):
            break
    return x


def laplace_approximation(
    model,
    map_epochs=1000,
    lr=1e-2,
    betas=(0.9, 0.999),
    newton_steps=20,
    hessian="fd",
    fd_step=None,
    jitter=1e-10,
    tolerance=1e-9,
    verbose=False,
    params=None,
    damping=1e-3,
):
    r"""Compute the Laplace approximation of the model posterior.

    Finds the maximum a posteriori (MAP) mode in unconstrained space (Adam steps from the
    current values, refined by Newton or Gauss-Newton steps) and approximates
    the posterior distribution by a Gaussian :math:`\mathcal{N}(\eta^*, (-H)^{-1})`, where
    :math:`H` is the Hessian of the unconstrained log joint. The model's parameters
    are left at the mode.

    Parameters
    ----------
    model : Model
        The probabilistic model to approximate.
    map_epochs : int, default 1000
        Number of Adam optimization steps to locate the mode. If 0, initial optimization is skipped.
    lr : float, default 1e-2
        Learning rate for Adam optimization.
    betas : tuple of float, default (0.9, 0.999)
        Betas momentum hyperparameters for Adam optimizer.
    newton_steps : int, default 20
        Maximum number of refinement steps after Adam.
    hessian : {"fd", "ggn"}, default "fd"
        Curvature used for the refinement and for the covariance.

        * ``"fd"``: the exact Hessian by central differences of the hand-written
          gradient (symmetrised), refined by Newton steps with a backtracking line search
          that stop early when :math:`-H` is not positive definite.
        * ``"ggn"``: the generalised Gauss-Newton matrix, which works for every model and
          is positive semi-definite by construction. Every likelihood term :math:`\log p(y \mid \eta)`
          contributes :math:`J^\top F J`, with :math:`F` the Fisher information of its
          distribution (``log_pdf_param_curvature``) and :math:`J` the Jacobian of the
          distribution's arguments w.r.t. the unconstrained latents (one reverse pass through
          the hand-written VJPs per argument element). Prior terms contribute their exact
          negative Hessian (including the :math:`\log|\det J|` of the transforms), projected
          onto the positive semi-definite matrices if it is not, as do likelihoods whose
          distribution has no Fisher information and the summed-out terms of enumerated latents
          (finite differences of the gradient). The refinement uses Levenberg-Marquardt steps.
          It is exact for models with a Gaussian likelihood of fixed scale and arguments linear
          in the latents; otherwise it replaces the observed information by the expected one.
    fd_step : float, optional
        Relative finite-difference step (default :math:`\epsilon^{1/3}` of the dtype).
    jitter : float, default 1e-10
        If :math:`-H` is not positive definite, a warning is issued and at least
        ``jitter`` times the mean absolute diagonal of :math:`-H` is added to it.
    tolerance : float, default 1e-9
        The refinement stops when the largest absolute gradient entry or step is below ``tolerance``.
    verbose : bool, default False
        Whether to display progress information during optimization.
    params : sequence of str, optional
        Names (or ``RandomParameter`` objects) of the latents to approximate, e.g. the last layer of a network. Adam optimises all
        latents, the refinement and the Hessian then only the selected ones with the others
        held at their values, so the result is the Gaussian approximation of the conditional
        posterior of the subset. Default: all latents.
    damping : float, default 1e-3
        Initial Levenberg-Marquardt damping of ``hessian="ggn"``, relative to the mean
        diagonal of the curvature.

    Returns
    -------
    LaplaceApproximation
        The fitted Gaussian approximation with sampling methods, predictive draws and marginal likelihood estimate.
    """
    model._capture_training_data()
    if hessian not in ("fd", "ggn"):
        raise ValueError(f"Unknown hessian method '{hessian}'; use 'fd' (finite differences of the gradient) or 'ggn' (Gauss-Newton).")
    names = None if params is None else [getattr(p, "name", p) for p in params]
    if names is not None:
        unknown = [name for name in names if name not in model.params]
        if unknown:
            raise ValueError(f"Unknown latent parameter(s) {unknown}.")
    layout = LatentLayout(model, purpose="The Laplace approximation", names=names)
    if map_epochs:
        find_MAP(model, lr=lr, epochs=map_epochs, betas=betas, verbose=verbose)
    x = layout.current(model)

    if hessian == "ggn":
        x = _gauss_newton(model, layout, x, newton_steps, tolerance, damping, fd_step, verbose)
    else:
        x = _newton(model, layout, x, newton_steps, tolerance, fd_step, verbose)

    for name, value in layout.unpack(x).items():
        model.params[name].set_unconstrained_value(value.clone())
    H = ggn_hessian(model, layout, x, fd_step) if hessian == "ggn" else finite_difference_hessian(model, layout, x, fd_step)
    return LaplaceApproximation(model, layout, x, H, _log_joint(model, layout, x), jitter, method=hessian)
