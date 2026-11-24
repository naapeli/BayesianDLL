"""Laplace approximation of the posterior in the unconstrained space."""

import math
from warnings import warn

import torch

from ._MAP import find_MAP
from ._groups import LatentLayout
from ._guide import trace_from_unconstrained


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
    if step is None:
        step = torch.finfo(flat.dtype).eps ** (1 / 3)
    dim = flat.numel()
    hessian = torch.empty(dim, dim, dtype=flat.dtype, device=flat.device)
    for j in range(dim):
        h = step * max(1.0, abs(float(flat[j])))
        shift = torch.zeros_like(flat)
        shift[j] = h
        hessian[:, j] = (_gradient(model, layout, flat + shift) - _gradient(model, layout, flat - shift)) / (2 * h)
    return 0.5 * (hessian + hessian.T)


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
    inverse of the negative Hessian of the unconstrained log-joint density.

    Parameters
    ----------
    model : Model
        The probabilistic model.
    layout : LatentLayout
        Layout of the latents in the flat unconstrained vector.
    mean : Tensor
        Mode :math:`\eta^*` in unconstrained space.
    hessian : Tensor
        Hessian matrix :math:`H` of the unconstrained log-joint density at the mode.
    log_joint : Tensor or float
        Unconstrained log-joint density :math:`\log p(\eta^*, y)` evaluated at the mode.
    jitter : float
        Diagonal jitter added to ensure positive definiteness of :math:`-H`.

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
    """

    def __init__(self, model, layout, mean, hessian, log_joint, jitter):
        self.model = model
        self.layout = layout
        self.names = list(layout.names)
        self.mean = mean
        self.hessian = hessian
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

    def __repr__(self):
        return f"LaplaceApproximation(params={self.names}, log_marginal_likelihood={float(self.log_marginal_likelihood):.4f})"


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
):
    r"""Compute the Laplace approximation of the model posterior.

    Finds the maximum a posteriori (MAP) mode in unconstrained space (Adam steps from the
    current values, refined by Newton steps with a backtracking line search) and approximates
    the posterior distribution by a Gaussian :math:`\mathcal{N}(\eta^*, (-H)^{-1})`, where
    :math:`H` is the Hessian of the unconstrained log joint (central finite differences of
    the hand-written gradient, symmetrised). The model's parameters are left at the mode.

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
        Maximum number of Newton refinement steps; Newton stops early when :math:`-H`
        is not positive definite.
    hessian : {"fd"}, default "fd"
        Method used to evaluate the Hessian matrix.
    fd_step : float, optional
        Relative finite-difference step (default :math:`\epsilon^{1/3}` of the dtype).
    jitter : float, default 1e-10
        If :math:`-H` is not positive definite, a warning is issued and at least
        ``jitter`` times the mean absolute diagonal of :math:`-H` is added to it.
    tolerance : float, default 1e-9
        Newton stops when the largest absolute gradient entry or step is below ``tolerance``.
    verbose : bool, default False
        Whether to display progress information during optimization.

    Returns
    -------
    LaplaceApproximation
        The fitted Gaussian approximation with sampling methods and marginal likelihood estimate.
    """
    model._capture_training_data()
    if hessian != "fd":
        raise ValueError(f"Unknown hessian method '{hessian}'; only 'fd' (finite differences of the gradient) is available.")
    layout = LatentLayout(model, purpose="The Laplace approximation")
    if map_epochs:
        find_MAP(model, lr=lr, epochs=map_epochs, betas=betas, verbose=verbose)
    x = layout.current(model)

    for step in range(newton_steps):
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

    for name, value in layout.unpack(x).items():
        model.params[name].set_unconstrained_value(value.clone())
    H = finite_difference_hessian(model, layout, x, fd_step)
    return LaplaceApproximation(model, layout, x, H, _log_joint(model, layout, x), jitter)
