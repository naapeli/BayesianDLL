"""Laplace approximation of the posterior in the unconstrained space."""

import math
from warnings import warn

import torch

from ._MAP import find_MAP
from ._auto_guides import LatentLayout, trace_from_unconstrained


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
    """Gaussian approximation ``N(mean, cov)`` of the posterior over the unconstrained latents.

    Attributes:
        mean: the MAP estimate (flat unconstrained vector).
        cov: ``(-H)^-1`` with ``H`` the Hessian of the log joint at the MAP.
        scale_tril: Cholesky factor of ``cov``.
        hessian: ``H``.
        log_marginal_likelihood: the Laplace estimate of ``log p(y)``,
            ``log p(theta*, y) + d/2 log(2 pi) - 1/2 log det(-H)``.
        names: latent names, in the order of the flat vector.
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
        """Constrained values at the MAP."""
        return self.layout.to_constrained(self.model, self.mean)

    def std(self):
        """Posterior standard deviations in the unconstrained space, per latent."""
        return self.layout.unpack(torch.sqrt(torch.diagonal(self.cov)))

    def sample_unconstrained(self, n):
        eps = torch.randn(n, self.layout.dim, dtype=self.mean.dtype, device=self.mean.device)
        return self.mean + eps @ self.scale_tril.T

    def sample(self, n_samples=1000, warmup_length=None, n_chains=1, **ignored):
        """Independent draws as a :class:`SamplingResult` in the constrained space (deterministics included).

        ``warmup_length`` and other sampler arguments are ignored.
        """
        draws = self.sample_unconstrained(n_chains * n_samples).reshape(n_chains, n_samples, self.layout.dim)
        return trace_from_unconstrained(self.model, self.layout, draws)

    def __repr__(self):
        return f"LaplaceApproximation(params={self.names}, log_marginal_likelihood={float(self.log_marginal_likelihood):.4f})"


def laplace_approximation(model, map_epochs=1000, lr=1e-2, betas=(0.9, 0.999), newton_steps=20, hessian="fd", fd_step=None, jitter=1e-10, tolerance=1e-9, verbose=False):
    """Laplace approximation of the posterior of ``model``.

    1. The MAP is found in the unconstrained space with :func:`find_MAP`
       (``map_epochs`` Adam steps from the current values) and refined with
       up to ``newton_steps`` damped Newton steps.
    2. The Hessian ``H`` of the unconstrained log joint is computed by
       central finite differences of the hand-written gradient
       ``model.joint_grad_log_prob`` (``hessian="fd"``, the only option) and
       symmetrised.
    3. The posterior is approximated by ``N(MAP, (-H)^-1)``; a warning is
       issued (and jitter added) when ``-H`` is not positive definite.

    The model's parameters are left at the MAP, like :func:`find_MAP`.
    Returns a :class:`LaplaceApproximation`.
    """
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
