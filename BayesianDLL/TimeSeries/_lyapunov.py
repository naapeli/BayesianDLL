"""Discrete Lyapunov equations and companion matrices."""
import torch


def _companion(phi):
    """Companion matrices ``(..., p, p)`` of AR coefficients ``(..., p)``."""
    p = phi.shape[-1]
    T = torch.zeros(*phi.shape, p, dtype=phi.dtype, device=phi.device)
    T[..., 0, :] = phi
    if p > 1:
        idx = torch.arange(p - 1, device=phi.device)
        T[..., idx + 1, idx] = 1.0
    return T


def _vec_system(T, transpose):
    """``I - kron(T, T)`` (or of the transposes) as ``(..., p^2, p^2)`` for row-major vectorised matrices."""
    p = T.shape[-1]
    A = T.transpose(-1, -2) if transpose else T
    K = torch.einsum("...ij,...kl->...ikjl", A, A).reshape(*T.shape[:-2], p * p, p * p)
    return torch.eye(p * p, dtype=T.dtype, device=T.device) - K


def _solve_lyapunov(T, Q, transpose=False):
    """Solve ``X = T X T' + Q`` (``transpose=False``) or ``X = T' X T + Q``."""
    p = T.shape[-1]
    X = torch.linalg.solve(_vec_system(T, transpose), Q.reshape(*Q.shape[:-2], p * p, 1))
    return X.reshape(*Q.shape[:-2], p, p)
