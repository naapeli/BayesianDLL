"""Kalman filter log likelihood of a linear Gaussian state-space model with a hand-written adjoint (backward) recursion.

System (``k`` observed components, ``m`` states)::

    y_t     = Z alpha_t + d_t + eps_t,        eps_t ~ N(0, H)
    alpha_{t+1} = T alpha_t + R eta_t,        eta_t ~ N(0, Q),      alpha_1 ~ N(a_1, P_1)

Prediction-form recursion with ``a_t = E[alpha_t | y_{1:t-1}]`` and ``P_t`` its covariance::

    v_t = y_t - Z a_t - d_t,   Y_t = P_t Z',   F_t = Z Y_t + H,   u_t = F_t^{-1} v_t
    a_{t+1} = T (a_t + Y_t u_t),   P_{t+1} = T (P_t - Y_t F_t^{-1} Y_t') T' + R Q R'
    log L = -1/2 sum_t [ n_t log 2 pi + log |F_t| + v_t' u_t ]

Missing observations (``mask = 0``) are handled exactly by the masking ``Z* = W Z``, ``F* = W F W + (I - W)``,
``v* = W v`` with ``W = diag(mask)``: the missing block of ``F*`` is the identity, so it contributes nothing to
the determinant or the quadratic form and no information to the update.

All arrays are time-major: ``y, mask, d`` have shape ``(N, *B, k)`` and the system matrices ``(*B, ., .)`` with
one common batch shape ``B``.

The functions take and return torch tensors. The recursions are written once against a small backend:
CPU tensors run through NumPy, whose per-operation overhead on these tiny matrices is several times lower than
torch's (the recursions are sequential in time, so their cost is the number of operations, not the arithmetic);
tensors on other devices stay on their device and run in torch. The dtype is kept either way.
"""
import math

import numpy as np
import torch

_LOG_2PI = math.log(2 * math.pi)


def _mv(A, x):
    return (A @ x[..., None])[..., 0]


def _t(A):
    return A.swapaxes(-1, -2)


class _NumPy:
    @staticmethod
    def eye(k, like):
        return np.eye(k, dtype=like.dtype)

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=like.dtype)

    zeros_like = staticmethod(np.zeros_like)
    stack = staticmethod(np.stack)
    log = staticmethod(np.log)
    inv = staticmethod(np.linalg.inv)

    @staticmethod
    def logdet(F):
        return np.linalg.slogdet(F)[1]

    @staticmethod
    def quiet():
        return np.errstate(all="ignore")


class _Torch:
    @staticmethod
    def eye(k, like):
        return torch.eye(k, dtype=like.dtype, device=like.device)

    @staticmethod
    def zeros(shape, like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)

    zeros_like = staticmethod(torch.zeros_like)
    stack = staticmethod(torch.stack)
    log = staticmethod(torch.log)
    inv = staticmethod(torch.linalg.inv)

    @staticmethod
    def logdet(F):
        return torch.linalg.slogdet(F)[1]

    @staticmethod
    def quiet():
        return torch.no_grad()


def _backend(tensors):
    """NumPy views of CPU tensors with the NumPy backend, or the tensors themselves with the torch backend."""
    if all(x.device.type == "cpu" for x in tensors):
        return [x.detach().numpy() for x in tensors], _NumPy
    return [x.detach() for x in tensors], _Torch


def _to_torch(x, like):
    return torch.as_tensor(x, device=like.device)


# Inputs and result of the last forward pass (see kalman_forward).
_last_forward = None


def kalman_forward(y, mask, d, Tm, Z, H, RQR, a, P):
    """Log likelihood per batch element, the cache the backward recursion needs and the final predicted ``a, P``.

    The last call is remembered: a log density followed by its gradient at the same values (as in every
    NUTS step) runs the recursion once. The result is a pure function of the inputs and is never modified
    by the callers.
    """
    global _last_forward
    inputs = (y, mask, d, Tm, Z, H, RQR, a, P)
    if _last_forward is not None and all(
        x.shape == c.shape and x.dtype == c.dtype and x.device == c.device and torch.equal(x, c) for x, c in zip(inputs, _last_forward[0])
    ):
        return _last_forward[1]
    arrays, xp = _backend(inputs)
    ll, cache, a_end, P_end = _kalman_forward(xp, *arrays)
    result = (_to_torch(ll, y), cache, _to_torch(a_end, y), _to_torch(P_end, y))
    _last_forward = (tuple(x.detach().clone() for x in inputs), result)
    return result


def _kalman_forward(xp, y, mask, d, Tm, Z, H, RQR, a, P):
    N, k = y.shape[0], y.shape[-1]
    mm = mask[..., :, None] * mask[..., None, :]
    Zs = Z[None] * mask[..., None]
    Zst = _t(Zs)
    Fadd = H[None] * mm + (1 - mask)[..., None] * xp.eye(k, y)
    vy = (y - d) * mask
    half_n_obs = 0.5 * _LOG_2PI * mask.sum(-1)
    Tt = _t(Tm)
    ll = xp.zeros(Tm.shape[:-2], y)
    a_list, P_list, Pn_list, Y_list, Finv_list, u_list = [], [], [], [], [], []
    with xp.quiet():
        for t in range(N):
            Zt = Zs[t]
            v = vy[t] - _mv(Zt, a)
            Y = P @ Zst[t]
            F = Zt @ Y + Fadd[t]
            if k == 1:
                Finv = 1.0 / F
                logdet = xp.log(F[..., 0, 0])
            else:
                Finv = xp.inv(F)
                logdet = xp.logdet(F)
            u = _mv(Finv, v)
            ll = ll - half_n_obs[t] - 0.5 * (logdet + (v * u).sum(-1))
            Pn = P - Y @ Finv @ _t(Y)
            a_list.append(a)
            P_list.append(P)
            Pn_list.append(Pn)
            Y_list.append(Y)
            Finv_list.append(Finv)
            u_list.append(u)
            a = _mv(Tm, a + _mv(Y, u))
            P = Tm @ Pn @ Tt + RQR
            P = 0.5 * (P + _t(P))
    return ll, (a_list, P_list, Pn_list, Y_list, Finv_list, u_list, Zs, mm, Z, H), a, P


def kalman_backward(cache, mask, Tm):
    """Gradient of the log likelihood with respect to the inputs of :func:`kalman_forward`.

    Returns a dict with ``y``, ``d`` (``(N, *B, k)``), ``T``, ``Z``, ``H``, ``RQR``, ``a`` and ``P`` (initial state
    mean and covariance); matrix gradients are those of the log likelihood as a function of the full matrix entries.
    The adjoints of the system matrices are accumulated after the time loop from the per-step adjoints.
    ``cache`` comes from :func:`kalman_forward`, and the gradients are torch tensors like ``Tm``.
    """
    like = Tm
    xp = _NumPy if isinstance(cache[-1], np.ndarray) else _Torch
    mask, Tm = (x.detach().numpy() if xp is _NumPy else x.detach() for x in (mask, Tm))
    a_list, P_list, Pn_list, Y_list, Finv_list, u_list, Zs, mm, Z, H = cache
    N = len(a_list)
    abar = xp.zeros_like(a_list[0])
    Pbar = xp.zeros_like(Tm)
    Tt = _t(Tm)
    abar_in, Pb_all, Ybar_all, S_all, vbar_all = [None] * N, [None] * N, [None] * N, [None] * N, [None] * N
    with xp.quiet():
        for t in range(N - 1, -1, -1):
            Y, Finv, u, Zt = Y_list[t], Finv_list[t], u_list[t], Zs[t]
            Pb = 0.5 * (Pbar + _t(Pbar))
            abar_in[t], Pb_all[t] = abar, Pb
            an = _mv(Tt, abar)
            Pn_bar = Tt @ Pb @ Tm
            Yt = _t(Y)
            Ybar = an[..., :, None] * u[..., None, :] - 2 * Pn_bar @ Y @ Finv
            ubar = _mv(Yt, an)
            Finv_u = _mv(Finv, ubar)
            vbar = Finv_u - u
            Fbar = Finv @ (Yt @ Pn_bar @ Y) @ Finv - Finv_u[..., :, None] * u[..., None, :] - 0.5 * (Finv - u[..., :, None] * u[..., None, :])
            S = 0.5 * (Fbar + _t(Fbar))
            Ztt = _t(Zt)
            Ybar = Ybar + Ztt @ S
            abar = an - _mv(Ztt, vbar)
            Pbar = Pn_bar + Ybar @ Zt
            Pbar = 0.5 * (Pbar + _t(Pbar))
            Ybar_all[t], S_all[t], vbar_all[t] = Ybar, S, vbar
    abar_in, Pb_all, Ybar_all, S_all, vbar_all = (xp.stack(x) for x in (abar_in, Pb_all, Ybar_all, S_all, vbar_all))
    a_all, P_all, Pn_all, Y_all, u_all = (xp.stack(x) for x in (a_list, P_list, Pn_list, Y_list, u_list))
    q = a_all + (Y_all @ u_all[..., None])[..., 0]
    Tbar = (abar_in[..., :, None] * q[..., None, :] + 2 * Pb_all @ Tm @ Pn_all).sum(0)
    Zstar = -vbar_all[..., :, None] * a_all[..., None, :] + S_all @ _t(Y_all) + _t(Ybar_all) @ P_all
    Zbar = (mask[..., :, None] * Zstar).sum(0)
    Hbar = (mm * S_all).sum(0)
    ybar = mask * vbar_all
    grads = {"y": ybar, "d": -ybar, "T": Tbar, "Z": Zbar, "H": Hbar, "RQR": Pb_all.sum(0), "a": abar, "P": Pbar}
    return {name: _to_torch(value, like) for name, value in grads.items()}
