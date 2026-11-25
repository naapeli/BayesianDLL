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
one common batch shape ``B``. The recursions run over tiny matrices, where the per-operation overhead of a tensor
library dominates the arithmetic; they are therefore written with NumPy arrays (a CPU copy of the inputs, no extra
dependency), which makes a likelihood-and-gradient evaluation several times faster than the equivalent torch ops.
"""
import math

import numpy as np

_LOG_2PI = math.log(2 * math.pi)


def _mv(A, x):
    return (A @ x[..., None])[..., 0]


def _t(A):
    return A.swapaxes(-1, -2)


def kalman_forward(y, mask, d, Tm, Z, H, RQR, a, P):
    """Log likelihood per batch element, the cache the backward recursion needs and the final predicted ``a, P``."""
    N, k = y.shape[0], y.shape[-1]
    mm = mask[..., :, None] * mask[..., None, :]
    Zs = Z[None] * mask[..., None]
    Zst = _t(Zs)
    Fadd = H[None] * mm + (1 - mask)[..., None] * np.eye(k, dtype=y.dtype)
    vy = (y - d) * mask
    half_n_obs = 0.5 * _LOG_2PI * mask.sum(-1)
    Tt = _t(Tm)
    ll = np.zeros(Tm.shape[:-2], dtype=y.dtype)
    a_list, P_list, Pn_list, Y_list, Finv_list, u_list = [], [], [], [], [], []
    with np.errstate(all="ignore"):
        for t in range(N):
            Zt = Zs[t]
            v = vy[t] - _mv(Zt, a)
            Y = P @ Zst[t]
            F = Zt @ Y + Fadd[t]
            if k == 1:
                Finv = 1.0 / F
                logdet = np.log(F[..., 0, 0])
            else:
                Finv = np.linalg.inv(F)
                logdet = np.linalg.slogdet(F)[1]
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
    """
    a_list, P_list, Pn_list, Y_list, Finv_list, u_list, Zs, mm, Z, H = cache
    N = len(a_list)
    abar = np.zeros_like(a_list[0])
    Pbar = np.zeros_like(Tm)
    Tt = _t(Tm)
    abar_in, Pb_all, Ybar_all, S_all, vbar_all = [None] * N, [None] * N, [None] * N, [None] * N, [None] * N
    with np.errstate(all="ignore"):
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
    abar_in, Pb_all, Ybar_all, S_all, vbar_all = (np.stack(x) for x in (abar_in, Pb_all, Ybar_all, S_all, vbar_all))
    a_all, P_all, Pn_all, Y_all, u_all = (np.stack(x) for x in (a_list, P_list, Pn_list, Y_list, u_list))
    q = a_all + (Y_all @ u_all[..., None])[..., 0]
    Tbar = (abar_in[..., :, None] * q[..., None, :] + 2 * Pb_all @ Tm @ Pn_all).sum(0)
    Zstar = -vbar_all[..., :, None] * a_all[..., None, :] + S_all @ _t(Y_all) + _t(Ybar_all) @ P_all
    Zbar = (mask[..., :, None] * Zstar).sum(0)
    Hbar = (mm * S_all).sum(0)
    ybar = mask * vbar_all
    return {"y": ybar, "d": -ybar, "T": Tbar, "Z": Zbar, "H": Hbar, "RQR": Pb_all.sum(0), "a": abar, "P": Pbar}
