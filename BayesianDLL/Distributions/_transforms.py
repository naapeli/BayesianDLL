import math

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from ._resolve import resolve
from .._batching import bsum as _bsum, n_events as _n_events


def _tiny(x):
    return torch.finfo(x.dtype).tiny


class Transform(ABC):
    """A bijection between a constrained space and an unconstrained space.

    Naming convention used throughout the library:

    * ``forward``: constrained -> unconstrained,
    * ``inverse``: unconstrained -> constrained,
    * ``derivative``: Jacobian of ``inverse`` at the unconstrained point. For
      element-wise transforms it is the element-wise derivative (same shape as
      the input). For vector transforms it is a matrix of shape
      ``(..., n_constrained, n_unconstrained)``.
    * ``log_abs_det_jacobian``: ``log|det d inverse / dz|`` summed to a scalar.
    """

    @abstractmethod
    def forward(self, x_constrained):
        """Map from constrained → unconstrained space."""
        pass

    @abstractmethod
    def inverse(self, x_unconstrained):
        """Map from unconstrained → constrained space."""
        pass

    @abstractmethod
    def derivative(self, x_unconstrained):
        """
        Jacobian of ``inverse``. Element-wise transforms return a tensor of the
        same shape as ``x_unconstrained``; vector transforms return the full
        Jacobian of shape ``(..., n_constrained, n_unconstrained)``.
        """
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, x_unconstrained):
        """
        Log absolute determinant of the Jacobian of ``inverse``.
        Returns a scalar (summed over all elements).
        """
        pass

    @abstractmethod
    def grad_log_abs_det_jacobian(self, x_unconstrained):
        """
        Gradient of log_abs_det_jacobian w.r.t. x_unconstrained.
        Returns a tensor of the same shape as x_unconstrained.
        """
        pass

    def unconstrained_shape(self, constrained_shape):
        """Shape of the unconstrained value for a constrained value of ``constrained_shape``."""
        return tuple(constrained_shape)

    def constrained_shape(self, unconstrained_shape):
        """Shape of the constrained value for an unconstrained value of ``unconstrained_shape``."""
        return tuple(unconstrained_shape)


def _apply_jacobian_transpose(transform, x_unconstrained, cotangent):
    """``J^T @ cotangent`` for the Jacobian of ``transform.inverse``."""
    vjp = getattr(transform, "vjp", None)
    if vjp is not None:
        return vjp(x_unconstrained, cotangent)
    deriv = transform.derivative(x_unconstrained)
    if deriv.shape == x_unconstrained.shape:
        return cotangent * deriv
    return (deriv.transpose(-2, -1) @ cotangent.unsqueeze(-1)).squeeze(-1)


class LogTransform(Transform):
    def __init__(self, border=0, side="larger"):
        self.border = border
        self.side = side

    def forward(self, x_constrained):
        border = resolve(self.border)
        if self.side == "larger":
            x = x_constrained - border
        else:
            x = border - x_constrained
        return torch.log(x.clamp(min=1e-8))

    def inverse(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return resolve(self.border) + sign * torch.exp(x_unconstrained)

    def derivative(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return sign * torch.exp(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return _bsum(x_unconstrained)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.ones_like(x_unconstrained)


class InverseSoftPlusTransform(Transform):
    def __init__(self, border=0, side="larger"):
        super().__init__()
        self.border = border
        self.side = side

    def forward(self, x_constrained):
        border = resolve(self.border)
        if self.side == "larger":
            x = x_constrained - border
        else:
            x = border - x_constrained
        x = x.clamp(min=1e-8)
        return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp(max=20))))

    def inverse(self, x_unconstrained):
        y = F.softplus(x_unconstrained)
        if self.side == "larger":
            return resolve(self.border) + y
        else:
            return resolve(self.border) - y

    def derivative(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return sign * torch.sigmoid(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return _bsum(-F.softplus(-x_unconstrained))

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.sigmoid(-x_unconstrained)


class LogitTransform(Transform):
    """``x = low + (high - low) * sigmoid(z)``.

    ``low`` and ``high`` may be numbers, tensors or model parameters; they are
    resolved every time the transform is used, so the map follows parameters
    whose value changes during sampling.
    """

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    @property
    def scale(self):
        return resolve(self.high) - resolve(self.low)

    def forward(self, x_constrained):
        low = resolve(self.low)
        scale = resolve(self.high) - low
        x_scaled = (x_constrained - low) / scale
        eps = torch.finfo(x_scaled.dtype).eps
        x_scaled = x_scaled.clamp(eps, 1 - eps)
        return torch.log(x_scaled) - torch.log1p(-x_scaled)

    def inverse(self, x_unconstrained):
        low = resolve(self.low)
        return low + (resolve(self.high) - low) * torch.sigmoid(x_unconstrained)

    def derivative(self, x_unconstrained):
        return self.scale * torch.sigmoid(x_unconstrained) * torch.sigmoid(-x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        z = x_unconstrained
        scale = torch.as_tensor(self.scale, dtype=z.dtype, device=z.device)
        return _bsum(torch.log(scale) + F.logsigmoid(z) + F.logsigmoid(-z))

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return 1 - 2 * torch.sigmoid(x_unconstrained)


class IdentityTransform(Transform):
    def forward(self, x_constrained):
        return x_constrained

    def inverse(self, x_unconstrained):
        return x_unconstrained

    def derivative(self, x_unconstrained):
        return torch.ones_like(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros((), dtype=x_unconstrained.dtype, device=x_unconstrained.device)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros_like(x_unconstrained)


class SoftMaxTransform(Transform):
    """Softmax map R^K -> simplex.

    .. warning:: This map is **not** a bijection (it is invariant to adding a
       constant to every coordinate), so it cannot be used as the sampling
       transform of a distribution on the simplex. Use
       :class:`StickBreakingTransform` instead. It is kept for backward
       compatibility.
    """

    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x_constrained):
        return torch.log(x_constrained.clamp(min=_tiny(x_constrained)))

    def inverse(self, x_unconstrained):
        return torch.softmax(x_unconstrained, dim=self.dim)

    def derivative(self, x_unconstrained):
        sm = self.inverse(x_unconstrained)
        return torch.diag_embed(sm) - sm.unsqueeze(-1) @ sm.unsqueeze(-2)

    def log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros((), dtype=x_unconstrained.dtype, device=x_unconstrained.device)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros_like(x_unconstrained)


class StickBreakingTransform(Transform):
    """Bijection between ``R^{K-1}`` and the open ``(K-1)``-simplex in ``R^K``.

    Stan's parameterisation: for ``k = 1, ..., K-1``

    .. math::
        y_k = \\sigma(z_k - \\log(K - k)), \\quad
        x_k = \\Big(1 - \\sum_{j<k} x_j\\Big) y_k, \\quad
        x_K = 1 - \\sum_{j<K} x_j .

    The offset ``log(K - k)`` makes ``z = 0`` map to the uniform simplex.
    ``derivative`` returns the ``(..., K, K-1)`` Jacobian of the inverse map.
    """

    def unconstrained_shape(self, constrained_shape):
        constrained_shape = tuple(constrained_shape)
        return constrained_shape[:-1] + (constrained_shape[-1] - 1,)

    def constrained_shape(self, unconstrained_shape):
        unconstrained_shape = tuple(unconstrained_shape)
        return unconstrained_shape[:-1] + (unconstrained_shape[-1] + 1,)

    @staticmethod
    def _offset(z):
        km1 = z.shape[-1]
        # log(K - k) for k = 1..K-1 (K - 1 = km1)
        return torch.log(torch.arange(km1, 0, -1, dtype=z.dtype, device=z.device))

    def _parts(self, z):
        """Return (u, log_x, y) where u = z - offset, y = sigmoid(u)."""
        u = z - self._offset(z)
        log_y = F.logsigmoid(u)
        log_1my = F.logsigmoid(-u)
        # log of the remaining stick before step k: exclusive cumsum of log(1 - y)
        cum = torch.cumsum(log_1my, dim=-1)
        log_r = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)  # (..., K)
        log_x = torch.cat([log_r[..., :-1] + log_y, log_r[..., -1:]], dim=-1)
        return u, log_x, torch.sigmoid(u)

    def log_simplex(self, x_unconstrained):
        """Numerically stable ``log(inverse(z))``."""
        return self._parts(x_unconstrained)[1]

    def forward(self, x_constrained):
        x = x_constrained.clamp(min=_tiny(x_constrained))
        # remaining mass after element k: sum_{j>k} x_j
        tail = torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])[..., 1:]
        km1 = x.shape[-1] - 1
        offset = torch.log(torch.arange(km1, 0, -1, dtype=x.dtype, device=x.device))
        return torch.log(x[..., :-1]) - torch.log(tail.clamp(min=_tiny(x))) + offset

    def inverse(self, x_unconstrained):
        return torch.exp(self.log_simplex(x_unconstrained))

    def derivative(self, x_unconstrained):
        z = x_unconstrained
        _, log_x, y = self._parts(z)
        x = torch.exp(log_x)
        K = x.shape[-1]
        i = torch.arange(K, device=z.device).unsqueeze(-1)
        k = torch.arange(K - 1, device=z.device).unsqueeze(0)
        diag = (i == k).to(z.dtype)
        lower = (k < i).to(z.dtype)
        # J[i, k] = x_i * (delta_ik (1 - y_k) - [k < i] y_k)
        inner = diag * (1 - y).unsqueeze(-2) - lower * y.unsqueeze(-2)
        return x.unsqueeze(-1) * inner

    def vjp(self, x_unconstrained, cotangent):
        """``J^T @ cotangent`` without forming the Jacobian."""
        _, log_x, y = self._parts(x_unconstrained)
        xv = torch.exp(log_x) * cotangent
        # sum_{i>k} x_i v_i for k = 0..K-2
        tail = torch.flip(torch.cumsum(torch.flip(xv, dims=[-1]), dim=-1), dims=[-1])[..., 1:]
        return xv[..., :-1] * (1 - y) - y * tail

    def _log_det_terms(self, x_unconstrained):
        u = x_unconstrained - self._offset(x_unconstrained)
        km1 = u.shape[-1]
        # K - k for k = 1..K-1
        counts = torch.arange(km1, 0, -1, dtype=u.dtype, device=u.device)
        return F.logsigmoid(u) + counts * F.logsigmoid(-u)

    def log_abs_det_jacobian(self, x_unconstrained):
        return _bsum(self._log_det_terms(x_unconstrained))

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        u = x_unconstrained - self._offset(x_unconstrained)
        km1 = u.shape[-1]
        counts = torch.arange(km1, 0, -1, dtype=u.dtype, device=u.device)
        y = torch.sigmoid(u)
        return (1 - y) - counts * y


def _log1m_tanh_sq(z):
    """Stable ``log(1 - tanh(z)^2)``."""
    a = z.abs()
    return 2 * (math.log(2.0) - a - F.softplus(-2 * a))


class CorrCholeskyTransform(Transform):
    """Bijection between ``R^{D(D-1)/2}`` and Cholesky factors of correlation matrices.

    The unconstrained vector is mapped through ``tanh`` to canonical partial
    correlations ``w`` (filled row-wise into the strictly lower triangle), and

    .. math::
        L_{ij} = w_{ij} \\sqrt{1 - \\sum_{k<j} L_{ik}^2}, \\quad
        L_{ii} = \\sqrt{1 - \\sum_{k<i} L_{ik}^2}.

    The constrained value is the full ``(..., D, D)`` lower-triangular matrix.
    Because the Jacobian is not a (vector -> vector) matrix, this transform
    provides :meth:`vjp` which computes ``J^T`` applied to a ``(..., D, D)``
    cotangent.
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self._n = self.dim * (self.dim - 1) // 2

    def unconstrained_shape(self, constrained_shape):
        constrained_shape = tuple(constrained_shape)
        return constrained_shape[:-2] + (self._n,)

    def constrained_shape(self, unconstrained_shape):
        unconstrained_shape = tuple(unconstrained_shape)
        return unconstrained_shape[:-1] + (self.dim, self.dim)

    def _indices(self, device):
        return torch.tril_indices(self.dim, self.dim, offset=-1, device=device)

    def _to_matrix(self, v):
        rows, cols = self._indices(v.device)
        out = torch.zeros(*v.shape[:-1], self.dim, self.dim, dtype=v.dtype, device=v.device)
        out[..., rows, cols] = v
        return out

    def _from_matrix(self, m):
        rows, cols = self._indices(m.device)
        return m[..., rows, cols]

    def _parts(self, z):
        W = self._to_matrix(torch.tanh(z))
        ell = self._to_matrix(_log1m_tanh_sq(z))
        log_s = torch.cumsum(ell, dim=-1) - ell  # log s_ij = sum_{k<j} log(1 - w_ik^2)
        sqrt_s = torch.exp(0.5 * log_s)
        eye = torch.eye(self.dim, dtype=z.dtype, device=z.device)
        L = (W + eye) * sqrt_s
        L = torch.tril(L)
        return W, sqrt_s, L

    def forward(self, x_constrained):
        L = x_constrained
        sq = L ** 2
        s = 1 - (torch.cumsum(sq, dim=-1) - sq)  # 1 - sum_{k<j} L_ik^2
        s = s.clamp(min=_tiny(L))
        w = self._from_matrix(L / torch.sqrt(s))
        eps = torch.finfo(L.dtype).eps
        return torch.atanh(w.clamp(-1 + eps, 1 - eps))

    def inverse(self, x_unconstrained):
        return self._parts(x_unconstrained)[2]

    def vjp(self, x_unconstrained, cotangent):
        W, sqrt_s, L = self._parts(x_unconstrained)
        b = torch.tril(cotangent)
        bL = b * L
        # T_ij = sum_{j < j' <= i} b_ij' L_ij'
        T = torch.flip(torch.cumsum(torch.flip(bL, dims=[-1]), dim=-1), dims=[-1]) - bL
        one_m_w2 = 1 - W ** 2
        grad = b * sqrt_s * one_m_w2 - W * T
        return self._from_matrix(grad)

    def derivative(self, x_unconstrained):
        """Full Jacobian of shape ``(..., D, D, D(D-1)/2)``."""
        z = x_unconstrained
        W, sqrt_s, L = self._parts(z)
        rows, cols = self._indices(z.device)
        D = self.dim
        jac = torch.zeros(*z.shape[:-1], D, D, self._n, dtype=z.dtype, device=z.device)
        jp = torch.arange(D, device=z.device)
        for m in range(self._n):
            i, j = int(rows[m]), int(cols[m])
            w = W[..., i, j]
            # j' = j
            jac[..., i, j, m] = sqrt_s[..., i, j] * (1 - w ** 2)
            # j < j' <= i
            later = jp[(jp > j) & (jp <= i)]
            jac[..., i, later, m] = -L[..., i, later] * w.unsqueeze(-1)
        return jac

    def _log_det_terms_matrix(self, z):
        ell = self._to_matrix(_log1m_tanh_sq(z))
        log_s = torch.cumsum(ell, dim=-1) - ell
        strict = torch.tril(torch.ones(self.dim, self.dim, dtype=z.dtype, device=z.device), diagonal=-1)
        return (0.5 * log_s + ell) * strict

    def log_abs_det_jacobian(self, x_unconstrained):
        return _bsum(self._log_det_terms_matrix(x_unconstrained))

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        z = x_unconstrained
        rows, cols = self._indices(z.device)
        w = torch.tanh(z)
        return -w * (rows - cols + 1).to(z.dtype)


class ChainTransform(Transform):
    """Composition: ``inverse(z) = second.inverse(first.inverse(z))``.

    ``forward(y) = first.forward(second.forward(y))``. Used by
    :class:`TransformedDistribution`, whose sampling space is the base
    distribution's unconstrained space.
    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def unconstrained_shape(self, constrained_shape):
        return self.first.unconstrained_shape(self.second.unconstrained_shape(constrained_shape))

    def constrained_shape(self, unconstrained_shape):
        return self.second.constrained_shape(self.first.constrained_shape(unconstrained_shape))

    def forward(self, x_constrained):
        return self.first.forward(self.second.forward(x_constrained))

    def inverse(self, x_unconstrained):
        return self.second.inverse(self.first.inverse(x_unconstrained))

    def derivative(self, x_unconstrained):
        z = x_unconstrained
        x = self.first.inverse(z)
        d1 = self.first.derivative(z)
        d2 = self.second.derivative(x)
        e1 = d1.shape == z.shape
        e2 = d2.shape == x.shape
        if e1 and e2:
            return d1 * d2
        if e1:
            return d2 * d1.unsqueeze(-2)
        if e2:
            return d2.unsqueeze(-1) * d1
        return d2 @ d1

    def vjp(self, x_unconstrained, cotangent):
        x = self.first.inverse(x_unconstrained)
        return _apply_jacobian_transpose(self.first, x_unconstrained, _apply_jacobian_transpose(self.second, x, cotangent))

    def log_abs_det_jacobian(self, x_unconstrained):
        x = self.first.inverse(x_unconstrained)
        return self.first.log_abs_det_jacobian(x_unconstrained) + self.second.log_abs_det_jacobian(x)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        x = self.first.inverse(x_unconstrained)
        return self.first.grad_log_abs_det_jacobian(x_unconstrained) + _apply_jacobian_transpose(
            self.first, x_unconstrained, self.second.grad_log_abs_det_jacobian(x))


class PositiveDefiniteTransform(Transform):
    """Bijection between ``R^{D(D+1)/2}`` and symmetric positive-definite ``D x D`` matrices.

    The unconstrained vector fills the lower triangle (row-wise, diagonal
    included) of a Cholesky factor ``L`` whose diagonal is exponentiated, and
    ``X = L L^T``. ``log|det J|`` is taken w.r.t. the ``D(D+1)/2`` unique
    entries of ``X`` (the measure Wishart densities are defined on):
    ``D log 2 + sum_i (D + 1 - i) z_ii`` (0-indexed ``i``). Since the Jacobian
    is not a (vector -> vector) matrix, :meth:`vjp` applies ``J^T`` to a full
    ``(..., D, D)`` cotangent (entries of ``X`` treated as independent).
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self._n = self.dim * (self.dim + 1) // 2

    def unconstrained_shape(self, constrained_shape):
        return tuple(constrained_shape)[:-2] + (self._n,)

    def constrained_shape(self, unconstrained_shape):
        return tuple(unconstrained_shape)[:-1] + (self.dim, self.dim)

    def _indices(self, device):
        return torch.tril_indices(self.dim, self.dim, offset=0, device=device)

    def _is_diag(self, device):
        rows, cols = self._indices(device)
        return rows == cols

    def _diag_weights(self, dtype, device):
        """(D + 1 - i) on diagonal positions, 0 elsewhere."""
        rows, cols = self._indices(device)
        return torch.where(rows == cols, (self.dim + 1 - rows).to(dtype), torch.zeros((), dtype=dtype, device=device))

    def cholesky(self, x_unconstrained):
        z = x_unconstrained
        rows, cols = self._indices(z.device)
        values = torch.where(rows == cols, torch.exp(z), z)
        L = torch.zeros(*z.shape[:-1], self.dim, self.dim, dtype=z.dtype, device=z.device)
        L[..., rows, cols] = values
        return L

    def forward(self, x_constrained):
        L = torch.linalg.cholesky(x_constrained)
        rows, cols = self._indices(L.device)
        values = L[..., rows, cols]
        tiny = torch.finfo(L.dtype).tiny
        return torch.where(rows == cols, torch.log(values.clamp(min=tiny)), values)

    def inverse(self, x_unconstrained):
        L = self.cholesky(x_unconstrained)
        return L @ L.transpose(-2, -1)

    def vjp(self, x_unconstrained, cotangent):
        L = self.cholesky(x_unconstrained)
        bar_L = (cotangent + cotangent.transpose(-2, -1)) @ L
        rows, cols = self._indices(L.device)
        bar = bar_L[..., rows, cols]
        return torch.where(rows == cols, bar * L[..., rows, cols], bar)

    def derivative(self, x_unconstrained):
        """Full Jacobian of shape ``(..., D, D, D(D+1)/2)``."""
        z = x_unconstrained
        L = self.cholesky(z)
        rows, cols = self._indices(z.device)
        D = self.dim
        jac = torch.zeros(*z.shape[:-1], D, D, self._n, dtype=z.dtype, device=z.device)
        for m in range(self._n):
            i, j = int(rows[m]), int(cols[m])
            # dL = E_ij scaled by L_ii on the diagonal; dX = dL L^T + L dL^T
            scale = L[..., i, j] if i == j else torch.ones_like(L[..., i, j])
            col = L[..., :, j] * scale.unsqueeze(-1)  # (..., D): row a of (E_ij L^T)[i, a] = L[a, j]
            jac[..., i, :, m] += col
            jac[..., :, i, m] += col
        return jac

    def log_abs_det_jacobian(self, x_unconstrained):
        z = x_unconstrained
        w = self._diag_weights(z.dtype, z.device)
        return _bsum(w * z) + math.log(2.0) * self.dim * _n_events(z, self._n)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        z = x_unconstrained
        return self._diag_weights(z.dtype, z.device).expand(z.shape).clone()


class OrderedTransform(Transform):
    """Bijection between ``R^K`` and strictly increasing vectors in ``R^K``.

    Stan's parameterisation of an ordered vector, applied along the last
    dimension (all leading dimensions, e.g. the chain dimension, are batch
    dimensions):

    .. math::
        x_0 = y_0, \\quad x_k = x_{k-1} + \\exp(y_k) \\quad (k \\geq 1),

    i.e. ``x = cumsum([y_0, exp(y_1), ..., exp(y_{K-1})])``. The Jacobian of
    the inverse map is lower triangular with ``J[i, k] = e_k`` for ``k <= i``
    (``e_0 = 1``, ``e_k = exp(y_k)``), so ``log|det J| = sum_{k>=1} y_k``.
    """

    @staticmethod
    def _increments(y):
        """``(y_0, exp(y_1), ..., exp(y_{K-1}))``: the derivative of ``x_i`` w.r.t. ``y_k`` for ``k <= i``."""
        return torch.cat([torch.ones_like(y[..., :1]), torch.exp(y[..., 1:])], dim=-1)

    def forward(self, x_constrained):
        x = x_constrained
        gaps = (x[..., 1:] - x[..., :-1]).clamp(min=_tiny(x))
        return torch.cat([x[..., :1], torch.log(gaps)], dim=-1)

    def inverse(self, x_unconstrained):
        y = x_unconstrained
        return torch.cumsum(torch.cat([y[..., :1], torch.exp(y[..., 1:])], dim=-1), dim=-1)

    def derivative(self, x_unconstrained):
        """Full Jacobian of shape ``(..., K, K)``."""
        y = x_unconstrained
        K = y.shape[-1]
        lower = torch.tril(torch.ones(K, K, dtype=y.dtype, device=y.device))
        return lower * self._increments(y).unsqueeze(-2)

    def vjp(self, x_unconstrained, cotangent):
        """``J^T @ cotangent``: ``e_k * sum_{i >= k} v_i`` (a reversed cumulative sum)."""
        tail = torch.flip(torch.cumsum(torch.flip(cotangent, dims=[-1]), dim=-1), dims=[-1])
        return self._increments(x_unconstrained) * tail

    def log_abs_det_jacobian(self, x_unconstrained):
        # Summed over the event dimension first so that K = 1 (no terms) works in a batched evaluation.
        return _bsum(x_unconstrained[..., 1:].sum(-1))

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        grad = torch.ones_like(x_unconstrained)
        grad[..., :1] = 0
        return grad
