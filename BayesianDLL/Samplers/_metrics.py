"""Inverse-metric (mass matrix) estimators for the NUTS warmup.

Every estimator works on a batch of chains: draws and gradients of shape
``(..., n, d)`` (``n`` warmup draws of a ``d``-dimensional state) give one
inverse metric per leading index. The estimates are computed in double
precision and returned in the dtype of the draws.
"""

import math
from collections import namedtuple

import torch


METRICS = ("low_rank", "dense", "diag", "ledoit_wolf", "oas", "auto")
# metric="auto" uses the dense metric up to this many dimensions and the
# low-rank metric above it.
AUTO_DENSE_MAX_DIM = 50
# Regularisation and eigenvalue cutoff of the low-rank estimator (the values
# of nutpie and of Seyboldt, Carlson and Carpenter, 2026).
LOW_RANK_GAMMA = 1e-5
LOW_RANK_CUTOFF = 2.0


class LowRankMetric(namedtuple("LowRankMetric", ["scale", "vectors", "values"])):
    """Inverse metric ``diag(scale) (I + vectors diag(values - 1) vectors^T) diag(scale)``.

    ``scale`` has shape ``(..., d)``, ``vectors`` ``(..., d, k)`` with
    orthonormal (or zero) columns and ``values`` ``(..., k)``; a column with
    value 1 does not contribute, which pads chains with fewer eigenvectors.
    """

    __slots__ = ()


def check_metric(metric):
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(repr(m) for m in METRICS)}; got {metric!r}.")


def resolve_metric(metric, dim):
    """The estimator used for ``metric`` and a state of ``dim`` dimensions (resolves ``"auto"``)."""
    if metric == "auto":
        return "dense" if dim <= AUTO_DENSE_MAX_DIM else "low_rank"
    return metric


def metric_kind(inv_metric, batch_dims=0):
    """``None`` (identity), ``"diag"``, ``"dense"`` or ``"low_rank"``, from the representation."""
    if inv_metric is None:
        return None
    if isinstance(inv_metric, LowRankMetric):
        return "low_rank"
    return "dense" if inv_metric.ndim == batch_dims + 2 else "diag"


# ------------------------------------------------------------------ dense
def stan_regularise(covariance, n):
    """Stan's ``n / (n + 5) * Sigma + 1e-3 * 5 / (n + 5) * I`` of a ``(..., d, d)`` covariance."""
    d = covariance.shape[-1]
    identity = torch.eye(d, dtype=covariance.dtype, device=covariance.device)
    inv_metric = (n / (n + 5.0)) * covariance + 1e-3 * (5.0 / (n + 5.0)) * identity
    return 0.5 * (inv_metric + inv_metric.transpose(-2, -1))


def dense_inverse_metric(covariance, n):
    """Regularised dense inverse metric from the sample covariance ``(..., d, d)`` of ``n`` warmup draws.

    A window with fewer than ``4 d`` draws gives a rank-deficient or very noisy
    covariance, and directions it has not seen would otherwise get a (near)
    zero variance, along which the sampler then barely moves. Its off-diagonal
    terms are therefore kept with weight ``n / (4 d)`` and the rest is taken
    from the diagonal; larger windows use the sample covariance as it is. The
    result is then regularised towards a small multiple of the identity, as
    in Stan.
    """
    d = covariance.shape[-1]
    weight = min(1.0, n / (4.0 * d))
    diagonal = torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
    return stan_regularise(weight * covariance + (1.0 - weight) * diagonal, n)


def shrinkage_inverse_metric(draws, method):
    """Ledoit-Wolf or OAS inverse metric from warmup draws ``(..., n, d)``.

    The draws are centred and standardised by their standard deviations, so
    the estimators shrink the sample correlation matrix ``R`` towards the
    identity (the covariance towards its diagonal) and the metric does not
    depend on the scales of the coordinates. With ``p = d``,
    ``R* = (1 - rho) R + rho (tr R / p) I`` and

    * ``"ledoit_wolf"`` (Ledoit and Wolf, 2004):
      ``rho = min(b2, d2) / d2`` with ``d2 = ||R - (tr R / p) I||_F^2 / p``
      and ``b2 = sum_k ||z_k z_k^T - R||_F^2 / (n^2 p)`` over the standardised
      draws ``z_k``;
    * ``"oas"`` (Chen, Wiesel, Eldar and Hero, 2010, eq. 23):
      ``rho = min(1, ((1 - 2/p) tr(R^2) + tr(R)^2) / ((n + 1 - 2/p) (tr(R^2) - tr(R)^2 / p)))``.

    The covariance ``D R* D``, with ``D`` the standard deviations (with
    ``n - 1`` in the denominator), is then regularised as in Stan.
    """
    dtype = draws.dtype
    x = draws.to(torch.float64)
    n, p = x.shape[-2], x.shape[-1]
    x = x - x.mean(-2, keepdim=True)
    sd = x.square().mean(-2).sqrt()
    z = x / torch.where(sd > 0, sd, torch.ones_like(sd)).unsqueeze(-2)
    corr = z.transpose(-2, -1) @ z / n
    trace = torch.diagonal(corr, dim1=-2, dim2=-1).sum(-1)
    trace_sq = corr.square().sum((-2, -1))
    if method == "ledoit_wolf":
        d2 = (trace_sq - trace.square() / p) / p
        fourth = z.square().sum(-1).square().sum(-1)
        b2 = torch.minimum((fourth - n * trace_sq) / (n * n * p), d2)
        rho = torch.where(d2 > 0, b2 / torch.where(d2 > 0, d2, torch.ones_like(d2)), torch.zeros_like(d2))
    elif method == "oas":
        numerator = (1.0 - 2.0 / p) * trace_sq + trace.square()
        denominator = (n + 1.0 - 2.0 / p) * (trace_sq - trace.square() / p)
        positive = denominator > 0
        rho = torch.where(positive, numerator / torch.where(positive, denominator, torch.ones_like(denominator)),
                          torch.ones_like(denominator))
    else:
        raise ValueError(f"Unknown shrinkage estimator {method!r}.")
    rho = rho.clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
    identity = torch.eye(p, dtype=corr.dtype, device=corr.device)
    shrunk = (1.0 - rho) * corr + rho * (trace / p).unsqueeze(-1).unsqueeze(-1) * identity
    sd = sd * math.sqrt(n / max(n - 1, 1))
    covariance = sd.unsqueeze(-1) * shrunk * sd.unsqueeze(-2)
    return stan_regularise(covariance, n).to(dtype)


# ------------------------------------------------------------------ low rank
def _spd_mean(a, b):
    """The symmetric positive-definite ``m`` with ``m b m = a`` (geometric mean of ``a`` and ``b^-1``)."""
    eig_b, vec_b = torch.linalg.eigh(b)
    eig_b = eig_b.clamp(min=torch.finfo(b.dtype).tiny)
    b_half = (vec_b * eig_b.sqrt().unsqueeze(-2)) @ vec_b.transpose(-2, -1)
    b_inv_half = (vec_b * eig_b.rsqrt().unsqueeze(-2)) @ vec_b.transpose(-2, -1)
    inner = b_half @ a @ b_half
    eig_m, vec_m = torch.linalg.eigh(0.5 * (inner + inner.transpose(-2, -1)))
    inner_half = (vec_m * eig_m.clamp(min=0.0).sqrt().unsqueeze(-2)) @ vec_m.transpose(-2, -1)
    mean = b_inv_half @ inner_half @ b_inv_half
    return 0.5 * (mean + mean.transpose(-2, -1))


def low_rank_inverse_metric(draws, gradients, gamma=LOW_RANK_GAMMA, cutoff=LOW_RANK_CUTOFF):
    """Low-rank plus diagonal inverse metric from warmup draws and their log-density gradients ``(..., n, d)``.

    The estimator of nutpie (Seyboldt, Carlson and Carpenter, 2026), which
    minimises the Fisher divergence between the preconditioned target and a
    standard normal:

    1. diagonal scales ``s_i = (var(x_i) / var(g_i))^(1/4)`` (``s_i^2`` is the
       variance for a Gaussian target);
    2. centred draws ``x / s`` and gradients ``g * s``, and an orthonormal
       basis ``Q`` of the space they span (all of ``R^d`` when ``2 n >= d``);
    3. in that basis, ``A = P_x P_x^T + gamma I`` and
       ``B = P_g P_g^T + gamma I`` from the projected draws and gradients,
       and the SPD matrix ``S`` with ``S B S = A``;
    4. the eigenpairs of ``S`` with eigenvalues outside
       ``[1 / cutoff, cutoff]`` give ``U = Q V`` and ``Lambda``.

    The inverse metric is ``diag(s) (I + U (Lambda - I) U^T) diag(s)``: the
    diagonal metric corrected along the directions in which the scaled
    posterior is much wider or narrower than one. Returns a
    :class:`LowRankMetric`; chains with fewer eigenpairs are padded with
    zero vectors of value 1.
    """
    dtype = draws.dtype
    x = draws.to(torch.float64)
    g = gradients.to(torch.float64)
    n, d = x.shape[-2], x.shape[-1]
    x = x - x.mean(-2, keepdim=True)
    g = g - g.mean(-2, keepdim=True)
    var_x, var_g = x.square().mean(-2), g.square().mean(-2)
    variance = (var_x / var_g).sqrt()
    # Coordinates without a usable ratio (constant draws or gradients) fall
    # back to the variance of the draws, or to one.
    fallback = torch.where(var_x > 0, var_x, torch.ones_like(var_x))
    variance = torch.where(torch.isfinite(variance) & (variance > 0), variance, fallback)
    scale = variance.sqrt().clamp(1e-10, 1e10)
    x = x / scale.unsqueeze(-2)
    g = g * scale.unsqueeze(-2)

    basis = None
    if 2 * n < d:
        basis = torch.linalg.svd(torch.cat([x, g], dim=-2).transpose(-2, -1), full_matrices=False)[0]
        x, g = x @ basis, g @ basis
    k = x.shape[-1]
    regulariser = gamma * torch.eye(k, dtype=x.dtype, device=x.device)
    mean = _spd_mean(x.transpose(-2, -1) @ x + regulariser, g.transpose(-2, -1) @ g + regulariser)
    values, vectors = torch.linalg.eigh(mean)
    keep = (values > cutoff) | (values < 1.0 / cutoff)

    rank = int(keep.sum(-1).max()) if keep.numel() else 0
    order = torch.argsort(keep.to(torch.int64), dim=-1, descending=True, stable=True)[..., :rank]
    kept = torch.gather(keep, -1, order)
    values = torch.where(kept, torch.gather(values, -1, order).clamp(1e-20, 1e20), torch.ones_like(values[..., :rank]))
    vectors = torch.gather(vectors, -1, order.unsqueeze(-2).expand(*vectors.shape[:-1], rank)) * kept.unsqueeze(-2)
    if basis is not None:
        vectors = basis @ vectors
    return LowRankMetric(scale.to(dtype), vectors.to(dtype), values.to(dtype))


def _project(vectors, r):
    """``vectors^T r`` for ``vectors`` ``(..., d, k)`` and ``r`` ``(..., d)``."""
    return (r.unsqueeze(-2) @ vectors).squeeze(-2)


def low_rank_velocity(r, metric):
    """``M^-1 r`` for a :class:`LowRankMetric` and momenta ``r`` ``(..., d)`` in ``O(d k)``."""
    q = metric.scale * r
    correction = (metric.vectors @ ((metric.values - 1.0) * _project(metric.vectors, q)).unsqueeze(-1)).squeeze(-1)
    return metric.scale * (q + correction)


def low_rank_momentum(z, metric):
    """Momenta with covariance ``M`` from standard normals ``z`` ``(..., d)`` in ``O(d k)``.

    ``r = diag(s)^-1 (I + U (Lambda^(-1/2) - I) U^T) z``; with orthonormal
    ``U`` the square of the middle factor is ``I + U (Lambda^-1 - I) U^T``,
    so ``r`` has covariance ``M``, the inverse of the inverse metric.
    """
    correction = (metric.vectors @ ((metric.values.rsqrt() - 1.0) * _project(metric.vectors, z)).unsqueeze(-1)).squeeze(-1)
    return (z + correction) / metric.scale


def low_rank_matrix(metric):
    """The dense ``(..., d, d)`` form of a :class:`LowRankMetric`."""
    d = metric.scale.shape[-1]
    identity = torch.eye(d, dtype=metric.scale.dtype, device=metric.scale.device)
    inner = identity + (metric.vectors * (metric.values - 1.0).unsqueeze(-2)) @ metric.vectors.transpose(-2, -1)
    return metric.scale.unsqueeze(-1) * inner * metric.scale.unsqueeze(-2)


def inverse_metric_matrix(inv_metric, batch_dims=0):
    """Dense form of any inverse-metric representation (``None`` for the identity)."""
    kind = metric_kind(inv_metric, batch_dims)
    if kind is None:
        return None
    if kind == "low_rank":
        return low_rank_matrix(inv_metric)
    if kind == "diag":
        return torch.diag_embed(inv_metric)
    return inv_metric


# ------------------------------------------------------------ shared helpers
def needs_draws(kind):
    """Whether the estimator of ``kind`` needs the window's draws (not only Welford moments)."""
    return kind in ("ledoit_wolf", "oas", "low_rank")


def draws_inverse_metric(kind, draws, gradients):
    """Inverse metric of ``kind`` from a window's draws (and gradients) ``(..., n, d)``."""
    if kind == "low_rank":
        return low_rank_inverse_metric(draws, gradients)
    return shrinkage_inverse_metric(draws, kind)


def cast_metric(inv_metric, like):
    """``inv_metric`` in the dtype and device of the tensor ``like``."""
    if isinstance(inv_metric, LowRankMetric):
        return LowRankMetric(*(t.to(dtype=like.dtype, device=like.device) for t in inv_metric))
    return inv_metric.to(dtype=like.dtype, device=like.device)


def identity_low_rank(like, rank=0):
    """The identity as a :class:`LowRankMetric` for ``like`` ``(..., d)`` (zero vectors of value 1)."""
    return LowRankMetric(torch.ones_like(like), like.new_zeros(*like.shape, rank), like.new_ones(*like.shape[:-1], rank))


def pad_rank(metric, rank):
    """``metric`` with zero vectors of value 1 appended up to ``rank`` columns."""
    extra = rank - metric.values.shape[-1]
    if extra <= 0:
        return metric
    vectors = torch.cat([metric.vectors, metric.vectors.new_zeros(*metric.vectors.shape[:-1], extra)], dim=-1)
    values = torch.cat([metric.values, metric.values.new_ones(*metric.values.shape[:-1], extra)], dim=-1)
    return LowRankMetric(metric.scale, vectors, values)


def low_rank_valid(metric):
    """Boolean ``(...)``: finite scales, vectors and values, with positive scales and values."""
    finite = torch.isfinite(metric.scale).all(-1) & torch.isfinite(metric.vectors).all((-2, -1)) & torch.isfinite(metric.values).all(-1)
    return finite & (metric.scale > 0).all(-1) & (metric.values > 0).all(-1)


def select_low_rank(valid, new, previous):
    """Per leading index, ``new`` where ``valid`` and ``previous`` (padded to a common rank) elsewhere."""
    rank = max(new.values.shape[-1], previous.values.shape[-1])
    new, previous = pad_rank(new, rank), pad_rank(previous, rank)
    return LowRankMetric(torch.where(valid.unsqueeze(-1), new.scale, previous.scale),
                         torch.where(valid[..., None, None], new.vectors, previous.vectors),
                         torch.where(valid.unsqueeze(-1), new.values, previous.values))
