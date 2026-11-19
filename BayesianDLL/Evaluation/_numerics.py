"""Small numerical helpers for the diagnostics, written with torch tensors only."""

import math

import torch


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Ranks ``1..n`` of a flattened tensor, ties replaced by their average rank.

    Matches ``scipy.stats.rankdata(values, method="average")``: the result is a
    float64 tensor of the input's (flattened) length, and all ranks are NaN if
    the input holds a NaN.
    """
    values = values.reshape(-1)
    n = values.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float64, device=values.device)
    if torch.isnan(values).any():
        return torch.full((n,), math.nan, dtype=torch.float64, device=values.device)

    order = torch.argsort(values, stable=True)
    _, counts = torch.unique_consecutive(values[order], return_counts=True)
    ends = torch.cumsum(counts, dim=0).to(torch.float64)  # 1-based rank of the last tie
    average = ends - (counts.to(torch.float64) - 1.0) / 2.0
    ranks = torch.empty(n, dtype=torch.float64, device=values.device)
    ranks[order] = torch.repeat_interleave(average, counts)
    return ranks


def z_scale(values: torch.Tensor) -> torch.Tensor:
    """Rank-normalise ``values`` with Blom's transform ``Phi^-1((r - 3/8) / (S + 1/4))``.

    Returns a float64 tensor of the input's shape.
    """
    size = values.numel()
    u = (rankdata(values) - 0.375) / (size + 0.25)
    u = u.clamp(1e-10, 1.0 - 1e-10)
    return torch.special.ndtri(u).reshape(values.shape)


def next_fast_len(n: int) -> int:
    """The smallest 5-smooth number (of the form ``2^a 3^b 5^c``) at least ``n``."""
    if n <= 1:
        return 1
    best = 1 << (n - 1).bit_length()  # the next power of two is always 5-smooth
    power5 = 1
    while power5 < best:
        power35 = power5
        while power35 < best:
            candidate = power35 << max(0, (math.ceil(n / power35) - 1).bit_length())
            best = min(best, candidate)
            power35 *= 3
        power5 *= 5
    return best


def autocov(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Biased autocovariance ``1/n sum_t (x_t - mean)(x_{t+k} - mean)`` along ``dim`` via FFT."""
    values = values.to(torch.float64)
    n = values.shape[dim]
    m = next_fast_len(2 * n)
    centered = values - values.mean(dim=dim, keepdim=True)
    spectrum = torch.fft.rfft(centered, n=m, dim=dim)
    power = spectrum * spectrum.conj()
    cov = torch.fft.irfft(power, n=m, dim=dim).narrow(dim, 0, n)
    return cov / n


def gaussian_kde(samples: torch.Tensor, grid: torch.Tensor, chunk_size: int = 2**22) -> torch.Tensor:
    """Gaussian kernel density estimate of 1-d ``samples`` evaluated at ``grid``.

    The bandwidth follows Scott's rule, ``sd * n^(-1/5)`` with the sample
    standard deviation (``ddof=1``), the default of ``scipy.stats.gaussian_kde``.
    The ``samples x grid`` kernel matrix is evaluated in pieces of at most
    ``chunk_size`` entries. Returns a float64 tensor of the grid's shape.
    """
    samples = torch.as_tensor(samples).detach().reshape(-1).to(torch.float64)
    grid = torch.as_tensor(grid).detach().to(dtype=torch.float64, device=samples.device)
    n = samples.numel()
    if n < 2:
        raise ValueError("A kernel density estimate needs at least two samples.")
    bandwidth = samples.std() * n ** (-1.0 / 5.0)
    if not torch.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("A kernel density estimate needs samples with a positive, finite standard deviation.")

    flat_grid = grid.reshape(-1)
    step = max(1, chunk_size // n)
    norm = n * bandwidth * math.sqrt(2.0 * math.pi)
    density = torch.empty_like(flat_grid)
    for start in range(0, flat_grid.numel(), step):
        points = flat_grid[start:start + step]
        z = (points[:, None] - samples[None, :]) / bandwidth
        density[start:start + step] = torch.exp(-0.5 * z * z).sum(dim=1) / norm
    return density.reshape(grid.shape)


def _log_beta(a: float, b: float) -> float:
    """``log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)``."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 100_000, tol: float = 1e-16) -> float:
    """Continued fraction of the incomplete beta function, evaluated with the modified Lentz algorithm.

    Returns ``K`` in ``I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * K`` where
    ``K = 1 / (1 + d_1 / (1 + d_2 / (1 + ...)))`` with
    ``d_{2m+1} = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1))`` and
    ``d_{2m} = m (b - m) x / ((a + 2m - 1)(a + 2m))``. It converges quickly
    for ``x < (a + 1) / (a + b + 2)``, in ``O(sqrt(max(a, b)))`` terms.
    """
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            return h
    raise RuntimeError(f"The incomplete beta continued fraction did not converge for a={a}, b={b}, x={x}.")


def _betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function ``I_x(a, b)``, for ``a, b > 0`` and ``0 <= x <= 1``.

    ``I_x(a, b) = B(a, b)^-1 int_0^x t^(a-1) (1 - t)^(b-1) dt``, the cdf of
    ``Beta(a, b)`` at ``x``. The prefactor ``x^a (1 - x)^b / B(a, b)`` is formed
    in log space with ``lgamma``, the remaining factor is the continued fraction
    of :func:`_betacf`, and for ``x > (a + 1) / (a + b + 2)``, where the
    fraction converges slowly, the symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)``
    is used. Matches ``scipy.special.betainc(a, b, x)``.
    """
    a, b, x = float(a), float(b), float(x)
    if not (a > 0.0 and b > 0.0) or not (0.0 <= x <= 1.0):
        return math.nan
    if x == 0.0 or x == 1.0:
        return x
    log_front = a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def _betaincinv(a: float, b: float, p: float, max_iter: int = 1000) -> float:
    """The inverse of :func:`_betainc` in ``x``: the ``p`` quantile of ``Beta(a, b)``.

    Solves ``I_x(a, b) = p`` on ``[0, 1]`` with a safeguarded Newton iteration:
    the root is kept in a bracket ``[lo, hi]`` that shrinks with each evaluation,
    a Newton step ``x - (I_x - p) / f(x)`` with the Beta density ``f`` is taken
    when it lands inside the bracket, and the bracket is bisected otherwise.
    Matches ``scipy.special.betaincinv(a, b, p)``.
    """
    a, b, p = float(a), float(b), float(p)
    if not (a > 0.0 and b > 0.0) or not (0.0 <= p <= 1.0):
        return math.nan
    if p == 0.0 or p == 1.0:
        return p
    log_b = _log_beta(a, b)
    lo, hi = 0.0, 1.0
    x = a / (a + b)
    for _ in range(max_iter):
        f = _betainc(a, b, x) - p
        if f == 0.0:
            return x
        if f < 0.0:
            lo = x
        else:
            hi = x
        log_density = (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - log_b
        density = math.exp(log_density) if log_density < 700.0 else math.inf
        candidate = x - f / density if density > 0.0 else math.nan
        if not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)
        if abs(candidate - x) <= 4.0 * math.ulp(x) or hi - lo <= 4.0 * math.ulp(x):
            return candidate
        x = candidate
    return x
