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
