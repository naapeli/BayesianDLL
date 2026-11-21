"""Means and covariances of a guide: the result object and Monte Carlo helpers."""

import torch


def transform_batch(function, values, shape):
    """Apply ``function`` to every draw of ``values`` ``(n, ...)``; the results have shape ``(n, *shape)``.

    The batch is transformed in one call when the transform supports a
    leading dimension, and draw by draw otherwise.
    """
    n = values.shape[0]
    expected = (n, *tuple(shape))
    try:
        out = function(values)
        if tuple(out.shape) == expected:
            return out
    except (RuntimeError, ValueError, IndexError):
        pass
    return torch.stack([function(value) for value in values]).reshape(expected)


def mc_moments(draws):
    """``(mean, covariance, standard_error_of_the_mean)`` of the draws ``(n, d)``."""
    n = draws.shape[0]
    mean = draws.mean(0)
    centred = draws - mean
    covariance = centred.T @ centred / max(n - 1, 1)
    standard_error = torch.sqrt(torch.diagonal(covariance) / n)
    return mean, covariance, standard_error


def _numel(shape):
    count = 1
    for size in shape:
        count *= int(size)
    return count


class GuideCovariance:
    """Mean and covariance of (part of) a guide over the concatenated flattened latents.

    Attributes:
        space: ``"unconstrained"`` or ``"constrained"``.
        names: the latents, in the order of the flat vector.
        shapes: ``{latent: shape}`` of each latent in ``space``.
        slices: ``{latent: (start, end)}`` of each latent in the flat vector.
        mean: the flat mean vector.
        matrix: the full covariance matrix. Blocks between independent parts
            of the guide (different groups or hand-written entries) are
            exactly zero.
        estimated: the latents whose moments are Monte Carlo estimates; every
            block that involves one of them is estimated, all other blocks
            are exact.
        standard_error: ``{latent: sd / sqrt(n)}`` per coordinate for the
            estimated latents (the standard error of the Monte Carlo mean).
        n_samples: the number of draws behind the estimates (``None`` when
            everything is exact).
    """

    def __init__(self, space, names, shapes, mean=None, matrix=None, dtype=None, device=None):
        self.space = space
        self.names = list(names)
        self.shapes = {name: tuple(shapes[name]) for name in self.names}
        self.slices = {}
        offset = 0
        for name in self.names:
            size = _numel(self.shapes[name])
            self.slices[name] = (offset, offset + size)
            offset += size
        self.dim = offset
        if mean is None:
            mean = torch.zeros(self.dim, dtype=dtype, device=device)
        if matrix is None:
            matrix = torch.zeros(self.dim, self.dim, dtype=mean.dtype, device=mean.device)
        self.mean = mean
        self.matrix = matrix
        self.estimated = []
        self.standard_error = {}
        self.n_samples = None

    def _index(self, names):
        return torch.cat([torch.arange(*self.slices[name], device=self.mean.device) for name in names])

    def set_block(self, names, mean, matrix):
        """Write the moments of the latents ``names`` (flat, in that order)."""
        index = self._index(names)
        self.mean = self.mean.clone()
        self.matrix = self.matrix.clone()
        self.mean[index] = mean.to(self.mean.dtype)
        self.matrix[index.unsqueeze(-1), index.unsqueeze(0)] = matrix.to(self.matrix.dtype)

    def set_standard_errors(self, names, standard_error, keep, n_samples):
        """Record the Monte Carlo standard errors of the latents in ``keep`` (``standard_error`` is flat over ``names``)."""
        offset = 0
        for name in names:
            size = _numel(self.shapes[name])
            if name in keep:
                self.standard_error[name] = standard_error[offset:offset + size].reshape(self.shapes[name])
            offset += size
        self.n_samples = int(n_samples)

    def means(self):
        """``{latent: mean}`` with each latent's shape."""
        return {name: self.mean[slice(*self.slices[name])].reshape(self.shapes[name]) for name in self.names}

    def covariance(self, first=None, second=None):
        """The full covariance matrix if arguments are omitted, or the block between ``first`` and ``second``."""
        if first is None and second is None:
            return self.matrix
        return self.block(first, second)

    def covariances(self):
        """``{latent: covariance_block}`` for each latent."""
        return {name: self.block(name) for name in self.names}

    def block(self, first, second=None):
        """The covariance block between the flattened latents ``first`` and ``second`` (default ``first``)."""
        second = first if second is None else second
        return self.matrix[slice(*self.slices[first]), slice(*self.slices[second])]

    def __getitem__(self, name):
        return self.block(name)

    def variances(self):
        """``{latent: variance}`` with each latent's shape."""
        diagonal = torch.diagonal(self.matrix)
        return {name: diagonal[slice(*self.slices[name])].reshape(self.shapes[name]) for name in self.names}

    @property
    def exact(self):
        """The latents whose moments are exact."""
        return [name for name in self.names if name not in self.estimated]

    @classmethod
    def combine(cls, space, parts, order):
        """The block-diagonal result of independent ``parts``, with the latents in ``order``."""
        shapes, owner = {}, {}
        for part in parts:
            for name in part.names:
                shapes[name] = part.shapes[name]
                owner[name] = part
        names = [name for name in order if name in owner]
        reference = parts[0].mean
        result = cls(space, names, shapes, dtype=reference.dtype, device=reference.device)
        for part in parts:
            members = [name for name in names if owner[name] is part]
            index = part._index(members)
            result.set_block(members, part.mean[index], part.matrix[index.unsqueeze(-1), index.unsqueeze(0)])
            result.estimated += [name for name in members if name in part.estimated]
            result.standard_error.update({name: part.standard_error[name] for name in members if name in part.standard_error})
            if part.n_samples is not None:
                result.n_samples = part.n_samples
        return result

    def __repr__(self):
        return (
            f"GuideCovariance(space={self.space!r}, dim={self.dim}, exact={self.exact}, "
            f"estimated={self.estimated}, n_samples={self.n_samples})"
        )
