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


class GuideMoments:
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

    def __init__(self, space, names, shapes, mean=None, matrix=None, dtype=None, device=None, groups=None):
        self.space = space
        self.names = list(names)
        self.shapes = {name: tuple(shapes[name]) for name in self.names}
        self.groups = dict(groups) if groups is not None else {}
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
        self._mean_vector = mean
        self.matrix = matrix
        self.estimated = []
        self.standard_error = {}
        self.n_samples = None

    def _index(self, names):
        return torch.cat([torch.arange(*self.slices[name], device=self._mean_vector.device) for name in names])

    def set_block(self, names, mean, matrix):
        """Write the moments of the latents ``names`` (flat, in that order)."""
        index = self._index(names)
        self._mean_vector = self._mean_vector.clone()
        self.matrix = self.matrix.clone()
        self._mean_vector[index] = mean.to(self._mean_vector.dtype)
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

    def mean(self, latent=None, group=None):
        """The posterior mean tensor.

        Parameters
        ----------
        latent : str or list of str, optional
            A single latent name or list of latent names. If a single name is
            given, returns the mean tensor reshaped to that parameter's shape.
            If a list is given, returns the 1-D concatenated mean vector.
        group : str, optional
            A named group. If given, returns the 1-D concatenated mean vector
            for all latents belonging to that group.

        Returns
        -------
        torch.Tensor
            The mean tensor. When neither ``latent`` nor ``group`` is given,
            returns the full 1-D mean vector of all parameters.
        """
        if group is not None:
            if latent is not None:
                raise ValueError("Specify either 'latent' or 'group', not both.")
            if group not in self.groups:
                raise KeyError(f"Group '{group}' not found in guide groups: {list(self.groups.keys())}")
            index = self._index(self.groups[group])
            return self._mean_vector[index]
        if latent is not None:
            if isinstance(latent, str):
                if latent not in self.slices:
                    raise KeyError(f"Latent '{latent}' not found in moments latents: {self.names}")
                return self._mean_vector[slice(*self.slices[latent])].reshape(self.shapes[latent])
            index = self._index(latent)
            return self._mean_vector[index]
        return self._mean_vector

    def means(self, latent=None, group=None):
        """Alias for :meth:`mean`."""
        return self.mean(latent=latent, group=group)

    _means = mean

    def variance(self, latent=None, group=None):
        """The posterior marginal variance tensor.

        Parameters
        ----------
        latent : str or list of str, optional
            A single latent name or list of latent names. If a single name is
            given, returns the variance tensor reshaped to that parameter's shape.
            If a list is given, returns the 1-D concatenated variance vector.
        group : str, optional
            A named group. If given, returns the 1-D concatenated variance vector
            for all latents belonging to that group.

        Returns
        -------
        torch.Tensor
            The variance tensor. When neither ``latent`` nor ``group`` is given,
            returns the full 1-D marginal variance vector of all parameters.
        """
        diagonal = torch.diagonal(self.matrix)
        if group is not None:
            if latent is not None:
                raise ValueError("Specify either 'latent' or 'group', not both.")
            if group not in self.groups:
                raise KeyError(f"Group '{group}' not found in guide groups: {list(self.groups.keys())}")
            index = self._index(self.groups[group])
            return diagonal[index]
        if latent is not None:
            if isinstance(latent, str):
                if latent not in self.slices:
                    raise KeyError(f"Latent '{latent}' not found in moments latents: {self.names}")
                return diagonal[slice(*self.slices[latent])].reshape(self.shapes[latent])
            index = self._index(latent)
            return diagonal[index]
        return diagonal

    def variances(self, latent=None, group=None):
        """Alias for :meth:`variance`."""
        return self.variance(latent=latent, group=group)

    def covariance(self, latent=None, group=None, second=None):
        """The posterior covariance matrix.

        Parameters
        ----------
        latent : str or list of str, optional
            A single latent name or list of latent names. If a single name is
            given, returns a 2-D covariance matrix for that parameter (a 1x1
            matrix for a scalar parameter). If a list is given, returns the
            joint covariance submatrix for those parameters.
        group : str, optional
            A named group. If given, returns the joint 2-D covariance matrix for
            all latents belonging to that group.
        second : str, optional
            If given alongside ``latent``, returns the cross-covariance block
            between ``latent`` and ``second``.

        Returns
        -------
        torch.Tensor
            A 2-D covariance matrix. When neither ``latent`` nor ``group`` is given,
            returns the full joint covariance matrix.
        """
        if second is not None:
            return self.block(latent, second)
        if group is not None:
            if latent is not None:
                raise ValueError("Specify either 'latent' or 'group', not both.")
            if group not in self.groups:
                raise KeyError(f"Group '{group}' not found in guide groups: {list(self.groups.keys())}")
            index = self._index(self.groups[group])
            return self.matrix[index.unsqueeze(-1), index.unsqueeze(0)]
        if latent is not None:
            if isinstance(latent, str):
                index = self._index([latent])
                return self.matrix[index.unsqueeze(-1), index.unsqueeze(0)]
            index = self._index(latent)
            return self.matrix[index.unsqueeze(-1), index.unsqueeze(0)]
        return self.matrix

    def covariances(self):
        """``{latent: covariance_block}`` for each latent."""
        return {name: self.covariance(name) for name in self.names}

    def block(self, first, second=None):
        """The covariance block between the flattened latents ``first`` and ``second`` (default ``first``)."""
        second = first if second is None else second
        return self.matrix[slice(*self.slices[first]), slice(*self.slices[second])]

    def __getitem__(self, name):
        return self.block(name)

    @property
    def exact(self):
        """The latents whose moments are exact."""
        return [name for name in self.names if name not in self.estimated]

    @classmethod
    def combine(cls, space, parts, order, groups=None):
        """The block-diagonal result of independent ``parts``, with the latents in ``order``."""
        shapes, owner = {}, {}
        for part in parts:
            for name in part.names:
                shapes[name] = part.shapes[name]
                owner[name] = part
        names = [name for name in order if name in owner]
        reference = parts[0]._mean_vector
        result_groups = {}
        for part in parts:
            if hasattr(part, "groups"):
                result_groups.update(part.groups)
        if groups:
            result_groups.update(groups)
        result = cls(space, names, shapes, dtype=reference.dtype, device=reference.device, groups=result_groups)
        for part in parts:
            members = [name for name in names if owner[name] is part]
            index = part._index(members)
            result.set_block(members, part._mean_vector[index], part.matrix[index.unsqueeze(-1), index.unsqueeze(0)])
            result.estimated += [name for name in members if name in part.estimated]
            result.standard_error.update({name: part.standard_error[name] for name in members if name in part.standard_error})
            if part.n_samples is not None:
                result.n_samples = part.n_samples
        return result

    @classmethod
    def compute(cls, guide, space="unconstrained", n_samples=4000, seed=0):
        """Compute the moments (mean and block-diagonal covariance) of a guide."""
        if space not in ("unconstrained", "constrained"):
            raise ValueError(f"space must be 'unconstrained' or 'constrained', got {space!r}.")
        parts = []
        for piece in guide.validate():
            if hasattr(piece, "moments"):
                parts.append(piece.moments(space, n_samples, seed))
            elif piece.__class__.__name__ == "StructuredPiece":
                parts.append(guide._piece_moments(piece, space, n_samples, seed))
            else:
                parts.append(guide._entry_moments(piece, space, n_samples, seed))
        groups = {}
        for group in guide.groups:
            if getattr(group, "name", None) is not None:
                groups[group.name] = list(group.names)
        return cls.combine(space, parts, list(guide.model.params), groups=groups)

    def __repr__(self):
        return (
            f"GuideMoments(space={self.space!r}, dim={self.dim}, exact={self.exact}, "
            f"estimated={self.estimated}, n_samples={self.n_samples})"
        )


