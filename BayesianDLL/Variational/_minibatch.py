"""Subsampling of plated observations for stochastic (minibatch) variational inference."""

from contextlib import contextmanager

import networkx as nx
import torch

from .._data import Data


def _plate_data(plate):
    size = plate._size
    return size if isinstance(size, Data) else None


class Subsampler:
    """Draw random minibatches along one plate whose size comes from ``Data``.

    Every observed node inside the plate is restricted to ``batch_size``
    random indices per minibatch: the ``Data`` nodes feeding them (covariates
    and the plate's size, when their batch dimension is the plate dimension)
    are sliced with ``Data.set_value`` and observed values stored as plain
    tensors are sliced directly. Everything is restored afterwards. The
    likelihood terms of those nodes are scaled by ``N / batch_size`` so the
    minibatch ELBO is an unbiased estimate of the full-data ELBO.
    """

    def __init__(self, model, batch_size, subsample_plate=None):
        if not model._compiled:
            model.compile()
        if getattr(model, "enumerated_params", {}):
            raise NotImplementedError("Minibatch inference does not support enumerated parameters.")
        candidates = {}
        for observed in model.observed_params.values():
            for plate in observed.plates:
                if subsample_plate is not None and plate.name != subsample_plate:
                    continue
                if _plate_data(plate) is not None:
                    candidates.setdefault(plate.name, plate)
        if not candidates:
            if subsample_plate is None:
                raise ValueError(
                    "Minibatch inference needs observations inside a plate whose size comes from Data, "
                    "e.g. `with plate('observations', x_data):`. Integer plate sizes and observations "
                    "outside a plate cannot be subsampled."
                )
            raise ValueError(f"No observation is inside a plate named '{subsample_plate}' whose size comes from Data.")
        if len(candidates) > 1:
            raise ValueError(f"Several plates can be subsampled ({sorted(candidates)}); choose one with subsample_plate.")
        self.plate_name, plate = next(iter(candidates.items()))
        # ``plate.dim`` is the dimension of the sizing Data that gives the plate
        # size; ``plate.position`` is where the plate sits among the leading
        # plate dimensions of the values (0 for the outermost plate).
        self.dim = plate.dim
        self.position = plate.position
        self.size_data = _plate_data(plate)
        self.n = int(plate.size)
        self.batch_size = int(batch_size)
        if not 0 < self.batch_size <= self.n:
            raise ValueError(f"batch_size must be between 1 and the plate size {self.n}, got {batch_size}.")

        self.observed = [o for o in model.observed_params.values() if any(p.name == self.plate_name for p in o.plates)]
        for name, parameter in model.params.items():
            if any(p.name == self.plate_name for p in getattr(parameter, "plates", [])):
                raise ValueError(f"The latent '{name}' is inside the plate '{self.plate_name}'; local latents cannot be subsampled.")

        # Data nodes to slice: the plate's size and the Data feeding the observations.
        ancestors = set()
        for observed in self.observed:
            ancestors |= nx.ancestors(model.graph, observed.name)
        # Each Data node is sliced along one axis of its value: the sizing Data
        # along the dimension that gives the plate size, the other Data (the
        # covariates of the observations) along the plate's position when that
        # dimension has the plate size.
        self.data = [(self.size_data, self._data_axis(self.size_data, self.dim))]
        for name in ancestors:
            data = model.data.get(name)
            if data is None or data is self.size_data:
                continue
            batch_shape = data.batch_shape
            if len(batch_shape) > self.position and batch_shape[self.position] == self.n:
                self.data.append((data, self.position))
        self.scales = {o.name: self.n / self.batch_size for o in self.observed}

    @staticmethod
    def _data_axis(data, dim):
        """Non-negative axis of ``data.value`` for a dimension of its batch shape."""
        return dim % len(data.batch_shape)

    @contextmanager
    def minibatch(self, generator=None):
        """Slice the model to a random minibatch; yields the likelihood scales."""
        index = torch.randperm(self.n, generator=generator)[: self.batch_size]
        saved_data = [(data, axis, data.value) for data, axis in self.data]
        saved_observed = [(observed, observed._observed_values) for observed in self.observed]
        try:
            for data, axis, value in saved_data:
                data.set_value(value.index_select(axis, index.to(value.device)))
            for observed, value in saved_observed:
                if isinstance(value, Data):
                    continue
                # Plate dimensions lead the observed values, outermost first.
                observed.observed_values = value.index_select(self.position, index.to(value.device))
            yield self.scales
        finally:
            for data, _, value in saved_data:
                data._value = value
            for observed, value in saved_observed:
                observed._observed_values = value
