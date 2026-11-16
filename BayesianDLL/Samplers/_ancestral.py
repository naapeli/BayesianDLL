"""Exact (ancestral) forward sampling of a model's graph.

Used by the predictive functions in :mod:`._sample` and by the variational
guides, which need draws of a given shape from a distribution.
"""

import math

import torch


def _event_and_batch_shape(distribution):
    try:
        return tuple(distribution.batch_shape) + tuple(distribution.event_shape)
    except Exception:
        return None


def _broadcast_index(source_shape, target_shape):
    """Map every element of ``target_shape`` to an element of ``source_shape``.

    ``source_shape`` broadcasts to ``target_shape``. Returns ``(index, rank, m)``
    where ``index[k]`` is the flat source element that target element ``k``
    repeats, ``rank[k]`` numbers the repetitions of that source element and
    ``m`` is the number of repetitions of each source element.
    """
    n_target = math.prod(target_shape)
    n_source = math.prod(source_shape)
    m = n_target // n_source
    padded = (1,) * (len(target_shape) - len(source_shape)) + tuple(source_shape)
    index = torch.arange(n_source).reshape(padded).expand(target_shape).reshape(-1)
    order = torch.argsort(index, stable=True)
    rank = torch.empty_like(index)
    rank[order] = torch.arange(n_target) - m * index[order]
    return index, rank, m


def draw_shaped(distribution, n, target_shape, reparametrized=False):
    """Draw ``n`` independent values of shape ``target_shape`` from ``distribution``.

    ``distribution.sample`` returns ``(n, *batch_shape, *event_shape)``. When
    that shape broadcasts to ``target_shape`` (e.g. a scalar prior used for a
    vector parameter, or a likelihood with scalar parameters inside a plate),
    every element of the result is still an independent draw: the
    distribution is sampled ``m`` times as often and the extra draws fill the
    broadcast positions.

    With ``reparametrized=True`` returns ``(samples, {param: d sample / d param})``
    with the derivatives rearranged in the same way.
    """
    target_shape = tuple(target_shape)
    source_shape = _event_and_batch_shape(distribution)
    if source_shape is not None and source_shape != target_shape:
        n_target, n_source = math.prod(target_shape), math.prod(source_shape)
        broadcastable = len(source_shape) <= len(target_shape)
        if broadcastable:
            try:
                padded = (1,) * (len(target_shape) - len(source_shape)) + source_shape
                broadcastable = tuple(torch.broadcast_shapes(padded, target_shape)) == target_shape
            except RuntimeError:
                broadcastable = False
        if broadcastable and n_target != n_source:
            index, rank, m = _broadcast_index(source_shape, target_shape)

            def rearrange(values):
                values = values.reshape(n, m, n_source)
                return values[:, rank, index].reshape(n, *target_shape)

            if reparametrized:
                samples, grads = distribution.sample(n * m, _reparametrization_trick_grad=True)
                grads = {
                    name: rearrange(torch.as_tensor(grad).expand(samples.shape))
                    for name, grad in grads.items()
                }
                return rearrange(samples), grads
            return rearrange(distribution.sample(n * m))
        if not broadcastable and n_target != n_source:
            raise ValueError(
                f"A {type(distribution).__name__} with batch and event shape {source_shape} "
                f"cannot produce draws of shape {target_shape}."
            )

    if reparametrized:
        samples, grads = distribution.sample(n, _reparametrization_trick_grad=True)
        grads = {name: torch.as_tensor(grad).expand(samples.shape).reshape(n, *target_shape) for name, grad in grads.items()}
        return samples.reshape(n, *target_shape), grads
    samples = distribution.sample(n)
    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)
    return samples.reshape(n, *target_shape)


def can_sample(distribution):
    """``True`` when ``distribution`` provides an exact sampler."""
    sampler = getattr(distribution, "sample", None)
    if sampler is None or not callable(sampler):
        return False
    from ..Distributions import Distribution
    if isinstance(distribution, Distribution) and type(distribution).sample is Distribution.sample:
        return False
    return True
