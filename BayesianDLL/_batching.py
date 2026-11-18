"""Helpers for evaluating a model for many MCMC chains at once.

Convention
----------
When a model is *batched* (``model.n_chains = C``), every latent parameter
(sampled or enumerated) holds values of shape ``(C, *value_shape)``: the
leading dimension is the chain dimension. ``Data``, observed values and
constants stay unbatched and broadcast. Deterministic nodes that depend on a
latent parameter are batched as well (shape ``(C, *value_shape)``).

Because plates are counted from the right, a batched tensor cannot simply be
broadcast against an unbatched one of higher rank: ``(C, 1)`` against data of
shape ``(N, M)`` would align ``C`` with ``N``. Every log density term is
therefore evaluated with its batched inputs *aligned*: the batch part of each
batched argument (its shape without the argument's event dimensions) is padded
with singleton dimensions after the chain dimension until every argument has
the same batch rank ``B`` (the largest batch rank among the term's arguments).
An argument with ``e`` event dimensions then has shape
``(C, 1, ..., 1, *value_shape)`` with ``1 + B + e`` dimensions, and ordinary
right-aligned broadcasting lines everything up. The alignment of a term is a
map ``id(node) -> rank`` that :func:`BayesianDLL.Distributions._resolve.resolve`
consults while the term is evaluated (see :func:`batched_evaluation`).

Reductions that sum a log density to a scalar (``_log_prob_unconstrained``,
``log_abs_det_jacobian``) use :func:`bsum`, which keeps the chain dimension
while a batched evaluation is active and is an ordinary ``sum()`` otherwise.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import torch


_state = threading.local()


def _stack():
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack


class _Frame:
    __slots__ = ("n_chains", "pad")

    def __init__(self, n_chains, pad):
        self.n_chains = n_chains
        self.pad = pad


def current():
    """The innermost active batched-evaluation frame, or ``None``."""
    stack = _stack()
    return stack[-1] if stack else None


def is_batched():
    frame = current()
    return frame is not None and frame.n_chains is not None


@contextmanager
def batched_evaluation(n_chains, pad=None):
    """Mark a batched evaluation (``n_chains=None``: an unbatched one)."""
    stack = _stack()
    stack.append(_Frame(n_chains, {} if pad is None else pad))
    try:
        yield stack[-1]
    finally:
        stack.pop()


def aligned(node, value):
    """``value`` of ``node`` padded to the rank of the current term alignment."""
    frame = current()
    if frame is None or not frame.pad:
        return value
    rank = frame.pad.get(id(node))
    if rank is None:
        return value
    return pad_to(value, rank)


def pad_to(value, rank):
    """Insert singleton dimensions after the chain dimension until ``value.ndim == rank``."""
    if not isinstance(value, torch.Tensor) or value.ndim >= rank:
        return value
    return value.reshape(value.shape[0], *([1] * (rank - value.ndim)), *value.shape[1:])


def bsum(value):
    """Sum to a scalar, or to one value per chain inside a batched evaluation."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0 or not is_batched():
        return value.sum()
    return value.reshape(value.shape[0], -1).sum(-1)


def n_events(value, event_numel):
    """Number of events of size ``event_numel`` per chain (all events when unbatched)."""
    total = value.numel()
    if is_batched() and value.ndim > 0:
        total = total // max(value.shape[0], 1)
    return total // event_numel


def sum_to_batched(grad, target_shape, batched_rank=None):
    """Sum a (broadcast) gradient onto ``target_shape = (C, *value_shape)``.

    ``grad`` has the chain dimension first when it depends on a batched
    value. ``batched_rank`` is the rank a batched gradient has (aligned
    gradients); a gradient of lower rank lacks the chain dimension and is
    broadcast over the chains. Without ``batched_rank`` a gradient is taken
    to be batched when its rank is at least that of the target.
    """
    target_shape = tuple(target_shape)
    if not isinstance(grad, torch.Tensor):
        grad = torch.as_tensor(grad)
    n_chains = target_shape[0]
    value_shape = target_shape[1:]
    if tuple(grad.shape) == target_shape:
        return grad
    if batched_rank is None:
        batched = grad.ndim >= len(target_shape) and grad.ndim > 0 and grad.shape[0] in (n_chains, 1)
    else:
        batched = grad.ndim >= batched_rank
    if not batched:
        grad = grad.unsqueeze(0)
    # Remove the extra leading (aligned / broadcast) dimensions of the batch part.
    extra = grad.ndim - 1 - len(value_shape)
    if extra > 0:
        grad = grad.sum(dim=tuple(range(1, 1 + extra)))
    elif extra < 0:
        grad = grad.reshape(grad.shape[0], *([1] * (-extra)), *grad.shape[1:])
    for dim, (g, t) in enumerate(zip(grad.shape[1:], value_shape), start=1):
        if g != t and t == 1:
            grad = grad.sum(dim=dim, keepdim=True)
    if grad.shape[0] != n_chains:
        grad = grad.expand(n_chains, *grad.shape[1:])
    if tuple(grad.shape[1:]) != value_shape:
        grad = torch.broadcast_to(grad, (n_chains, *value_shape))
    return grad
