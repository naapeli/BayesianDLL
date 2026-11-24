"""Differentiable building blocks for model graphs.

Every block is a :class:`~BayesianDLL.DeterministicParameter` with a
hand-written vector-Jacobian product (``vjp_func``), so the model's reverse
pass never builds a dense Jacobian. Inputs may be model nodes
(``RandomParameter``, ``DeterministicParameter``, ``Data``), constant tensors or
Python numbers; numbers adopt the floating dtype and device of the tensor
inputs. Elementwise blocks broadcast their inputs like ``torch`` does, and the
gradient of a broadcast input is summed back onto its shape.

The first argument of every block is the name of the new deterministic
node. ``Data`` inputs receive no gradient. With vectorized chains the values
of batched nodes carry a leading chain dimension; ``dim`` arguments address
the same axis with and without it (negative dimensions count from the right).
"""

from __future__ import annotations

from numbers import Number

import torch
import torch.nn.functional as F

from .._data import Data
from .._parameters import DeterministicParameter, ObservedParameter, RandomParameter, VariationalParameter


_NODE_TYPES = (RandomParameter, DeterministicParameter, ObservedParameter, Data)


def _is_node(value):
    return isinstance(value, _NODE_TYPES)


def _sum_to_shape(grad, shape):
    from .._model import _sum_to_shape as sum_to_shape
    return sum_to_shape(grad, tuple(shape))


def _negative_dim(dim, rank):
    """``dim`` counted from the right of an unbatched tensor of rank ``rank``.

    Negative dimensions address the same axis of a batched tensor (with a
    leading chain dimension) and of the unbatched one.
    """
    if isinstance(dim, (tuple, list)):
        return tuple(_negative_dim(d, rank) for d in dim)
    return dim - rank if dim >= 0 else dim


def _reference(values):
    """Floating dtype and device shared by the tensor-valued arguments."""
    dtype = None
    device = None
    for value in values:
        if isinstance(value, torch.Tensor):
            if device is None:
                device = value.device
            if value.is_floating_point():
                dtype = value.dtype if dtype is None else torch.promote_types(dtype, value.dtype)
    return (torch.get_default_dtype() if dtype is None else dtype), device


class _Block(DeterministicParameter):
    """Base class: maps model-node values to argument values and reduces VJPs.

    Subclasses implement ``_compute(values)`` (the forward value from the list
    of argument values), ``_arg_grads(g, values, out, needs)`` (the gradient
    w.r.t. every argument for which ``needs[i]`` is true, ``None`` otherwise;
    gradients may have the broadcast shape) and optionally
    ``_local(values, out, needs)`` (elementwise partial derivatives, which
    makes :meth:`derivatives` cheap).
    """

    _non_differentiable = ()  # argument positions that never receive a gradient

    def __init__(self, name, args):
        args = list(args)
        for i, arg in enumerate(args):
            if not (
                _is_node(arg) or isinstance(arg, torch.Tensor | VariationalParameter)
                or (isinstance(arg, Number) and not isinstance(arg, bool))
            ):
                raise TypeError(
                    f"Argument {i} of '{name}' must be a model node, a VariationalParameter, a tensor or a number; "
                    f"got {type(arg).__name__}."
                )
        nodes = []
        node_index = {}
        positions = []
        for arg in args:
            if _is_node(arg):
                if arg.name not in node_index:
                    node_index[arg.name] = len(nodes)
                    nodes.append(arg)
                positions.append(node_index[arg.name])
            else:
                positions.append(None)
        self._args = args
        self._positions = positions
        self._needs = tuple(
            _is_node(arg) and not isinstance(arg, Data) and i not in self._non_differentiable
            for i, arg in enumerate(args)
        )
        # VariationalParameter arguments (blocks inside a guide) are read at
        # every evaluation and receive gradients through variational_vjp.
        self._vp_needs = tuple(
            isinstance(arg, VariationalParameter) and i not in self._non_differentiable for i, arg in enumerate(args)
        )
        self.variational_parameters = {arg.name: arg for arg in args if isinstance(arg, VariationalParameter)}
        derivative = self._derivative_from_nodes if type(self)._local is not _Block._local else None
        super().__init__(name, self._forward_from_nodes, derivative, nodes, vjp_func=self._vjp_from_nodes)

    # ------------------------------------------------------------ plumbing
    def _values(self, node_values):
        raw = [
            node_values[p] if p is not None else (arg.value if isinstance(arg, VariationalParameter) else arg)
            for p, arg in zip(self._positions, self._args)
        ]
        dtype, device = _reference(raw)
        values = []
        for value in raw:
            if isinstance(value, torch.Tensor):
                values.append(value if device is None or value.device == device else value.to(device))
            else:
                values.append(torch.as_tensor(value, dtype=dtype, device=device))
        return values

    def _forward_from_nodes(self, *node_values):
        return self._compute(self._values(node_values))

    def _vjp_from_nodes(self, cotangent, *node_values):
        values = self._values(node_values)
        out = self._compute(values)
        grads = self._arg_grads(cotangent, values, out, self._needs)
        result = {}
        for arg, need, value, grad in zip(self._args, self._needs, values, grads):
            if not need or grad is None:
                continue
            grad = _sum_to_shape(grad, value.shape)
            result[arg.name] = result[arg.name] + grad if arg.name in result else grad
        return result

    def variational_vjp(self, cotangent):
        """Gradients ``{name: cotangent^T J}`` w.r.t. the VariationalParameter arguments at the current inputs."""
        if not self.variational_parameters:
            return {}
        values = self._values(self.input_values())
        out = self._compute(values)
        grads = self._arg_grads(cotangent, values, out, self._vp_needs)
        result = {}
        for arg, need, grad in zip(self._args, self._vp_needs, grads):
            if not need or grad is None:
                continue
            grad = _sum_to_shape(grad, arg.value.shape)
            result[arg.name] = result[arg.name] + grad if arg.name in result else grad
        return result

    def _derivative_from_nodes(self, *node_values):
        values = self._values(node_values)
        out = self._compute(values)
        locals_ = self._local(values, out, self._needs)
        result = {}
        for arg, need, local in zip(self._args, self._needs, locals_):
            if not need or local is None:
                continue
            local = torch.broadcast_to(local, out.shape)
            result[arg.name] = result[arg.name] + local if arg.name in result else local
        return result

    # ------------------------------------------------------------ batching
    #: Whether batched arguments are aligned (padded after the chain dimension
    #: to a common batch rank) before ``_compute_batched``.
    _align_arguments = True

    def _arg_event_ndims(self, unbatched_ndims):
        """Trailing event dimensions of every argument (elementwise blocks: none)."""
        return [0] * len(unbatched_ndims)

    def _batched_setup(self, node_values):
        values = self._values(node_values)
        node_flags = self._batched_flags()
        flags = [node_flags[p] if p is not None else False for p in self._positions]
        unbatched_ndims = [v.ndim - int(f) for v, f in zip(values, flags)]
        if self._align_arguments:
            values, _ = self._align_batched(values, flags, self._arg_event_ndims(unbatched_ndims))
        return values, flags, unbatched_ndims

    def batched_forward(self, input_values):
        values, flags, unbatched_ndims = self._batched_setup(input_values)
        return self._to_batched_output(self._compute_batched(values, flags, unbatched_ndims))

    def batched_vjp(self, cotangent, input_values):
        from .._batching import pad_to, sum_to_batched
        values, flags, unbatched_ndims = self._batched_setup(input_values)
        out = self._compute_batched(values, flags, unbatched_ndims)
        g = pad_to(cotangent, out.ndim)
        grads = self._arg_grads_batched(g, values, out, self._needs, flags, unbatched_ndims)
        result = {}
        for arg, need, flag, position, grad in zip(self._args, self._needs, flags, self._positions, grads):
            if not need or not flag or grad is None:
                continue
            grad = sum_to_batched(grad, input_values[position].shape)
            result[arg.name] = result[arg.name] + grad if arg.name in result else grad
        return result

    def _compute_batched(self, values, flags, unbatched_ndims):
        return self._compute(values)

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        return self._arg_grads(g, values, out, needs)

    # ------------------------------------------------------------ to override
    def _compute(self, values):
        raise NotImplementedError

    def _arg_grads(self, g, values, out, needs):
        raise NotImplementedError

    def _local(self, values, out, needs):
        raise NotImplementedError


class _ElementwiseBlock(_Block):
    """Elementwise block: the VJP is ``g * local`` for every argument."""

    def _arg_grads(self, g, values, out, needs):
        return [None if local is None else g * local for local in self._local(values, out, needs)]


# ============================================================ arithmetic
class Add(_ElementwiseBlock):
    r"""Elementwise addition node :math:`y = x_1 + x_2 + \dots`.

    Parameters
    ----------
    name : str
        Node name.
    *inputs : Parameter or Data or torch.Tensor
        Inputs to add.
    """

    def __init__(self, name, *inputs):
        if len(inputs) < 1:
            raise ValueError("Add needs at least one input.")
        super().__init__(name, inputs)

    def _compute(self, values):
        out = values[0]
        for value in values[1:]:
            out = out + value
        return out

    def _local(self, values, out, needs):
        return [torch.ones_like(out) if need else None for need in needs]


class Subtract(_ElementwiseBlock):
    r"""Elementwise subtraction node :math:`y = a - b`.

    Parameters
    ----------
    name : str
        Node name.
    a : Parameter or Data or torch.Tensor
        Minuend tensor.
    b : Parameter or Data or torch.Tensor
        Subtrahend tensor.
    """

    def __init__(self, name, a, b):
        super().__init__(name, [a, b])

    def _compute(self, values):
        return values[0] - values[1]

    def _local(self, values, out, needs):
        return [torch.ones_like(out) if needs[0] else None, -torch.ones_like(out) if needs[1] else None]


class Multiply(_ElementwiseBlock):
    r"""Elementwise multiplication node :math:`y = x_1 \cdot x_2 \cdot \dots`.

    Parameters
    ----------
    name : str
        Node name.
    *inputs : Parameter or Data or torch.Tensor
        Inputs to multiply.
    """

    def __init__(self, name, *inputs):
        if len(inputs) < 1:
            raise ValueError("Multiply needs at least one input.")
        super().__init__(name, inputs)

    def _compute(self, values):
        out = values[0]
        for value in values[1:]:
            out = out * value
        return out

    def _local(self, values, out, needs):
        locals_ = []
        for i, need in enumerate(needs):
            if not need:
                locals_.append(None)
                continue
            # Product of the other factors (exact also when a factor is zero).
            other = torch.ones_like(out)
            for j, value in enumerate(values):
                if j != i:
                    other = other * value
            locals_.append(other)
        return locals_


class Divide(_ElementwiseBlock):
    r"""Elementwise division node :math:`y = a / b`.

    Parameters
    ----------
    name : str
        Node name.
    a : Parameter or Data or torch.Tensor
        Numerator tensor.
    b : Parameter or Data or torch.Tensor
        Denominator tensor.
    """

    def __init__(self, name, a, b):
        super().__init__(name, [a, b])

    def _compute(self, values):
        return values[0] / values[1]

    def _local(self, values, out, needs):
        a, b = values
        return [1 / b if needs[0] else None, -out / b if needs[1] else None]


class Affine(_ElementwiseBlock):
    r"""Elementwise affine (location-scale) transform :math:`y = \text{loc} + \text{scale} \cdot x`.

    The building block of the non-centred parameterisation of hierarchical
    models: sampling ``z ~ Normal(0, 1)`` and setting
    ``theta = Affine("theta", z, loc=mu, scale=tau)`` removes the funnel that
    ``theta ~ Normal(mu, tau)`` causes for NUTS when the data are weak.

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Input tensor.
    loc : Parameter or Data or torch.Tensor or float, default 0.0
        Location parameter.
    scale : Parameter or Data or torch.Tensor or float, default 1.0
        Scale parameter.
    """

    def __init__(self, name, x, loc=0.0, scale=1.0):
        super().__init__(name, [x, loc, scale])

    def _compute(self, values):
        x, loc, scale = values
        return loc + scale * x

    def _local(self, values, out, needs):
        x, loc, scale = values
        return [
            torch.broadcast_to(scale, out.shape) if needs[0] else None,
            torch.ones_like(out) if needs[1] else None,
            torch.broadcast_to(x, out.shape) if needs[2] else None,
        ]


class Elementwise(_ElementwiseBlock):
    """A user-defined elementwise function with a hand-written derivative.

    ``fn(*values)`` computes the output from the input values and
    ``dfn(*values)`` returns a tuple with the elementwise partial derivative
    of the output w.r.t. every input (broadcastable to the output; ``None``
    for an input without gradient). Example::

        Elementwise("y", lambda x: x ** 3, lambda x: (3 * x ** 2,), x)
    """

    def __init__(self, name, fn, dfn, *inputs):
        if not callable(fn) or not callable(dfn):
            raise TypeError("fn and dfn must be callable.")
        if len(inputs) < 1:
            raise ValueError("Elementwise needs at least one input.")
        self.fn = fn
        self.dfn = dfn
        super().__init__(name, inputs)

    def _compute(self, values):
        return self.fn(*values)

    def _local(self, values, out, needs):
        partials = self.dfn(*values)
        if isinstance(partials, torch.Tensor):
            partials = (partials,)
        partials = tuple(partials)
        if len(partials) != len(values):
            raise ValueError(f"dfn of '{self.name}' returned {len(partials)} partial derivatives for {len(values)} inputs.")
        return [
            None if (not need or partial is None) else torch.as_tensor(partial, dtype=out.dtype, device=out.device)
            for need, partial in zip(needs, partials)
        ]


# ============================================================ unary nonlinearities
class Softplus(_ElementwiseBlock):
    r"""Elementwise softplus node :math:`y = \ln(1 + e^x)`.

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Input tensor.
    """

    def __init__(self, name, x):
        super().__init__(name, [x])

    def _compute(self, values):
        return F.softplus(values[0])

    def _local(self, values, out, needs):
        return [torch.sigmoid(values[0]) if needs[0] else None]


class Tanh(_ElementwiseBlock):
    r"""Elementwise hyperbolic tangent node :math:`y = \tanh(x)`.

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Input tensor.
    """

    def __init__(self, name, x):
        super().__init__(name, [x])

    def _compute(self, values):
        return torch.tanh(values[0])

    def _local(self, values, out, needs):
        return [1 - out ** 2 if needs[0] else None]


class Softmax(_Block):
    r"""Softmax node mapping real vectors to probability simplex along a dimension.

    .. math::

        \mathrm{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    For a categorical likelihood prefer ``Categorical(logits=...)``, which is
    numerically more stable.

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Input tensor.
    dim : int, default -1
        Dimension along which softmax is computed.
    """

    def __init__(self, name, x, dim=-1):
        self.dim = dim
        super().__init__(name, [x])

    def _compute(self, values):
        return torch.softmax(values[0], dim=self.dim)

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None]
        return [out * (g - (g * out).sum(dim=self.dim, keepdim=True))]

    def _compute_batched(self, values, flags, unbatched_ndims):
        return torch.softmax(values[0], dim=_negative_dim(self.dim, unbatched_ndims[0]))

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        if not needs[0]:
            return [None]
        dim = _negative_dim(self.dim, unbatched_ndims[0])
        return [out * (g - (g * out).sum(dim=dim, keepdim=True))]


# ============================================================ reductions and shapes
class Sum(_Block):
    r"""Sum reduction node over specified dimension(s).

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Input tensor.
    dim : int or tuple of int, optional
        Dimension(s) to sum over. If ``None``, sums over all elements.
    keepdim : bool, default False
        Whether the output tensor has ``dim`` retained with size 1.
    """

    def __init__(self, name, x, dim=None, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim
        super().__init__(name, [x])

    def _compute(self, values):
        x = values[0]
        if self.dim is None:
            out = x.sum()
            return out.reshape([1] * x.ndim) if self.keepdim else out
        return x.sum(dim=self.dim, keepdim=self.keepdim)

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None]
        x = values[0]
        if self.dim is None:
            return [g.reshape(()).expand(x.shape) if g.numel() == 1 else g.expand(x.shape)]
        if not self.keepdim:
            dims = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
            for d in sorted(d % x.ndim for d in dims):
                g = g.unsqueeze(d)
        return [g.expand(x.shape)]

    def _compute_batched(self, values, flags, unbatched_ndims):
        x = values[0]
        if self.dim is None:
            out = x.reshape(x.shape[0], -1).sum(-1)
            return out.reshape(x.shape[0], *([1] * unbatched_ndims[0])) if self.keepdim else out
        return x.sum(dim=_negative_dim(self.dim, unbatched_ndims[0]), keepdim=self.keepdim)

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        if not needs[0]:
            return [None]
        x = values[0]
        if self.dim is None:
            return [g.reshape(x.shape[0], *([1] * (x.ndim - 1))).expand(x.shape)]
        if not self.keepdim:
            dims = _negative_dim(self.dim, unbatched_ndims[0])
            dims = (dims,) if isinstance(dims, int) else tuple(dims)
            for d in sorted(d % x.ndim for d in dims):
                g = g.unsqueeze(d)
        return [g.expand(x.shape)]


class Concat(_Block):
    r"""Concatenate tensors along an existing dimension.

    Parameters
    ----------
    name : str
        Node name.
    *inputs : Parameter or Data or torch.Tensor
        Tensors to concatenate; scalars are treated as vectors of length one.
    dim : int, default -1
        Dimension along which to concatenate.
    """

    def __init__(self, name, *inputs, dim=-1):
        if len(inputs) < 1:
            raise ValueError("Concat needs at least one input.")
        self.dim = dim
        super().__init__(name, inputs)

    def _compute(self, values):
        values = [v.reshape(1) if v.ndim == 0 else v for v in values]
        return torch.cat(values, dim=self.dim)

    def _arg_grads(self, g, values, out, needs):
        sizes = [1 if v.ndim == 0 else v.shape[self.dim] for v in values]
        pieces = torch.split(g, sizes, dim=self.dim)
        return [piece.reshape(v.shape) if need else None for piece, v, need in zip(pieces, values, needs)]

    def _batched_pieces(self, values, flags, unbatched_ndims):
        n_chains = next(v.shape[0] for v, f in zip(values, flags) if f)
        rank = max(max(unbatched_ndims), 1)
        pieces = []
        for value, flag, ndim in zip(values, flags, unbatched_ndims):
            if not flag:
                value = value.reshape(1) if value.ndim == 0 else value
                value = value.unsqueeze(0).expand(n_chains, *value.shape)
            elif ndim == 0:
                value = value.reshape(n_chains, 1)
            pieces.append(value)
        return pieces, _negative_dim(self.dim, rank)

    def _compute_batched(self, values, flags, unbatched_ndims):
        pieces, dim = self._batched_pieces(values, flags, unbatched_ndims)
        return torch.cat(pieces, dim=dim)

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        pieces, dim = self._batched_pieces(values, flags, unbatched_ndims)
        split = torch.split(g, [p.shape[dim] for p in pieces], dim=dim)
        return [part.reshape(v.shape) if need and flag else None for part, v, need, flag in zip(split, values, needs, flags)]


class Stack(_Block):
    r"""Stack tensors along a new dimension.

    Parameters
    ----------
    name : str
        Node name.
    *inputs : Parameter or Data or torch.Tensor
        Tensors to stack; they are first broadcast to a common shape.
    dim : int, default 0
        Dimension along which to stack.
    """

    def __init__(self, name, *inputs, dim=0):
        if len(inputs) < 1:
            raise ValueError("Stack needs at least one input.")
        self.dim = dim
        super().__init__(name, inputs)

    def _compute(self, values):
        shape = torch.broadcast_shapes(*(v.shape for v in values))
        return torch.stack([v.expand(shape) for v in values], dim=self.dim)

    def _arg_grads(self, g, values, out, needs):
        return [piece if need else None for piece, need in zip(torch.unbind(g, dim=self.dim), needs)]

    def _compute_batched(self, values, flags, unbatched_ndims):
        shape = torch.broadcast_shapes(*(v.shape for v in values))
        rank = len(shape) - 1
        return torch.stack([v.expand(shape) for v in values], dim=_negative_dim(self.dim, rank + 1))

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        rank = out.ndim - 2
        pieces = torch.unbind(g, dim=_negative_dim(self.dim, rank + 1))
        return [piece if need else None for piece, need in zip(pieces, needs)]


# ============================================================ linear algebra and indexing
class MatMul(_Block):
    r"""Matrix product node :math:`y = A B` with ``torch.matmul`` semantics (vectors and broadcast batch dimensions).

    Parameters
    ----------
    name : str
        Node name.
    a : Parameter or Data or torch.Tensor
        First tensor :math:`A`.
    b : Parameter or Data or torch.Tensor
        Second tensor :math:`B`.
    """

    def __init__(self, name, a, b):
        super().__init__(name, [a, b])

    def _compute(self, values):
        return torch.matmul(values[0], values[1])

    def _arg_grads(self, g, values, out, needs):
        a, b = values
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError(f"MatMul '{self.name}' needs inputs with at least one dimension.")
        a2 = a.unsqueeze(0) if a.ndim == 1 else a
        b2 = b.unsqueeze(-1) if b.ndim == 1 else b
        g2 = g
        if b.ndim == 1:
            g2 = g2.unsqueeze(-1)
        if a.ndim == 1:
            g2 = g2.unsqueeze(-2)
        grad_a = grad_b = None
        if needs[0]:
            grad_a = _sum_to_shape(g2 @ b2.transpose(-2, -1), a2.shape).reshape(a.shape)
        if needs[1]:
            grad_b = _sum_to_shape(a2.transpose(-2, -1) @ g2, b2.shape).reshape(b.shape)
        return [grad_a, grad_b]

    def _arg_event_ndims(self, unbatched_ndims):
        return [min(ndim, 2) for ndim in unbatched_ndims]

    def _promoted(self, values, unbatched_ndims):
        a, b = values
        if unbatched_ndims[0] == 0 or unbatched_ndims[1] == 0:
            raise ValueError(f"MatMul '{self.name}' needs inputs with at least one dimension.")
        a2 = a.unsqueeze(-2) if unbatched_ndims[0] == 1 else a
        b2 = b.unsqueeze(-1) if unbatched_ndims[1] == 1 else b
        return a2, b2

    def _compute_batched(self, values, flags, unbatched_ndims):
        a2, b2 = self._promoted(values, unbatched_ndims)
        out = a2 @ b2
        if unbatched_ndims[1] == 1:
            out = out.squeeze(-1)
        if unbatched_ndims[0] == 1:
            out = out.squeeze(-2)
        return out

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        a2, b2 = self._promoted(values, unbatched_ndims)
        g2 = g
        if unbatched_ndims[1] == 1:
            g2 = g2.unsqueeze(-1)
        if unbatched_ndims[0] == 1:
            g2 = g2.unsqueeze(-2)
        grad_a = grad_b = None
        if needs[0]:
            grad_a = g2 @ b2.transpose(-2, -1)
            if unbatched_ndims[0] == 1:
                grad_a = grad_a.squeeze(-2)
        if needs[1]:
            grad_b = a2.transpose(-2, -1) @ g2
            if unbatched_ndims[1] == 1:
                grad_b = grad_b.squeeze(-1)
        return [grad_a, grad_b]


class Index(_Block):
    r"""Gather elements along a dimension using integer indices.

    Parameters
    ----------
    name : str
        Node name.
    x : Parameter or Data or torch.Tensor
        Source tensor.
    index : Parameter or Data or torch.Tensor
        Integer index tensor, e.g. group labels. A ``Data`` index can be replaced
        with ``Data.set_value`` to predict for other observations or groups; a
        discrete ``RandomParameter`` also works.
    dim : int, default 0
        Dimension along which to index.

    Notes
    -----
    The output has shape ``x.shape[:dim] + index.shape + x.shape[dim + 1:]``,
    e.g. ``theta[group_idx]``. The index receives no gradient.
    """

    _non_differentiable = (1,)

    def __init__(self, name, x, index, dim=0):
        if isinstance(index, (list, tuple)):
            index = torch.as_tensor(index, dtype=torch.long)
        if isinstance(index, torch.Tensor) and index.is_floating_point():
            raise TypeError("A constant index must be an integer tensor.")
        self.dim = dim
        self.index = index
        super().__init__(name, [x, index])

    def _values(self, node_values):
        x_position, index_position = self._positions
        x = node_values[x_position] if x_position is not None else self._args[0]
        index = node_values[index_position] if index_position is not None else self._args[1]
        x = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        index = torch.as_tensor(index, device=x.device).long()
        return [x, index]

    def _compute(self, values):
        x, index = values
        dim = self.dim % x.ndim
        out = torch.index_select(x, dim, index.reshape(-1))
        return out.reshape(*x.shape[:dim], *index.shape, *x.shape[dim + 1:])

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None, None]
        x, index = values
        dim = self.dim % x.ndim
        g = g.reshape(*x.shape[:dim], index.numel(), *x.shape[dim + 1:])
        grad = torch.zeros(x.shape, dtype=g.dtype, device=g.device)
        grad.index_add_(dim, index.reshape(-1), g)
        return [grad, None]

    _align_arguments = False

    def _gather_layout(self, values, flags, unbatched_ndims):
        x, index = values
        n_chains = index.shape[0]
        dim = self.dim % unbatched_ndims[0]
        expanded = x if flags[0] else x.unsqueeze(0).expand(n_chains, *x.shape)
        moved = expanded.movedim(1 + dim, 1)  # (C, n, *pre, *post)
        n_index = index.ndim - 1
        chain = torch.arange(n_chains, device=index.device).reshape(n_chains, *([1] * n_index))
        n_out = 1 + n_index + moved.ndim - 2
        # gathered layout (C, *index, *pre, *post) -> output (C, *pre, *index, *post)
        permutation = [0] + list(range(1 + n_index, 1 + n_index + dim)) + list(range(1, 1 + n_index)) + list(range(1 + n_index + dim, n_out))
        return moved, chain, permutation, dim

    def _compute_batched(self, values, flags, unbatched_ndims):
        x, index = values
        if not flags[1]:
            rank = unbatched_ndims[0]
            dim = (self.dim % rank - rank) % x.ndim
            out = torch.index_select(x, dim, index.reshape(-1))
            return out.reshape(*x.shape[:dim], *index.shape, *x.shape[dim + 1:])
        moved, chain, permutation, _ = self._gather_layout(values, flags, unbatched_ndims)
        return moved[chain, index].permute(permutation)

    def _arg_grads_batched(self, g, values, out, needs, flags, unbatched_ndims):
        if not needs[0] or not flags[0]:
            return [None, None]
        x, index = values
        if not flags[1]:
            rank = unbatched_ndims[0]
            dim = (self.dim % rank - rank) % x.ndim
            g = g.reshape(*x.shape[:dim], index.numel(), *x.shape[dim + 1:])
            grad = torch.zeros(x.shape, dtype=g.dtype, device=g.device)
            grad.index_add_(dim, index.reshape(-1), g)
            return [grad, None]
        moved, chain, permutation, dim = self._gather_layout(values, flags, unbatched_ndims)
        inverse = [0] * len(permutation)
        for position, source in enumerate(permutation):
            inverse[source] = position
        gathered = g.permute(inverse)
        grad = torch.zeros(moved.shape, dtype=g.dtype, device=g.device)
        grad.index_put_((chain.expand_as(index), index), gathered, accumulate=True)
        return [grad.movedim(1, 1 + dim), None]


class CholeskyCovariance(_Block):
    r"""Construct Cholesky factor :math:`\mathbf{L} = \mathrm{diag}(\boldsymbol{\sigma}) \mathbf{L}_{\text{corr}}` of covariance matrix.

    Parameters
    ----------
    name : str
        Node name.
    L_corr : Parameter or torch.Tensor
        Lower-triangular Cholesky factor of correlation matrix, shape ``(..., D, D)``.
    sigma : Parameter or torch.Tensor
        Standard deviations vector of shape ``(..., D)``.
    """

    def __init__(self, name, L_corr, sigma):
        super().__init__(name, [L_corr, sigma])

    def _arg_event_ndims(self, unbatched_ndims):
        return [2, 1]

    def _compute(self, values):
        L, sigma = values
        return sigma.unsqueeze(-1) * L

    def _arg_grads(self, g, values, out, needs):
        L, sigma = values
        grad_L = sigma.unsqueeze(-1) * g if needs[0] else None
        grad_sigma = (g * L).sum(-1) if needs[1] else None
        return [grad_L, grad_sigma]


class CovarianceFromCholesky(_Block):
    r"""Reconstruct covariance matrix :math:`\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top` from Cholesky factor.

    Parameters
    ----------
    name : str
        Node name.
    L : Parameter or torch.Tensor
        Lower-triangular Cholesky factor of shape ``(..., D, D)``.
    """

    def __init__(self, name, L):
        super().__init__(name, [L])

    def _compute(self, values):
        L = values[0]
        return L @ L.transpose(-2, -1)

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None]
        L = values[0]
        return [(g + g.transpose(-2, -1)) @ L]
