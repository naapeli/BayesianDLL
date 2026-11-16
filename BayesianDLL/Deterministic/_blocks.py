"""Differentiable building blocks for model graphs.

Every block is a :class:`~BayesianDLL.DeterministicParameter` with a
hand-written vector-Jacobian product (``vjp_func``), so the model's reverse
pass never builds a dense Jacobian. Inputs may be model nodes
(``RandomParameter``, ``DeterministicParameter``, ``Data``), constant tensors or
Python numbers; numbers adopt the floating dtype and device of the tensor
inputs. Elementwise blocks broadcast their inputs like ``torch`` does, and the
gradient of a broadcast input is summed back onto its shape.
"""

from __future__ import annotations

from numbers import Number

import torch
import torch.nn.functional as F

from .._data import Data
from .._parameters import DeterministicParameter, ObservedParameter, RandomParameter


_NODE_TYPES = (RandomParameter, DeterministicParameter, ObservedParameter, Data)


def _is_node(value):
    return isinstance(value, _NODE_TYPES)


def _sum_to_shape(grad, shape):
    from .._model import _sum_to_shape as sum_to_shape
    return sum_to_shape(grad, tuple(shape))


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
            if not (_is_node(arg) or isinstance(arg, torch.Tensor) or (isinstance(arg, Number) and not isinstance(arg, bool))):
                raise TypeError(
                    f"Argument {i} of '{name}' must be a model node, a tensor or a number; got {type(arg).__name__}."
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
        derivative = self._derivative_from_nodes if type(self)._local is not _Block._local else None
        super().__init__(name, self._forward_from_nodes, derivative, nodes, vjp_func=self._vjp_from_nodes)

    # ------------------------------------------------------------ plumbing
    def _values(self, node_values):
        raw = [node_values[p] if p is not None else arg for p, arg in zip(self._positions, self._args)]
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
    """Elementwise sum ``inputs[0] + inputs[1] + ...`` with broadcasting.

    Example: ``Add("mu", alpha, beta_x)``.
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
    """Elementwise difference ``a - b`` with broadcasting."""

    def __init__(self, name, a, b):
        super().__init__(name, [a, b])

    def _compute(self, values):
        return values[0] - values[1]

    def _local(self, values, out, needs):
        return [torch.ones_like(out) if needs[0] else None, -torch.ones_like(out) if needs[1] else None]


class Multiply(_ElementwiseBlock):
    """Elementwise product ``inputs[0] * inputs[1] * ...`` with broadcasting."""

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
    """Elementwise quotient ``a / b`` with broadcasting."""

    def __init__(self, name, a, b):
        super().__init__(name, [a, b])

    def _compute(self, values):
        return values[0] / values[1]

    def _local(self, values, out, needs):
        a, b = values
        return [1 / b if needs[0] else None, -out / b if needs[1] else None]


class Affine(_ElementwiseBlock):
    """Location-scale transform ``loc + scale * x`` (elementwise, broadcasting).

    This is the building block of the **non-centered parameterisation** of
    hierarchical models. A centered model samples group effects directly,
    ``theta ~ Normal(mu, tau)``; when the data are weak, the posterior of
    ``(theta, tau)`` forms a funnel (small ``tau`` forces every ``theta`` close
    to ``mu``) that gradient-based samplers such as NUTS cannot explore,
    which shows up as divergent transitions. The non-centered model samples
    standardised effects ``z ~ Normal(0, 1)`` and sets
    ``theta = Affine("theta", z, loc=mu, scale=tau)``. The prior geometry of
    ``z`` no longer depends on ``tau``, which removes the funnel.

    ``loc`` and ``scale`` may be model nodes, tensors or numbers.
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
    """Elementwise ``log(1 + exp(x))``, a smooth map to the positive reals."""

    def __init__(self, name, x):
        super().__init__(name, [x])

    def _compute(self, values):
        return F.softplus(values[0])

    def _local(self, values, out, needs):
        return [torch.sigmoid(values[0]) if needs[0] else None]


class Tanh(_ElementwiseBlock):
    """Elementwise hyperbolic tangent."""

    def __init__(self, name, x):
        super().__init__(name, [x])

    def _compute(self, values):
        return torch.tanh(values[0])

    def _local(self, values, out, needs):
        return [1 - out ** 2 if needs[0] else None]


class Softmax(_Block):
    """Softmax along ``dim``: maps real vectors to the probability simplex.

    The VJP is ``s * (g - sum(g * s, dim))`` with ``s`` the output. For a
    categorical likelihood prefer ``Categorical(logits=...)``, which is
    numerically more stable; ``Softmax`` is for simplex-valued quantities
    used elsewhere in a model.
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


# ============================================================ reductions and shapes
class Sum(_Block):
    """Sum of ``x`` over ``dim`` (all elements when ``dim`` is ``None``)."""

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


class Concat(_Block):
    """Concatenate inputs along ``dim`` (``torch.cat``)."""

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


class Stack(_Block):
    """Stack equally shaped inputs along a new dimension ``dim`` (``torch.stack``)."""

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


# ============================================================ linear algebra and indexing
class MatMul(_Block):
    """Matrix product ``a @ b`` with ``torch.matmul`` semantics.

    Vectors and batch dimensions are supported (batch dimensions broadcast).
    The VJP is ``grad_a = g @ b^T`` and ``grad_b = a^T @ g``.
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


class Index(_Block):
    """Gather ``x`` along ``dim`` with an integer index, e.g. ``theta[group_idx]``.

    ``index`` is an integer tensor (or list), or a model node holding integer
    values: a :class:`~BayesianDLL.Data` object -- which can be replaced with
    ``Data.set_value`` to predict for other observations or groups -- or a
    discrete ``RandomParameter``. The output has shape
    ``x.shape[:dim] + index.shape + x.shape[dim + 1:]``. The index receives no
    gradient; the VJP w.r.t. ``x`` is a scatter-add (``index_add_``) of the
    cotangent.
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


class CholeskyCovariance(_Block):
    """Cholesky factor ``diag(sigma) @ L_corr`` of a covariance matrix.

    Combines the Cholesky factor of a correlation matrix (e.g. from
    ``LKJCholesky``) with standard deviations ``sigma`` (last dimension ``D``)
    into the lower-triangular factor of ``Sigma = diag(sigma) R diag(sigma)``,
    ready for ``MultivariateNormal(mu, scale_tril=...)``.
    """

    def __init__(self, name, L_corr, sigma):
        super().__init__(name, [L_corr, sigma])

    def _compute(self, values):
        L, sigma = values
        return sigma.unsqueeze(-1) * L

    def _arg_grads(self, g, values, out, needs):
        L, sigma = values
        grad_L = sigma.unsqueeze(-1) * g if needs[0] else None
        grad_sigma = (g * L).sum(-1) if needs[1] else None
        return [grad_L, grad_sigma]


class CovarianceFromCholesky(_Block):
    """Covariance ``L @ L^T`` from a (lower-triangular) Cholesky factor ``L``.

    ``MultivariateNormal`` also accepts ``scale_tril=L`` directly, which is
    cheaper; use this block when the covariance matrix itself is needed.
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
