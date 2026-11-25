"""Multi-output covariance kernels (intrinsic and linear models of coregionalisation).

The outputs of a multi-output GP are stacked into one long vector: every row of the
input matrix is a ``(x, i)`` pair, the input features ``x`` plus one column holding
the integer index ``i`` of the output the observation belongs to. The kernels here
read that column and are otherwise ordinary :class:`~BayesianDLL.GP.Kernel` objects,
so they work with :class:`~BayesianDLL.GP.ExactGP`, :class:`~BayesianDLL.GP.LatentGP`
(also with inducing points) and the composition operators.
"""

from __future__ import annotations

import torch

from ..Distributions._resolve import resolve
from ._kernels import (
    Kernel,
    _accumulate_derivative,
    _bparam,
    _batch_size,
    _times_matrix,
)


def _take(values, index):
    """``values (B or 1, n, ...)`` gathered at ``index (B or 1, N)`` -> ``(B, N, ...)``."""
    size = max(values.shape[0], index.shape[0])
    values = values.expand(size, *values.shape[1:])
    index = index.expand(size, index.shape[1])
    return values[torch.arange(size, device=index.device).unsqueeze(-1), index]


def _expand(tensor, size):
    return tensor.expand(size, *tensor.shape[1:])


class LinearCoregionalization(Kernel):
    r"""Linear model of coregionalisation (LMC) kernel for multi-output GPs.

    .. math::

        k\big((\mathbf{x}, i), (\mathbf{x}', j)\big) = \sum_{q=1}^{Q} B_q[i, j]\, k_q(\mathbf{x}, \mathbf{x}'),
        \qquad B_q = W_q W_q^\top + \operatorname{diag}(\boldsymbol{\kappa}_q),

    for outputs :math:`i, j \in \{0, \dots, P - 1\}`: :math:`Q` independent latent
    processes with kernels :math:`k_q` are mixed into :math:`P` outputs by the
    positive semi-definite coregionalisation matrices :math:`B_q`
    (:math:`W_q \in \mathbb{R}^{P \times R_q}`, :math:`\kappa_q \geq 0`).
    The inputs stack the outputs: each row holds the features :math:`\mathbf{x}`
    and, in column ``output_dimension``, the integer output index :math:`i`
    (see the example below). That column is not an input of the sub-kernels and
    receives no input gradient.

    Parameters
    ----------
    kernels : sequence of Kernel
        The latent kernels :math:`k_q` (on the input features without the index column).
    weights : sequence of (torch.Tensor or Parameter)
        Mixing matrices :math:`W_q` of shape ``(n_outputs, rank_q)``, one per kernel.
    kappa : sequence of (float or torch.Tensor or Parameter or None), optional
        Non-negative independent variances :math:`\kappa_q` per output (shape
        ``(n_outputs,)``, or a scalar shared by the outputs), one per kernel;
        ``None`` entries mean zero. Default is zero for all (use a positive
        parameter such as an exponential for learning).
    output_dimension : int, default -1
        The input column holding the output index.

    Notes
    -----
    The gradients with respect to :math:`W_q` and :math:`\kappa_q` are exact:
    :math:`\partial B_q[i, j] / \partial W_q[p, s] = \delta_{ip} W_q[j, s] + \delta_{jp} W_q[i, s]`
    and :math:`\partial B_q[i, j] / \partial \kappa_q[p] = \delta_{ip} \delta_{jp}`,
    multiplied elementwise by :math:`k_q`. Rows with an index outside
    ``0 .. n_outputs - 1`` raise a ``ValueError``.

    Examples
    --------
    >>> import torch
    >>> from BayesianDLL.GP import LinearCoregionalization, RBF, Matern32
    >>> kernel = LinearCoregionalization([RBF(1.0), Matern32(0.5)], [torch.ones(2, 1), torch.eye(2)])
    >>> x = torch.linspace(0, 1, 5)
    >>> inputs = torch.cat([torch.stack([x, torch.zeros(5)], 1), torch.stack([x, torch.ones(5)], 1)])
    >>> kernel(inputs).shape
    torch.Size([10, 10])
    """

    def __init__(self, kernels, weights, kappa=None, output_dimension=-1):
        kernels, weights = tuple(kernels), tuple(weights)
        if not kernels or len(kernels) != len(weights):
            raise ValueError("LinearCoregionalization needs one weight matrix per kernel (and at least one kernel).")
        for kernel in kernels:
            if not isinstance(kernel, Kernel):
                raise TypeError("LinearCoregionalization kernels must be instances of BayesianDLL.GP.Kernel.")
        kappa = (None,) * len(kernels) if kappa is None else tuple(kappa)
        if len(kappa) != len(kernels):
            raise ValueError("LinearCoregionalization needs one kappa entry (or None) per kernel.")
        if isinstance(output_dimension, bool) or not isinstance(output_dimension, int):
            raise TypeError("LinearCoregionalization output_dimension must be an integer.")
        shapes = [tuple(torch.as_tensor(resolve(w)).shape) for w in weights]
        if any(len(shape) != 2 for shape in shapes) or len({shape[0] for shape in shapes}) != 1:
            raise ValueError(
                f"LinearCoregionalization weights must be matrices with the same number of rows (outputs); got shapes {shapes}."
            )
        self.n_outputs = shapes[0][0]
        for entry in kappa:
            if entry is not None and tuple(torch.as_tensor(resolve(entry)).shape) not in ((), (self.n_outputs,)):
                raise ValueError(f"LinearCoregionalization kappa must be a scalar or have shape ({self.n_outputs},).")
        self.kernels = kernels
        self.weights = weights
        self.kappa = kappa
        self.output_dimension = output_dimension
        super().__init__(
            *(p for kernel in kernels for p in kernel.parameters), *weights, *(k for k in kappa if k is not None)
        )

    # -- helpers ------------------------------------------------------------
    def _split(self, x):
        """``(index (B, N) long, features (B, N, D - 1))`` of stacked inputs ``(B, N, D)``."""
        dimension = self.output_dimension
        if x.shape[-1] < 2 or not -x.shape[-1] <= dimension < x.shape[-1]:
            raise ValueError(
                "LinearCoregionalization inputs need the features and one output-index column "
                f"(output_dimension={dimension}); got {x.shape[-1]} columns."
            )
        dimension %= x.shape[-1]
        index = x[..., dimension].round().long()
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= self.n_outputs):
            raise ValueError(f"LinearCoregionalization output indices must lie in 0..{self.n_outputs - 1}.")
        return index, torch.cat([x[..., :dimension], x[..., dimension + 1:]], dim=-1), dimension

    def _mixing(self, q, parameter_values, reference):
        """``(W (B, P, R), κ (B, P), kappa value or None, B_q (B, P, P))``."""
        weight = _bparam(self.weights[q], parameter_values, reference)
        if weight.ndim != 3:
            raise ValueError("LinearCoregionalization weights must be matrices.")
        kappa_parameter = self.kappa[q]
        kappa_value = None if kappa_parameter is None else _bparam(kappa_parameter, parameter_values, reference)
        matrix = weight @ weight.transpose(-2, -1)
        kappa = None
        if kappa_value is not None:
            kappa = kappa_value.expand(kappa_value.shape[0], self.n_outputs)
            matrix = matrix + torch.diag_embed(kappa)
        return weight, kappa, kappa_value, matrix

    @staticmethod
    def _entries(matrix, rows, columns):
        """``B[i_a, j_b]`` ``(B, N, M)`` from matrices ``(B, P, P)`` and indices ``(B, N)``, ``(B, M)``."""
        size = max(matrix.shape[0], rows.shape[0], columns.shape[0])
        flat = rows.expand(size, -1).unsqueeze(-1) * matrix.shape[-1] + columns.expand(size, -1).unsqueeze(-2)
        values = _expand(matrix, size).reshape(size, -1)
        return torch.gather(values, 1, flat.reshape(size, -1)).reshape(flat.shape)

    def coregionalization_matrices(self, parameter_values=None):
        """The matrices :math:`B_q = W_q W_q^\\top + \\operatorname{diag}(\\kappa_q)`, a list of ``(n_outputs, n_outputs)`` tensors."""
        values = self._single_values(parameter_values)
        reference = torch.zeros(1, 1, 1, dtype=torch.get_default_dtype())
        return [self._mixing(q, values, reference)[3][0] for q in range(len(self.kernels))]

    # -- batched kernel API ----------------------------------------------------
    def batch_evaluate(self, x1, x2, parameter_values):
        rows, features1, _ = self._split(x1)
        columns, features2, _ = self._split(x2)
        result = 0
        for q, kernel in enumerate(self.kernels):
            matrix = self._mixing(q, parameter_values, x1)[3]
            result = result + self._entries(matrix, rows, columns) * kernel.batch_evaluate(features1, features2, parameter_values)
        return result

    def batch_derivatives(self, x1, x2, parameter_values):
        rows, features1, _ = self._split(x1)
        columns, features2, _ = self._split(x2)
        derivatives = {}
        for q, kernel in enumerate(self.kernels):
            weight, _, kappa_value, matrix = self._mixing(q, parameter_values, x1)
            base = kernel.batch_evaluate(features1, features2, parameter_values)
            for name, derivative in kernel.batch_derivatives(features1, features2, parameter_values).items():
                _accumulate_derivative(derivatives, name, _times_matrix(derivative, self._entries(matrix, rows, columns)))
            size = max(base.shape[0], weight.shape[0], rows.shape[0], columns.shape[0])
            base, weight = _expand(base, size), _expand(weight, size)
            hot_rows = torch.nn.functional.one_hot(_expand(rows, size), self.n_outputs).to(base.dtype)
            hot_columns = torch.nn.functional.one_hot(_expand(columns, size), self.n_outputs).to(base.dtype)
            if _is_parameter(self.weights[q]):
                at_columns, at_rows = _take(weight, columns), _take(weight, rows)  # W[j_b, :], W[i_a, :]
                derivative = (
                    torch.einsum("bap,bms,bam->bpsam", hot_rows, at_columns, base)
                    + torch.einsum("bmp,bas,bam->bpsam", hot_columns, at_rows, base)
                )
                _accumulate_derivative(derivatives, self.weights[q].name, derivative)
            if _is_parameter(self.kappa[q]):
                derivative = torch.einsum("bap,bmp,bam->bpam", hot_rows, hot_columns, base)
                if kappa_value.shape[1] == 1:
                    derivative = derivative.sum(dim=1, keepdim=True)
                _accumulate_derivative(derivatives, self.kappa[q].name, derivative)
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """The sub-kernels' input VJPs with the cotangent ``G ⊙ B_q[i_a, j_b]``; the index column gets zero."""
        rows, features1, dimension = self._split(x1)
        columns, features2, _ = self._split(x2)
        result = 0
        for q, kernel in enumerate(self.kernels):
            matrix = self._mixing(q, parameter_values, x1)[3]
            weighted = cotangent * self._entries(matrix, rows, columns)
            result = result + kernel.batch_input_vjp(features1, features2, weighted, parameter_values)
        return self._embed(result, x1, dimension, cotangent)

    @staticmethod
    def _embed(gradient, x, dimension, cotangent):
        """Insert the zero gradient of the index column into ``gradient (B, N, D - 1)``."""
        size = _batch_size(x, cotangent)
        gradient = _expand(gradient, size) if torch.is_tensor(gradient) else torch.zeros(size, x.shape[1], x.shape[-1] - 1, dtype=x.dtype)
        zero = torch.zeros(*gradient.shape[:-1], 1, dtype=gradient.dtype, device=gradient.device)
        return torch.cat([gradient[..., :dimension], zero, gradient[..., dimension:]], dim=-1)

    def batch_diag(self, x, parameter_values, chunk_size=256):
        rows, features, _ = self._split(x)
        result = 0
        for q, kernel in enumerate(self.kernels):
            matrix = self._mixing(q, parameter_values, x)[3]
            result = result + self._diagonal_entries(matrix, rows) * kernel.batch_diag(features, parameter_values, chunk_size)
        return result

    @staticmethod
    def _diagonal_entries(matrix, rows):
        """``B[i_a, i_a]`` ``(B, N)``."""
        return _take(torch.diagonal(matrix, dim1=-2, dim2=-1), rows)

    def batch_diag_derivatives(self, x, parameter_values, chunk_size=256):
        rows, features, _ = self._split(x)
        derivatives = {}
        for q, kernel in enumerate(self.kernels):
            weight, _, kappa_value, matrix = self._mixing(q, parameter_values, x)
            base = kernel.batch_diag(features, parameter_values, chunk_size)
            diagonal = self._diagonal_entries(matrix, rows)
            for name, derivative in kernel.batch_diag_derivatives(features, parameter_values, chunk_size).items():
                _accumulate_derivative(
                    derivatives, name, derivative * diagonal.reshape(diagonal.shape[0], *([1] * (derivative.ndim - 2)), -1)
                )
            size = max(base.shape[0], weight.shape[0], rows.shape[0])
            base, weight = _expand(base, size), _expand(weight, size)
            hot = torch.nn.functional.one_hot(_expand(rows, size), self.n_outputs).to(base.dtype)
            if _is_parameter(self.weights[q]):
                # d B[i, i] / d W[p, s] = 2 δ_ip W[i, s]
                derivative = 2.0 * torch.einsum("bap,bas,ba->bpsa", hot, _take(weight, rows), base)
                _accumulate_derivative(derivatives, self.weights[q].name, derivative)
            if _is_parameter(self.kappa[q]):
                derivative = torch.einsum("bap,ba->bpa", hot, base)
                if kappa_value.shape[1] == 1:
                    derivative = derivative.sum(dim=1, keepdim=True)
                _accumulate_derivative(derivatives, self.kappa[q].name, derivative)
        return derivatives

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        rows, features, dimension = self._split(x)
        result = 0
        for q, kernel in enumerate(self.kernels):
            matrix = self._mixing(q, parameter_values, x)[3]
            weighted = cotangent * self._diagonal_entries(matrix, rows)
            result = result + kernel.batch_diag_input_vjp(features, weighted, parameter_values, chunk_size)
        return self._embed(result, x, dimension, cotangent)


def _is_parameter(value):
    return isinstance(getattr(value, "name", None), str)


class IntrinsicCoregionalization(LinearCoregionalization):
    r"""Intrinsic coregionalisation model (ICM) kernel: one latent kernel shared by all outputs.

    .. math::

        k\big((\mathbf{x}, i), (\mathbf{x}', j)\big) = B[i, j]\, k(\mathbf{x}, \mathbf{x}'),
        \qquad B = W W^\top + \operatorname{diag}(\boldsymbol{\kappa}),

    the :class:`LinearCoregionalization` with a single term: the outputs share one
    input-correlation structure and differ by the coregionalisation matrix
    :math:`B`. See :class:`LinearCoregionalization` for the input layout.

    Parameters
    ----------
    kernel : Kernel
        The shared kernel :math:`k` (on the input features without the index column).
    weights : torch.Tensor or Parameter
        Matrix :math:`W` of shape ``(n_outputs, rank)``.
    kappa : float or torch.Tensor or Parameter, optional
        Non-negative variances :math:`\kappa` per output (shape ``(n_outputs,)`` or a scalar). Default zero.
    output_dimension : int, default -1
        The input column holding the output index.
    """

    def __init__(self, kernel, weights, kappa=None, output_dimension=-1):
        super().__init__([kernel], [weights], None if kappa is None else [kappa], output_dimension=output_dimension)
