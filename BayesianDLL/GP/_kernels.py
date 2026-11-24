"""Composable covariance kernels with explicit analytic derivatives.

Batched form
------------
Every built-in kernel is implemented once, in a *batched form* that the GP
nodes use for vectorized MCMC chains (see :mod:`BayesianDLL._batching`):

* inputs have shape ``(B, N, D)`` (a 2-D ``(N, D)`` or 1-D ``(N,)`` input is
  shared by the whole batch and treated as ``B = 1``);
* ``parameter_values`` maps the name of every named kernel parameter to a
  tensor with a leading batch dimension, ``(B, *parameter_shape)``; unnamed
  constants are shared;
* leading dimensions of size ``1`` broadcast against ``B``.

The batched methods are :meth:`Kernel.batch_evaluate` (``(B, N, M)``),
:meth:`Kernel.batch_derivatives` (``(B, *parameter_shape, N, M)``),
:meth:`Kernel.batch_diag` (``(B, N)``), :meth:`Kernel.batch_diag_derivatives`
(``(B, *parameter_shape, N)``), :meth:`Kernel.batch_input_vjp`
(``(B, N, D)``) and :meth:`Kernel.batch_diag_input_vjp` (``(B, N, D)``, the
input gradient of ``g · diag K(X, X)``). The ordinary methods
(``kernel(x1, x2)``, ``derivatives``, ``diag``, ``diag_derivatives``,
``input_vjp``, ``diag_input_vjp``) are the ``B = 1`` case and keep their
unbatched shapes.

A user kernel that only overrides the unbatched ``evaluate`` (and
``derivatives`` / ``input_vjp``) still works in batched form: the default
batched methods loop over the batch.
"""

from __future__ import annotations

import math
from abc import ABC

import torch

from ..Distributions._resolve import resolve


def _is_named(value):
    """Whether ``value`` is a named model node (``torch.Tensor.name`` is ``None``)."""
    return isinstance(getattr(value, "name", None), str)


def _parameter_value(parameter, parameter_values=None):
    if parameter_values is not None and _is_named(parameter) and parameter.name in parameter_values:
        return parameter_values[parameter.name]
    return resolve(parameter)


def _feature_matrix(x):
    x = torch.as_tensor(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _batch_inputs(x):
    """``(B, N, D)`` from a batched 3-D input or a shared ``(N, D)`` / ``(N,)`` input."""
    x = torch.as_tensor(x)
    if x.ndim == 3:
        return x
    return _feature_matrix(x).unsqueeze(0)


def _unique_named(values):
    result = []
    seen = set()
    for value in values:
        if not _is_named(value) or value.name in seen:
            continue
        seen.add(value.name)
        result.append(value)
    return tuple(result)


def _bparam(parameter, parameter_values, reference):
    """Batched value ``(B, *shape)`` of a kernel parameter (``B = 1`` for constants)."""
    if _is_named(parameter):
        value = parameter_values[parameter.name]
    else:
        value = torch.as_tensor(resolve(parameter)).unsqueeze(0)
    value = torch.as_tensor(value).to(dtype=reference.dtype, device=reference.device)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    return value


def _bscalar(parameter, parameter_values, reference, name, kernel_name):
    """Batched scalar parameter as shape ``(B,)``."""
    value = _bparam(parameter, parameter_values, reference)
    flat = value.reshape(value.shape[0], -1)
    if flat.shape[1] != 1:
        raise ValueError(f"{kernel_name} {name} must be a scalar.")
    return flat[:, 0]


def _blengthscale(parameter, parameter_values, x, kernel_name):
    """Batched length-scale: the parameter value ``(B, *shape)`` and ``(B, 1 or D)``."""
    value = _bparam(parameter, parameter_values, x)
    flat = value.reshape(value.shape[0], -1)
    if flat.shape[1] not in (1, x.shape[-1]):
        raise ValueError(
            f"{kernel_name} lengthscale must be scalar or have one value per feature; "
            f"got {flat.shape[1]} values for {x.shape[-1]} features."
        )
    return value, flat


def _scalar_derivative(matrix):
    """``(B, N, M)`` -> ``(B, 1, N, M)``: the derivative w.r.t. a scalar parameter."""
    return matrix.unsqueeze(1)


def _lengthscale_derivative(value, per_feature):
    """Shape a per-feature derivative ``(B, N, M, D)`` like the parameter.

    A scalar length-scale shared by all features gets the sum over the
    features, ``(B, 1, N, M)``; an ARD length-scale gets
    ``(B, *parameter_shape, N, M)``, e.g. ``(B, D, N, M)``.
    """
    if value[0].numel() == 1:
        return per_feature.sum(dim=-1).unsqueeze(1)
    per_feature = per_feature.movedim(-1, 1)
    return per_feature.reshape(per_feature.shape[0], *value.shape[1:], *per_feature.shape[-2:])


def _view(value, ndim):
    """Reshape a ``(B,)`` or ``(B, F)`` batched value to broadcast against ``ndim``-D tensors."""
    if value.ndim == 1:
        return value.reshape(value.shape[0], *([1] * (ndim - 1)))
    return value.reshape(value.shape[0], *([1] * (ndim - 2)), value.shape[-1])


def _radial_input_vjp(weights, x1, x2, scale=None):
    r"""``Σ_j W_ij (x1_i - x2_j)`` for batched ``W`` ``(B, N, M)`` without an ``N x M x D`` tensor.

    Returns ``(rowsum(W) x1 - W x2) * scale`` with an optional per-feature
    ``scale`` of shape ``(B, 1 or D)``.
    """
    result = weights.sum(dim=-1, keepdim=True) * x1 - weights @ x2
    if scale is not None:
        result = result * scale.unsqueeze(-2)
    return result


def _differences(x1, x2):
    """``(B, N, M, D)`` pairwise differences of batched inputs."""
    return x1.unsqueeze(-2) - x2.unsqueeze(-3)


def _times_matrix(derivative, matrix):
    """Multiply a derivative ``(B, *P, N, M)`` by a matrix ``(B, N, M)`` (broadcasting ``B``)."""
    return derivative * matrix.reshape(matrix.shape[0], *([1] * (derivative.ndim - 3)), *matrix.shape[-2:])


def _zero_input_vjp(x, cotangent):
    """Zero input gradient ``(B, N, D)`` (for kernels whose value does not depend on the inputs)."""
    return torch.zeros((_batch_size(x, cotangent), *x.shape[1:]), dtype=x.dtype, device=x.device)


def _slice_batch(value, index):
    return value[index] if value.shape[0] > 1 else value[0]


def _batch_size(*tensors):
    size = 1
    for tensor in tensors:
        if tensor.shape[0] != 1:
            size = tensor.shape[0]
    return size


_QUADRATURE_NODES = 96
_QUADRATURE_TAIL = 40.0


def _log_cosh(x):
    x = x.abs()
    return x + torch.log1p(torch.exp(-2.0 * x)) - math.log(2.0)


def _power_bessel_k(power, order, argument, nodes=_QUADRATURE_NODES):
    r"""Evaluate ``a^power * K_order(a)`` for ``a > 0`` in pure torch.

    Uses ``K_ν(a) = ∫_0^∞ exp(-a cosh t) cosh(ν t) dt`` (with
    ``K_{-ν} = K_ν``) and the trapezoidal rule, which converges exponentially
    for this analytic, rapidly decaying integrand. The upper limit ``T``
    adapts to ``a`` so that ``a (cosh T - 1) - ν T >= 40``, i.e. the
    truncated tail is below ``exp(-40)`` relative to the integral. The
    integrand is evaluated in log space as
    ``power log a - a - a (cosh t - 1) + log cosh(ν t)``, so neither tiny nor
    large arguments overflow. Entries with ``a <= 0`` return an arbitrary
    finite value; callers handle the limit ``a -> 0`` themselves.

    Measured accuracy in float64 against ``scipy.special.kv`` (with the
    default 96 nodes, orders ``ν`` and ``ν - 1`` for ``ν`` in 0.3–3.7): the
    largest relative error is ``6e-14`` for ``a`` in ``[1e-6, 50]`` and
    ``1.2e-13`` for ``a`` in ``[1e-8, 630]``.
    """
    order = abs(float(order))
    a = torch.where(argument > 0, argument, torch.ones_like(argument))
    # Fixed-point iterations for T = arccosh(1 + (40 + ν T) / a).
    upper = torch.acosh(1.0 + _QUADRATURE_TAIL / a)
    for _ in range(4):
        upper = torch.acosh(1.0 + (_QUADRATURE_TAIL + order * upper) / a)
    fractions = torch.linspace(0.0, 1.0, nodes, dtype=a.dtype, device=a.device)
    t = upper.unsqueeze(-1) * fractions  # (..., nodes)
    step = upper / (nodes - 1)
    a_column = a.unsqueeze(-1)
    log_integrand = (
        power * torch.log(a_column)
        - a_column
        - a_column * (torch.cosh(t) - 1.0)
        + _log_cosh(order * t)
    )
    weights = torch.ones(nodes, dtype=a.dtype, device=a.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    return step * (torch.exp(log_integrand) * weights).sum(dim=-1)


def _half_integer_order(nu):
    """Return ``p`` when ``nu = p + 1/2`` for an integer ``p >= 0``, else ``None``."""
    p = nu - 0.5
    if p >= 0 and abs(p - round(p)) < 1e-12:
        return int(round(p))
    return None


def _half_integer_matern(p, u):
    r"""Return ``f(u)`` and ``f'(u)`` for the half-integer Matérn correlation.

    ``f(u) = exp(-u) p!/(2p)! Σ_{i=0}^{p} (p+i)!/(i!(p-i)!) (2u)^{p-i}`` with
    ``u = sqrt(2ν) r / l`` and ``ν = p + 1/2``; ``f'(u) = exp(-u)(P'(u) - P(u))``
    for the polynomial ``P`` in front of the exponential.
    """
    leading = math.factorial(p) / math.factorial(2 * p)
    # coefficients[d] multiplies (2u)^d; P and P' are evaluated with Horner's rule.
    coefficients = [0.0] * (p + 1)
    for i in range(p + 1):
        coefficients[p - i] = leading * math.factorial(p + i) / (math.factorial(i) * math.factorial(p - i))
    exponential = torch.exp(-u)
    if p == 0:
        return exponential, -exponential
    two_u = 2.0 * u
    polynomial = torch.full_like(u, coefficients[p])
    for degree in range(p - 1, -1, -1):
        polynomial.mul_(two_u).add_(coefficients[degree])
    # P'(u) = Σ_d 2 d c_d (2u)^(d-1).
    derivative = torch.full_like(u, 2.0 * p * coefficients[p])
    for degree in range(p - 1, 0, -1):
        derivative.mul_(two_u).add_(2.0 * degree * coefficients[degree])
    value = polynomial.mul_(exponential)
    slope = derivative.mul_(exponential).sub_(value)
    return value, slope


class Kernel(ABC):
    r"""Base class for covariance kernels :math:`k(\mathbf{x}_1, \mathbf{x}_2)`.
    """

    def __init__(self, *parameters):
        self._parameters = _unique_named(parameters)

    @property
    def parameters(self):
        """The named model nodes among the kernel's hyperparameters (each once)."""
        return self._parameters

    # ------------------------------------------------------------------
    # Unbatched API (the B = 1 case of the batched form)
    # ------------------------------------------------------------------
    def _single_values(self, parameter_values):
        """Every named parameter's value with a leading batch dimension of one."""
        values = {}
        for parameter in self.parameters:
            value = _parameter_value(parameter, parameter_values)
            values[parameter.name] = torch.as_tensor(value).unsqueeze(0)
        return values

    def evaluate(self, x1, x2, parameter_values=None):
        """Covariance matrix ``(N, M)`` of inputs ``x1`` ``(N, D)`` / ``(N,)`` and ``x2`` ``(M, D)`` / ``(M,)``.

        ``parameter_values`` optionally maps parameter names to values that
        replace the nodes' current values (e.g. a posterior draw).
        """
        if type(self).batch_evaluate is Kernel.batch_evaluate:
            raise NotImplementedError(f"{type(self).__name__} must implement evaluate or batch_evaluate.")
        return self.batch_evaluate(_batch_inputs(x1), _batch_inputs(x2), self._single_values(parameter_values))[0]

    def derivatives(self, x1, x2, parameter_values=None):
        """Return derivatives keyed by model-parameter name.

        Each value has parameter axes first and covariance axes last. Thus a
        scalar parameter has derivative shape ``(1, N, M)``.
        """
        if type(self).batch_derivatives is Kernel.batch_derivatives:
            return {}
        derivatives = self.batch_derivatives(_batch_inputs(x1), _batch_inputs(x2), self._single_values(parameter_values))
        return {name: value[0] for name, value in derivatives.items()}

    def diag(self, x, parameter_values=None, chunk_size=256):
        """Return ``diag(k(x, x))`` without forming the full ``N x N`` matrix."""
        return self.batch_diag(_batch_inputs(x), self._single_values(parameter_values), chunk_size=chunk_size)[0]

    def diag_derivatives(self, x, parameter_values=None, chunk_size=256):
        """Derivatives of :meth:`diag`; each value has shape ``parameter_shape + (N,)``."""
        derivatives = self.batch_diag_derivatives(_batch_inputs(x), self._single_values(parameter_values), chunk_size=chunk_size)
        return {name: value[0] for name, value in derivatives.items()}

    def input_vjp(self, x1, x2, cotangent, parameter_values=None):
        r"""Vector-Jacobian product w.r.t. the first input locations.

        Returns ``∂⟨G, K(x1, x2)⟩ / ∂x1`` with shape ``(N, D)`` for the
        cotangent ``G`` of shape ``(N, M)``, i.e.
        ``out[i, d] = Σ_j G_ij ∂k(x1_i, x2_j)/∂x1_id``. Each row of ``x1``
        only affects one row of ``K``, so no ``N x M x N x D`` Jacobian is
        formed. The gradient of ``⟨G, K(Z, Z)⟩`` w.r.t. ``Z`` (both
        arguments) is ``input_vjp(Z, Z, G + Gᵀ)`` because kernels are
        symmetric, ``k(a, b) = k(b, a)``.
        """
        if type(self).batch_input_vjp is Kernel.batch_input_vjp:
            raise NotImplementedError(f"{type(self).__name__} does not implement input_vjp.")
        return self.batch_input_vjp(
            _batch_inputs(x1), _batch_inputs(x2), torch.as_tensor(cotangent).unsqueeze(0), self._single_values(parameter_values)
        )[0]

    def input_vjp_symmetric(self, x, cotangent, parameter_values=None):
        """``∂⟨G, K(x, x)⟩ / ∂x`` with both arguments moving (shape ``(N, D)``)."""
        return self.input_vjp(x, x, cotangent + cotangent.transpose(-2, -1), parameter_values=parameter_values)

    def diag_input_vjp(self, x, cotangent, parameter_values=None):
        r"""``∂(g · diag K(x, x)) / ∂x``: ``out[i] = g_i d k(x_i, x_i) / d x_i`` (shape ``(N, D)``).

        ``cotangent`` ``g`` has shape ``(N,)``. Stationary kernels have a
        constant diagonal and give zero; see :meth:`batch_diag_input_vjp`.
        """
        return self.batch_diag_input_vjp(
            _batch_inputs(x), torch.as_tensor(cotangent).unsqueeze(0), self._single_values(parameter_values)
        )[0]

    def __call__(self, x1, x2=None, parameter_values=None):
        """``kernel(x1, x2)`` is :meth:`evaluate`; ``x2`` defaults to ``x1``."""
        if x2 is None:
            x2 = x1
        return self.evaluate(x1, x2, parameter_values=parameter_values)

    # ------------------------------------------------------------------
    # Batched API; the defaults loop over the batch (for user kernels)
    # ------------------------------------------------------------------
    def _loop(self, function, x1, x2, parameter_values, *extra):
        size = _batch_size(x1, x2, *extra, *(v for v in parameter_values.values()))
        results = []
        for index in range(size):
            values = {name: _slice_batch(v, index) for name, v in parameter_values.items()}
            results.append(function(_slice_batch(x1, index), _slice_batch(x2, index), values, *(_slice_batch(e, index) for e in extra)))
        return results

    def batch_evaluate(self, x1, x2, parameter_values):
        """``(B, N, M)`` covariance of batched inputs ``(B, N, D)``, ``(B, M, D)``."""
        if type(self).evaluate is Kernel.evaluate:
            raise NotImplementedError(f"{type(self).__name__} must implement evaluate or batch_evaluate.")
        return torch.stack(self._loop(lambda a, b, v: self.evaluate(a, b, parameter_values=v), x1, x2, parameter_values))

    def batch_derivatives(self, x1, x2, parameter_values):
        """``{name: (B, *parameter_shape, N, M)}``."""
        if type(self).derivatives is Kernel.derivatives:
            return {}
        results = self._loop(lambda a, b, v: self.derivatives(a, b, parameter_values=v), x1, x2, parameter_values)
        return {name: torch.stack([r[name] for r in results]) for name in results[0]}

    def batch_diag(self, x, parameter_values, chunk_size=256):
        """``(B, N)`` diagonal from square blocks of at most ``chunk_size`` points."""
        if type(self).diag is not Kernel.diag:
            return torch.stack(self._loop(lambda a, b, v: self.diag(a, parameter_values=v), x, x, parameter_values))
        blocks = []
        for start in range(0, x.shape[1], chunk_size):
            block = x[:, start:start + chunk_size]
            blocks.append(torch.diagonal(self.batch_evaluate(block, block, parameter_values), dim1=-2, dim2=-1))
        return torch.cat(blocks, dim=-1)

    def batch_diag_derivatives(self, x, parameter_values, chunk_size=256):
        """``{name: (B, *parameter_shape, N)}``."""
        if type(self).diag_derivatives is not Kernel.diag_derivatives:
            results = self._loop(lambda a, b, v: self.diag_derivatives(a, parameter_values=v), x, x, parameter_values)
            return {name: torch.stack([r[name] for r in results]) for name in results[0]}
        pieces = {}
        for start in range(0, x.shape[1], chunk_size):
            block = x[:, start:start + chunk_size]
            for name, derivative in self.batch_derivatives(block, block, parameter_values).items():
                pieces.setdefault(name, []).append(torch.diagonal(derivative, dim1=-2, dim2=-1))
        return {name: torch.cat(values, dim=-1) for name, values in pieces.items()}

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """``(B, N, D)``: ``∂⟨G_b, K_b(x1_b, x2_b)⟩ / ∂x1_b`` for every batch element."""
        if type(self).input_vjp is Kernel.input_vjp:
            raise NotImplementedError(f"{type(self).__name__} does not implement input_vjp.")
        return torch.stack(self._loop(
            lambda a, b, v, g: self.input_vjp(a, b, g, parameter_values=v), x1, x2, parameter_values, cotangent
        ))

    def batch_input_vjp_symmetric(self, x, cotangent, parameter_values):
        return self.batch_input_vjp(x, x, cotangent + cotangent.transpose(-2, -1), parameter_values)

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        r"""``(B, N, D)``: ``∂(g_b · diag K_b(x_b, x_b)) / ∂x_b`` for a cotangent ``g`` ``(B, N)``.

        Generic fallback: by symmetry, ``d k(x_i, x_i) / d x_i = 2 ∂_1 k(x_i, x_i)``,
        so the result is :meth:`batch_input_vjp` of square blocks of at most
        ``chunk_size`` points with the diagonal cotangent ``2 diag(g)`` (no
        ``N x N`` matrix). The built-in kernels override it with closed forms.
        """
        blocks = []
        for start in range(0, x.shape[1], chunk_size):
            block = x[:, start:start + chunk_size]
            weights = 2.0 * torch.diag_embed(cotangent[:, start:start + chunk_size])
            blocks.append(self.batch_input_vjp(block, block, weights, parameter_values))
        return torch.cat(blocks, dim=1)

    def __add__(self, other):
        return SumKernel(self, other)

    def __radd__(self, other):
        return SumKernel(other, self)

    def __mul__(self, other):
        return ProductKernel(self, other)

    def __rmul__(self, other):
        return ProductKernel(other, self)


class Constant(Kernel):
    r"""Constant covariance kernel.

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v

    Parameters
    ----------
    variance : float or torch.Tensor or Parameter, default 1.0
        Constant variance parameter :math:`v > 0`.
    """

    def __init__(self, variance=1.0):
        super().__init__(variance)
        self.variance = variance

    def batch_evaluate(self, x1, x2, parameter_values):
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "Constant")
        ones = torch.ones((1, x1.shape[1], x2.shape[1]), dtype=x1.dtype, device=x1.device)
        return variance.reshape(-1, 1, 1) * ones

    def batch_derivatives(self, x1, x2, parameter_values):
        if not _is_named(self.variance):
            return {}
        ones = torch.ones((1, 1, x1.shape[1], x2.shape[1]), dtype=x1.dtype, device=x1.device)
        return {self.variance.name: ones}

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """Zero: the kernel does not depend on the input locations."""
        return _zero_input_vjp(x1, cotangent)

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """Zero: ``k(x, x) = v``."""
        return _zero_input_vjp(x, cotangent)


class WhiteNoise(Kernel):
    r"""Independent white noise covariance kernel: :math:`\mathbf{K}(X_1, X_2) = v \mathbf{I}` if :math:`X_1` and :math:`X_2` are the same inputs, else :math:`\mathbf{0}`.

    "The same inputs" means equal tensors (same shape and values): the kernel adds
    :math:`v` to the diagonal of :math:`\mathbf{K}(X, X)` and nothing to
    cross-covariances, even at coincident points.

    Parameters
    ----------
    variance : float or torch.Tensor or Parameter, default 1e-6
        Scalar noise variance :math:`v > 0`.
    """

    def __init__(self, variance=1e-6):
        super().__init__(variance)
        self.variance = variance

    @staticmethod
    def _pattern(x1, x2):
        if x1.shape[1] == x2.shape[1] and x1.shape == x2.shape and torch.equal(x1, x2):
            return torch.eye(x1.shape[1], dtype=x1.dtype, device=x1.device).unsqueeze(0)
        return torch.zeros((1, x1.shape[1], x2.shape[1]), dtype=x1.dtype, device=x1.device)

    def batch_evaluate(self, x1, x2, parameter_values):
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "WhiteNoise")
        return variance.reshape(-1, 1, 1) * self._pattern(x1, x2)

    def batch_derivatives(self, x1, x2, parameter_values):
        if not _is_named(self.variance):
            return {}
        return {self.variance.name: self._pattern(x1, x2).unsqueeze(1)}

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """Zero: the identity pattern is piecewise constant in the inputs."""
        return _zero_input_vjp(x1, cotangent)

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """Zero: ``k(x, x) = v``."""
        return _zero_input_vjp(x, cotangent)


class _StationaryDiagonal:
    """``k(x, x) = variance`` for stationary kernels: an ``O(N)`` diagonal.

    Only the variance changes the diagonal; length-scales and periods have
    zero diagonal derivatives.
    """

    def batch_diag(self, x, parameter_values, chunk_size=256):
        variance = _bscalar(self.variance, parameter_values, x, "variance", type(self).__name__)
        return variance.reshape(-1, 1).expand(variance.shape[0], x.shape[1])

    def batch_diag_derivatives(self, x, parameter_values, chunk_size=256):
        derivatives = {}
        for parameter in self.parameters:
            value = _bparam(parameter, parameter_values, x)
            derivatives[parameter.name] = torch.zeros((1, *value.shape[1:], x.shape[1]), dtype=x.dtype, device=x.device)
        if _is_named(self.variance):
            derivatives[self.variance.name] = torch.ones((1, 1, x.shape[1]), dtype=x.dtype, device=x.device)
        return derivatives

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """Zero: the diagonal ``k(x, x) = v`` does not depend on ``x``."""
        return _zero_input_vjp(x, cotangent)


class RBF(_StationaryDiagonal, Kernel):
    r"""Squared exponential (Radial Basis Function / Gaussian) covariance kernel.

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v \exp\left( -\frac{1}{2} \sum_{d=1}^D \frac{(x_{1d} - x_{2d})^2}{l_d^2} \right)

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Characteristic lengthscale :math:`l`. Scalar or shape ``(D,)`` for automatic relevance determination (ARD).
    variance : float or torch.Tensor or Parameter, default 1.0
        Output variance :math:`v > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale, variance)
        self.lengthscale = lengthscale
        self.variance = variance

    def _geometry(self, x1, x2, parameter_values):
        value, lengthscale = _blengthscale(self.lengthscale, parameter_values, x1, "RBF")
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "RBF")
        scaled_square = (_differences(x1, x2) / _view(lengthscale, 4)).square()  # (B, N, M, D)
        covariance = variance.reshape(-1, 1, 1) * torch.exp(-0.5 * scaled_square.sum(dim=-1))
        return value, lengthscale, variance, scaled_square, covariance

    def batch_evaluate(self, x1, x2, parameter_values):
        return self._geometry(x1, x2, parameter_values)[-1]

    def batch_derivatives(self, x1, x2, parameter_values):
        value, lengthscale, variance, scaled_square, covariance = self._geometry(x1, x2, parameter_values)
        derivatives = {}
        if _is_named(self.lengthscale):
            per_feature = covariance.unsqueeze(-1) * scaled_square / _view(lengthscale, 4)
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            derivatives[self.variance.name] = _scalar_derivative(covariance / variance.reshape(-1, 1, 1))
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        _, lengthscale, _, _, covariance = self._geometry(x1, x2, parameter_values)
        return -_radial_input_vjp(cotangent * covariance, x1, x2, 1.0 / lengthscale.square())


class Periodic(_StationaryDiagonal, Kernel):
    r"""Periodic (exponentiated sine squared) covariance kernel.

    For scalar lengthscale :math:`l` and period :math:`p`:

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v \exp\left( -\frac{2\sin^2\left(\frac{\pi \|\mathbf{x}_1 - \mathbf{x}_2\|}{p}\right)}{l^2} \right)

    With an ARD lengthscale the sine is applied per feature,
    :math:`v \exp\left(-2 \sum_d \sin^2(\pi (x_{1d} - x_{2d}) / p) / l_d^2\right)`.

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Lengthscale :math:`l > 0`, scalar or ARD vector of shape ``(D,)``.
    variance : float or torch.Tensor or Parameter, default 1.0
        Scalar output variance :math:`v > 0`.
    period : float or torch.Tensor or Parameter, default 1.0
        Scalar period of the repetition :math:`p > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        super().__init__(lengthscale, variance, period)
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def _geometry(self, x1, x2, parameter_values):
        value, lengthscale = _blengthscale(self.lengthscale, parameter_values, x1, "Periodic")
        period = _bscalar(self.period, parameter_values, x1, "period", "Periodic")
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "Periodic")
        difference = _differences(x1, x2)
        if lengthscale.shape[-1] == 1:
            # (B, N, M, 1): the Euclidean distance between the inputs.
            distance = difference.square().sum(dim=-1, keepdim=True).sqrt()
        else:
            # (B, N, M, D): per-feature distances for ARD length-scales.
            distance = difference.abs()
        sine = torch.sin(torch.pi * distance / _view(period, 4))
        exponent = (sine.square() / _view(lengthscale, 4).square()).sum(dim=-1)
        covariance = variance.reshape(-1, 1, 1) * torch.exp(-2.0 * exponent)
        return value, lengthscale, period, variance, difference, distance, sine, covariance

    def batch_evaluate(self, x1, x2, parameter_values):
        return self._geometry(x1, x2, parameter_values)[-1]

    def batch_derivatives(self, x1, x2, parameter_values):
        value, lengthscale, period, variance, _, distance, sine, covariance = self._geometry(x1, x2, parameter_values)
        lengthscale_view = _view(lengthscale, 4)
        derivatives = {}
        if _is_named(self.lengthscale):
            per_feature = covariance.unsqueeze(-1) * (4.0 * sine.square() / lengthscale_view.pow(3))
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            derivatives[self.variance.name] = _scalar_derivative(covariance / variance.reshape(-1, 1, 1))
        if _is_named(self.period):
            period_view = _view(period, 4)
            phase = torch.pi * distance / period_view
            period_derivative = covariance * (
                4.0 * sine * torch.cos(phase) * (torch.pi * distance / period_view.square()) / lengthscale_view.square()
            ).sum(dim=-1)
            derivatives[self.period.name] = _scalar_derivative(period_derivative)
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        r"""Input gradient of the periodic kernel.

        With ``φ = π δ / p`` and ``d sin²φ / dδ = (π/p) sin 2φ``:
        scalar length-scale (``δ = ‖x1-x2‖``):
        ``∂k/∂x1 = -k (2π / (p l²)) sin(2φ) (x1 - x2) / δ``;
        ARD (``δ_d = x1_d - x2_d``):
        ``∂k/∂x1_d = -k (2π / (p l_d²)) sin(2π δ_d / p)``.
        """
        _, lengthscale, period, _, difference, distance, _, covariance = self._geometry(x1, x2, parameter_values)
        weighted = cotangent * covariance
        factor = -2.0 * torch.pi / (period.reshape(-1, 1) * lengthscale.square())  # (B, 1 or D)
        if lengthscale.shape[-1] == 1:
            distance = distance[..., 0]
            double_sine = torch.sin(2.0 * torch.pi * distance / period.reshape(-1, 1, 1))
            safe = torch.where(distance > 0, distance, torch.ones_like(distance))
            radial = torch.where(distance > 0, double_sine / safe, torch.zeros_like(distance))
            return _radial_input_vjp(weighted * radial, x1, x2, factor)
        double_sine = torch.sin(2.0 * torch.pi * difference / _view(period, 4))
        per_feature = (weighted.unsqueeze(-1) * double_sine).sum(dim=-2)  # (B, N, D)
        return factor.unsqueeze(-2) * per_feature


class Matern(_StationaryDiagonal, Kernel):
    r"""Matérn covariance kernel for arbitrary smoothness :math:`\nu > 0`.

    The covariance between two points at distance :math:`r = \sqrt{\sum_{d=1}^D (x_{1d} - x_{2d})^2 / l_d^2}` is:

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} r\right)^\nu K_\nu\left(\sqrt{2\nu} r\right)

    where :math:`K_\nu` is the modified Bessel function of the second kind.

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Lengthscale parameter :math:`l`, scalar or ARD vector of shape ``(D,)``.
    variance : float or torch.Tensor or Parameter, default 1.0
        Output variance :math:`v > 0`.
    nu : float, default 1.5
        Smoothness parameter :math:`\nu > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, nu=1.5):
        if not isinstance(nu, (int, float)) or nu <= 0 or not math.isfinite(float(nu)):
            raise ValueError("Matern nu must be a positive finite number.")
        super().__init__(lengthscale, variance)
        self.lengthscale = lengthscale
        self.variance = variance
        self.nu = float(nu)
        self._half_integer = _half_integer_order(self.nu)

    def _geometry(self, x1, x2, parameter_values):
        value, lengthscale = _blengthscale(self.lengthscale, parameter_values, x1, "Matern")
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "Matern")
        difference = _differences(x1, x2)
        scaled_distance = (difference / _view(lengthscale, 4)).square().sum(-1).sqrt()
        argument = math.sqrt(2.0 * self.nu) * scaled_distance
        return value, lengthscale, variance, difference, scaled_distance, argument

    def _log_coefficient(self, reference):
        nu = torch.as_tensor(self.nu, dtype=reference.dtype, device=reference.device)
        return (1.0 - nu) * math.log(2.0) - torch.lgamma(nu)

    def _correlation(self, argument, derivative=False):
        """Return ``f(a)`` (and ``f'(a)`` when ``derivative``) with ``f(0) = 1``."""
        if self._half_integer is not None:
            value, slope = _half_integer_matern(self._half_integer, argument)
            return (value, slope) if derivative else value
        coefficient = torch.exp(self._log_coefficient(argument))
        positive = argument > 0
        value = coefficient * _power_bessel_k(self.nu, self.nu, argument)
        value = torch.where(positive, value, torch.ones_like(value))
        if not derivative:
            return value
        slope = -coefficient * _power_bessel_k(self.nu, self.nu - 1.0, argument)
        # The slope only enters multiplied by (x_d - x'_d)^2 / s, which
        # vanishes at s = 0.
        slope = torch.where(positive, slope, torch.zeros_like(slope))
        return value, slope

    @staticmethod
    def _radial(scaled_distance, slope):
        """``f'(a) / s``, set to zero at ``s = 0`` (where it multiplies a zero difference)."""
        positive = scaled_distance > 0
        safe = torch.where(positive, scaled_distance, torch.ones_like(scaled_distance))
        return torch.where(positive, slope / safe, torch.zeros_like(slope))

    def batch_evaluate(self, x1, x2, parameter_values):
        _, _, variance, _, _, argument = self._geometry(x1, x2, parameter_values)
        return variance.reshape(-1, 1, 1) * self._correlation(argument)

    def batch_derivatives(self, x1, x2, parameter_values):
        value, lengthscale, variance, difference, scaled_distance, argument = self._geometry(x1, x2, parameter_values)
        correlation, slope = self._correlation(argument, derivative=True)
        derivatives = {}
        if _is_named(self.lengthscale):
            radial = -self._radial(scaled_distance, slope)
            per_feature = (
                variance.reshape(-1, 1, 1, 1)
                * radial.unsqueeze(-1)
                * math.sqrt(2.0 * self.nu)
                * difference.square()
                / _view(lengthscale, 4).pow(3)
            )
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            derivatives[self.variance.name] = _scalar_derivative(correlation)
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """Coincident points contribute zero (for ``ν = 1/2`` the derivative at zero distance is not defined)."""
        _, lengthscale, variance, _, scaled_distance, argument = self._geometry(x1, x2, parameter_values)
        _, slope = self._correlation(argument, derivative=True)
        weights = cotangent * variance.reshape(-1, 1, 1) * math.sqrt(2.0 * self.nu) * self._radial(scaled_distance, slope)
        return _radial_input_vjp(weights, x1, x2, 1.0 / lengthscale.square())


class Matern32(Matern):
    r"""Matérn 3/2 covariance kernel (:math:`\nu = 1.5`).

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v (1 + \sqrt{3}r) \exp(-\sqrt{3}r)

    where :math:`r = \sqrt{\sum_{d=1}^D (x_{1d} - x_{2d})^2 / l_d^2}`.

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Lengthscale parameter :math:`l`, scalar or ARD vector of shape ``(D,)``.
    variance : float or torch.Tensor or Parameter, default 1.0
        Output variance :math:`v > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale=lengthscale, variance=variance, nu=1.5)


class Matern52(Matern):
    r"""Matérn 5/2 covariance kernel (:math:`\nu = 2.5`).

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v \left(1 + \sqrt{5}r + \frac{5}{3}r^2\right) \exp(-\sqrt{5}r)

    where :math:`r = \sqrt{\sum_{d=1}^D (x_{1d} - x_{2d})^2 / l_d^2}`.

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Lengthscale parameter :math:`l`, scalar or ARD vector of shape ``(D,)``.
    variance : float or torch.Tensor or Parameter, default 1.0
        Output variance :math:`v > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale=lengthscale, variance=variance, nu=2.5)


class RationalQuadratic(_StationaryDiagonal, Kernel):
    r"""Rational quadratic covariance kernel.

    Equivalent to a continuous scale mixture of RBF kernels with Gamma-distributed inverse squared lengthscales:

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v \left( 1 + \frac{r^2}{2\alpha} \right)^{-\alpha}, \quad r^2 = \sum_{d=1}^D \frac{(x_{1d} - x_{2d})^2}{l_d^2}

    Parameters
    ----------
    lengthscale : float or torch.Tensor or Parameter, default 1.0
        Lengthscale parameter :math:`l`, scalar or ARD vector of shape ``(D,)``.
    variance : float or torch.Tensor or Parameter, default 1.0
        Output variance :math:`v > 0`.
    alpha : float or torch.Tensor or Parameter, default 1.0
        Scale mixture shape parameter :math:`\alpha > 0`.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, alpha=1.0):
        super().__init__(lengthscale, variance, alpha)
        self.lengthscale = lengthscale
        self.variance = variance
        self.alpha = alpha

    def _geometry(self, x1, x2, parameter_values):
        value, lengthscale = _blengthscale(self.lengthscale, parameter_values, x1, "RationalQuadratic")
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "RationalQuadratic")
        alpha = _bscalar(self.alpha, parameter_values, x1, "alpha", "RationalQuadratic").reshape(-1, 1, 1)
        scaled_square = (_differences(x1, x2) / _view(lengthscale, 4)).square()
        squared_distance = scaled_square.sum(dim=-1)
        base = 1.0 + squared_distance / (2.0 * alpha)
        covariance = variance.reshape(-1, 1, 1) * base.pow(-alpha)
        return value, lengthscale, variance, alpha, scaled_square, squared_distance, base, covariance

    def batch_evaluate(self, x1, x2, parameter_values):
        return self._geometry(x1, x2, parameter_values)[-1]

    def batch_derivatives(self, x1, x2, parameter_values):
        value, lengthscale, variance, alpha, scaled_square, squared_distance, base, covariance = self._geometry(
            x1, x2, parameter_values
        )
        derivatives = {}
        if _is_named(self.lengthscale):
            per_feature = (covariance / base).unsqueeze(-1) * scaled_square / _view(lengthscale, 4)
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            derivatives[self.variance.name] = _scalar_derivative(covariance / variance.reshape(-1, 1, 1))
        if _is_named(self.alpha):
            alpha_derivative = covariance * (squared_distance / (2.0 * alpha * base) - torch.log(base))
            derivatives[self.alpha.name] = _scalar_derivative(alpha_derivative)
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        _, lengthscale, _, _, _, _, base, covariance = self._geometry(x1, x2, parameter_values)
        return -_radial_input_vjp(cotangent * covariance / base, x1, x2, 1.0 / lengthscale.square())


class Linear(Kernel):
    r"""Linear covariance kernel with offset.

    .. math::

        k(\mathbf{x}_1, \mathbf{x}_2) = v (\mathbf{x}_1 - \mathbf{c})^\top (\mathbf{x}_2 - \mathbf{c})

    Parameters
    ----------
    variance : float or torch.Tensor or Parameter, default 1.0
        Variance scaling factor :math:`v > 0`.
    offset : float or torch.Tensor or Parameter, default 0.0
        Offset vector :math:`\mathbf{c}`, scalar or shape ``(D,)``.
    """

    def __init__(self, variance=1.0, offset=0.0):
        super().__init__(variance, offset)
        self.variance = variance
        self.offset = offset

    def _centred(self, x1, x2, parameter_values):
        variance = _bscalar(self.variance, parameter_values, x1, "variance", "Linear")
        offset_value = _bparam(self.offset, parameter_values, x1)
        offset = offset_value.reshape(offset_value.shape[0], 1, -1)
        return variance, offset_value, offset, x1 - offset, x2 - offset

    def batch_evaluate(self, x1, x2, parameter_values):
        variance, _, _, left, right = self._centred(x1, x2, parameter_values)
        return variance.reshape(-1, 1, 1) * (left @ right.transpose(-2, -1))

    def batch_derivatives(self, x1, x2, parameter_values):
        variance, offset_value, offset, left, right = self._centred(x1, x2, parameter_values)
        derivatives = {}
        if _is_named(self.variance):
            derivatives[self.variance.name] = _scalar_derivative(left @ right.transpose(-2, -1))
        if _is_named(self.offset):
            per_feature = variance.reshape(-1, 1, 1, 1) * (2 * offset.unsqueeze(1) - x1.unsqueeze(-2) - x2.unsqueeze(-3))
            if offset_value[0].numel() == 1:
                derivatives[self.offset.name] = per_feature.sum(dim=-1).unsqueeze(1)
            else:
                derivatives[self.offset.name] = per_feature.movedim(-1, 1)
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """``∂k/∂x1 = v (x2 - c)``, so the VJP is ``v G (x2 - c)``."""
        variance, _, _, _, right = self._centred(x1, x2, parameter_values)
        return variance.reshape(-1, 1, 1) * (cotangent @ right)

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """``k(x, x) = v ‖x - c‖²``, so the VJP is ``2 v (x - c) g``."""
        variance, _, _, left, _ = self._centred(x, x, parameter_values)
        return 2.0 * variance.reshape(-1, 1, 1) * left * cotangent.unsqueeze(-1)


class SumKernel(Kernel):
    r"""Sum of covariance kernels :math:`k = k_1 + k_2 + \dots` (also built with ``k1 + k2``).

    Parameters
    ----------
    *kernels : Kernel or float or callable
        Kernels to sum. A number is a constant kernel without gradient, a plain
        callable ``k(x1, x2)`` gets no gradients (unbatched inputs only).
    """

    def __init__(self, *kernels):
        self.kernels = tuple(kernels)
        super().__init__(*(parameter for kernel in self.kernels for parameter in _kernel_parameters(kernel)))

    def batch_evaluate(self, x1, x2, parameter_values):
        return sum(_batch_evaluate_kernel(kernel, x1, x2, parameter_values) for kernel in self.kernels)

    def batch_derivatives(self, x1, x2, parameter_values):
        derivatives = {}
        for kernel in self.kernels:
            for name, derivative in _batch_kernel_derivatives(kernel, x1, x2, parameter_values).items():
                derivatives[name] = derivatives[name] + derivative if name in derivatives else derivative
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        return sum(_batch_kernel_input_vjp(kernel, x1, x2, cotangent, parameter_values) for kernel in self.kernels)

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """Sum rule: the parts' diagonal input VJPs added up."""
        return sum(_batch_kernel_diag_input_vjp(kernel, x, cotangent, parameter_values, chunk_size) for kernel in self.kernels)


class ProductKernel(Kernel):
    r"""Elementwise product of covariance kernels :math:`k = k_1 \cdot k_2 \cdot \dots` (also built with ``k1 * k2``).

    Parameters
    ----------
    *kernels : Kernel or float or callable
        Kernels to multiply, as for :class:`SumKernel`.
    """

    def __init__(self, *kernels):
        self.kernels = tuple(kernels)
        super().__init__(*(parameter for kernel in self.kernels for parameter in _kernel_parameters(kernel)))

    def batch_evaluate(self, x1, x2, parameter_values):
        result = None
        for kernel in self.kernels:
            value = _batch_evaluate_kernel(kernel, x1, x2, parameter_values)
            result = value if result is None else result * value
        return result

    def _others(self, x1, x2, parameter_values):
        values = [_batch_evaluate_kernel(kernel, x1, x2, parameter_values) for kernel in self.kernels]
        others = []
        for index in range(len(values)):
            other = None
            for other_index, value in enumerate(values):
                if other_index != index:
                    other = value if other is None else other * value
            others.append(other)
        return others

    def batch_derivatives(self, x1, x2, parameter_values):
        derivatives = {}
        for kernel, other in zip(self.kernels, self._others(x1, x2, parameter_values)):
            for name, derivative in _batch_kernel_derivatives(kernel, x1, x2, parameter_values).items():
                contribution = derivative if other is None else _times_matrix(derivative, other)
                derivatives[name] = derivatives[name] + contribution if name in derivatives else contribution
        return derivatives

    def batch_input_vjp(self, x1, x2, cotangent, parameter_values):
        """Product rule: kernel ``k`` receives the cotangent ``G ⊙ Π_{j≠k} K_j``."""
        result = 0
        for kernel, other in zip(self.kernels, self._others(x1, x2, parameter_values)):
            weighted = cotangent if other is None else cotangent * other
            result = result + _batch_kernel_input_vjp(kernel, x1, x2, weighted, parameter_values)
        return result

    def batch_diag_input_vjp(self, x, cotangent, parameter_values, chunk_size=256):
        """Product rule on the diagonal: part ``k`` receives the cotangent ``g ⊙ Π_{j≠k} diag K_j``."""
        diagonals = [_batch_kernel_diag(kernel, x, parameter_values, chunk_size) for kernel in self.kernels]
        result = 0
        for index, kernel in enumerate(self.kernels):
            weighted = cotangent
            for other_index, diagonal in enumerate(diagonals):
                if other_index != index:
                    weighted = weighted * diagonal
            result = result + _batch_kernel_diag_input_vjp(kernel, x, weighted, parameter_values, chunk_size)
        return result


def _kernel_parameters(kernel):
    if isinstance(kernel, Kernel):
        return kernel.parameters
    return tuple(getattr(kernel, "parameters", ())) if callable(kernel) else ()


def _shared_inputs(x1, x2, what):
    if x1.shape[0] != 1 or x2.shape[0] != 1:
        raise NotImplementedError(f"A plain callable inside a composite kernel cannot {what} batched inputs.")
    return x1[0], x2[0]


def _batch_evaluate_kernel(kernel, x1, x2, parameter_values):
    if isinstance(kernel, Kernel):
        return kernel.batch_evaluate(x1, x2, parameter_values)
    if callable(kernel):
        a, b = _shared_inputs(x1, x2, "evaluate")
        return torch.as_tensor(kernel(a, b)).unsqueeze(0)
    return torch.as_tensor(kernel, dtype=x1.dtype, device=x1.device).reshape(1, 1, 1)


def _batch_kernel_derivatives(kernel, x1, x2, parameter_values):
    if isinstance(kernel, Kernel):
        return kernel.batch_derivatives(x1, x2, parameter_values)
    return {}


def _batch_kernel_diag(kernel, x, parameter_values, chunk_size=256):
    """``(B, N)`` diagonal of a part of a composite kernel (a kernel, a plain callable or a number)."""
    if isinstance(kernel, Kernel):
        return kernel.batch_diag(x, parameter_values, chunk_size=chunk_size)
    if callable(kernel):
        a, _ = _shared_inputs(x, x, "evaluate")
        return torch.diagonal(torch.as_tensor(kernel(a, a))).unsqueeze(0)
    return torch.as_tensor(kernel, dtype=x.dtype, device=x.device).reshape(1, 1)


def _batch_kernel_diag_input_vjp(kernel, x, cotangent, parameter_values, chunk_size=256):
    if isinstance(kernel, Kernel):
        return kernel.batch_diag_input_vjp(x, cotangent, parameter_values, chunk_size=chunk_size)
    if callable(kernel):
        raise NotImplementedError("Input gradients are not available for plain callables inside a composite kernel.")
    return _zero_input_vjp(x, cotangent)


def _batch_kernel_input_vjp(kernel, x1, x2, cotangent, parameter_values):
    if isinstance(kernel, Kernel):
        return kernel.batch_input_vjp(x1, x2, cotangent, parameter_values)
    if callable(kernel):
        raise NotImplementedError("Input gradients are not available for plain callables inside a composite kernel.")
    return torch.zeros((_batch_size(x1, cotangent), *x1.shape[1:]), dtype=x1.dtype, device=x1.device)

