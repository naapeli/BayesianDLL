"""Composable covariance kernels with explicit analytic derivatives."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.special import gamma, kv

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


def _unique_named(values):
    result = []
    seen = set()
    for value in values:
        if not _is_named(value) or value.name in seen:
            continue
        seen.add(value.name)
        result.append(value)
    return tuple(result)


def _parameter_derivative(value, matrix):
    """Put parameter axes before the two covariance axes."""
    value = value.reshape(-1)
    if value.numel() == 1:
        return matrix.reshape(1, *matrix.shape)
    return matrix


def _lengthscale_vector(parameter, parameter_values, x, kernel_name):
    """Return the length-scale as a flat tensor with 1 or ``D`` entries."""
    value = _parameter_value(parameter, parameter_values).to(dtype=x.dtype, device=x.device)
    flat = value.reshape(-1)
    if flat.numel() not in (1, x.shape[-1]):
        raise ValueError(
            f"{kernel_name} lengthscale must be scalar or have one value per feature; "
            f"got {flat.numel()} values for {x.shape[-1]} features."
        )
    return value, flat


def _lengthscale_derivative(value, per_feature):
    """Shape a per-feature derivative stack ``(D, N, M)`` like the parameter.

    A scalar length-scale shared by all features receives the sum over the
    feature axis (shape ``(1, N, M)``); an ARD length-scale with ``D`` values
    receives ``value.shape + (N, M)``, e.g. ``(D, N, M)``.
    """
    if value.numel() == 1:
        return per_feature.sum(dim=0, keepdim=True)
    return per_feature.reshape(*value.shape, *per_feature.shape[-2:])


def _scipy_bessel_k(order, argument, dtype, device):
    """Evaluate K_order numerically without automatic differentiation."""
    values = kv(order, argument.detach().cpu().numpy())
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


class Kernel(ABC):
    """Base class for covariance functions ``k(X1, X2)``."""

    def __init__(self, *parameters):
        self._parameters = _unique_named(parameters)

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def evaluate(self, x1, x2, parameter_values=None):
        pass

    def derivatives(self, x1, x2, parameter_values=None):
        """Return derivatives keyed by model-parameter name.

        Each value has parameter axes first and covariance axes last. Thus a
        scalar parameter has derivative shape ``(1, N, M)``.
        """
        return {}

    def diag(self, x, parameter_values=None, chunk_size=256):
        """Return ``diag(k(x, x))`` without forming the full ``N x N`` matrix.

        The default evaluates square blocks of at most ``chunk_size`` points
        along the diagonal, so it costs ``O(N * chunk_size)``.
        """
        x = _feature_matrix(x)
        blocks = []
        for start in range(0, x.shape[0], chunk_size):
            block = x[start:start + chunk_size]
            blocks.append(torch.diagonal(self.evaluate(block, block, parameter_values=parameter_values), dim1=-2, dim2=-1))
        return torch.cat(blocks, dim=-1)

    def diag_derivatives(self, x, parameter_values=None, chunk_size=256):
        """Derivatives of :meth:`diag`; each value has shape ``parameter_shape + (N,)``."""
        x = _feature_matrix(x)
        pieces = {}
        for start in range(0, x.shape[0], chunk_size):
            block = x[start:start + chunk_size]
            for name, derivative in self.derivatives(block, block, parameter_values=parameter_values).items():
                pieces.setdefault(name, []).append(torch.diagonal(derivative, dim1=-2, dim2=-1))
        return {name: torch.cat(values, dim=-1) for name, values in pieces.items()}

    def __call__(self, x1, x2=None, parameter_values=None):
        if x2 is None:
            x2 = x1
        return self.evaluate(x1, x2, parameter_values=parameter_values)

    def __add__(self, other):
        return SumKernel(self, other)

    def __radd__(self, other):
        return SumKernel(other, self)

    def __mul__(self, other):
        return ProductKernel(self, other)

    def __rmul__(self, other):
        return ProductKernel(other, self)


class Constant(Kernel):
    def __init__(self, variance=1.0):
        super().__init__(variance)
        self.variance = variance

    def evaluate(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        return variance * torch.ones((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)

    def derivatives(self, x1, x2, parameter_values=None):
        if not _is_named(self.variance):
            return {}
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        matrix = torch.ones((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)
        value = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        return {self.variance.name: _parameter_derivative(value, matrix)}


class WhiteNoise(Kernel):
    """Independent noise on matching input points."""

    def __init__(self, variance=1e-6):
        super().__init__(variance)
        self.variance = variance

    def evaluate(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        if x1.shape[0] == x2.shape[0] and torch.equal(x1, x2):
            return variance * torch.eye(x1.shape[0], dtype=x1.dtype, device=x1.device)
        return torch.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)

    def derivatives(self, x1, x2, parameter_values=None):
        if not _is_named(self.variance):
            return {}
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        matrix = (
            torch.eye(x1.shape[0], dtype=x1.dtype, device=x1.device)
            if x1.shape[0] == x2.shape[0] and torch.equal(x1, x2)
            else torch.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)
        )
        value = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        return {self.variance.name: _parameter_derivative(value, matrix)}


class _StationaryDiagonal:
    """``k(x, x) = variance`` for stationary kernels: an ``O(N)`` diagonal.

    Only the variance changes the diagonal; length-scales and periods have
    zero diagonal derivatives.
    """

    def diag(self, x, parameter_values=None, chunk_size=256):
        x = _feature_matrix(x)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x.dtype, device=x.device)
        return variance.reshape(()).expand(x.shape[0]) if variance.numel() == 1 else variance.reshape(-1)

    def diag_derivatives(self, x, parameter_values=None, chunk_size=256):
        x = _feature_matrix(x)
        derivatives = {}
        for parameter in self.parameters:
            value = _parameter_value(parameter, parameter_values).to(dtype=x.dtype, device=x.device)
            shape = (1,) if value.numel() == 1 else tuple(value.shape)
            derivatives[parameter.name] = torch.zeros((*shape, x.shape[0]), dtype=x.dtype, device=x.device)
        if _is_named(self.variance):
            derivatives[self.variance.name] = torch.ones((1, x.shape[0]), dtype=x.dtype, device=x.device)
        return derivatives


class RBF(_StationaryDiagonal, Kernel):
    r"""Squared-exponential kernel with optional ARD length-scales.

    .. math::

        k(x_1, x_2) = v\exp\Bigl(-\tfrac12\sum_{d=1}^D (x_{1d}-x_{2d})^2 / l_d^2\Bigr).

    ``lengthscale`` is either a scalar shared by all ``D`` features or a
    vector with one value per feature (automatic relevance determination).
    It may be a constant, a ``RandomParameter`` (e.g. with ``shape=D`` or
    inside a plate of size ``D``) or a ``DeterministicParameter``. The
    derivative w.r.t. an ARD length-scale has shape ``(D, N, M)``:
    ``dk/dl_d = k (x_{1d}-x_{2d})^2 / l_d^3``.
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale, variance)
        self.lengthscale = lengthscale
        self.variance = variance

    def _geometry(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        value, lengthscale = _lengthscale_vector(self.lengthscale, parameter_values, x1, "RBF")
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        # (D, N, M) squared scaled differences.
        scaled_square = ((x1[:, None, :] - x2[None, :, :]) / lengthscale.reshape(1, 1, -1)).square().permute(2, 0, 1)
        covariance = variance * torch.exp(-0.5 * scaled_square.sum(dim=0))
        return value, lengthscale, variance, scaled_square, covariance

    def evaluate(self, x1, x2, parameter_values=None):
        return self._geometry(x1, x2, parameter_values)[-1]

    def derivatives(self, x1, x2, parameter_values=None):
        value, lengthscale, variance, scaled_square, covariance = self._geometry(x1, x2, parameter_values)
        derivatives = {}
        if _is_named(self.lengthscale):
            per_feature = covariance.unsqueeze(0) * scaled_square / lengthscale.reshape(-1, 1, 1)
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            derivatives[self.variance.name] = _parameter_derivative(variance, covariance / variance)
        return derivatives


class Periodic(_StationaryDiagonal, Kernel):
    r"""Exponentiated-sine-squared kernel for periodic relationships.

    With a scalar length-scale the kernel is

    .. math::

        k(x_1, x_2) = v\exp\left(
            -\frac{2\sin^2(\pi\lVert x_1-x_2\rVert / p)}{l^2}
        \right),

    where ``v`` is the variance, ``l`` is the length-scale, and ``p`` is
    the (scalar) period. With an ARD length-scale holding one value per
    feature, the sine is applied feature by feature:

    .. math::

        k(x_1, x_2) = v\exp\Bigl(
            -2\sum_{d=1}^D \sin^2(\pi (x_{1d}-x_{2d}) / p) / l_d^2
        \Bigr).

    Both forms agree for one-dimensional inputs.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        super().__init__(lengthscale, variance, period)
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def _geometry(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        value, lengthscale = _lengthscale_vector(self.lengthscale, parameter_values, x1, "Periodic")
        period = _parameter_value(self.period, parameter_values).to(dtype=x1.dtype, device=x1.device).reshape(-1)
        if period.numel() != 1:
            raise ValueError("Periodic period must be a scalar.")
        if lengthscale.numel() == 1:
            # (1, N, M): the Euclidean distance between the inputs.
            distance = torch.cdist(x1, x2).unsqueeze(0)
        else:
            # (D, N, M): per-feature distances for ARD length-scales.
            distance = (x1[:, None, :] - x2[None, :, :]).abs().permute(2, 0, 1)
        phase = torch.pi * distance / period
        sine = torch.sin(phase)
        return x1, value, lengthscale, period, sine, distance

    def _covariance(self, x1, lengthscale, sine, parameter_values):
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        exponent = (sine.square() / lengthscale.square().reshape(-1, 1, 1)).sum(dim=0)
        return variance, variance * torch.exp(-2.0 * exponent)

    def evaluate(self, x1, x2, parameter_values=None):
        x1, _, lengthscale, _, sine, _ = self._geometry(x1, x2, parameter_values)
        return self._covariance(x1, lengthscale, sine, parameter_values)[1]

    def derivatives(self, x1, x2, parameter_values=None):
        x1, value, lengthscale, period, sine, distance = self._geometry(x1, x2, parameter_values)
        variance, covariance = self._covariance(x1, lengthscale, sine, parameter_values)
        lengthscale_column = lengthscale.reshape(-1, 1, 1)
        derivatives = {}

        if _is_named(self.lengthscale):
            per_feature = covariance * (4.0 * sine.square() / lengthscale_column.pow(3))
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)

        if _is_named(self.variance):
            derivatives[self.variance.name] = _parameter_derivative(variance, covariance / variance)

        if _is_named(self.period):
            phase = torch.pi * distance / period
            period_derivative = covariance * (
                4.0 * sine * torch.cos(phase) * (torch.pi * distance / period.square()) / lengthscale_column.square()
            ).sum(dim=0)
            derivatives[self.period.name] = _parameter_derivative(period, period_derivative)

        return derivatives


class Matern(_StationaryDiagonal, Kernel):
    r"""Matérn kernel for any fixed positive finite smoothness ``nu``.

    With ``s = sqrt(sum_d (x_{1d}-x_{2d})^2 / l_d^2)`` and
    ``a = sqrt(2 nu) s``, ``k = v 2^{1-nu}/Gamma(nu) a^nu K_nu(a)``.
    ``lengthscale`` may be scalar or hold one value per feature (ARD). Using
    ``d/da [a^nu K_nu(a)] = -a^nu K_{nu-1}(a)``, the derivative w.r.t. each
    length-scale component is

    ``dk/dl_d = v 2^{1-nu}/Gamma(nu) a^nu K_{nu-1}(a) sqrt(2 nu) (x_{1d}-x_{2d})^2 / (s l_d^3)``,

    which tends to zero as ``s -> 0``.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, nu=1.5):
        if not isinstance(nu, (int, float)) or nu <= 0 or not math.isfinite(float(nu)):
            raise ValueError("Matern nu must be a positive finite number.")
        super().__init__(lengthscale, variance)
        self.lengthscale = lengthscale
        self.variance = variance
        self.nu = float(nu)

    def _geometry(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        value, lengthscale = _lengthscale_vector(self.lengthscale, parameter_values, x1, "Matern")
        difference = x1[:, None, :] - x2[None, :, :]
        scaled_distance = (difference / lengthscale.reshape(1, 1, -1)).square().sum(-1).sqrt()
        argument = math.sqrt(2.0 * self.nu) * scaled_distance
        return x1, value, lengthscale, difference, scaled_distance, argument

    def evaluate(self, x1, x2, parameter_values=None):
        x1, _, _, _, _, argument = self._geometry(x1, x2, parameter_values)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        safe_argument = torch.where(argument == 0, torch.ones_like(argument), argument)
        coefficient = 2.0 ** (1.0 - self.nu) / gamma(self.nu)
        correlation = coefficient * safe_argument.pow(self.nu) * _scipy_bessel_k(self.nu, safe_argument, x1.dtype, x1.device)
        correlation = torch.where(argument == 0, torch.ones_like(correlation), correlation)
        return variance * correlation

    def derivatives(self, x1, x2, parameter_values=None):
        x1, value, lengthscale, difference, scaled_distance, argument = self._geometry(x1, x2, parameter_values)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        derivatives = {}
        if _is_named(self.lengthscale):
            safe_argument = torch.where(argument == 0, torch.ones_like(argument), argument)
            coefficient = 2.0 ** (1.0 - self.nu) / gamma(self.nu)
            bessel_previous = _scipy_bessel_k(self.nu - 1.0, safe_argument, x1.dtype, x1.device)
            radial_factor = coefficient * safe_argument.pow(self.nu) * bessel_previous
            radial_factor = torch.where(argument == 0, torch.zeros_like(radial_factor), radial_factor)
            safe_distance = torch.where(scaled_distance == 0, torch.ones_like(scaled_distance), scaled_distance)
            per_feature = (
                variance
                * radial_factor
                * math.sqrt(2.0 * self.nu)
                * difference.square().permute(2, 0, 1)
                / (safe_distance.unsqueeze(0) * lengthscale.reshape(-1, 1, 1).pow(3))
            )
            derivatives[self.lengthscale.name] = _lengthscale_derivative(value, per_feature)
        if _is_named(self.variance):
            covariance = self.evaluate(x1, x2, parameter_values=parameter_values)
            derivatives[self.variance.name] = _parameter_derivative(variance, covariance / variance)
        return derivatives


class Matern32(Matern):
    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale=lengthscale, variance=variance, nu=1.5)


class Matern52(Matern):
    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale=lengthscale, variance=variance, nu=2.5)


class Linear(Kernel):
    def __init__(self, variance=1.0, offset=0.0):
        super().__init__(variance, offset)
        self.variance = variance
        self.offset = offset

    def evaluate(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        offset = _parameter_value(self.offset, parameter_values).to(dtype=x1.dtype, device=x1.device)
        return variance * ((x1 - offset) @ (x2 - offset).transpose(-2, -1))

    def derivatives(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        offset = _parameter_value(self.offset, parameter_values).to(dtype=x1.dtype, device=x1.device)
        derivatives = {}
        if _is_named(self.variance):
            derivatives[self.variance.name] = _parameter_derivative(variance, ((x1 - offset) @ (x2 - offset).transpose(-2, -1)))
        if _is_named(self.offset):
            derivatives[self.offset.name] = variance * (2 * offset - x1[:, None, :] - x2[None, :, :]).permute(2, 0, 1)
        return derivatives


class SumKernel(Kernel):
    def __init__(self, *kernels):
        self.kernels = tuple(kernels)
        super().__init__(*(parameter for kernel in self.kernels for parameter in _kernel_parameters(kernel)))

    def evaluate(self, x1, x2, parameter_values=None):
        return sum(_evaluate_kernel(kernel, x1, x2, parameter_values) for kernel in self.kernels)

    def derivatives(self, x1, x2, parameter_values=None):
        derivatives = {}
        for kernel in self.kernels:
            for name, derivative in _kernel_derivatives(kernel, x1, x2, parameter_values).items():
                derivatives[name] = derivatives.get(name, 0) + derivative
        return derivatives


class ProductKernel(Kernel):
    def __init__(self, *kernels):
        self.kernels = tuple(kernels)
        super().__init__(*(parameter for kernel in self.kernels for parameter in _kernel_parameters(kernel)))

    def evaluate(self, x1, x2, parameter_values=None):
        result = None
        for kernel in self.kernels:
            value = _evaluate_kernel(kernel, x1, x2, parameter_values)
            result = value if result is None else result * value
        return result

    def derivatives(self, x1, x2, parameter_values=None):
        values = [_evaluate_kernel(kernel, x1, x2, parameter_values) for kernel in self.kernels]
        derivatives = {}
        for index, kernel in enumerate(self.kernels):
            other = None
            for other_index, value in enumerate(values):
                if other_index != index:
                    other = value if other is None else other * value
            for name, derivative in _kernel_derivatives(kernel, x1, x2, parameter_values).items():
                contribution = derivative if other is None else derivative * other
                derivatives[name] = derivatives.get(name, 0) + contribution
        return derivatives


def _kernel_parameters(kernel):
    if isinstance(kernel, Kernel):
        return kernel.parameters
    return tuple(getattr(kernel, "parameters", ())) if callable(kernel) else ()


def _evaluate_kernel(kernel, x1, x2, parameter_values=None):
    if isinstance(kernel, Kernel):
        return kernel.evaluate(x1, x2, parameter_values=parameter_values)
    if callable(kernel):
        return kernel(x1, x2)
    return torch.as_tensor(kernel, dtype=_feature_matrix(x1).dtype, device=_feature_matrix(x1).device)


def _kernel_derivatives(kernel, x1, x2, parameter_values=None):
    if isinstance(kernel, Kernel):
        return kernel.derivatives(x1, x2, parameter_values=parameter_values)
    return {}
