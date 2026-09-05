"""Composable covariance kernels with explicit analytic derivatives."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.special import gamma, kv

from ..Distributions._resolve import resolve


def _parameter_value(parameter, parameter_values=None):
    if parameter_values is not None and hasattr(parameter, "name") and parameter.name in parameter_values:
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
        if not hasattr(value, "name") or value.name in seen:
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
        if not hasattr(self.variance, "name"):
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
        if not hasattr(self.variance, "name"):
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


class RBF(Kernel):
    """Squared-exponential kernel with optional ARD length-scales."""

    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__(lengthscale, variance)
        self.lengthscale = lengthscale
        self.variance = variance

    def evaluate(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        lengthscale = _parameter_value(self.lengthscale, parameter_values).to(dtype=x1.dtype, device=x1.device).reshape(-1)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        if lengthscale.numel() not in (1, x1.shape[-1]):
            raise ValueError(f"RBF lengthscale must be scalar or have one value per feature; got {lengthscale.numel()} values for {x1.shape[-1]} features.")
        difference = (x1[:, None, :] - x2[None, :, :]) / lengthscale.reshape(1, 1, -1)
        return variance * torch.exp(-0.5 * difference.square().sum(dim=-1))

    def derivatives(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        lengthscale = _parameter_value(self.lengthscale, parameter_values).to(dtype=x1.dtype, device=x1.device).reshape(-1)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        difference = (x1[:, None, :] - x2[None, :, :]) / lengthscale.reshape(1, 1, -1)
        covariance = self.evaluate(x1, x2, parameter_values=parameter_values)
        derivatives = {}
        if hasattr(self.lengthscale, "name"):
            derivatives[self.lengthscale.name] = covariance.unsqueeze(0) * difference.square().permute(2, 0, 1) / lengthscale.reshape(-1, 1, 1)
        if hasattr(self.variance, "name"):
            derivatives[self.variance.name] = _parameter_derivative(variance, covariance / variance)
        return derivatives


class Periodic(Kernel):
    r"""Exponentiated-sine-squared kernel for periodic relationships.

    The kernel is

    .. math::

        k(x_1, x_2) = v\exp\left(
            -\frac{2\sin^2(\pi\lVert x_1-x_2\rVert / p)}{l^2}
        \right),

    where ``v`` is the variance, ``l`` is the length-scale, and ``p`` is
    the period.  ``lengthscale`` and ``period`` are scalar parameters, as in
    the original implementation this kernel is based on.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        super().__init__(lengthscale, variance, period)
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def _geometry(self, x1, x2, parameter_values=None):
        x1 = _feature_matrix(x1)
        x2 = _feature_matrix(x2)
        lengthscale = _parameter_value(self.lengthscale, parameter_values).to(
            dtype=x1.dtype, device=x1.device
        ).reshape(-1)
        period = _parameter_value(self.period, parameter_values).to(
            dtype=x1.dtype, device=x1.device
        ).reshape(-1)
        if lengthscale.numel() != 1:
            raise ValueError("Periodic lengthscale must be a scalar.")
        if period.numel() != 1:
            raise ValueError("Periodic period must be a scalar.")

        distance = torch.cdist(x1, x2)
        phase = torch.pi * distance / period
        sine = torch.sin(phase)
        return x1, lengthscale, period, sine, distance

    def evaluate(self, x1, x2, parameter_values=None):
        x1, lengthscale, period, sine, _ = self._geometry(
            x1, x2, parameter_values
        )
        variance = _parameter_value(self.variance, parameter_values).to(
            dtype=x1.dtype, device=x1.device
        )
        return variance * torch.exp(-2.0 * sine.square() / lengthscale.square())

    def derivatives(self, x1, x2, parameter_values=None):
        x1, lengthscale, period, sine, distance = self._geometry(
            x1, x2, parameter_values
        )
        variance = _parameter_value(self.variance, parameter_values).to(
            dtype=x1.dtype, device=x1.device
        )
        covariance = self.evaluate(x1, x2, parameter_values=parameter_values)
        derivatives = {}

        if hasattr(self.lengthscale, "name"):
            lengthscale_derivative = covariance * (
                4.0 * sine.square() / lengthscale.pow(3)
            )
            derivatives[self.lengthscale.name] = _parameter_derivative(
                lengthscale, lengthscale_derivative
            )

        if hasattr(self.variance, "name"):
            derivatives[self.variance.name] = _parameter_derivative(
                variance, covariance / variance
            )

        if hasattr(self.period, "name"):
            phase = torch.pi * distance / period
            period_derivative = covariance * (
                4.0
                * sine
                * torch.cos(phase)
                * (torch.pi * distance / period.square())
                / lengthscale.square()
            )
            derivatives[self.period.name] = _parameter_derivative(
                period, period_derivative
            )

        return derivatives


class Matern(Kernel):
    """Matérn kernel for any fixed positive finite smoothness ``nu``."""

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
        lengthscale = _parameter_value(self.lengthscale, parameter_values).to(dtype=x1.dtype, device=x1.device).reshape(-1)
        if lengthscale.numel() not in (1, x1.shape[-1]):
            raise ValueError(f"Matern lengthscale must be scalar or have one value per feature; got {lengthscale.numel()} values for {x1.shape[-1]} features.")
        difference = x1[:, None, :] - x2[None, :, :]
        scaled_distance = (difference / lengthscale.reshape(1, 1, -1)).square().sum(-1).sqrt()
        argument = math.sqrt(2.0 * self.nu) * scaled_distance
        return x1, lengthscale, difference, scaled_distance, argument

    def evaluate(self, x1, x2, parameter_values=None):
        x1, _, _, _, argument = self._geometry(x1, x2, parameter_values)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        safe_argument = torch.where(argument == 0, torch.ones_like(argument), argument)
        coefficient = 2.0 ** (1.0 - self.nu) / gamma(self.nu)
        correlation = coefficient * safe_argument.pow(self.nu) * _scipy_bessel_k(self.nu, safe_argument, x1.dtype, x1.device)
        correlation = torch.where(argument == 0, torch.ones_like(correlation), correlation)
        return variance * correlation

    def derivatives(self, x1, x2, parameter_values=None):
        x1, lengthscale, difference, scaled_distance, argument = self._geometry(x1, x2, parameter_values)
        variance = _parameter_value(self.variance, parameter_values).to(dtype=x1.dtype, device=x1.device)
        covariance = self.evaluate(x1, x2, parameter_values=parameter_values)
        derivatives = {}
        if hasattr(self.lengthscale, "name"):
            safe_argument = torch.where(argument == 0, torch.ones_like(argument), argument)
            coefficient = 2.0 ** (1.0 - self.nu) / gamma(self.nu)
            bessel_previous = _scipy_bessel_k(self.nu - 1.0, safe_argument, x1.dtype, x1.device)
            radial_factor = coefficient * safe_argument.pow(self.nu) * bessel_previous
            radial_factor = torch.where(argument == 0, torch.zeros_like(radial_factor), radial_factor)
            safe_distance = torch.where(scaled_distance == 0, torch.ones_like(scaled_distance), scaled_distance)
            derivatives[self.lengthscale.name] = (
                variance
                * radial_factor
                * math.sqrt(2.0 * self.nu)
                * difference.square().permute(2, 0, 1)
                / (safe_distance.unsqueeze(0) * lengthscale.reshape(-1, 1, 1).pow(3))
            )
        if hasattr(self.variance, "name"):
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
        if hasattr(self.variance, "name"):
            derivatives[self.variance.name] = _parameter_derivative(variance, ((x1 - offset) @ (x2 - offset).transpose(-2, -1)))
        if hasattr(self.offset, "name"):
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
