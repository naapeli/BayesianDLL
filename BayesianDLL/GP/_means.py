"""Mean functions ``m(X)`` of Gaussian processes, with explicit analytic derivatives.

A mean function plays the role of a kernel for the first moment of a GP:
every GP class takes it as ``mean=`` and evaluates it at its training
inputs, and the predictive functions evaluate it at the new inputs draw by
draw. Its parameters may be constants, ``RandomParameter`` or
``DeterministicParameter`` nodes; they are discovered and registered in the
model graph exactly like kernel parameters.

Batched form
------------
As for kernels (see :mod:`BayesianDLL.GP._kernels`), every built-in mean is
implemented once in batched form:

* inputs have shape ``(B, N, D)`` (a 2-D ``(N, D)`` or 1-D ``(N,)`` input is
  shared by the batch and treated as ``B = 1``);
* ``parameter_values`` maps the name of every named parameter to a tensor
  with a leading batch dimension ``(B, *parameter_shape)``;
* leading dimensions of size ``1`` broadcast against ``B``.

The batched methods are :meth:`MeanFunction.batch_evaluate` (``(B, N)``),
:meth:`MeanFunction.batch_derivatives` (``{name: (B, *parameter_shape, N)}``)
and :meth:`MeanFunction.batch_input_vjp` (``(B, N, D)``). The ordinary
methods (``mean(x)``, ``derivatives``, ``input_vjp``) are the ``B = 1`` case.
A user mean function that only overrides the unbatched ``evaluate`` (and
``derivatives`` / ``input_vjp``) still works: the batched defaults loop over
the batch.
"""

from __future__ import annotations

import numbers
from abc import ABC

import torch

from ._kernels import (
    _batch_inputs,
    _batch_size,
    _bparam,
    _bscalar,
    _is_named,
    _parameter_value,
    _slice_batch,
    _unique_named,
)


class MeanFunction(ABC):
    """Base class for GP mean functions ``m(X)``.

    Subclasses implement the batched methods (``batch_evaluate`` and, where
    available, ``batch_derivatives`` and ``batch_input_vjp``; see the module
    docstring) or the unbatched ``evaluate`` / ``derivatives`` /
    ``input_vjp``. Mean functions are added with ``+`` (:class:`SumMean`); a
    number in a sum is a :class:`ConstantMean`.
    """

    def __init__(self, *parameters):
        self._parameters = _unique_named(parameters)

    @property
    def parameters(self):
        """The named model nodes among the mean's parameters (each once)."""
        return self._parameters

    # -- unbatched API (the B = 1 case) ----------------------------------
    def _single_values(self, parameter_values):
        values = {}
        for parameter in self.parameters:
            values[parameter.name] = torch.as_tensor(_parameter_value(parameter, parameter_values)).unsqueeze(0)
        return values

    def evaluate(self, x, parameter_values=None):
        """Mean ``(N,)`` at inputs ``x`` of shape ``(N, D)`` or ``(N,)``.

        ``parameter_values`` optionally maps parameter names to values that
        replace the nodes' current values (e.g. a posterior draw).
        """
        if type(self).batch_evaluate is MeanFunction.batch_evaluate:
            raise NotImplementedError(f"{type(self).__name__} must implement evaluate or batch_evaluate.")
        return self.batch_evaluate(_batch_inputs(x), self._single_values(parameter_values))[0]

    def derivatives(self, x, parameter_values=None):
        """``{name: (*parameter_shape, N)}``: derivatives of the mean w.r.t. its named parameters."""
        if type(self).batch_derivatives is MeanFunction.batch_derivatives:
            return {}
        derivatives = self.batch_derivatives(_batch_inputs(x), self._single_values(parameter_values))
        return {name: value[0] for name, value in derivatives.items()}

    def input_vjp(self, x, cotangent, parameter_values=None):
        r"""``∂⟨g, m(X)⟩ / ∂X`` with shape ``(N, D)`` for a cotangent ``g`` of shape ``(N,)``.

        Row ``i`` of ``X`` only changes ``m_i``, so
        ``out[i, d] = g_i ∂m(x_i)/∂x_id``.
        """
        if type(self).batch_input_vjp is MeanFunction.batch_input_vjp:
            raise NotImplementedError(f"{type(self).__name__} does not implement input_vjp.")
        return self.batch_input_vjp(
            _batch_inputs(x), torch.as_tensor(cotangent).unsqueeze(0), self._single_values(parameter_values)
        )[0]

    def __call__(self, x, parameter_values=None):
        return self.evaluate(x, parameter_values=parameter_values)

    # -- batched API; the defaults loop over the batch ---------------------
    def _loop(self, function, x, parameter_values, *extra):
        size = _batch_size(x, *extra, *parameter_values.values())
        results = []
        for index in range(size):
            values = {name: _slice_batch(v, index) for name, v in parameter_values.items()}
            results.append(function(_slice_batch(x, index), values, *(_slice_batch(e, index) for e in extra)))
        return results

    def batch_evaluate(self, x, parameter_values):
        """``(B, N)`` mean of batched inputs ``(B, N, D)``."""
        if type(self).evaluate is MeanFunction.evaluate:
            raise NotImplementedError(f"{type(self).__name__} must implement evaluate or batch_evaluate.")
        return torch.stack(self._loop(lambda a, v: self.evaluate(a, parameter_values=v), x, parameter_values))

    def batch_derivatives(self, x, parameter_values):
        """``{name: (B, *parameter_shape, N)}``."""
        if type(self).derivatives is MeanFunction.derivatives:
            return {}
        results = self._loop(lambda a, v: self.derivatives(a, parameter_values=v), x, parameter_values)
        return {name: torch.stack([r[name] for r in results]) for name in results[0]}

    def batch_input_vjp(self, x, cotangent, parameter_values):
        """``(B, N, D)``: ``∂⟨g_b, m_b(x_b)⟩ / ∂x_b`` for a cotangent ``(B, N)``."""
        if type(self).input_vjp is MeanFunction.input_vjp:
            raise NotImplementedError(f"{type(self).__name__} does not implement input_vjp.")
        return torch.stack(self._loop(
            lambda a, v, g: self.input_vjp(a, g, parameter_values=v), x, parameter_values, cotangent
        ))

    def __add__(self, other):
        return SumMean(self, other)

    def __radd__(self, other):
        return SumMean(other, self)


def _zeros_like_inputs(x, *others):
    return torch.zeros((_batch_size(x, *others), *x.shape[1:]), dtype=x.dtype, device=x.device)


class ZeroMean(MeanFunction):
    """``m(x) = 0``: the default mean of every GP. It has no parameters and adds no work to the gradients."""

    def __init__(self):
        super().__init__()

    def batch_evaluate(self, x, parameter_values):
        return torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)

    def batch_derivatives(self, x, parameter_values):
        return {}

    def batch_input_vjp(self, x, cotangent, parameter_values):
        return _zeros_like_inputs(x, cotangent)


class ConstantMean(MeanFunction):
    """``m(x) = c`` with a scalar ``value`` ``c``: a number, a 0-d tensor or a named scalar node.

    ``∂m/∂c = 1``; the input VJP is zero.
    """

    def __init__(self, value=0.0):
        _check_scalar(value, "ConstantMean value")
        super().__init__(value)
        self.value = value

    def batch_evaluate(self, x, parameter_values):
        value = _bscalar(self.value, parameter_values, x, "value", "ConstantMean")
        return value.reshape(-1, 1).expand(value.shape[0], x.shape[1])

    def batch_derivatives(self, x, parameter_values):
        if not _is_named(self.value):
            return {}
        return {self.value.name: torch.ones((1, 1, x.shape[1]), dtype=x.dtype, device=x.device)}

    def batch_input_vjp(self, x, cotangent, parameter_values):
        return _zeros_like_inputs(x, cotangent)


class LinearMean(MeanFunction):
    r"""``m(x) = x·w + b``: a linear trend with ``coefficients`` ``w`` and an ``intercept`` ``b``.

    ``w`` has one value per input feature, shape ``(D,)``, or is a scalar
    for one-dimensional inputs; ``b`` is a scalar (default ``0.0``). Both may
    be constants or named nodes. Derivatives: ``∂m_i/∂w_d = x_id`` (shape
    ``(D, N)``), ``∂m_i/∂b = 1``; the input VJP is ``out[i, d] = g_i w_d``.

    Unlike a :class:`~BayesianDLL.GP.Linear` *kernel*, which puts a Gaussian
    prior on a random slope and integrates it out, the mean function is a
    fixed trend whose slope is a parameter of the model (learned with its
    own prior); the GP then models the deviations from that trend.
    """

    def __init__(self, coefficients, intercept=0.0):
        _check_scalar(intercept, "LinearMean intercept")
        super().__init__(coefficients, intercept)
        self.coefficients = coefficients
        self.intercept = intercept

    def _weights(self, x, parameter_values):
        value = _bparam(self.coefficients, parameter_values, x)
        flat = value.reshape(value.shape[0], -1)
        if flat.shape[1] != x.shape[-1]:
            raise ValueError(
                f"LinearMean coefficients must have one value per input feature ({x.shape[-1]}); "
                f"got {flat.shape[1]} values."
            )
        return value, flat

    def batch_evaluate(self, x, parameter_values):
        _, weights = self._weights(x, parameter_values)
        intercept = _bscalar(self.intercept, parameter_values, x, "intercept", "LinearMean")
        return (x @ weights.unsqueeze(-1)).squeeze(-1) + intercept.unsqueeze(-1)

    def batch_derivatives(self, x, parameter_values):
        derivatives = {}
        if _is_named(self.coefficients):
            value, _ = self._weights(x, parameter_values)
            per_feature = x.transpose(-2, -1)  # (B, D, N)
            derivatives[self.coefficients.name] = per_feature.reshape(x.shape[0], *value.shape[1:], x.shape[1])
        if _is_named(self.intercept):
            ones = torch.ones((1, 1, x.shape[1]), dtype=x.dtype, device=x.device)
            derivatives[self.intercept.name] = derivatives[self.intercept.name] + ones if self.intercept.name in derivatives else ones
        return derivatives

    def batch_input_vjp(self, x, cotangent, parameter_values):
        _, weights = self._weights(x, parameter_values)
        return cotangent.unsqueeze(-1) * weights.unsqueeze(-2)


class SumMean(MeanFunction):
    """``m = m_1 + m_2 + ...`` (also built with ``m1 + m2``); numbers become :class:`ConstantMean`.

    A parameter shared by several parts gets the sum of their derivatives.
    """

    def __init__(self, *means):
        self.means = tuple(as_mean_function(mean, "SumMean part") for mean in means)
        super().__init__(*(parameter for mean in self.means for parameter in mean.parameters))

    def batch_evaluate(self, x, parameter_values):
        result = None
        for mean in self.means:
            value = mean.batch_evaluate(x, parameter_values)
            result = value if result is None else result + value
        return result

    def batch_derivatives(self, x, parameter_values):
        derivatives = {}
        for mean in self.means:
            for name, derivative in mean.batch_derivatives(x, parameter_values).items():
                derivatives[name] = derivatives[name] + derivative if name in derivatives else derivative
        return derivatives

    def batch_input_vjp(self, x, cotangent, parameter_values):
        result = _zeros_like_inputs(x, cotangent)
        for mean in self.means:
            result = result + mean.batch_input_vjp(x, cotangent, parameter_values)
        return result


def _scalar_count(value):
    if _is_named(value):
        from ._process import _input_value
        value = _input_value(value)
    return torch.as_tensor(value).numel()


def _check_scalar(value, what):
    if _scalar_count(value) != 1:
        raise ValueError(f"{what} must be a scalar; got {_scalar_count(value)} values.")


def as_mean_function(mean, what="GP"):
    """Convert a ``mean=`` argument to a :class:`MeanFunction`.

    ``None`` is :class:`ZeroMean`; a number, a 0-d (or one-element) tensor or
    a named scalar node is a :class:`ConstantMean`. Anything with more than
    one value (a per-input vector of means) is rejected: write the mean as a
    function of the inputs (e.g. :class:`LinearMean` or a custom
    :class:`MeanFunction`) so that it can be evaluated at new inputs.
    """
    if mean is None:
        return ZeroMean()
    if isinstance(mean, MeanFunction):
        return mean
    if isinstance(mean, numbers.Number) or isinstance(mean, torch.Tensor) or _is_named(mean):
        count = _scalar_count(mean)
        if count != 1:
            raise ValueError(
                f"The {what} mean must be a MeanFunction or a scalar; got {count} values. A per-input vector of "
                "means is not accepted because it cannot be evaluated at new inputs: use a mean function such as "
                "ConstantMean, LinearMean or a custom BayesianDLL.GP.MeanFunction."
            )
        return ConstantMean(mean)
    raise TypeError(f"The {what} mean must be a MeanFunction, a scalar or a named scalar node; got {type(mean).__name__}.")


def _is_zero(mean):
    """Whether ``mean`` is a :class:`ZeroMean` (the fast path: nothing to evaluate or differentiate)."""
    return isinstance(mean, ZeroMean)
