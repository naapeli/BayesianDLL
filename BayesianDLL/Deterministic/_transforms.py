"""Common deterministic transformations for Bayesian models."""

from __future__ import annotations

import torch

from .._parameters import DeterministicParameter


def _require_named(value, description):
    if getattr(value, "name", None) is None:
        raise TypeError(f"{description} must be a named model parameter or Data node.")


def _is_named(value):
    return getattr(value, "name", None) is not None


def _as_tensor(value, *, dtype=None, device=None):
    if isinstance(value, torch.Tensor):
        if dtype is not None or device is not None:
            return value.to(dtype=dtype, device=device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _linear_output(inputs, coefficients, intercept):
    """Evaluate a scalar-output linear predictor."""
    inputs = _as_tensor(inputs)
    coefficients = _as_tensor(coefficients, dtype=inputs.dtype, device=inputs.device)
    intercept = _as_tensor(intercept, dtype=inputs.dtype, device=inputs.device)

    if coefficients.numel() == 1:
        return inputs * coefficients.reshape(()) + intercept

    if inputs.ndim < 2:
        raise ValueError(
            "A vector of regression coefficients requires inputs with shape "
            "(n_observations, n_features)."
        )
    coefficients = coefficients.reshape(-1)
    if inputs.shape[-1] != coefficients.numel():
        raise ValueError(
            "The number of regression coefficients must equal the input feature "
            f"count ({inputs.shape[-1]}); got {coefficients.numel()}."
        )
    return inputs @ coefficients + intercept


class Linear(DeterministicParameter):
    """A linear predictor with an optional intercept.

    For one-dimensional inputs and a scalar coefficient, this evaluates
    ``inputs * slope + intercept``. For a design matrix with shape ``(N, D)``
    and a coefficient vector with shape ``(D,)``, it evaluates
    ``inputs @ coefficients + intercept``.

    Parameters may be model nodes or constants. Named parameters are included
    in the model graph and receive analytic derivatives. ``slope`` and
    ``weights`` are accepted as readable aliases for ``coefficients``.
    """

    def __init__(
        self,
        name,
        inputs,
        coefficients=None,
        intercept=0.0,
        *,
        slope=None,
        weights=None,
    ):
        _require_named(inputs, "inputs")
        supplied_coefficients = [
            value for value in (coefficients, slope, weights) if value is not None
        ]
        if len(supplied_coefficients) != 1:
            raise TypeError(
                "Provide exactly one of coefficients, slope, or weights."
            )
        coefficients = supplied_coefficients[0]

        named_inputs = [inputs]
        for value, description in (
            (coefficients, "coefficients"),
            (intercept, "intercept"),
        ):
            if _is_named(value):
                named_inputs.append(value)
            elif not isinstance(value, (torch.Tensor, int, float, complex)):
                raise TypeError(
                    f"{description} must be a model node or a numeric tensor/value."
                )

        def forward(*values):
            value_index = 1
            coefficient_value = values[value_index] if _is_named(coefficients) else coefficients
            if _is_named(coefficients):
                value_index += 1
            intercept_value = values[value_index] if _is_named(intercept) else intercept
            return _linear_output(values[0], coefficient_value, intercept_value)

        def derivative(*values):
            inputs_value = _as_tensor(values[0])
            coefficient_value = values[1] if _is_named(coefficients) else coefficients
            coefficient_value = _as_tensor(
                coefficient_value,
                dtype=inputs_value.dtype,
                device=inputs_value.device,
            )
            output = _linear_output(
                inputs_value,
                coefficient_value,
                values[-1] if _is_named(intercept) else intercept,
            )
            derivatives = {}

            if _is_named(coefficients):
                if coefficient_value.numel() == 1:
                    derivatives[coefficients.name] = inputs_value.reshape(-1, 1)
                else:
                    derivatives[coefficients.name] = inputs_value

            if _is_named(intercept):
                if intercept.constrained_value.numel() == 1:
                    derivatives[intercept.name] = torch.ones(
                        (output.numel(), 1), dtype=output.dtype, device=output.device
                    )
                else:
                    if intercept.constrained_value.numel() != output.numel():
                        raise ValueError(
                            "A vector intercept must contain one value per model output."
                        )
                    derivatives[intercept.name] = torch.eye(
                        output.numel(), dtype=output.dtype, device=output.device
                    )
            return derivatives

        super().__init__(name, forward, derivative, named_inputs)
        self.regression_inputs = inputs
        self.coefficients = coefficients
        self.weights = coefficients
        self.slope = coefficients
        self.intercept = intercept


class Exp(DeterministicParameter):
    """Apply the elementwise exponential transformation to a model node."""

    def __init__(self, name, input):
        _require_named(input, "input")

        def forward(value):
            return torch.exp(value)

        def derivative(value):
            return {input.name: torch.exp(value)}

        super().__init__(name, forward, derivative, [input])
        self.input = input


class Log(DeterministicParameter):
    """Apply the elementwise natural logarithm to a positive model node."""

    def __init__(self, name, input):
        _require_named(input, "input")

        def forward(value):
            return torch.log(value)

        def derivative(value):
            return {input.name: value.reciprocal()}

        super().__init__(name, forward, derivative, [input])
        self.input = input


class Sigmoid(DeterministicParameter):
    """Map an unconstrained model node to the interval ``(0, 1)``."""

    def __init__(self, name, input):
        _require_named(input, "input")

        def forward(value):
            return torch.sigmoid(value)

        def derivative(value):
            probability = torch.sigmoid(value)
            return {input.name: probability * (1 - probability)}

        super().__init__(name, forward, derivative, [input])
        self.input = input
