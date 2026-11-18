"""Common deterministic transformations for Bayesian models."""

from __future__ import annotations

import torch

from .._parameters import DeterministicParameter, RandomParameter


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


def _is_scalar_slope(input_ndim, n_features, n_coefficients):
    """Whether a linear predictor multiplies its inputs by a scalar slope.

    A single coefficient is a scalar slope unless the inputs are a design
    matrix with one feature (shape ``(..., N, 1)``): then it is a coefficient
    vector of length one and the output drops the feature dimension, exactly
    as for ``D > 1`` features.
    """
    return n_coefficients == 1 and not (input_ndim >= 2 and n_features == 1)


def _linear_output(inputs, coefficients, intercept):
    """Evaluate a scalar-output linear predictor."""
    inputs = _as_tensor(inputs)
    coefficients = _as_tensor(coefficients, dtype=inputs.dtype, device=inputs.device)
    intercept = _as_tensor(intercept, dtype=inputs.dtype, device=inputs.device)

    n_features = inputs.shape[-1] if inputs.ndim else 1
    if _is_scalar_slope(inputs.ndim, n_features, coefficients.numel()):
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
    ``inputs @ coefficients + intercept`` with shape ``(N,)``; this includes
    a single feature, ``D = 1``, where the coefficient may also be a scalar.

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

        def vjp(cotangent, *values):
            inputs_value = _as_tensor(values[0])
            value_index = 1
            if _is_named(coefficients):
                coefficient_value = values[value_index]
                value_index += 1
            else:
                coefficient_value = coefficients
            coefficient_value = _as_tensor(coefficient_value, device=inputs_value.device)
            grads = {}
            n_features = inputs_value.shape[-1] if inputs_value.ndim else 1
            if _is_scalar_slope(inputs_value.ndim, n_features, coefficient_value.numel()):
                if _is_named(coefficients):
                    grads[coefficients.name] = (cotangent * inputs_value).sum().reshape(coefficient_value.shape)
                if isinstance(inputs, (RandomParameter, DeterministicParameter)):
                    grads[inputs.name] = cotangent * coefficient_value.reshape(())
            else:
                flat = coefficient_value.reshape(-1)
                if _is_named(coefficients):
                    grads[coefficients.name] = (
                        (cotangent.unsqueeze(-1) * inputs_value).reshape(-1, flat.numel()).sum(0).reshape(coefficient_value.shape)
                    )
                if isinstance(inputs, (RandomParameter, DeterministicParameter)):
                    grads[inputs.name] = cotangent.unsqueeze(-1) * flat
            if _is_named(intercept):
                grads[intercept.name] = cotangent
            return grads

        super().__init__(name, forward, derivative, named_inputs, vjp_func=vjp)
        self.regression_inputs = inputs
        self.coefficients = coefficients
        self.weights = coefficients
        self.slope = coefficients
        self.intercept = intercept

    # ------------------------------------------------------------ batching
    def _batched_parts(self, input_values):
        flags = self._batched_flags()
        inputs = _as_tensor(input_values[0])
        if not inputs.is_floating_point():
            inputs = inputs.to(torch.get_default_dtype())
        index = 1
        if _is_named(self.coefficients):
            coefficients, coefficients_batched = input_values[index], flags[index]
            index += 1
        else:
            coefficients, coefficients_batched = _as_tensor(self.coefficients, dtype=inputs.dtype, device=inputs.device), False
        if _is_named(self.intercept):
            intercept, intercept_batched = input_values[index], flags[index]
        else:
            intercept, intercept_batched = _as_tensor(self.intercept, dtype=inputs.dtype, device=inputs.device), False
        n_chains = self.owner_model.n_chains
        per_chain = coefficients[0].numel() if coefficients_batched else coefficients.numel()
        input_ndim = inputs.ndim - 1 if flags[0] else inputs.ndim
        scalar = _is_scalar_slope(input_ndim, inputs.shape[-1] if input_ndim else 1, per_chain)
        if coefficients_batched:
            coefficients = coefficients.reshape(n_chains) if scalar else coefficients.reshape(n_chains, per_chain)
        else:
            coefficients = coefficients.reshape(()) if scalar else coefficients.reshape(per_chain)
        return (inputs, flags[0]), (coefficients, coefficients_batched), (intercept, intercept_batched), scalar

    def _batched_terms(self, input_values):
        (x, fx), (w, fw), (b, fb), scalar = self._batched_parts(input_values)
        if scalar:
            (xa, wa), _ = self._align_batched([x, w], [fx, fw])
            product = xa * wa
        else:
            (xa, wa), _ = self._align_batched([x, w], [fx, fw], [1, 1])
            product = (xa * wa).sum(-1)
        (pa, ba), _ = self._align_batched([product, b], [fx or fw, fb])
        return xa, wa, pa + ba, scalar

    def batched_forward(self, input_values):
        return self._to_batched_output(self._batched_terms(input_values)[2])

    def batched_vjp(self, cotangent, input_values):
        from .._batching import pad_to
        xa, wa, out, scalar = self._batched_terms(input_values)
        g = pad_to(cotangent, out.ndim)
        grads = {}
        if scalar:
            if _is_named(self.coefficients):
                grads[self.coefficients.name] = g * xa
            if isinstance(self.regression_inputs, (RandomParameter, DeterministicParameter)):
                grads[self.regression_inputs.name] = g * wa
        else:
            if _is_named(self.coefficients):
                grads[self.coefficients.name] = g.unsqueeze(-1) * xa
            if isinstance(self.regression_inputs, (RandomParameter, DeterministicParameter)):
                grads[self.regression_inputs.name] = g.unsqueeze(-1) * wa
        if _is_named(self.intercept):
            grads[self.intercept.name] = g
        return grads


class Exp(DeterministicParameter):
    """Apply the elementwise exponential transformation to a model node."""

    def __init__(self, name, input):
        _require_named(input, "input")

        def forward(value):
            return torch.exp(value)

        def derivative(value):
            return {input.name: torch.exp(value)}

        def vjp(cotangent, value):
            return {input.name: cotangent * derivative(value)[input.name]}

        super().__init__(name, forward, derivative, [input], vjp_func=vjp)
        self.input = input


class Log(DeterministicParameter):
    """Apply the elementwise natural logarithm to a positive model node."""

    def __init__(self, name, input):
        _require_named(input, "input")

        def forward(value):
            return torch.log(value)

        def derivative(value):
            return {input.name: value.reciprocal()}

        def vjp(cotangent, value):
            return {input.name: cotangent * derivative(value)[input.name]}

        super().__init__(name, forward, derivative, [input], vjp_func=vjp)
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

        def vjp(cotangent, value):
            return {input.name: cotangent * derivative(value)[input.name]}

        super().__init__(name, forward, derivative, [input], vjp_func=vjp)
        self.input = input
