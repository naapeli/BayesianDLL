"""Latent and exact Gaussian processes for MCMC and variational inference."""

from __future__ import annotations

import torch

from .._data import Data
from .._parameters import DeterministicParameter, RandomParameter
from ..Distributions._distributions import Distribution
from ..Distributions._resolve import resolve
from ..Distributions import Normal
from ._kernels import Kernel


def stable_cholesky(matrix, jitter=1e-6, max_tries=6):
    """Compute a Cholesky factor, increasing diagonal jitter when necessary."""
    matrix = (matrix + matrix.transpose(-2, -1)) / 2
    if not torch.isfinite(matrix).all():
        raise ValueError("GP covariance contains non-finite values.")
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    diagonal_scale = torch.diagonal(matrix, dim1=-2, dim2=-1).abs().max().clamp_min(1.0)
    current_jitter = torch.as_tensor(jitter, dtype=matrix.dtype, device=matrix.device) * diagonal_scale
    current_jitter = current_jitter.clamp_min(torch.finfo(matrix.dtype).eps * diagonal_scale)
    error = None
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(matrix + current_jitter * eye)
        except RuntimeError as exc:
            error = exc
            current_jitter = current_jitter * 10

    # A covariance assembled in finite precision can have a small negative
    # eigenvalue even when the kernel is theoretically positive semidefinite.
    # Correct the smallest eigenvalue directly as a final, differentiability-
    # free fallback for difficult MCMC proposals.
    minimum_eigenvalue = torch.linalg.eigvalsh(matrix).amin()
    correction = (-minimum_eigenvalue + torch.finfo(matrix.dtype).eps * diagonal_scale).clamp_min(current_jitter)
    try:
        return torch.linalg.cholesky(matrix + correction * eye)
    except RuntimeError as exc:
        error = exc
    raise RuntimeError(
        f"Could not compute a positive-definite Cholesky factor after {max_tries} jitter attempts."
    ) from error


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _input_value(value):
    if isinstance(value, Data):
        return value.value
    if hasattr(value, "constrained_value"):
        return value.constrained_value
    return _as_tensor(value)


def _named_model_inputs(values):
    return {value.name: tensor for value, tensor in values if hasattr(value, "name")}


def _feature_count(x):
    x = _input_value(x)
    return x.shape[0] if x.ndim > 0 else 1


def _mean_value(mean, x):
    if hasattr(mean, "name"):
        mean = _input_value(mean)
    mean = _as_tensor(mean).to(dtype=x.dtype, device=x.device)
    if mean.ndim == 0:
        return mean.expand(x.shape[0])
    return mean


def _resolved_value(value, dtype, device, parameter_values=None):
    if parameter_values is not None and hasattr(value, "name") and value.name in parameter_values:
        value = parameter_values[value.name]
    else:
        value = resolve(value)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _cholesky_output_derivative(factor, covariance_derivative, latent):
    """Differentiate ``chol(K) @ latent`` from a covariance differential.

    For ``K = L Lᵀ`` and ``A = L⁻¹ dK L⁻ᵀ``:

    ``dL = L @ (tril(A, -1) + 0.5 * diag(diag(A)))``.

    The leading dimensions of ``covariance_derivative`` are parameter
    dimensions. They are returned after the output dimension so they match
    the Jacobian convention used by ``Model.joint_grad_log_prob``.
    """
    if covariance_derivative.ndim == 2:
        covariance_derivative = covariance_derivative.unsqueeze(0)
    parameter_shape = covariance_derivative.shape[:-2]
    flat_derivatives = covariance_derivative.reshape(-1, factor.shape[-1], factor.shape[-1])
    output_derivatives = []
    for derivative in flat_derivatives:
        left_solve = torch.linalg.solve_triangular(factor, derivative, upper=False)
        normalized = torch.linalg.solve_triangular(
            factor,
            left_solve.transpose(-2, -1),
            upper=False,
        ).transpose(-2, -1)
        lower = torch.tril(normalized, diagonal=-1)
        diagonal = 0.5 * torch.diag(torch.diagonal(normalized))
        factor_derivative = factor @ (lower + diagonal)
        output_derivatives.append(factor_derivative @ latent)
    return torch.stack(output_derivatives, dim=-1).reshape(latent.numel(), *parameter_shape)


class LatentGP(DeterministicParameter):
    """An explicit whitened latent GP function node.

    The object is a ``DeterministicParameter`` whose value is
    ``mean(inputs) + L @ z``. If ``latent`` is omitted, a standard-normal
    latent vector named ``<name>_white`` is created automatically. This is a
    non-centred parameterization intended for NUTS and variational inference.

    ``inputs`` should normally be a ``Data`` object containing shape ``(N, D)``
    or ``(N,)``. The likelihood can then use the returned function directly,
    for example ``ObservedParameter("y", Normal(f, noise), y)``.
    """

    def __init__(
        self,
        name,
        inputs,
        kernel: Kernel,
        latent=None,
        mean=0.0,
        jitter=1e-6,
        latent_name=None,
    ):
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not hasattr(inputs, "name"):
            raise TypeError("inputs must be a named Data or model parameter node.")

        n = _feature_count(inputs)
        if latent is None:
            latent = RandomParameter(latent_name or f"{name}_white", Normal(0.0, 1.0), shape=n)
        if not hasattr(latent, "name"):
            raise TypeError("latent must be a named model parameter node.")
        if latent.constrained_value.numel() != n:
            raise ValueError(
                f"latent must contain one value per training input ({n}); "
                f"got {latent.constrained_value.numel()}."
            )

        kernel_parameters = list(kernel.parameters)
        model_inputs = [inputs, latent, *kernel_parameters]
        mean_input = mean if hasattr(mean, "name") else None
        if mean_input is not None:
            model_inputs.append(mean_input)

        def forward(*values):
            x = values[0]
            z = values[1].reshape(-1)
            if not isinstance(x, torch.Tensor):
                x = _as_tensor(x)
            if x.dtype != z.dtype:
                x = x.to(dtype=z.dtype)
            kernel_values = _named_model_inputs(zip(model_inputs[2:], values[2:]))
            covariance = kernel(x, x, parameter_values=kernel_values)
            factor = stable_cholesky(covariance, jitter=jitter)
            mean_value = values[2 + len(kernel_parameters)] if mean_input is not None else mean
            return _mean_value(mean_value, x) + factor @ z

        def derivative(*values):
            latent_value = values[1].reshape(-1)
            x = values[0]
            if not isinstance(x, torch.Tensor):
                x = _as_tensor(x)
            if x.dtype != latent_value.dtype:
                x = x.to(dtype=latent_value.dtype)
            kernel_values = _named_model_inputs(zip(model_inputs[2:], values[2:]))
            covariance = kernel(x, x, parameter_values=kernel_values)
            factor = stable_cholesky(covariance, jitter=jitter)
            derivatives = {latent.name: factor}

            covariance_derivatives = kernel.derivatives(x, x, parameter_values=kernel_values)
            for parameter in kernel_parameters:
                if parameter.name in covariance_derivatives:
                    derivatives[parameter.name] = _cholesky_output_derivative(
                        factor,
                        covariance_derivatives[parameter.name],
                        latent_value,
                    )

            if mean_input is not None:
                mean_value = values[2 + len(kernel_parameters)]
                mean_tensor = _as_tensor(mean_value)
                if mean_tensor.ndim == 0 or mean_tensor.numel() == 1:
                    derivatives[mean_input.name] = torch.ones(
                        (x.shape[0], 1), dtype=x.dtype, device=x.device
                    )
                else:
                    derivatives[mean_input.name] = torch.eye(
                        x.shape[0], dtype=x.dtype, device=x.device
                    )
            return derivatives

        super().__init__(name, forward, derivative, model_inputs)
        self.gp_inputs = inputs
        self.gp_latent = latent
        # Public aliases make the latent variable easy to inspect when it was
        # supplied explicitly, while retaining the more specific GP metadata.
        self.latent = latent
        self.latent_name = latent.name
        self.gp_kernel = kernel
        self.gp_mean = mean
        self.gp_jitter = jitter


# Backwards-compatible name retained for existing models and notebooks.
GaussianProcess = LatentGP


class ExactGP(Distribution):
    """Gaussian-process marginal likelihood with the latent function integrated out.

    If ``f ~ Normal(mean, K)`` and ``y | f ~ Normal(f, noise_variance * I)``,
    this distribution evaluates the exact marginal likelihood

    ``y ~ Normal(mean, K + noise_variance * I)``.

    The object is used as the distribution of an ``ObservedParameter``::

        gp = ExactGP("function", inputs, kernel, noise_variance=0.05)
        ObservedParameter("observations", gp, y)

    Unlike :class:`LatentGP`, this does not create a latent random
    variable. Kernel and noise hyperparameters remain ordinary model
    parameters and can be sampled with MCMC.
    """

    def __init__(self, name, inputs, kernel: Kernel, noise_variance=1e-6, mean=0.0, jitter=1e-6):
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not hasattr(inputs, "name"):
            raise TypeError("inputs must be a named Data or model parameter node.")
        self.name = name
        self.inputs = inputs
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.mean = mean
        self.jitter = jitter

        self.add_dependency(inputs)
        for parameter in kernel.parameters:
            self.add_dependency(parameter)
        self.add_dependency(noise_variance)
        self.add_dependency(mean)

    @property
    def event_shape(self):
        return (_feature_count(self.inputs),)

    @property
    def batch_shape(self):
        return torch.Size()

    def _components(self, observed, parameter_values=None):
        x = _input_value(self.inputs)
        if x.ndim == 0:
            x = x.reshape(1)
        x = torch.as_tensor(x)
        observed = torch.as_tensor(observed, dtype=x.dtype, device=x.device).reshape(-1)
        n = _feature_count(x)
        if observed.numel() != n:
            raise ValueError(
                f"Exact GP observations must contain one value per input ({n}); "
                f"got {observed.numel()} values."
            )

        kernel_values = parameter_values
        covariance = self.kernel(x, x, parameter_values=kernel_values)
        covariance = (covariance + covariance.transpose(-2, -1)) / 2
        noise = _resolved_value(
            self.noise_variance,
            dtype=x.dtype,
            device=x.device,
            parameter_values=parameter_values,
        ).reshape(-1)
        if noise.numel() != 1:
            raise ValueError("Exact GP noise_variance must be scalar.")

        mean = _resolved_value(
            self.mean,
            dtype=x.dtype,
            device=x.device,
            parameter_values=parameter_values,
        )
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(n)
        else:
            mean = mean.reshape(-1)
            if mean.numel() != n:
                raise ValueError(
                    f"Exact GP mean must be scalar or contain one value per input ({n}); "
                    f"got {mean.numel()} values."
                )

        covariance = covariance + noise[0] * torch.eye(n, dtype=x.dtype, device=x.device)
        factor = stable_cholesky(covariance, jitter=self.jitter)
        residual = observed - mean
        alpha = torch.cholesky_solve(residual.reshape(-1, 1), factor).reshape(-1)
        log_det = 2 * torch.log(torch.diagonal(factor)).sum()
        log_probability = -0.5 * (
            residual.dot(alpha) + log_det + n * torch.log(torch.as_tensor(2 * torch.pi, dtype=x.dtype))
        )
        return x, covariance, factor, residual, alpha, log_probability

    def _named_parameter_values(self):
        values = {}
        parameters = [*self.kernel.parameters, self.noise_variance, self.mean]
        for parameter in parameters:
            if hasattr(parameter, "name"):
                values[parameter.name] = resolve(parameter)
        return values

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))

    def log_pdf(self, x):
        return self._components(x)[-1]

    def log_pdf_grad(self, x):
        return -self._components(x)[4]

    def log_pdf_param_grads(self, x):
        x_values, covariance, factor, _, alpha, _ = self._components(x)
        n = x_values.shape[0]
        inverse = torch.cholesky_solve(
            torch.eye(n, dtype=x_values.dtype, device=x_values.device),
            factor,
        )
        sensitivity = 0.5 * (alpha[:, None] * alpha[None, :] - inverse)
        parameter_values = self._named_parameter_values()
        gradients = {}

        for name, derivative in self.kernel.derivatives(
            x_values,
            x_values,
            parameter_values=parameter_values,
        ).items():
            if derivative.ndim == 2:
                derivative = derivative.unsqueeze(0)
            gradients[name] = (sensitivity * derivative).sum(dim=(-2, -1))

        if hasattr(self.noise_variance, "name"):
            gradients[self.noise_variance.name] = 0.5 * (
                alpha.square().sum() - torch.diagonal(inverse).sum()
            ).reshape(resolve(self.noise_variance).shape)

        if hasattr(self.mean, "name"):
            gradients[self.mean.name] = alpha

        return gradients

    def sample(self, n_samples=1):
        x = _input_value(self.inputs)
        zero = torch.zeros(_feature_count(x), dtype=x.dtype, device=x.device)
        _, _, factor, _, _, _ = self._components(zero)
        mean = _resolved_value(self.mean, x.dtype, x.device)
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(zero.numel())
        return mean + torch.randn(
            (n_samples, zero.numel()), dtype=zero.dtype, device=zero.device
        ) @ factor.transpose(-2, -1)


# Backwards-compatible name retained for existing exact-GP models.
ExactGaussianProcess = ExactGP


def gp_predictive(gp, trace, inputs, n_samples=1, observation_noise=None):
    """Sample latent GP values at new inputs from posterior latent traces.

    ``trace`` may be a ``SamplingResult`` or a mapping containing the GP's
    deterministic trace and kernel-parameter traces. The result has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not hasattr(gp, "gp_kernel"):
        raise TypeError("gp must be a LatentGP function returned by BayesianDLL.GP.LatentGP.")
    if not hasattr(trace, "__getitem__"):
        raise TypeError("trace must be a SamplingResult or mapping-like object.")

    function_trace = trace[gp.name]
    latent_draws = function_trace.reshape(-1, function_trace.shape[-1])
    training_values = _input_value(gp.gp_inputs).to(dtype=latent_draws.dtype, device=latent_draws.device)
    new_values = _input_value(inputs).to(dtype=latent_draws.dtype, device=latent_draws.device)
    total_draws = latent_draws.shape[0]
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    outputs = []
    for draw_index in range(total_draws):
        parameter_values = {}
        for parameter in gp.gp_kernel.parameters:
            if parameter.name in trace:
                values = trace[parameter.name].reshape(-1, *trace[parameter.name].shape[2:])
                parameter_values[parameter.name] = values[draw_index]

        covariance = gp.gp_kernel(training_values, training_values, parameter_values=parameter_values)
        cross_covariance = gp.gp_kernel(training_values, new_values, parameter_values=parameter_values)
        new_covariance = gp.gp_kernel(new_values, new_values, parameter_values=parameter_values)
        factor = stable_cholesky(covariance, jitter=gp.gp_jitter)
        residual = latent_draws[draw_index] - _mean_value(gp.gp_mean, training_values)
        alpha = torch.cholesky_solve(residual.reshape(-1, 1), factor).reshape(-1)
        predictive_mean = _mean_value(gp.gp_mean, new_values) + cross_covariance.transpose(-2, -1) @ alpha
        conditional_covariance = new_covariance - cross_covariance.transpose(-2, -1) @ torch.cholesky_solve(cross_covariance, factor)
        conditional_covariance = (conditional_covariance + conditional_covariance.transpose(-2, -1)) / 2
        if observation_noise is not None:
            observation_noise = _input_value(observation_noise).to(
                dtype=conditional_covariance.dtype,
                device=conditional_covariance.device,
            )
            conditional_covariance = conditional_covariance + observation_noise * torch.eye(
                conditional_covariance.shape[-1], dtype=conditional_covariance.dtype, device=conditional_covariance.device
            )
        prediction_factor = stable_cholesky(conditional_covariance, jitter=gp.gp_jitter)
        noise = torch.randn(
            (n_samples, predictive_mean.numel()),
            dtype=predictive_mean.dtype,
            device=predictive_mean.device,
        )
        outputs.append(predictive_mean + noise @ prediction_factor.transpose(-2, -1))
    return torch.stack(outputs)


def _trace_draws(trace, parameter, n_draws, dtype, device):
    if hasattr(parameter, "name"):
        if parameter.name not in trace:
            raise KeyError(
                f"Trace does not contain the model parameter '{parameter.name}'."
            )
        values = trace[parameter.name]
        if values.ndim >= 2:
            values = values.reshape(-1, *values.shape[2:])
        else:
            values = values.reshape(-1)
        return values.to(dtype=dtype, device=device)

    value = _resolved_value(parameter, dtype=dtype, device=device).reshape(1, -1)
    return value.expand(n_draws, *value.shape[1:])


def exact_gp_predictive(
    gp,
    trace,
    observed_values,
    inputs,
    n_samples=1,
    observation_noise=None,
):
    """Sample exact-GP predictions conditional on observed training values.

    ``trace`` contains only hyperparameter draws; the training function is
    integrated out during inference and is sampled here from its conditional
    Gaussian distribution. The returned tensor has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, ExactGP):
        raise TypeError(
            "gp must be an ExactGP distribution returned by BayesianDLL.GP.ExactGP."
        )
    if not hasattr(trace, "__getitem__") or not hasattr(trace, "keys"):
        raise TypeError("trace must be a mapping-like sampling result.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    training_values = _input_value(gp.inputs)
    new_values = _input_value(inputs)
    if training_values.ndim == 0:
        training_values = training_values.reshape(1)
    if new_values.ndim == 0:
        new_values = new_values.reshape(1)
    observed_values = torch.as_tensor(
        _input_value(observed_values),
        dtype=training_values.dtype,
        device=training_values.device,
    ).reshape(-1)
    n_training = _feature_count(training_values)
    if observed_values.numel() != n_training:
        raise ValueError(
            f"observed_values must contain one value per training input ({n_training}); "
            f"got {observed_values.numel()} values."
        )

    first_value = next(iter(trace.values()))
    n_draws = first_value.shape[0] * first_value.shape[1] if first_value.ndim >= 2 else first_value.shape[0]
    dtype = training_values.dtype
    device = training_values.device
    parameter_draws = {
        parameter.name: _trace_draws(trace, parameter, n_draws, dtype, device)
        for parameter in gp.kernel.parameters
        if hasattr(parameter, "name")
    }
    noise_draws = _trace_draws(trace, gp.noise_variance, n_draws, dtype, device)
    mean_draws = _trace_draws(trace, gp.mean, n_draws, dtype, device)

    outputs = []
    for draw_index in range(n_draws):
        kernel_values = {
            name: values[draw_index]
            for name, values in parameter_draws.items()
        }
        covariance = gp.kernel(
            training_values,
            training_values,
            parameter_values=kernel_values,
        )
        cross_covariance = gp.kernel(
            training_values,
            new_values,
            parameter_values=kernel_values,
        )
        new_covariance = gp.kernel(
            new_values,
            new_values,
            parameter_values=kernel_values,
        )
        noise = noise_draws[draw_index].reshape(-1)
        if noise.numel() != 1:
            raise ValueError("Exact GP noise_variance must be scalar.")

        factor = stable_cholesky(
            covariance + noise[0] * torch.eye(n_training, dtype=dtype, device=device),
            jitter=gp.jitter,
        )
        mean_train = mean_draws[draw_index].reshape(-1)
        if mean_train.numel() == 1:
            mean_train = mean_train.expand(n_training)
        mean_new = mean_train.new_zeros(new_values.shape[0])
        if mean_draws[draw_index].numel() == 1:
            mean_new = mean_draws[draw_index].reshape(()).expand(new_values.shape[0])
        elif mean_draws[draw_index].numel() == new_values.shape[0]:
            mean_new = mean_draws[draw_index].reshape(-1)
        else:
            raise ValueError("Exact GP mean must be scalar or match prediction inputs.")

        alpha = torch.cholesky_solve(
            (observed_values - mean_train).reshape(-1, 1),
            factor,
        ).reshape(-1)
        predictive_mean = mean_new + cross_covariance.transpose(-2, -1) @ alpha
        predictive_covariance = new_covariance - cross_covariance.transpose(-2, -1) @ torch.cholesky_solve(
            cross_covariance,
            factor,
        )
        if observation_noise is not None:
            predictive_covariance = predictive_covariance + _resolved_value(
                observation_noise,
                dtype=dtype,
                device=device,
            ).reshape(-1)[0] * torch.eye(new_values.shape[0], dtype=dtype, device=device)
        predictive_covariance = (predictive_covariance + predictive_covariance.transpose(-2, -1)) / 2
        prediction_factor = stable_cholesky(predictive_covariance, jitter=gp.jitter)
        outputs.append(
            predictive_mean
            + torch.randn((n_samples, new_values.shape[0]), dtype=dtype, device=device)
            @ prediction_factor.transpose(-2, -1)
        )

    return torch.stack(outputs)
