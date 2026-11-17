"""Latent and exact Gaussian processes for MCMC and variational inference."""

from __future__ import annotations

import math

import torch

from .._data import Data
from .._parameters import DeterministicParameter, RandomParameter
from ..Distributions._distributions import Distribution
from ..Distributions._resolve import resolve
from ..Distributions import Normal
from ._kernels import Kernel, _is_named


def stable_cholesky(matrix, jitter=1e-6, max_tries=6, return_jitter=False):
    """Compute a Cholesky factor, increasing diagonal jitter when necessary.

    The jitter added to the diagonal is ``jitter * max(1, max|diag(matrix)|)``,
    multiplied by 10 for every failed attempt. With ``return_jitter=True`` the
    result is ``(factor, jitter_info)``, where ``jitter_info`` describes the
    added jitter so that derivatives can account for its dependence on the
    covariance (see :func:`_jitter_derivative`).
    """
    matrix = (matrix + matrix.transpose(-2, -1)) / 2
    if not torch.isfinite(matrix).all():
        raise ValueError("GP covariance contains non-finite values.")
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1).abs()
    diagonal_scale = diagonal.max().clamp_min(1.0)
    # The jitter is proportional to the largest diagonal entry only when that
    # entry exceeds one; otherwise the scale is the constant 1.
    scale_index = int(diagonal.argmax()) if bool(diagonal.max() > 1.0) else None
    current_jitter = torch.as_tensor(jitter, dtype=matrix.dtype, device=matrix.device) * diagonal_scale
    current_jitter = current_jitter.clamp_min(torch.finfo(matrix.dtype).eps * diagonal_scale)
    error = None
    for _ in range(max_tries):
        factor, info = torch.linalg.cholesky_ex(matrix + current_jitter * eye)
        if int(info) == 0:
            if return_jitter:
                return factor, (current_jitter, diagonal_scale, scale_index)
            return factor
        error = RuntimeError(f"Cholesky decomposition failed (info={int(info)}).")
        current_jitter = current_jitter * 10

    # A covariance assembled in finite precision can have a small negative
    # eigenvalue even when the kernel is theoretically positive semidefinite.
    # Correct the smallest eigenvalue directly as a final, differentiability-
    # free fallback for difficult MCMC proposals.
    minimum_eigenvalue = torch.linalg.eigvalsh(matrix).amin()
    correction = (-minimum_eigenvalue + torch.finfo(matrix.dtype).eps * diagonal_scale).clamp_min(current_jitter)
    try:
        factor = torch.linalg.cholesky(matrix + correction * eye)
        if return_jitter:
            # The eigenvalue correction is treated as a constant.
            return factor, (correction, diagonal_scale, None)
        return factor
    except RuntimeError as exc:
        error = exc
    raise RuntimeError(
        f"Could not compute a positive-definite Cholesky factor after {max_tries} jitter attempts."
    ) from error


def _jitter_derivative(jitter_info, covariance_derivative):
    """Add the derivative of the scale-dependent jitter to ``dK``.

    ``stable_cholesky`` factorizes ``K + c * max(1, max|diag K|) * I``. When the
    largest diagonal entry exceeds one, the jitter changes with the covariance
    and its derivative ``c * d(K_ii)`` (``i`` being the largest diagonal entry)
    must be included for the gradient to match the factorized matrix.
    """
    jitter, scale, index = jitter_info
    if index is None:
        return covariance_derivative
    n = covariance_derivative.shape[-1]
    eye = torch.eye(n, dtype=covariance_derivative.dtype, device=covariance_derivative.device)
    scale_derivative = covariance_derivative[..., index, index]
    return covariance_derivative + (jitter / scale) * scale_derivative[..., None, None] * eye


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
    return {value.name: tensor for value, tensor in values if _is_named(value)}


def _feature_count(x):
    x = _input_value(x)
    return x.shape[0] if x.ndim > 0 else 1


def _mean_value(mean, x):
    if _is_named(mean):
        mean = _input_value(mean)
    mean = _as_tensor(mean).to(dtype=x.dtype, device=x.device)
    if mean.ndim == 0:
        return mean.expand(x.shape[0])
    return mean


def _resolved_value(value, dtype, device, parameter_values=None):
    if parameter_values is not None and _is_named(value) and value.name in parameter_values:
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
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")

        n = _feature_count(inputs)
        if latent is None:
            latent = RandomParameter(latent_name or f"{name}_white", Normal(0.0, 1.0), shape=n)
        if not _is_named(latent):
            raise TypeError("latent must be a named model parameter node.")
        if latent.constrained_value.numel() != n:
            raise ValueError(
                f"latent must contain one value per training input ({n}); "
                f"got {latent.constrained_value.numel()}."
            )

        kernel_parameters = list(kernel.parameters)
        model_inputs = [inputs, latent, *kernel_parameters]
        mean_input = mean if _is_named(mean) else None
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
            factor, jitter_info = stable_cholesky(covariance, jitter=jitter, return_jitter=True)
            derivatives = {latent.name: factor}

            covariance_derivatives = kernel.derivatives(x, x, parameter_values=kernel_values)
            for parameter in kernel_parameters:
                if parameter.name in covariance_derivatives:
                    covariance_derivative = covariance_derivatives[parameter.name]
                    if covariance_derivative.ndim == 2:
                        covariance_derivative = covariance_derivative.unsqueeze(0)
                    # Flatten the parameter axes into one Jacobian column per
                    # parameter component, as ``Model.joint_grad_log_prob``
                    # expects for vector-valued (e.g. ARD) parameters.
                    derivatives[parameter.name] = _cholesky_output_derivative(
                        factor,
                        _jitter_derivative(jitter_info, covariance_derivative),
                        latent_value,
                    ).reshape(latent_value.numel(), -1)

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
        if not _is_named(inputs):
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
        factor, self._jitter_info = stable_cholesky(covariance, jitter=self.jitter, return_jitter=True)
        residual = observed - mean
        alpha = torch.cholesky_solve(residual.reshape(-1, 1), factor).reshape(-1)
        log_det = 2 * torch.log(torch.diagonal(factor)).sum()
        log_probability = -0.5 * (
            residual.dot(alpha) + log_det + n * math.log(2 * math.pi)
        )
        return x, covariance, factor, residual, alpha, log_probability

    def _named_parameter_values(self):
        values = {}
        parameters = [*self.kernel.parameters, self.noise_variance, self.mean]
        for parameter in parameters:
            if _is_named(parameter):
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
        jitter_info = self._jitter_info
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
            derivative = _jitter_derivative(jitter_info, derivative)
            gradients[name] = (sensitivity * derivative).sum(dim=(-2, -1))

        if _is_named(self.noise_variance):
            # d(K + noise * I + jitter * I) / d(noise) = (1 + d(jitter)/d(noise)) * I.
            jitter, scale, index = jitter_info
            noise_scale = 1.0 + (jitter / scale if index is not None else 0.0)
            gradients[self.noise_variance.name] = (0.5 * (
                alpha.square().sum() - torch.diagonal(inverse).sum()
            ) * noise_scale).reshape(resolve(self.noise_variance).shape)

        if _is_named(self.mean):
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


_SPARSE_METHODS = ("vfe", "dtc", "fitc")


def _with_parameter_axis(derivative):
    if derivative is not None and derivative.ndim == 2:
        return derivative.unsqueeze(0)
    return derivative


class SparseGP(Distribution):
    r"""Inducing-point approximation of the GP marginal likelihood.

    With ``m`` inducing inputs ``Z`` and ``Q_nn = K_nm K_mm⁻¹ K_mn``, the
    log density of the observations ``y`` (one value per training input) is

    * ``method="vfe"`` (default): Titsias' collapsed variational bound,
      ``log N(y | mean, Q_nn + σ² I) - tr(K_nn - Q_nn) / (2σ²)``. It is a lower
      bound on the exact log marginal likelihood and equals it when ``Z``
      equals the training inputs.
    * ``method="dtc"``: ``log N(y | mean, Q_nn + σ² I)`` (deterministic
      training conditional).
    * ``method="fitc"``: ``log N(y | mean, Q_nn + diag(K_nn - Q_nn) + σ² I)``
      (fully independent training conditional).

    Everything is evaluated with Cholesky factors of ``m x m`` matrices, the
    Woodbury identity and the matrix determinant lemma, i.e. in ``O(n m²)``
    time and ``O(n m)`` memory; no ``n x n`` matrix is ever formed.

    The inducing inputs are fixed (a tensor or a ``Data`` node). Kernel
    hyperparameters, ``noise_variance`` and ``mean`` can be model parameters;
    their gradients are derived by hand in :meth:`log_pdf_param_grads`.
    Use it as the distribution of an ``ObservedParameter``::

        gp = SparseGP("function", inputs, kernel, inducing_points=Z, noise_variance=noise)
        ObservedParameter("observations", gp, y)
    """

    def __init__(self, name, inputs, kernel: Kernel, inducing_points, noise_variance=1e-6, mean=0.0, method="vfe", jitter=1e-6):
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")
        if isinstance(inducing_points, (RandomParameter, DeterministicParameter)):
            raise TypeError("SparseGP inducing_points must be fixed (a tensor or a Data node).")
        method = str(method).lower()
        if method not in _SPARSE_METHODS:
            raise ValueError(f"SparseGP method must be one of {_SPARSE_METHODS}; got {method!r}.")
        self.name = name
        self.inputs = inputs
        self.kernel = kernel
        self.inducing_points = inducing_points
        self.noise_variance = noise_variance
        self.mean = mean
        self.method = method
        self.jitter = jitter

        self.add_dependency(inputs)
        self.add_dependency(inducing_points)
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

    def _named_parameter_values(self):
        values = {}
        for parameter in [*self.kernel.parameters, self.noise_variance, self.mean]:
            if _is_named(parameter):
                values[parameter.name] = resolve(parameter)
        return values

    def _training_inputs(self):
        x = torch.as_tensor(_input_value(self.inputs))
        if x.ndim == 0:
            x = x.reshape(1)
        return x

    def _inducing_inputs(self, dtype, device):
        z = torch.as_tensor(_input_value(self.inducing_points)).to(dtype=dtype, device=device)
        if z.ndim == 0:
            z = z.reshape(1)
        return z

    def _components(self, observed):
        x = self._training_inputs()
        dtype, device = x.dtype, x.device
        z = self._inducing_inputs(dtype, device)
        observed = torch.as_tensor(observed, dtype=dtype, device=device).reshape(-1)
        n = _feature_count(x)
        if observed.numel() != n:
            raise ValueError(
                f"Sparse GP observations must contain one value per input ({n}); got {observed.numel()} values."
            )
        parameter_values = self._named_parameter_values()
        noise = _resolved_value(self.noise_variance, dtype, device).reshape(-1)
        if noise.numel() != 1:
            raise ValueError("Sparse GP noise_variance must be scalar.")
        noise = noise[0]
        mean = _resolved_value(self.mean, dtype, device)
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(n)
        else:
            mean = mean.reshape(-1)
            if mean.numel() != n:
                raise ValueError(
                    f"Sparse GP mean must be scalar or contain one value per input ({n}); got {mean.numel()} values."
                )

        k_mm = self.kernel(z, z, parameter_values=parameter_values)
        k_mn = self.kernel(z, x, parameter_values=parameter_values)
        k_diag = self.kernel.diag(x, parameter_values=parameter_values)
        factor, jitter_info = stable_cholesky(k_mm, jitter=self.jitter, return_jitter=True)
        # V = L⁻¹ K_mn, so Q_nn = Vᵀ V and diag(Q_nn) = colsum(V ⊙ V).
        v = torch.linalg.solve_triangular(factor, k_mn, upper=False)
        q_diag = v.square().sum(dim=0)
        if self.method == "fitc":
            lam = (k_diag - q_diag).clamp_min(0.0) + noise
        else:
            lam = noise.expand(n)
        residual = observed - mean
        eye_m = torch.eye(v.shape[0], dtype=dtype, device=device)
        v_scaled = v / lam
        # Σ = Vᵀ V + Λ and B = I + V Λ⁻¹ Vᵀ = L_B L_Bᵀ, so |Σ| = |Λ| |B|.
        factor_b = torch.linalg.cholesky(eye_m + v_scaled @ v.transpose(-2, -1))
        # Woodbury: Σ⁻¹ r = Λ⁻¹ r - Λ⁻¹ Vᵀ L_B⁻ᵀ c with c = L_B⁻¹ V Λ⁻¹ r.
        c = torch.linalg.solve_triangular(factor_b, (v_scaled @ residual).reshape(-1, 1), upper=False)
        correction = torch.linalg.solve_triangular(factor_b.transpose(-2, -1), c, upper=True).reshape(-1)
        alpha = (residual - v.transpose(-2, -1) @ correction) / lam
        log_det = torch.log(lam).sum() + 2 * torch.log(torch.diagonal(factor_b)).sum()
        quadratic = (residual.square() / lam).sum() - c.square().sum()
        log_probability = -0.5 * (quadratic + log_det + n * math.log(2 * math.pi))
        trace_term = (k_diag - q_diag).sum()
        if self.method == "vfe":
            log_probability = log_probability - 0.5 * trace_term / noise
        return {
            "x": x, "z": z, "noise": noise, "factor": factor, "jitter_info": jitter_info,
            "v": v, "lam": lam, "factor_b": factor_b, "alpha": alpha, "trace_term": trace_term,
            "parameter_values": parameter_values, "log_probability": log_probability,
        }

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))

    def log_pdf(self, x):
        return self._components(x)["log_probability"]

    def log_pdf_grad(self, x):
        return -self._components(x)["alpha"]

    def log_pdf_param_grads(self, x):
        r"""Hand-derived gradients of the approximate log marginal likelihood.

        With ``Σ = Q_nn + Λ``, ``α = Σ⁻¹ r`` and ``W = K_mm⁻¹ K_mn``,
        ``dQ = dK_nm W + Wᵀ dK_mn - Wᵀ dK_mm W`` and
        ``d log N(y | mean, Σ) = ½ αᵀ dΣ α - ½ tr(Σ⁻¹ dΣ)``. The traces need
        only ``P = W Σ⁻¹`` (``m x n``) and ``diag(Σ⁻¹)``, both from Woodbury
        in ``O(n m²)``:

        * ``½ αᵀ dQ α = (dK_mn α)·(Wα) - ½ (Wα)ᵀ dK_mm (Wα)``,
        * ``½ tr(Σ⁻¹ dQ) = Σ(P ⊙ dK_mn) - ½ Σ((P Wᵀ) ⊙ dK_mm)``,
        * a diagonal ``dΛ`` contributes ``½ Σ_i dΛ_i (α_i² - (Σ⁻¹)_ii)``, with
          ``dΛ = dσ² I`` and, for FITC, also ``d diag(K_nn) - diag(dQ)``,
          ``diag(dQ) = 2 colsum(dK_mn ⊙ W) - colsum(W ⊙ dK_mm W)``,
        * the VFE trace term adds ``-(tr dK_nn - tr dQ) / (2σ²)`` to the
          kernel gradients and ``tr(K_nn - Q_nn) / (2σ⁴)`` to the noise one.

        ``dK_mm`` includes the derivative of the scale-dependent jitter.
        """
        parts = self._components(x)
        x_values, z = parts["x"], parts["z"]
        factor, factor_b, v, lam, alpha = parts["factor"], parts["factor_b"], parts["v"], parts["lam"], parts["alpha"]
        noise, jitter_info, parameter_values = parts["noise"], parts["jitter_info"], parts["parameter_values"]

        w = torch.linalg.solve_triangular(factor.transpose(-2, -1), v, upper=True)  # K_mm⁻¹ K_mn, (m, n)
        w_alpha = w @ alpha
        # Σ⁻¹ = Λ⁻¹ - Gᵀ G with G = L_B⁻¹ V Λ⁻¹.
        g = torch.linalg.solve_triangular(factor_b, v / lam, upper=False)
        sigma_inverse_diag = 1.0 / lam - g.square().sum(dim=0)
        p = w / lam - (w @ g.transpose(-2, -1)) @ g  # W Σ⁻¹, (m, n)
        p_w = p @ w.transpose(-2, -1)  # W Σ⁻¹ Wᵀ, (m, m)
        w_w = w @ w.transpose(-2, -1)
        diag_weight = 0.5 * (alpha.square() - sigma_inverse_diag)
        is_vfe = self.method == "vfe"
        is_fitc = self.method == "fitc"

        cross = self.kernel.derivatives(z, x_values, parameter_values=parameter_values)
        inducing = self.kernel.derivatives(z, z, parameter_values=parameter_values)
        diagonal = self.kernel.diag_derivatives(x_values, parameter_values=parameter_values) if (is_vfe or is_fitc) else {}
        gradients = {}
        for name in {**cross, **inducing, **diagonal}:
            d_mn = _with_parameter_axis(cross.get(name))
            d_mm = _with_parameter_axis(inducing.get(name))
            d_nn = diagonal.get(name)
            if d_mm is not None:
                d_mm = _jitter_derivative(jitter_info, d_mm)
            if d_nn is not None and d_nn.ndim == 1:
                d_nn = d_nn.unsqueeze(0)
            if d_mn is not None:
                shape = d_mn.shape[:-2]
            elif d_mm is not None:
                shape = d_mm.shape[:-2]
            else:
                shape = d_nn.shape[:-1]
            gradient = alpha.new_zeros(shape)
            trace_dq = alpha.new_zeros(shape)
            diag_dq = alpha.new_zeros((*shape, alpha.numel()))
            if d_mn is not None:
                gradient = gradient + (d_mn @ alpha) @ w_alpha - (p * d_mn).sum(dim=(-2, -1))
                trace_dq = trace_dq + 2 * (d_mn * w).sum(dim=(-2, -1))
                if is_fitc:
                    diag_dq = diag_dq + 2 * (d_mn * w).sum(dim=-2)
            if d_mm is not None:
                gradient = gradient - 0.5 * (d_mm @ w_alpha) @ w_alpha + 0.5 * (p_w * d_mm).sum(dim=(-2, -1))
                trace_dq = trace_dq - (w_w * d_mm).sum(dim=(-2, -1))
                if is_fitc:
                    diag_dq = diag_dq - ((d_mm @ w) * w).sum(dim=-2)
            if is_fitc:
                # The clamp in Λ is inactive for a valid kernel; it is treated
                # as the identity when differentiating.
                d_lam = -diag_dq if d_nn is None else d_nn - diag_dq
                gradient = gradient + (d_lam * diag_weight).sum(dim=-1)
            if is_vfe:
                trace_d_knn = d_nn.sum(dim=-1) if d_nn is not None else 0.0
                gradient = gradient - 0.5 * (trace_d_knn - trace_dq) / noise
            gradients[name] = gradient

        if _is_named(self.noise_variance):
            noise_gradient = diag_weight.sum()
            if is_vfe:
                noise_gradient = noise_gradient + 0.5 * parts["trace_term"] / noise.square()
            gradients[self.noise_variance.name] = noise_gradient.reshape(resolve(self.noise_variance).shape)

        if _is_named(self.mean):
            gradients[self.mean.name] = alpha

        return gradients

    def sample(self, n_samples=1):
        """Draw from the approximate prior ``Normal(mean, Q_nn + Λ)``.

        Uses ``Vᵀ ε₁ + sqrt(Λ) ε₂`` with ``V = L⁻¹ K_mn``, so no ``n x n``
        covariance is formed.
        """
        x = self._training_inputs()
        parts = self._components(torch.zeros(_feature_count(x), dtype=x.dtype, device=x.device))
        v, lam = parts["v"], parts["lam"]
        mean = _resolved_value(self.mean, x.dtype, x.device)
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(lam.numel())
        first = torch.randn((n_samples, v.shape[0]), dtype=x.dtype, device=x.device) @ v
        second = torch.randn((n_samples, lam.numel()), dtype=x.dtype, device=x.device) * lam.sqrt()
        return mean + first + second


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
    if _is_named(parameter):
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
        if _is_named(parameter)
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


def sparse_gp_predictive(
    gp,
    trace,
    observed_values,
    inputs,
    n_samples=1,
    observation_noise=None,
):
    r"""Sample sparse-GP predictions conditional on observed training values.

    For every hyperparameter draw in ``trace`` the inducing-point posterior
    ``q(u) = N(K_mm Σ_u K_mn Λ⁻¹ r, K_mm Σ_u K_mm)`` with
    ``Σ_u = (K_mm + K_mn Λ⁻¹ K_nm)⁻¹`` (Titsias' optimal ``q(u)`` for VFE,
    the DTC posterior, and for FITC the same with its diagonal ``Λ``) is
    projected to the new inputs:

    ``mean = K_*m Σ_u K_mn Λ⁻¹ r`` and ``cov = K_** - Q_** + K_*m Σ_u K_m*``.

    All solves use ``m x m`` Cholesky factors, so the cost is linear in the
    number of training points. The result has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, SparseGP):
        raise TypeError("gp must be a SparseGP distribution returned by BayesianDLL.GP.SparseGP.")
    if not hasattr(trace, "__getitem__") or not hasattr(trace, "keys"):
        raise TypeError("trace must be a mapping-like sampling result.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    training_values = gp._training_inputs()
    dtype = training_values.dtype
    device = training_values.device
    inducing_values = gp._inducing_inputs(dtype, device)
    new_values = torch.as_tensor(_input_value(inputs)).to(dtype=dtype, device=device)
    if new_values.ndim == 0:
        new_values = new_values.reshape(1)
    n_training = _feature_count(training_values)
    n_new = _feature_count(new_values)
    observed_values = torch.as_tensor(_input_value(observed_values), dtype=dtype, device=device).reshape(-1)
    if observed_values.numel() != n_training:
        raise ValueError(
            f"observed_values must contain one value per training input ({n_training}); "
            f"got {observed_values.numel()} values."
        )

    first_value = next(iter(trace.values()))
    n_draws = first_value.shape[0] * first_value.shape[1] if first_value.ndim >= 2 else first_value.shape[0]
    parameter_draws = {
        parameter.name: _trace_draws(trace, parameter, n_draws, dtype, device)
        for parameter in gp.kernel.parameters
        if _is_named(parameter)
    }
    noise_draws = _trace_draws(trace, gp.noise_variance, n_draws, dtype, device)
    mean_draws = _trace_draws(trace, gp.mean, n_draws, dtype, device)
    eye_m = torch.eye(_feature_count(inducing_values), dtype=dtype, device=device)

    outputs = []
    for draw_index in range(n_draws):
        kernel_values = {name: values[draw_index] for name, values in parameter_draws.items()}
        noise = noise_draws[draw_index].reshape(-1)
        if noise.numel() != 1:
            raise ValueError("Sparse GP noise_variance must be scalar.")
        mean_value = mean_draws[draw_index].reshape(-1)
        if mean_value.numel() != 1:
            raise ValueError("sparse_gp_predictive supports only a scalar mean.")
        mean_train = mean_value.reshape(()).expand(n_training)
        mean_new = mean_value.reshape(()).expand(n_new)

        k_mm = gp.kernel(inducing_values, inducing_values, parameter_values=kernel_values)
        k_mn = gp.kernel(inducing_values, training_values, parameter_values=kernel_values)
        k_ms = gp.kernel(inducing_values, new_values, parameter_values=kernel_values)
        k_ss = gp.kernel(new_values, new_values, parameter_values=kernel_values)
        factor = stable_cholesky(k_mm, jitter=gp.jitter)
        v = torch.linalg.solve_triangular(factor, k_mn, upper=False)
        v_new = torch.linalg.solve_triangular(factor, k_ms, upper=False)
        if gp.method == "fitc":
            k_diag = gp.kernel.diag(training_values, parameter_values=kernel_values)
            lam = (k_diag - v.square().sum(dim=0)).clamp_min(0.0) + noise[0]
        else:
            lam = noise[0].expand(n_training)
        # Σ_u = L⁻ᵀ B⁻¹ L⁻¹ with B = I + V Λ⁻¹ Vᵀ = L_B L_Bᵀ.
        factor_b = torch.linalg.cholesky(eye_m + (v / lam) @ v.transpose(-2, -1))
        c = torch.linalg.solve_triangular(
            factor_b, ((v / lam) @ (observed_values - mean_train)).reshape(-1, 1), upper=False
        )
        projected = torch.linalg.solve_triangular(factor_b, v_new, upper=False)  # L_B⁻¹ L⁻¹ K_m*
        predictive_mean = mean_new + (projected.transpose(-2, -1) @ c).reshape(-1)
        predictive_covariance = (
            k_ss
            - v_new.transpose(-2, -1) @ v_new
            + projected.transpose(-2, -1) @ projected
        )
        if observation_noise is not None:
            predictive_covariance = predictive_covariance + _resolved_value(
                observation_noise, dtype=dtype, device=device
            ).reshape(-1)[0] * torch.eye(n_new, dtype=dtype, device=device)
        predictive_covariance = (predictive_covariance + predictive_covariance.transpose(-2, -1)) / 2
        prediction_factor = stable_cholesky(predictive_covariance, jitter=gp.jitter)
        outputs.append(
            predictive_mean
            + torch.randn((n_samples, n_new), dtype=dtype, device=device) @ prediction_factor.transpose(-2, -1)
        )

    return torch.stack(outputs)


# Batched (vectorized) chains evaluate GP nodes chain by chain (see Model.batch).
LatentGP._batchable = False
ExactGP._batchable = False
SparseGP._batchable = False
