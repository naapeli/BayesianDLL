"""Latent, exact and sparse Gaussian processes for MCMC and variational inference.

Vectorized chains
-----------------
All GP nodes are evaluated in a *batched form*: every quantity carries a
leading batch dimension ``B`` that is either the number of chains ``C`` (for
values that depend on a latent parameter of a batched model, see
:mod:`BayesianDLL._batching`) or ``1`` (shared values: data, constants, and
everything in an unbatched model). Kernels evaluate batched hyperparameters
``(C, *shape)`` and batched inputs ``(C, N, D)`` to ``(C, N, M)`` matrices,
and the linear algebra (Cholesky factorisations, triangular solves) is
batched over ``C``. An unbatched evaluation is the ``B = 1`` case, so both
paths run the same code. Data-dependent jitter (:func:`stable_cholesky`
retries) is chosen per chain.

What is *not* batched:

* A user kernel that only implements the unbatched ``evaluate`` /
  ``derivatives`` / ``input_vjp`` is called once per chain inside the
  batched computation (the GP linear algebra stays batched).
* A plain callable (not a :class:`Kernel`) inside a ``SumKernel`` /
  ``ProductKernel`` cannot evaluate per-chain inputs (learnable inputs or
  inducing points). It raises, and ``Model.batch`` then falls back to the
  chain-by-chain loop mode with a warning.
* The predictive functions and ``sample()`` work draw by draw (unbatched),
  except :func:`sparse_latent_gp_predictive`, which batches the draws.
* Gradients w.r.t. learnable grid inputs of ``KroneckerGP`` /
  ``ToeplitzGP`` are not implemented (in batched or unbatched mode).
"""

from __future__ import annotations

import math

import torch

from .. import _batching
from .._data import Data
from .._parameters import DeterministicParameter, RandomParameter
from ..Distributions._distributions import Distribution
from ..Distributions._resolve import resolve
from ..Distributions import Normal
from ._kernels import Kernel, _is_named


# ----------------------------------------------------------------------------
# Cholesky factorisation with per-chain jitter
# ----------------------------------------------------------------------------


def batched_stable_cholesky(matrix, jitter=1e-6, max_tries=6):
    """Batched :func:`stable_cholesky` for ``(B, n, n)`` matrices; the jitter is chosen per matrix.

    Returns ``(factor, (jitter, scale, index))`` with tensors of shape
    ``(B,)``: the jitter added to each diagonal, ``max(1, max|diag|)`` and the
    index of the largest diagonal entry when it exceeds one (``-1``
    otherwise; then the jitter does not depend on the matrix). A matrix that
    needs a larger jitter does not change the others.
    """
    matrix = (matrix + matrix.transpose(-2, -1)) / 2
    if not torch.isfinite(matrix).all():
        raise ValueError("GP covariance contains non-finite values.")
    n = matrix.shape[-1]
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1).abs()
    largest, argmax = diagonal.max(dim=-1)
    scale = largest.clamp_min(1.0)
    # The jitter is proportional to the largest diagonal entry only when that
    # entry exceeds one; otherwise the scale is the constant 1.
    index = torch.where(largest > 1.0, argmax, torch.full_like(argmax, -1))
    current = (torch.as_tensor(jitter, dtype=matrix.dtype, device=matrix.device) * scale).clamp_min(
        torch.finfo(matrix.dtype).eps * scale
    )
    failed = None
    for _ in range(max_tries):
        factor, info = torch.linalg.cholesky_ex(matrix + current[:, None, None] * eye)
        failed = info != 0
        if not bool(failed.any()):
            return factor, (current, scale, index)
        current = torch.where(failed, current * 10, current)

    # A covariance assembled in finite precision can have a small negative
    # eigenvalue even when the kernel is theoretically positive semidefinite.
    # Correct the smallest eigenvalue of the failing matrices directly as a
    # final fallback for difficult MCMC proposals; the correction is treated
    # as a constant.
    minimum = torch.linalg.eigvalsh(matrix).amin(dim=-1)
    correction = (-minimum + torch.finfo(matrix.dtype).eps * scale).clamp_min(current)
    current = torch.where(failed, correction, current)
    index = torch.where(failed, torch.full_like(index, -1), index)
    factor, info = torch.linalg.cholesky_ex(matrix + current[:, None, None] * eye)
    if bool((info != 0).any()):
        raise RuntimeError(f"Could not compute a positive-definite Cholesky factor after {max_tries} jitter attempts.")
    return factor, (current, scale, index)


def stable_cholesky(matrix, jitter=1e-6, max_tries=6, return_jitter=False):
    """Compute a Cholesky factor, increasing diagonal jitter when necessary.

    The jitter added to the diagonal is ``jitter * max(1, max|diag(matrix)|)``,
    multiplied by 10 for every failed attempt. With ``return_jitter=True`` the
    result is ``(factor, jitter_info)``, where ``jitter_info`` describes the
    added jitter so that derivatives can account for its dependence on the
    covariance (see :func:`_jitter_derivative`). A batch ``(B, n, n)`` is
    factorised with a jitter per matrix (see :func:`batched_stable_cholesky`).
    """
    if matrix.ndim == 3:
        factor, info = batched_stable_cholesky(matrix, jitter=jitter, max_tries=max_tries)
        return (factor, info) if return_jitter else factor
    factor, (current, scale, index) = batched_stable_cholesky(matrix.unsqueeze(0), jitter=jitter, max_tries=max_tries)
    if return_jitter:
        position = int(index[0])
        return factor[0], (current[0], scale[0], None if position < 0 else position)
    return factor[0]


def _jitter_coefficient(jitter_info):
    """``d(jitter) / d(K_ii*)`` per matrix: ``c`` when the scale is a diagonal entry, else 0."""
    current, scale, index = jitter_info
    return torch.where(index >= 0, current / scale, torch.zeros_like(current))


def _jitter_cotangent(jitter_info, cotangent):
    """Fold the scale-dependent jitter into a cotangent ``G`` of ``K`` (``(B, n, n)``).

    ``stable_cholesky`` factorises ``K + c max(1, max|diag K|) I``; when the
    largest diagonal entry ``K_ii*`` exceeds one, ``⟨G, dK + c dK_i*i* I⟩ =
    ⟨G + c tr(G) e_i* e_i*ᵀ, dK⟩``.
    """
    current, scale, index = jitter_info
    coefficient = _jitter_coefficient(jitter_info)
    n = cotangent.shape[-1]
    one_hot = (torch.arange(n, device=cotangent.device) == index.clamp_min(0)[:, None]).to(cotangent.dtype)
    trace = torch.diagonal(cotangent, dim1=-2, dim2=-1).sum(-1)
    return cotangent + torch.diag_embed(one_hot * (coefficient * trace)[:, None])


def _jitter_derivative(jitter_info, covariance_derivative):
    """Add the derivative of the scale-dependent (unbatched) jitter to ``dK`` ``(..., n, n)``."""
    jitter, scale, index = jitter_info
    if index is None:
        return covariance_derivative
    n = covariance_derivative.shape[-1]
    eye = torch.eye(n, dtype=covariance_derivative.dtype, device=covariance_derivative.device)
    scale_derivative = covariance_derivative[..., index, index]
    return covariance_derivative + (jitter / scale) * scale_derivative[..., None, None] * eye


# ----------------------------------------------------------------------------
# Values in batched form
# ----------------------------------------------------------------------------


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


def _feature_count(x):
    x = _input_value(x)
    return x.shape[0] if x.ndim > 0 else 1


def _resolved_value(value, dtype, device, parameter_values=None):
    if parameter_values is not None and _is_named(value) and value.name in parameter_values:
        value = parameter_values[value.name]
    else:
        value = resolve(value)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _mean_value(mean, x):
    if _is_named(mean):
        mean = _input_value(mean)
    mean = _as_tensor(mean).to(dtype=x.dtype, device=x.device)
    if mean.ndim == 0:
        return mean.expand(x.shape[0])
    return mean


class _BatchContext:
    """Which nodes carry a chain dimension in the current evaluation.

    Inside a batched term evaluation (``Model.batch`` in tensor mode) the
    batched nodes of the term are the keys of the active alignment (see
    :mod:`BayesianDLL._batching`); their values have shape
    ``(C, *value_shape)``. Every other value is shared and gets a leading
    batch dimension of one.
    """

    def __init__(self, flags=None):
        self._flags = flags
        frame = _batching.current()
        self.pad = frame.pad if frame is not None and frame.n_chains is not None and frame.pad else {}

    def is_batched(self, node):
        if self._flags is not None:
            return self._flags.get(id(node), False)
        return id(node) in self.pad

    def any(self, nodes):
        return any(self.is_batched(node) for node in nodes if _is_named(node))

    def value(self, node, dtype=None, device=None, raw=None):
        """Value of ``node`` with a leading batch dimension (``raw`` overrides the node's value)."""
        if raw is not None:
            value = _as_tensor(raw)
        elif _is_named(node):
            value = _as_tensor(_input_value(node))
        else:
            value = _as_tensor(node)
        if dtype is not None:
            value = value.to(dtype=dtype, device=device)
        return value if (_is_named(node) and self.is_batched(node)) else value.unsqueeze(0)

    def inputs(self, node, dtype=None, device=None, raw=None):
        """Inputs as ``(B, N, D)``."""
        value = self.value(node, dtype, device, raw)
        if value.ndim == 1:
            return value.reshape(value.shape[0], 1, 1)
        if value.ndim == 2:
            return value.unsqueeze(-1)
        return value

    def kernel_values(self, kernel, dtype, device, raw=None):
        raw = raw or {}
        return {p.name: self.value(p, dtype, device, raw.get(p.name)) for p in kernel.parameters}


def _contract(cotangent, derivative):
    """``Σ_ij G_bij dK_b...ij``: cotangent ``(B, N, M)``, derivative ``(B', *P, N, M)`` -> ``(B, *P)``."""
    if derivative.ndim == 3:
        derivative = derivative.unsqueeze(1)
    view = cotangent.reshape(cotangent.shape[0], *([1] * (derivative.ndim - 3)), *cotangent.shape[-2:])
    return (view * derivative).sum(dim=(-2, -1))


def _contract_diagonal(weights, derivative):
    """``Σ_i g_bi d_b...i``: weights ``(B, N)``, derivative ``(B', *P, N)`` -> ``(B, *P)``."""
    if derivative.ndim == 2:
        derivative = derivative.unsqueeze(1)
    view = weights.reshape(weights.shape[0], *([1] * (derivative.ndim - 2)), weights.shape[-1])
    return (view * derivative).sum(dim=-1)


def _parameter_shape(context, node):
    value = _as_tensor(_input_value(node))
    return tuple(value.shape[1:]) if context.is_batched(node) else tuple(value.shape)


def _to_parameter_shape(gradient, shape):
    """Reshape a batched gradient ``(B, *g)`` to ``(B, *shape)``; scalar parameters get the sum."""
    if tuple(gradient.shape[1:]) == tuple(shape):
        return gradient
    if math.prod(gradient.shape[1:]) == math.prod(shape):
        return gradient.reshape(gradient.shape[0], *shape)
    if math.prod(shape) == 1:
        return gradient.reshape(gradient.shape[0], -1).sum(-1).reshape(gradient.shape[0], *shape)
    raise ValueError(f"Cannot map a gradient of shape {tuple(gradient.shape[1:])} to a parameter of shape {tuple(shape)}.")


def _batched_mean(context, mean, n, dtype, device, what, raw=None):
    value = context.value(mean, dtype, device, raw)
    flat = value.reshape(value.shape[0], -1)
    if flat.shape[1] == 1:
        return flat.expand(flat.shape[0], n)
    if flat.shape[1] != n:
        raise ValueError(f"{what} mean must be scalar or contain one value per input ({n}); got {flat.shape[1]} values.")
    return flat


def _expand_batch(tensor, size):
    return tensor.expand(size, *tensor.shape[1:]) if tensor.shape[0] != size else tensor


def _finish(value, batched):
    """Drop the batch dimension of an unbatched evaluation."""
    if batched:
        return value
    if isinstance(value, dict):
        return {name: v[0] for name, v in value.items()}
    return value[0]


# ----------------------------------------------------------------------------
# Latent (whitened) GP
# ----------------------------------------------------------------------------


class LatentGP(DeterministicParameter):
    r"""An explicit whitened latent GP function node.

    The object is a ``DeterministicParameter`` whose value is
    ``mean + L @ z`` with ``L Lᵀ = K(X, X) + jitter I`` (the jitter of
    :func:`stable_cholesky`). If ``latent`` is omitted, a standard-normal
    latent vector of ``N`` values named ``latent_name`` (default
    ``<name>_white``) is created automatically. This is a non-centred
    parameterization intended for NUTS and variational inference.

    Arguments:

    * ``inputs``: a named node, normally a ``Data`` object, of shape
      ``(N, D)`` or ``(N,)``; a learnable ``RandomParameter`` /
      ``DeterministicParameter`` also gets gradients;
    * ``kernel``: a :class:`Kernel`;
    * ``latent``: a named ``RandomParameter`` with ``N`` values and a
      ``Normal(0, 1)`` prior (default ``None``: created automatically);
    * ``mean``: a scalar, one value per input, or a named node (default ``0.0``);
    * ``jitter``: relative diagonal jitter (default ``1e-6``);
    * ``latent_name``: name of the automatically created latent.

    The node stores them as ``gp_inputs``, ``gp_latent``, ``gp_kernel``,
    ``gp_mean`` and ``gp_jitter``. The likelihood can use the returned
    function directly, for example
    ``ObservedParameter("y", Normal(f, noise), y)``; see
    :func:`gp_predictive` for predictions at new inputs.

    **Gradients** (vector-Jacobian product, hand-derived). For a cotangent
    ``ḡ`` of ``f``: ``z`` gets ``Lᵀ ḡ``; with ``dL = L Φ(L⁻¹ dK L⁻ᵀ)``
    (``Φ`` takes the lower triangle and halves the diagonal) every kernel
    hyperparameter gets ``⟨G_K, ∂K/∂θ⟩`` with
    ``G_K = sym(L⁻ᵀ S L⁻¹)``, ``S = Φ(Lᵀ ḡ zᵀ)``. Learnable inputs get
    ``kernel.input_vjp(X, X, G_K + G_Kᵀ)``. With vectorized chains all of
    this is batched over the chain dimension.
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
        self._kernel_parameters = kernel_parameters
        self._mean_input = mean_input

        def forward(*values):
            return self._batched_function(values, [False] * len(values))[0][0]

        def derivative(*values):
            return self._jacobians(values)

        def vjp(cotangent, *values):
            flags = [False] * len(values)
            return _finish(self._batched_vjp(_as_tensor(cotangent).unsqueeze(0), values, flags), False)

        super().__init__(name, forward, derivative, model_inputs, vjp_func=vjp)
        self.gp_inputs = inputs
        self.gp_latent = latent
        self.gp_kernel = kernel
        self.gp_mean = mean
        self.gp_jitter = jitter

    def _context(self, values, flags):
        return _BatchContext({id(node): bool(flag) for node, flag in zip(self.inputs, flags)})

    def _batched_function(self, values, flags):
        """``(f, pieces)`` in batched form for input ``values`` with chain flags ``flags``."""
        context = self._context(values, flags)
        z = context.value(self.inputs[1], raw=values[1])
        z = z.reshape(z.shape[0], -1)
        dtype, device = z.dtype, z.device
        x = context.inputs(self.inputs[0], dtype, device, raw=values[0])
        raw = {p.name: v for p, v in zip(self.inputs[2:2 + len(self._kernel_parameters)], values[2:])}
        kernel_values = context.kernel_values(self.gp_kernel, dtype, device, raw)
        covariance = self.gp_kernel.batch_evaluate(x, x, kernel_values)
        size = max(covariance.shape[0], z.shape[0])
        factor, jitter_info = batched_stable_cholesky(_expand_batch(covariance, size), jitter=self.gp_jitter)
        n = z.shape[-1]
        if self._mean_input is not None:
            mean = _batched_mean(context, self._mean_input, n, dtype, device, "LatentGP", raw=values[-1])
        else:
            mean = _mean_value(self.gp_mean, torch.zeros(n, dtype=dtype, device=device)).reshape(1, n)
        f = mean + (factor @ z.unsqueeze(-1)).squeeze(-1)
        return f, (context, x, z, kernel_values, factor, jitter_info)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, x, z, kernel_values, factor, jitter_info) = self._batched_function(values, flags)
        cotangent = cotangent.reshape(cotangent.shape[0], -1).to(dtype=z.dtype)
        projected = (factor.transpose(-2, -1) @ cotangent.unsqueeze(-1)).squeeze(-1)  # Lᵀ ḡ
        latent = self.inputs[1]
        latent_shape = tuple(_as_tensor(values[1]).shape[1:]) if context.is_batched(latent) else tuple(_as_tensor(values[1]).shape)
        grads = {latent.name: projected.reshape(projected.shape[0], *latent_shape)}
        outer = projected.unsqueeze(-1) * z.unsqueeze(-2)
        lower = torch.tril(outer, diagonal=-1) + 0.5 * torch.diag_embed(torch.diagonal(outer, dim1=-2, dim2=-1))
        left = torch.linalg.solve_triangular(factor.transpose(-2, -1), lower, upper=True)
        covariance_cotangent = torch.linalg.solve_triangular(factor, left, upper=False, left=False)
        covariance_cotangent = 0.5 * (covariance_cotangent + covariance_cotangent.transpose(-2, -1))
        covariance_cotangent = _jitter_cotangent(jitter_info, covariance_cotangent)
        for name, derivative in self.gp_kernel.batch_derivatives(x, x, kernel_values).items():
            parameter = next(p for p in self._kernel_parameters if p.name == name)
            value = values[self.inputs.index(parameter)]
            shape = tuple(_as_tensor(value).shape[1:]) if context.is_batched(parameter) else tuple(_as_tensor(value).shape)
            grads[name] = _to_parameter_shape(_contract(covariance_cotangent, derivative), shape)
        inputs_node = self.inputs[0]
        if not isinstance(inputs_node, Data) and isinstance(inputs_node, (RandomParameter, DeterministicParameter)):
            input_gradient = self.gp_kernel.batch_input_vjp_symmetric(x, covariance_cotangent, kernel_values)
            shape = tuple(_as_tensor(values[0]).shape[1:]) if context.is_batched(inputs_node) else tuple(_as_tensor(values[0]).shape)
            grads[inputs_node.name] = input_gradient.reshape(input_gradient.shape[0], *shape)
        if self._mean_input is not None:
            value = values[-1]
            shape = tuple(_as_tensor(value).shape[1:]) if context.is_batched(self._mean_input) else tuple(_as_tensor(value).shape)
            grads[self._mean_input.name] = _to_parameter_shape(cotangent, shape)
        return grads

    def _jacobians(self, values):
        """Dense Jacobians ``(n, parameter.numel())`` of ``f`` (unbatched; for ``derivatives()``)."""
        n = _as_tensor(values[1]).numel()
        dtype = _as_tensor(values[1]).dtype
        rows = []
        for i in range(n):
            unit = torch.zeros(n, dtype=dtype, device=_as_tensor(values[1]).device)
            unit[i] = 1.0
            rows.append(self.vjp_func(unit, *values))
        jacobians = {}
        for name in rows[0]:
            jacobians[name] = torch.stack([row[name].reshape(-1) for row in rows])
        latent_name = self.inputs[1].name
        if latent_name in jacobians:
            jacobians[latent_name] = jacobians[latent_name].reshape(n, n)
        return jacobians

    # -- vectorized chains -------------------------------------------------
    def batched_forward(self, input_values):
        f, _ = self._batched_function(input_values, self._batched_flags())
        return _expand_batch(f, self.owner_model.n_chains)

    def batched_vjp(self, cotangent, input_values):
        flags = self._batched_flags()
        grads = self._batched_vjp(cotangent, input_values, flags)
        result = {}
        for node, flag in zip(self.inputs, flags):
            name = getattr(node, "name", None)
            if flag and name in grads:
                result[name] = _expand_batch(grads[name], self.owner_model.n_chains)
        return result


# ----------------------------------------------------------------------------
# Exact GP
# ----------------------------------------------------------------------------


class _GPDistribution(Distribution):
    """Shared plumbing of the GP likelihood nodes.

    The kernel's parameter nodes are stored in ``_kernel_parameter_nodes`` so
    that the model's batching machinery sees them as arguments of the term
    (and marks the batched ones).
    """

    def _register_kernel(self, kernel):
        self._kernel_parameter_nodes = list(kernel.parameters)
        for parameter in kernel.parameters:
            self.add_dependency(parameter)

    @property
    def batch_shape(self):
        return torch.Size()

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))


class ExactGP(_GPDistribution):
    """Gaussian-process marginal likelihood with the latent function integrated out.

    If ``f ~ Normal(mean, K)`` and ``y | f ~ Normal(f, noise_variance * I)``,
    this distribution evaluates the exact marginal likelihood

    ``y ~ Normal(mean, K + noise_variance * I)``.

    The object is used as the distribution of an ``ObservedParameter``::

        gp = ExactGP("function", inputs, kernel, noise_variance=0.05)
        ObservedParameter("observations", gp, y)

    Unlike :class:`LatentGP`, this does not create a latent random
    variable. Kernel and noise hyperparameters remain ordinary model
    parameters and can be sampled with MCMC (or optimised with
    :func:`~BayesianDLL.Variational.find_MAP`, a type-II MAP estimate when
    they have priors). With vectorized chains the covariance matrices of all
    chains are factorised as one batch.

    Arguments: ``name`` (a label for the node), ``inputs`` (a named ``Data``
    or model node of shape ``(N, D)`` or ``(N,)``), ``kernel`` (a
    :class:`Kernel`), ``noise_variance`` (the scalar noise **variance**
    ``σ²``, default ``1e-6``), ``mean`` (scalar, one value per input or a
    named node; default ``0.0``) and ``jitter`` (default ``1e-6``; the
    relative diagonal jitter of :func:`batched_stable_cholesky`, applied to
    ``K + σ² I`` and included in the gradients). The observations have
    shape ``(N,)``; ``sample`` draws ``y`` including the noise. Use
    :func:`exact_gp_predictive` for predictions at new inputs.
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
        self._register_kernel(kernel)
        self.add_dependency(noise_variance)
        self.add_dependency(mean)

    @property
    def event_shape(self):
        return (_feature_count(self.inputs),)

    def _named_parameter_values(self):
        values = {}
        for parameter in [*self.kernel.parameters, self.noise_variance, self.mean]:
            if _is_named(parameter):
                values[parameter.name] = resolve(parameter)
        return values

    def _components(self, observed):
        context = _BatchContext()
        x_value = _as_tensor(_input_value(self.inputs))
        dtype = x_value.dtype if x_value.is_floating_point() else torch.get_default_dtype()
        device = x_value.device
        x = context.inputs(self.inputs, dtype, device)
        n = x.shape[1]
        observed = torch.as_tensor(observed, dtype=dtype, device=device).reshape(-1)
        if observed.numel() != n:
            raise ValueError(
                f"Exact GP observations must contain one value per input ({n}); got {observed.numel()} values."
            )
        kernel_values = context.kernel_values(self.kernel, dtype, device)
        covariance = self.kernel.batch_evaluate(x, x, kernel_values)
        noise = context.value(self.noise_variance, dtype, device)
        noise = noise.reshape(noise.shape[0], -1)
        if noise.shape[1] != 1:
            raise ValueError("Exact GP noise_variance must be scalar.")
        mean = _batched_mean(context, self.mean, n, dtype, device, "Exact GP")
        size = max(covariance.shape[0], noise.shape[0], mean.shape[0])
        eye = torch.eye(n, dtype=dtype, device=device)
        covariance = _expand_batch(covariance, size) + noise[:, :1, None] * eye
        factor, jitter_info = batched_stable_cholesky(covariance, jitter=self.jitter)
        residual = observed - mean
        alpha = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
        alpha = _expand_batch(alpha, size)
        log_det = 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
        log_probability = -0.5 * ((residual * alpha).sum(-1) + log_det + n * math.log(2 * math.pi))
        batched = context.any([self.inputs, self.noise_variance, self.mean, *self.kernel.parameters])
        return {
            "context": context, "x": x, "kernel_values": kernel_values, "factor": factor, "jitter_info": jitter_info,
            "alpha": alpha, "log_probability": log_probability, "batched": batched,
        }

    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        return _finish(-parts["alpha"], parts["batched"])

    def log_pdf_param_grads(self, x):
        r"""``d log p = ⟨½(ααᵀ - Σ⁻¹), dΣ⟩`` for ``Σ = K + σ² I`` (+ jitter), batched over chains."""
        parts = self._components(x)
        context, factor, alpha, jitter_info = parts["context"], parts["factor"], parts["alpha"], parts["jitter_info"]
        inverse = torch.cholesky_inverse(factor)
        sensitivity = 0.5 * (alpha.unsqueeze(-1) * alpha.unsqueeze(-2) - inverse)
        sensitivity = _jitter_cotangent(jitter_info, sensitivity)
        gradients = {}
        x_values = parts["x"]
        for name, derivative in self.kernel.batch_derivatives(x_values, x_values, parts["kernel_values"]).items():
            parameter = next(p for p in self.kernel.parameters if p.name == name)
            gradients[name] = _to_parameter_shape(_contract(sensitivity, derivative), _parameter_shape(context, parameter))
        inputs = self.inputs
        if isinstance(inputs, (RandomParameter, DeterministicParameter)):
            input_gradient = self.kernel.batch_input_vjp_symmetric(x_values, sensitivity, parts["kernel_values"])
            shape = _parameter_shape(context, inputs)
            gradients[inputs.name] = input_gradient.reshape(input_gradient.shape[0], *shape)
        if _is_named(self.noise_variance):
            # d(K + σ² I + jitter I) / dσ² = I + c e_i* e_i*ᵀ is folded into the sensitivity.
            noise_gradient = torch.diagonal(sensitivity, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            gradients[self.noise_variance.name] = _to_parameter_shape(noise_gradient, _parameter_shape(context, self.noise_variance))
        if _is_named(self.mean):
            gradients[self.mean.name] = _to_parameter_shape(alpha, _parameter_shape(context, self.mean))
        return _finish(gradients, parts["batched"])

    def sample(self, n_samples=1):
        x = _input_value(self.inputs)
        zero = torch.zeros(_feature_count(x), dtype=x.dtype, device=x.device)
        factor = self._components(zero)["factor"][0]
        mean = _resolved_value(self.mean, x.dtype, x.device)
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(zero.numel())
        return mean + torch.randn(
            (n_samples, zero.numel()), dtype=zero.dtype, device=zero.device
        ) @ factor.transpose(-2, -1)


# ----------------------------------------------------------------------------
# Sparse (inducing-point) GP
# ----------------------------------------------------------------------------


_SPARSE_METHODS = ("vfe", "dtc", "fitc")


class SparseGP(_GPDistribution):
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
    time and ``O(n m)`` memory; no ``n x n`` matrix is ever formed. With
    vectorized chains all matrices carry a leading chain dimension.

    ``inducing_points`` (shape ``(m, D)`` or ``(m,)``) is a tensor, a ``Data``
    node, or a learnable ``RandomParameter`` / ``DeterministicParameter``,
    e.g. ``RandomParameter("Z", Normal(Z0, 0.5), initial_value=Z0)`` with
    ``Z0`` of shape ``(m, D)``. Kernel hyperparameters, ``noise_variance``,
    ``mean`` and learnable inducing inputs get hand-derived gradients (see
    :meth:`log_pdf_param_grads`); the inducing-input gradient uses
    :meth:`Kernel.input_vjp`.

    **Learnable inducing inputs: use VFE.** The VFE bound can only increase
    towards the exact marginal likelihood as ``Z`` improves, so optimizing or
    sampling ``Z`` is safe. DTC and FITC are not bounds: moving ``Z`` can
    raise their likelihood above the exact one (FITC in particular is known
    to collapse inducing points and underestimate the noise), so they can
    overfit ``Z``.

    The other arguments are as for :class:`ExactGP`: ``noise_variance`` is
    the scalar noise variance ``σ²`` (default ``1e-6``), ``mean`` defaults to
    ``0.0``, ``method`` is ``"vfe"``, ``"dtc"`` or ``"fitc"`` (case
    insensitive) and ``jitter`` (default ``1e-6``) is the relative jitter of
    the ``K_mm`` Cholesky factor. Use it as the distribution of an
    ``ObservedParameter``::

        gp = SparseGP("function", inputs, kernel, inducing_points=Z, noise_variance=noise)
        ObservedParameter("observations", gp, y)

    Use :func:`sparse_gp_predictive` for predictions at new inputs.
    """

    def __init__(self, name, inputs, kernel: Kernel, inducing_points, noise_variance=1e-6, mean=0.0, method="vfe", jitter=1e-6):
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")
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
        self._register_kernel(kernel)
        self.add_dependency(noise_variance)
        self.add_dependency(mean)

    @property
    def event_shape(self):
        return (_feature_count(self.inputs),)

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

    def _learnable_inducing(self):
        return _is_named(self.inducing_points) and not isinstance(self.inducing_points, Data)

    def _components(self, observed):
        context = _BatchContext()
        x_value = self._training_inputs()
        dtype = x_value.dtype if x_value.is_floating_point() else torch.get_default_dtype()
        device = x_value.device
        x = context.inputs(self.inputs, dtype, device)
        z = context.inputs(self.inducing_points, dtype, device)
        n = x.shape[1]
        observed = torch.as_tensor(observed, dtype=dtype, device=device).reshape(-1)
        if observed.numel() != n:
            raise ValueError(
                f"Sparse GP observations must contain one value per input ({n}); got {observed.numel()} values."
            )
        noise = context.value(self.noise_variance, dtype, device)
        noise = noise.reshape(noise.shape[0], -1)
        if noise.shape[1] != 1:
            raise ValueError("Sparse GP noise_variance must be scalar.")
        noise = noise[:, 0]
        mean = _batched_mean(context, self.mean, n, dtype, device, "Sparse GP")
        kernel_values = context.kernel_values(self.kernel, dtype, device)

        k_mm = self.kernel.batch_evaluate(z, z, kernel_values)
        k_mn = self.kernel.batch_evaluate(z, x, kernel_values)
        k_diag = self.kernel.batch_diag(x, kernel_values)
        size = max(k_mm.shape[0], k_mn.shape[0], k_diag.shape[0], noise.shape[0], mean.shape[0])
        k_mm, k_mn = _expand_batch(k_mm, size), _expand_batch(k_mn, size)
        k_diag, noise = _expand_batch(k_diag, size), _expand_batch(noise, size)
        factor, jitter_info = batched_stable_cholesky(k_mm, jitter=self.jitter)
        # V = L⁻¹ K_mn, so Q_nn = Vᵀ V and diag(Q_nn) = colsum(V ⊙ V).
        v = torch.linalg.solve_triangular(factor, k_mn, upper=False)
        q_diag = v.square().sum(dim=-2)
        if self.method == "fitc":
            lam = (k_diag - q_diag).clamp_min(0.0) + noise[:, None]
        else:
            lam = noise[:, None].expand(size, n)
        residual = _expand_batch(observed - mean, size)
        eye_m = torch.eye(v.shape[-2], dtype=dtype, device=device)
        v_scaled = v / lam.unsqueeze(-2)
        # Σ = Vᵀ V + Λ and B = I + V Λ⁻¹ Vᵀ = L_B L_Bᵀ, so |Σ| = |Λ| |B|.
        factor_b = torch.linalg.cholesky(eye_m + v_scaled @ v.transpose(-2, -1))
        # Woodbury: Σ⁻¹ r = Λ⁻¹ r - Λ⁻¹ Vᵀ L_B⁻ᵀ c with c = L_B⁻¹ V Λ⁻¹ r.
        c = torch.linalg.solve_triangular(factor_b, (v_scaled @ residual.unsqueeze(-1)), upper=False)
        correction = torch.linalg.solve_triangular(factor_b.transpose(-2, -1), c, upper=True).squeeze(-1)
        alpha = (residual - (v.transpose(-2, -1) @ correction.unsqueeze(-1)).squeeze(-1)) / lam
        log_det = torch.log(lam).sum(-1) + 2 * torch.log(torch.diagonal(factor_b, dim1=-2, dim2=-1)).sum(-1)
        quadratic = (residual.square() / lam).sum(-1) - c.square().sum(dim=(-2, -1))
        log_probability = -0.5 * (quadratic + log_det + n * math.log(2 * math.pi))
        trace_term = (k_diag - q_diag).sum(-1)
        if self.method == "vfe":
            log_probability = log_probability - 0.5 * trace_term / noise
        batched = context.any([self.inputs, self.inducing_points, self.noise_variance, self.mean, *self.kernel.parameters])
        return {
            "context": context, "x": x, "z": z, "noise": noise, "factor": factor, "jitter_info": jitter_info,
            "v": v, "lam": lam, "factor_b": factor_b, "alpha": alpha, "trace_term": trace_term,
            "kernel_values": kernel_values, "log_probability": log_probability, "batched": batched,
        }

    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        return _finish(-parts["alpha"], parts["batched"])

    def _cotangents(self, parts):
        r"""Cotangents of the log density w.r.t. ``K_mn``, ``K_mm`` and ``diag(K_nn)`` (batched).

        Every gradient below is linear in the covariance differentials, so it
        is ``⟨G_mn, dK_mn⟩ + ⟨G_mm, dK_mm⟩ + ⟨g_nn, d diag(K_nn)⟩``. With
        ``Σ = Q_nn + Λ``, ``α = Σ⁻¹ r``, ``W = K_mm⁻¹ K_mn``, ``P = W Σ⁻¹`` and
        ``ω = ½ (α² - diag(Σ⁻¹))``:

        * Gaussian term (all methods), from ``dQ = dK_nm W + Wᵀ dK_mn - Wᵀ dK_mm W``
          and ``½ αᵀ dΣ α - ½ tr(Σ⁻¹ dΣ)``:
          ``G_mn = (Wα) αᵀ - P`` and ``G_mm = -½ (Wα)(Wα)ᵀ + ½ P Wᵀ``;
        * FITC (``dΛ = d diag(K_nn) - diag(dQ)``):
          ``G_mn -= 2 W diag(ω)``, ``G_mm += W diag(ω) Wᵀ``, ``g_nn = ω``;
        * VFE trace term ``-tr(K_nn - Q_nn) / (2σ²)``:
          ``G_mn += W / σ²``, ``G_mm -= ½ W Wᵀ / σ²``, ``g_nn = -1 / (2σ²)``.

        ``G_mm`` is finally corrected for the scale-dependent jitter
        ``c max(1, max diag K_mm)``: ``G_mm[i*, i*] += c tr(G_mm)``.
        """
        factor, factor_b, v, lam, alpha = parts["factor"], parts["factor_b"], parts["v"], parts["lam"], parts["alpha"]
        noise = parts["noise"]
        w = torch.linalg.solve_triangular(factor.transpose(-2, -1), v, upper=True)  # K_mm⁻¹ K_mn, (B, m, n)
        w_alpha = (w @ alpha.unsqueeze(-1)).squeeze(-1)
        # Σ⁻¹ = Λ⁻¹ - Gᵀ G with G = L_B⁻¹ V Λ⁻¹.
        g = torch.linalg.solve_triangular(factor_b, v / lam.unsqueeze(-2), upper=False)
        sigma_inverse_diag = 1.0 / lam - g.square().sum(dim=-2)
        p = w / lam.unsqueeze(-2) - (w @ g.transpose(-2, -1)) @ g  # W Σ⁻¹, (B, m, n)
        diag_weight = 0.5 * (alpha.square() - sigma_inverse_diag)
        g_mn = w_alpha.unsqueeze(-1) * alpha.unsqueeze(-2) - p
        g_mm = -0.5 * w_alpha.unsqueeze(-1) * w_alpha.unsqueeze(-2) + 0.5 * p @ w.transpose(-2, -1)
        g_nn = None
        if self.method == "fitc":
            g_mn = g_mn - 2.0 * w * diag_weight.unsqueeze(-2)
            g_mm = g_mm + (w * diag_weight.unsqueeze(-2)) @ w.transpose(-2, -1)
            g_nn = diag_weight
        elif self.method == "vfe":
            g_mn = g_mn + w / noise[:, None, None]
            g_mm = g_mm - 0.5 * (w @ w.transpose(-2, -1)) / noise[:, None, None]
            g_nn = (-0.5 / noise)[:, None].expand_as(alpha)
        g_mm = _jitter_cotangent(parts["jitter_info"], g_mm)
        return g_mn, g_mm, g_nn, diag_weight

    def log_pdf_param_grads(self, x):
        r"""Hand-derived gradients of the approximate log marginal likelihood.

        The cotangents ``G_mn``, ``G_mm`` and ``g_nn`` of :meth:`_cotangents`
        are contracted with the kernel derivatives:

        * a kernel hyperparameter ``θ`` gets
          ``Σ(G_mn ⊙ ∂K_mn/∂θ) + Σ(G_mm ⊙ ∂K_mm/∂θ) + g_nn · ∂diag(K_nn)/∂θ``;
        * the noise variance gets ``Σ_i ω_i`` (plus ``tr(K_nn - Q_nn)/(2σ⁴)``
          for VFE);
        * learnable inducing inputs ``Z`` get
          ``kernel.input_vjp(Z, X, G_mn) + kernel.input_vjp(Z, Z, G_mm + G_mmᵀ)``
          (``K_nn`` does not depend on ``Z``);
        * the mean gets ``α``.
        """
        parts = self._components(x)
        context, x_values, z = parts["context"], parts["x"], parts["z"]
        noise, alpha, kernel_values = parts["noise"], parts["alpha"], parts["kernel_values"]
        g_mn, g_mm, g_nn, diag_weight = self._cotangents(parts)

        cross = self.kernel.batch_derivatives(z, x_values, kernel_values)
        inducing = self.kernel.batch_derivatives(z, z, kernel_values)
        diagonal = self.kernel.batch_diag_derivatives(x_values, kernel_values) if g_nn is not None else {}
        gradients = {}
        for name in {**cross, **inducing, **diagonal}:
            gradient = 0.0
            if name in cross:
                gradient = gradient + _contract(g_mn, cross[name])
            if name in inducing:
                gradient = gradient + _contract(g_mm, inducing[name])
            if name in diagonal:
                gradient = gradient + _contract_diagonal(g_nn, diagonal[name])
            parameter = next(p for p in self.kernel.parameters if p.name == name)
            gradients[name] = _to_parameter_shape(gradient, _parameter_shape(context, parameter))

        if self._learnable_inducing():
            z_gradient = self.kernel.batch_input_vjp(z, x_values, g_mn, kernel_values)
            z_gradient = z_gradient + self.kernel.batch_input_vjp_symmetric(z, g_mm, kernel_values)
            shape = _parameter_shape(context, self.inducing_points)
            gradients[self.inducing_points.name] = z_gradient.reshape(z_gradient.shape[0], *shape)

        if _is_named(self.noise_variance):
            noise_gradient = diag_weight.sum(-1)
            if self.method == "vfe":
                noise_gradient = noise_gradient + 0.5 * parts["trace_term"] / noise.square()
            gradients[self.noise_variance.name] = _to_parameter_shape(noise_gradient.unsqueeze(-1), _parameter_shape(context, self.noise_variance))

        if _is_named(self.mean):
            gradients[self.mean.name] = _to_parameter_shape(alpha, _parameter_shape(context, self.mean))

        return _finish(gradients, parts["batched"])

    def sample(self, n_samples=1):
        """Draw from the approximate prior ``Normal(mean, Q_nn + Λ)``.

        Uses ``Vᵀ ε₁ + sqrt(Λ) ε₂`` with ``V = L⁻¹ K_mn``, so no ``n x n``
        covariance is formed.
        """
        x = self._training_inputs()
        parts = self._components(torch.zeros(_feature_count(x), dtype=x.dtype, device=x.device))
        v, lam = parts["v"][0], parts["lam"][0]
        mean = _resolved_value(self.mean, x.dtype, x.device)
        if mean.numel() == 1:
            mean = mean.reshape(()).expand(lam.numel())
        first = torch.randn((n_samples, v.shape[0]), dtype=x.dtype, device=x.device) @ v
        second = torch.randn((n_samples, lam.numel()), dtype=x.dtype, device=x.device) * lam.sqrt()
        return mean + first + second


def gp_predictive(gp, trace, inputs, n_samples=1, observation_noise=None, new_mean=None):
    """Sample latent GP values at new inputs from posterior latent traces.

    ``gp`` is a :class:`LatentGP`. ``trace`` may be a ``SamplingResult`` or
    a mapping with values of shape ``(n_chains, n_draws, *shape)``
    containing the GP's deterministic trace (``trace[gp.name]``). Kernel
    parameters, a learned (named) ``gp_mean`` and learnable training inputs
    are read from the trace draw by draw; a node missing from the trace
    keeps its current value. For every draw ``f`` the function at ``inputs``
    is drawn ``n_samples`` times from the noise-free GP conditional
    ``N(m* + K_*X K⁻¹ (f - m), K_** - K_*X K⁻¹ K_X*)``; a given
    ``observation_noise`` variance is added to the diagonal.

    ``new_mean`` is the mean ``m*`` at the new inputs (shape ``(n_new,)``,
    or ``(n_parameter_draws, n_new)`` for one per draw). A scalar mean is
    used at the new inputs when it is omitted; a mean given per training
    input (a vector or vector-valued node) cannot be evaluated at new
    inputs, so ``new_mean`` is then required (``ValueError`` otherwise).
    The result has shape ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not hasattr(gp, "gp_kernel"):
        raise TypeError("gp must be a LatentGP function returned by BayesianDLL.GP.LatentGP.")
    if not hasattr(trace, "__getitem__"):
        raise TypeError("trace must be a SamplingResult or mapping-like object.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    function_trace = torch.as_tensor(trace[gp.name])
    latent_draws = function_trace.reshape(-1, function_trace.shape[-1])
    dtype, device = latent_draws.dtype, latent_draws.device
    total_draws = latent_draws.shape[0]
    training_shape = _input_value(gp.gp_inputs).shape
    training_draws = _draws_of(trace, gp.gp_inputs, total_draws, dtype, device)
    new_values = _input_value(inputs).to(dtype=dtype, device=device)
    n_new = new_values.shape[0] if new_values.ndim > 0 else 1
    mean_train, mean_new = _prediction_means(
        _draws_of(trace, gp.gp_mean, total_draws, dtype, device), new_mean,
        _feature_count(gp.gp_inputs), n_new, total_draws, dtype, device, "LatentGP",
    )
    kernel_draws = {
        parameter.name: _draws_of(trace, parameter, total_draws, dtype, device)
        for parameter in gp.gp_kernel.parameters
    }

    outputs = []
    for draw_index in range(total_draws):
        parameter_values = {
            name: draws[draw_index if draws.shape[0] > 1 else 0] for name, draws in kernel_draws.items()
        }
        training_values = training_draws[draw_index if training_draws.shape[0] > 1 else 0].reshape(training_shape)

        covariance = gp.gp_kernel(training_values, training_values, parameter_values=parameter_values)
        cross_covariance = gp.gp_kernel(training_values, new_values, parameter_values=parameter_values)
        new_covariance = gp.gp_kernel(new_values, new_values, parameter_values=parameter_values)
        factor = stable_cholesky(covariance, jitter=gp.gp_jitter)
        residual = latent_draws[draw_index] - mean_train[draw_index]
        alpha = torch.cholesky_solve(residual.reshape(-1, 1), factor).reshape(-1)
        predictive_mean = mean_new[draw_index] + cross_covariance.transpose(-2, -1) @ alpha
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


def _prediction_means(mean_draws, new_mean, n_training, n_new, n_draws, dtype, device, what):
    """Per-draw means ``(n_draws, n_training)`` at the training and ``(n_draws, n_new)`` at the new inputs.

    ``mean_draws`` has shape ``(n_draws or 1, *mean_shape)``. A scalar mean
    (a number or a named scalar node) is used at the new inputs as well. A
    mean given per training input (a vector or a vector-valued node) cannot
    be evaluated at new inputs: the mean there must then be passed as
    ``new_mean`` (shape ``(n_new,)``, or ``(n_draws, n_new)`` for one per
    draw), whatever the number of new inputs. A given ``new_mean`` is also
    used with a scalar mean.
    """
    flat = torch.as_tensor(mean_draws).to(dtype=dtype, device=device)
    flat = flat.reshape(flat.shape[0], -1)
    if flat.shape[0] not in (1, n_draws):
        raise ValueError(f"The trace of the {what} mean has {flat.shape[0]} draws; expected {n_draws}.")
    if flat.shape[1] == 1:
        mean_train = flat.expand(n_draws, n_training)
        if new_mean is None:
            return mean_train, flat.expand(n_draws, n_new)
    elif flat.shape[1] == n_training:
        mean_train = flat.expand(n_draws, n_training)
        if new_mean is None:
            raise ValueError(
                f"The {what} mean is given per training input and cannot be evaluated at the new inputs; "
                f"pass new_mean= (the mean at the {n_new} new inputs, shape (n_new,) or (n_draws, n_new))."
            )
    else:
        raise ValueError(
            f"The {what} mean must be scalar or contain one value per training input ({n_training}); got {flat.shape[1]} values."
        )
    value = _input_value(new_mean) if _is_named(new_mean) else _as_tensor(new_mean)
    value = torch.as_tensor(value).to(dtype=dtype, device=device)
    if value.numel() == 1:
        return mean_train, value.reshape(1, 1).expand(n_draws, n_new)
    if value.shape[-1] != n_new or value.reshape(-1, n_new).shape[0] not in (1, n_draws):
        raise ValueError(f"new_mean must have shape ({n_new},) or ({n_draws}, {n_new}); got {tuple(value.shape)}.")
    return mean_train, value.reshape(-1, n_new).expand(n_draws, n_new)


def exact_gp_predictive(
    gp,
    trace,
    observed_values,
    inputs,
    n_samples=1,
    observation_noise=None,
    new_mean=None,
):
    """Sample exact-GP predictions conditional on observed training values.

    ``gp`` is an :class:`ExactGP`, ``observed_values`` the training
    observations ``y`` (``(N,)``) and ``inputs`` the new inputs. ``trace``
    (values of shape ``(n_chains, n_draws, *shape)``) contains only
    hyperparameter draws; the latent function is integrated out during
    inference and is sampled here from the GP posterior
    ``mean = m* + K_*X (K + σ² I)⁻¹ (y - m)`` and
    ``cov = K_** - K_*X (K + σ² I)⁻¹ K_X*``. Every named kernel parameter,
    the noise variance and the mean must be in the trace. A given
    ``observation_noise`` variance is added to the diagonal to predict
    noisy observations.

    ``new_mean`` is the mean ``m*`` at the new inputs (shape ``(n_new,)``,
    or ``(n_parameter_draws, n_new)`` for one per draw). A scalar mean is
    used at the new inputs when it is omitted; a mean given per training
    input (a vector or vector-valued node) cannot be evaluated at new
    inputs, so ``new_mean`` is then required (``ValueError`` otherwise,
    even when the number of new inputs equals the number of training
    inputs). The returned tensor has shape
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
    mean_train_draws, mean_new_draws = _prediction_means(
        _trace_draws(trace, gp.mean, n_draws, dtype, device), new_mean,
        n_training, new_values.shape[0], n_draws, dtype, device, "ExactGP",
    )

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
        mean_train = mean_train_draws[draw_index]
        mean_new = mean_new_draws[draw_index]

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
    new_mean=None,
):
    r"""Sample sparse-GP predictions conditional on observed training values.

    For every hyperparameter draw in ``trace`` the inducing-point posterior
    ``q(u) = N(K_mm Σ_u K_mn Λ⁻¹ r, K_mm Σ_u K_mm)`` with
    ``Σ_u = (K_mm + K_mn Λ⁻¹ K_nm)⁻¹`` (Titsias' optimal ``q(u)`` for VFE,
    the DTC posterior, and for FITC the same with its diagonal ``Λ``) is
    projected to the new inputs:

    ``mean = K_*m Σ_u K_mn Λ⁻¹ r`` and ``cov = K_** - Q_** + K_*m Σ_u K_m*``.

    Here ``r = y - mean`` with ``y = observed_values``. All solves use
    ``m x m`` Cholesky factors, so the cost is linear in the number of
    training points. Kernel parameters, the noise variance and a named
    mean must be in ``trace``; learnable inducing inputs are read from the
    trace draw by draw. A given ``observation_noise`` variance is added to
    the diagonal.

    ``new_mean`` is the mean at the new inputs (shape ``(n_new,)``, or
    ``(n_parameter_draws, n_new)`` for one per draw). A scalar mean is used
    at the new inputs when it is omitted; a mean given per training input
    cannot be evaluated at new inputs, so ``new_mean`` is then required
    (``ValueError`` otherwise). The result has shape
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
    mean_train_draws, mean_new_draws = _prediction_means(
        _trace_draws(trace, gp.mean, n_draws, dtype, device), new_mean,
        n_training, n_new, n_draws, dtype, device, "SparseGP",
    )
    eye_m = torch.eye(_feature_count(inducing_values), dtype=dtype, device=device)
    learnable_inducing = _is_named(gp.inducing_points) and not isinstance(gp.inducing_points, Data)
    if learnable_inducing:
        inducing_draws = _trace_draws(trace, gp.inducing_points, n_draws, dtype, device)

    outputs = []
    for draw_index in range(n_draws):
        kernel_values = {name: values[draw_index] for name, values in parameter_draws.items()}
        if learnable_inducing:
            inducing_values = inducing_draws[draw_index].reshape(gp._inducing_inputs(dtype, device).shape)
        noise = noise_draws[draw_index].reshape(-1)
        if noise.numel() != 1:
            raise ValueError("Sparse GP noise_variance must be scalar.")
        mean_train = mean_train_draws[draw_index]
        mean_new = mean_new_draws[draw_index]

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


# ----------------------------------------------------------------------------
# Sparse latent (whitened inducing-point) GP
# ----------------------------------------------------------------------------


class SparseLatentGP(DeterministicParameter):
    r"""A whitened inducing-point latent GP function node (DTC / projected process).

    With ``m`` inducing inputs ``Z``, ``L Lᵀ = K_mm(Z) + jitter I`` and a
    standard-normal *whitened* vector ``v`` of shape ``(m,)``, the value is

    ``f(X) = mean + K_nm L⁻ᵀ v``.

    Equivalently ``u = L v ~ Normal(0, K_mm)`` are the function values at
    ``Z`` and ``f = K_nm K_mm⁻¹ u`` is their kernel interpolation, so the
    prior covariance of ``f`` is the Nyström / DTC approximation
    ``Q_nn = K_nm K_mm⁻¹ K_mn`` of ``K_nn``. Only ``m x m`` matrices are
    factorised and ``m`` latent values are sampled, whatever the number of
    training inputs ``n``; with ``Z = X`` (``m = n``) the node equals
    :class:`LatentGP` up to the jitter.

    No FITC-style diagonal correction ``diag(K_nn - Q_nn)`` is added to the
    prior: it would need ``n`` further latent values and a square root whose
    derivative is infinite where the correction vanishes. The prior variance
    of ``f`` is therefore smaller than ``k(x, x)`` away from the inducing
    inputs; this is the DTC approximation, and it is why the inducing inputs
    should cover the data (or be learned).

    Arguments:

    * ``inputs``: a named ``Data`` node (or a learnable parameter) of shape
      ``(n, D)`` or ``(n,)``;
    * ``inducing_points``: ``(m, D)`` or ``(m,)``; a fixed tensor, a ``Data``
      node, or a learnable ``RandomParameter`` / ``DeterministicParameter``
      (e.g. ``RandomParameter("Z", Normal(Z0, 1.0), initial_value=Z0)``);
    * ``whitened_values``: a ``RandomParameter`` with ``m`` values and a
      ``Normal(0, 1)`` prior; ``None`` creates ``<name>_white``;
    * ``mean``: a scalar, a vector with one value per input, or a named
      parameter (default ``0.0``);
    * ``jitter``: relative diagonal jitter of ``K_mm`` (default ``1e-6``).

    The node stores them as ``gp_inputs``, ``gp_inducing_points``,
    ``gp_latent`` (the whitened values), ``gp_kernel``, ``gp_mean`` and
    ``gp_jitter``.

    **Gradients** (hand-derived VJP). For a cotangent ``g`` of ``f`` let
    ``a = L⁻ᵀ v``, so that ``f = mean + K_nm a``:

    * ``v`` gets ``c = L⁻¹ K_mn g``;
    * ``K_mn = K(Z, X)`` gets ``G_mn = a gᵀ``;
    * ``L`` gets ``L̄ = -a cᵀ`` (from ``da = -L⁻ᵀ dLᵀ a``), and the Cholesky
      VJP maps it to ``G_mm = sym(L⁻ᵀ Φ(Lᵀ L̄) L⁻¹)`` with ``Lᵀ L̄ = -v cᵀ``
      (``Φ`` takes the lower triangle and halves the diagonal), plus the
      correction for the scale-dependent jitter;
    * a kernel hyperparameter ``θ`` gets
      ``⟨G_mn, ∂K_mn/∂θ⟩ + ⟨G_mm, ∂K_mm/∂θ⟩``;
    * learnable inducing inputs get
      ``kernel.input_vjp(Z, X, G_mn) + kernel.input_vjp_symmetric(Z, G_mm)``
      and learnable training inputs ``kernel.input_vjp(X, Z, G_mnᵀ)``;
    * the mean gets ``g``.

    With vectorized chains everything is batched over the chain dimension
    (``v`` ``(C, m)``, ``Z`` ``(C, m, D)``, hyperparameters ``(C, ...)``).
    Use :func:`sparse_latent_gp_predictive` for predictions at new inputs.
    """

    def __init__(
        self,
        name,
        inputs,
        kernel: Kernel,
        inducing_points,
        whitened_values=None,
        mean=0.0,
        jitter=1e-6,
    ):
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")
        if not _is_named(inducing_points):
            inducing_points = _as_tensor(inducing_points)
            if not inducing_points.is_floating_point():
                inducing_points = inducing_points.to(torch.get_default_dtype())
        m = _feature_count(inducing_points)
        if whitened_values is None:
            whitened_values = RandomParameter(f"{name}_white", Normal(0.0, 1.0), shape=m)
        if not _is_named(whitened_values) or isinstance(whitened_values, Data):
            raise TypeError("whitened_values must be a named model parameter node.")
        if whitened_values.constrained_value.numel() != m:
            raise ValueError(
                f"whitened_values must contain one value per inducing input ({m}); "
                f"got {whitened_values.constrained_value.numel()}."
            )

        kernel_parameters = list(kernel.parameters)
        model_inputs = [inputs, whitened_values]
        self._inducing_index = None
        if _is_named(inducing_points):
            self._inducing_index = len(model_inputs)
            model_inputs.append(inducing_points)
        self._kernel_start = len(model_inputs)
        model_inputs.extend(kernel_parameters)
        mean_input = mean if _is_named(mean) else None
        if mean_input is not None:
            model_inputs.append(mean_input)
        self._kernel_parameters = kernel_parameters
        self._mean_input = mean_input
        self._fixed_inducing = None if _is_named(inducing_points) else inducing_points

        def forward(*values):
            return self._batched_function(values, [False] * len(values))[0][0]

        def vjp(cotangent, *values):
            flags = [False] * len(values)
            return _finish(self._batched_vjp(_as_tensor(cotangent).unsqueeze(0), values, flags), False)

        super().__init__(name, forward, None, model_inputs, vjp_func=vjp)
        self.gp_inputs = inputs
        self.gp_inducing_points = inducing_points
        self.gp_latent = whitened_values
        self.gp_kernel = kernel
        self.gp_mean = mean
        self.gp_jitter = jitter

    @staticmethod
    def _learnable(node):
        return not isinstance(node, Data) and isinstance(node, (RandomParameter, DeterministicParameter))

    def _batched_function(self, values, flags):
        """``(f, pieces)`` in batched form for input ``values`` with chain flags ``flags``."""
        context = _BatchContext({id(node): bool(flag) for node, flag in zip(self.inputs, flags)})
        v = context.value(self.inputs[1], raw=values[1])
        v = v.reshape(v.shape[0], -1)
        dtype, device = v.dtype, v.device
        x = context.inputs(self.inputs[0], dtype, device, raw=values[0])
        if self._inducing_index is not None:
            z = context.inputs(self.inputs[self._inducing_index], dtype, device, raw=values[self._inducing_index])
        else:
            z = context.inputs(self._fixed_inducing, dtype, device)
        stop = self._kernel_start + len(self._kernel_parameters)
        raw = {p.name: value for p, value in zip(self.inputs[self._kernel_start:stop], values[self._kernel_start:stop])}
        kernel_values = context.kernel_values(self.gp_kernel, dtype, device, raw)
        k_mm = self.gp_kernel.batch_evaluate(z, z, kernel_values)
        k_mn = self.gp_kernel.batch_evaluate(z, x, kernel_values)
        size = max(k_mm.shape[0], k_mn.shape[0], v.shape[0])
        factor, jitter_info = batched_stable_cholesky(_expand_batch(k_mm, size), jitter=self.gp_jitter)
        # a = L⁻ᵀ v, so that f = mean + K_nm a.
        a = torch.linalg.solve_triangular(factor.transpose(-2, -1), v.unsqueeze(-1), upper=True)
        n = x.shape[1]
        if self._mean_input is not None:
            mean = _batched_mean(context, self._mean_input, n, dtype, device, "SparseLatentGP", raw=values[-1])
        else:
            mean = _mean_value(self.gp_mean, torch.zeros(n, dtype=dtype, device=device)).reshape(1, n)
        f = mean + (k_mn.transpose(-2, -1) @ a).squeeze(-1)
        return f, (context, x, z, v, kernel_values, k_mn, factor, jitter_info, a)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, x, z, v, kernel_values, k_mn, factor, jitter_info, a) = self._batched_function(values, flags)
        g = cotangent.reshape(cotangent.shape[0], -1).to(dtype=v.dtype)

        def shape_of(node, value):
            value = _as_tensor(value)
            return tuple(value.shape[1:]) if context.is_batched(node) else tuple(value.shape)

        grads = {}
        # v gets c = L⁻¹ K_mn g.
        c = torch.linalg.solve_triangular(factor, k_mn @ g.unsqueeze(-1), upper=False)  # (B, m, 1)
        latent = self.inputs[1]
        grads[latent.name] = c.reshape(c.shape[0], *shape_of(latent, values[1]))
        # K_mn gets G_mn = a gᵀ.
        g_mn = a * g.unsqueeze(-2)  # (B, m, n)
        # L gets L̄ = -a cᵀ; Lᵀ L̄ = -v cᵀ, which the Cholesky VJP maps to K_mm.
        outer = -(v.unsqueeze(-1) * c.transpose(-2, -1))
        lower = torch.tril(outer, diagonal=-1) + 0.5 * torch.diag_embed(torch.diagonal(outer, dim1=-2, dim2=-1))
        left = torch.linalg.solve_triangular(factor.transpose(-2, -1), lower, upper=True)
        g_mm = torch.linalg.solve_triangular(factor, left, upper=False, left=False)
        g_mm = _jitter_cotangent(jitter_info, 0.5 * (g_mm + g_mm.transpose(-2, -1)))
        size = max(g_mn.shape[0], g_mm.shape[0])
        g_mn, g_mm = _expand_batch(g_mn, size), _expand_batch(g_mm, size)

        cross = self.gp_kernel.batch_derivatives(z, x, kernel_values)
        inducing = self.gp_kernel.batch_derivatives(z, z, kernel_values)
        for name in {**cross, **inducing}:
            gradient = 0.0
            if name in cross:
                gradient = gradient + _contract(g_mn, cross[name])
            if name in inducing:
                gradient = gradient + _contract(g_mm, inducing[name])
            parameter = next(p for p in self._kernel_parameters if p.name == name)
            value = values[self.inputs.index(parameter)]
            grads[name] = _to_parameter_shape(gradient, shape_of(parameter, value))

        if self._inducing_index is not None and self._learnable(self.inputs[self._inducing_index]):
            node = self.inputs[self._inducing_index]
            z_full = _expand_batch(z, size)
            z_gradient = self.gp_kernel.batch_input_vjp(z_full, x, g_mn, kernel_values)
            z_gradient = z_gradient + self.gp_kernel.batch_input_vjp_symmetric(z_full, g_mm, kernel_values)
            grads[node.name] = z_gradient.reshape(size, *shape_of(node, values[self._inducing_index]))
        if self._learnable(self.inputs[0]):
            node = self.inputs[0]
            x_gradient = self.gp_kernel.batch_input_vjp(_expand_batch(x, size), z, g_mn.transpose(-2, -1), kernel_values)
            grads[node.name] = x_gradient.reshape(size, *shape_of(node, values[0]))
        if self._mean_input is not None:
            grads[self._mean_input.name] = _to_parameter_shape(g, shape_of(self._mean_input, values[-1]))
        return grads

    # -- vectorized chains -------------------------------------------------
    def batched_forward(self, input_values):
        f, _ = self._batched_function(input_values, self._batched_flags())
        return _expand_batch(f, self.owner_model.n_chains)

    def batched_vjp(self, cotangent, input_values):
        flags = self._batched_flags()
        grads = self._batched_vjp(cotangent, input_values, flags)
        result = {}
        for node, flag in zip(self.inputs, flags):
            name = getattr(node, "name", None)
            if flag and name in grads:
                result[name] = _expand_batch(grads[name], self.owner_model.n_chains)
        return result


def _draws_of(trace, node, n_draws, dtype, device):
    """``(n_draws, *shape)`` draws of ``node`` from ``trace``, or ``(1, *shape)`` when it is not traced."""
    if _is_named(node) and not isinstance(node, Data) and node.name in trace:
        values = torch.as_tensor(trace[node.name])
        values = values.reshape(-1, *values.shape[2:]) if values.ndim >= 2 else values.reshape(-1, 1)
        if values.shape[0] != n_draws:
            raise ValueError(f"The trace of '{node.name}' has {values.shape[0]} draws; expected {n_draws}.")
        return values.to(dtype=dtype, device=device)
    value = _input_value(node) if _is_named(node) else _as_tensor(node)
    return torch.as_tensor(value).to(dtype=dtype, device=device).unsqueeze(0)


def sparse_latent_gp_predictive(gp, trace, new_inputs, n_samples=1, conditional_variance=False, chunk_size=64, new_mean=None):
    r"""Values of a :class:`SparseLatentGP` at new inputs for every draw of ``trace``.

    Under the DTC approximation the function is deterministic given the
    whitened values ``v``, the inducing inputs and the hyperparameters:
    ``f* = mean + K_*m L⁻ᵀ v`` at ``new_inputs``, so the ``n_samples``
    samples per draw are identical by default. With
    ``conditional_variance=True`` (default ``False``)
    independent Normal noise with the FITC-style variance
    ``diag(K_** - Q_**)`` (the prior variance that the inducing values do not
    explain) is added to every sample.

    ``trace`` is a ``SamplingResult`` (from NUTS or ``guide.sample``) or a
    mapping with values of shape ``(n_chains, n_draws, *shape)``. It must
    contain the whitened values; learnable inducing inputs, kernel
    parameters and a named mean are read from it as well (a node missing
    from the trace keeps its current value). Draws are evaluated in batches
    of ``chunk_size`` (default ``64``) with the batched kernel API.

    ``new_mean`` is the mean at the new inputs (shape ``(n_new,)``, or
    ``(n_parameter_draws, n_new)`` for one per draw). A scalar mean is used
    at the new inputs when it is omitted; a mean given per training input
    cannot be evaluated at new inputs, so ``new_mean`` is then required
    (``ValueError`` otherwise). The result has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, SparseLatentGP):
        raise TypeError("gp must be a SparseLatentGP node returned by BayesianDLL.GP.SparseLatentGP.")
    if not hasattr(trace, "__getitem__"):
        raise TypeError("trace must be a SamplingResult or mapping-like object.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    latent = torch.as_tensor(trace[gp.gp_latent.name])
    dtype, device = latent.dtype, latent.device
    v = latent.reshape(-1, latent.shape[-1])
    n_draws = v.shape[0]

    z = _draws_of(trace, gp.gp_inducing_points, n_draws, dtype, device)
    if z.ndim == 2:
        z = z.unsqueeze(-1)
    new = torch.as_tensor(_input_value(new_inputs)).to(dtype=dtype, device=device)
    new = (new.reshape(-1, 1) if new.ndim <= 1 else new).unsqueeze(0)
    kernel_draws = {p.name: _draws_of(trace, p, n_draws, dtype, device) for p in gp.gp_kernel.parameters}
    _, mean = _prediction_means(
        _draws_of(trace, gp.gp_mean, n_draws, dtype, device), new_mean,
        _feature_count(gp.gp_inputs), new.shape[1], n_draws, dtype, device, "SparseLatentGP",
    )

    def take(tensor, start, stop):
        return tensor if tensor.shape[0] == 1 else tensor[start:stop]

    outputs = []
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        size = stop - start
        values = {name: take(draws, start, stop) for name, draws in kernel_draws.items()}
        z_chunk = take(z, start, stop)
        k_mm = gp.gp_kernel.batch_evaluate(z_chunk, z_chunk, values)
        k_ms = gp.gp_kernel.batch_evaluate(z_chunk, new, values)
        factor, _ = batched_stable_cholesky(_expand_batch(k_mm, size), jitter=gp.gp_jitter)
        a = torch.linalg.solve_triangular(factor.transpose(-2, -1), v[start:stop].unsqueeze(-1), upper=True)
        f = take(mean, start, stop) + (k_ms.transpose(-2, -1) @ a).squeeze(-1)  # (size, n_new)
        f = f.unsqueeze(1).expand(size, n_samples, f.shape[-1])
        if conditional_variance:
            projected = torch.linalg.solve_triangular(factor, _expand_batch(k_ms, size), upper=False)
            k_diag = _expand_batch(gp.gp_kernel.batch_diag(new, values), size)
            variance = (k_diag - projected.square().sum(-2)).clamp_min(0.0)
            f = f + variance.sqrt().unsqueeze(1) * torch.randn(f.shape, dtype=dtype, device=device)
        outputs.append(f)
    return torch.cat(outputs)
