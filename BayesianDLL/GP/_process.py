"""Latent and exact Gaussian processes (dense or with inducing points) for MCMC and variational inference.

:class:`ExactGP` and :class:`LatentGP` are the public classes. Each builds one
of two private variants: the dense one, or with ``inducing_points`` the
inducing-point one (``_SparseExactGP``, ``_SparseLatentGP``). Both variants
are instances of the public class.

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
  except :func:`_gp_predictive` with inducing points, which batches the draws.

Learnable inputs
----------------
Every GP class except ``ToeplitzGP`` accepts learnable training inputs: a
``RandomParameter`` or ``DeterministicParameter`` passed as ``inputs`` (for
``KroneckerGP``, as any grid axis) gets hand-derived gradients through the
kernel's input VJPs and the mean's input VJP, batched over chains like every
other parameter. Typical uses are GP latent variable models (the inputs are
latent coordinates with a prior), uncertain or noisy inputs, and input
warping (inputs computed by deterministic blocks from fixed data). The
predictive functions read learnable inputs from the trace draw by draw.
``ToeplitzGP`` needs fixed, equally spaced inputs and raises a ``TypeError``.
"""

from __future__ import annotations

import inspect
import math

import torch

from .. import _batching
from .._data import Data
from .._parameters import DeterministicParameter, RandomParameter
from ..Distributions._distributions import Distribution
from ..Distributions._resolve import resolve
from ..Distributions import Normal
from ._kernels import Kernel, _is_named
from ._means import MeanFunction, _is_zero, as_mean_function


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


def _is_learnable(node):
    """Whether ``node`` is a learnable model node (a ``RandomParameter`` or ``DeterministicParameter``, not ``Data``)."""
    return not isinstance(node, Data) and isinstance(node, (RandomParameter, DeterministicParameter))


def _feature_count(x):
    x = _input_value(x)
    return x.shape[0] if x.ndim > 0 else 1


def _resolved_value(value, dtype, device, parameter_values=None):
    if parameter_values is not None and _is_named(value) and value.name in parameter_values:
        value = parameter_values[value.name]
    else:
        value = resolve(value)
    return torch.as_tensor(value, dtype=dtype, device=device)


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


def _accumulate(gradients, name, gradient):
    gradients[name] = gradients[name] + gradient if name in gradients else gradient


def _noise_view(value, target, what="GP"):
    """``value`` (shape ``s``) reshaped to broadcast against ``target``; ``ValueError`` if it does not.

    A noise variance is a scalar or has one value per observation: its shape
    must broadcast to the observations' shape ``target`` (e.g. ``(N,)``, or
    ``(n_outputs, N)``). Leading singleton dimensions beyond ``target`` are dropped.
    """
    shape = list(value.shape)
    while len(shape) > len(target) and shape[0] == 1:
        shape.pop(0)
    padded = [1] * (len(target) - len(shape)) + shape
    if len(padded) != len(target) or any(a not in (1, b) for a, b in zip(padded, target)):
        raise ValueError(
            f"{what} noise_variance must be a scalar or have one value per observation (a shape that broadcasts "
            f"to {tuple(target)}); got shape {tuple(value.shape)}."
        )
    return value.reshape(padded)


def _batched_noise(context, node, dtype, device, target, what="GP"):
    """Batched noise variance ``(B, *t)`` broadcastable to ``(B, *target)``, and its unbatched shape."""
    value = context.value(node, dtype, device)
    shape = tuple(value.shape[1:])
    view = _noise_view(value[0], target, what)
    return value.reshape(value.shape[0], *view.shape), shape


def _sum_to_shape(gradient, shape):
    """Sum a batched gradient ``(B, *full)`` to a value of shape ``shape`` that broadcasts to ``full``: ``(B, *shape)``."""
    full = tuple(gradient.shape[1:])
    stripped = list(shape)
    while len(stripped) > len(full) and stripped[0] == 1:
        stripped.pop(0)
    padded = [1] * (len(full) - len(stripped)) + stripped
    dims = tuple(1 + i for i, (a, b) in enumerate(zip(padded, full)) if a == 1 and b != 1)
    if dims:
        gradient = gradient.sum(dim=dims, keepdim=True)
    return gradient.reshape(gradient.shape[0], *shape)


def _evaluate_mean(context, mean, x, dtype, device, raw=None):
    """Mean ``m(X)`` ``(B, n)`` of batched inputs ``x`` ``(B, n, D)`` (``raw`` overrides parameter values)."""
    if _is_zero(mean):
        return torch.zeros((1, x.shape[1]), dtype=dtype, device=device)
    return mean.batch_evaluate(x, context.kernel_values(mean, dtype, device, raw))


def _add_mean_gradients(gradients, mean, context, x, cotangent, shape_of, dtype, device, raw=None, input_node=None, input_shape=None):
    r"""Add ``⟨g, ∂m(X)/∂θ⟩`` for every named mean parameter ``θ`` to ``gradients``.

    ``cotangent`` ``g`` ``(B, n)`` is the gradient w.r.t. ``m(X)``;
    ``shape_of(node)`` gives a parameter's (unbatched) value shape. When
    ``input_node`` (the learnable inputs ``X``) is given, it also gets the
    mean's input VJP ``∂⟨g, m(X)⟩/∂X`` reshaped to ``input_shape``. A
    :class:`ZeroMean` adds nothing.
    """
    if _is_zero(mean):
        return gradients
    values = context.kernel_values(mean, dtype, device, raw)
    for name, derivative in mean.batch_derivatives(x, values).items():
        parameter = next(p for p in mean.parameters if p.name == name)
        _accumulate(gradients, name, _to_parameter_shape(_contract_diagonal(cotangent, derivative), shape_of(parameter)))
    if input_node is not None:
        vjp = mean.batch_input_vjp(x, cotangent, values)
        _accumulate(gradients, input_node.name, vjp.reshape(vjp.shape[0], *input_shape))
    return gradients


def _expand_batch(tensor, size):
    return tensor.expand(size, *tensor.shape[1:]) if tensor.shape[0] != size else tensor


def _finish(value, batched):
    """Drop the batch dimension of an unbatched evaluation."""
    if batched:
        return value
    if isinstance(value, dict):
        return {name: v[0] for name, v in value.items()}
    return value[0]


def _with_parameters(nodes, parameters):
    """``nodes`` followed by the named ``parameters`` that are not among them yet."""
    nodes = list(nodes)
    names = {node.name for node in nodes if _is_named(node)}
    for parameter in parameters:
        if parameter.name not in names:
            names.add(parameter.name)
            nodes.append(parameter)
    return nodes


def _raw_values(nodes, values):
    """``{name: value}`` of the named input nodes of a deterministic node."""
    return {node.name: value for node, value in zip(nodes, values) if _is_named(node)}


# ----------------------------------------------------------------------------
# Latent (whitened) GP
# ----------------------------------------------------------------------------


class LatentGP(DeterministicParameter):
    r"""Explicit whitened latent Gaussian Process function node.

    Represents a latent GP function :math:`\mathbf{f}` via non-centred reparameterization:

    .. math::

        \mathbf{f} = \mathbf{m}(X) + \mathbf{L} \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

    where :math:`\mathbf{L}` is the Cholesky factor of :math:`\mathbf{K}(X, X) + \text{jitter} \cdot \mathbf{I}`.
    This non-centred form is intended for NUTS and variational inference.

    With ``inducing_points`` the function is the whitened inducing-point (DTC)
    approximation :math:`\mathbf{f} = \mathbf{m}(X) + \mathbf{K}_{XZ} \mathbf{L}_Z^{-\top} \mathbf{z}`
    with :math:`\mathbf{L}_Z \mathbf{L}_Z^\top = \mathbf{K}_{ZZ} + \text{jitter} \cdot \mathbf{I}`
    and :math:`M` latent values, whatever the number of inputs. Its prior variance is
    smaller than :math:`k(x, x)` away from the inducing inputs, so they should cover
    the data (or be learned).

    With ``n_basis`` the function is the Hilbert-space approximation (Solin & Särkkä,
    2020; Riutort-Mayol et al., 2023) of a stationary kernel,
    :math:`\mathbf{f} = \mathbf{m}(X) + \boldsymbol{\Phi}(X) \left(\sqrt{S(\sqrt{\boldsymbol{\lambda}})} \odot \mathbf{z}\right)`,
    with the Laplacian eigenfunctions :math:`\phi_j(x) = L^{-1/2} \sin(\pi j (x - c + L) / (2L))`
    and eigenvalues :math:`\lambda_j = (\pi j / (2L))^2` on :math:`[c - L, c + L]` per input
    feature (tensor products for :math:`D > 1`) and the kernel's spectral density :math:`S`
    (:class:`~BayesianDLL.GP.RBF`, :class:`~BayesianDLL.GP.Matern`,
    :class:`~BayesianDLL.GP.SpectralMixture`, sums of these and products with
    :class:`~BayesianDLL.GP.Constant`). It costs :math:`\mathcal{O}(N M)` per evaluation with no
    matrix factorisation, and :math:`M = \prod_d m_d` latent values. With ``centred=True``
    the latent values are the basis weights themselves,
    :math:`\mathbf{f} = \mathbf{m}(X) + \boldsymbol{\Phi}(X) \boldsymbol{\beta}` with
    :math:`\beta_k \sim \mathcal{N}(0, S(\sqrt{\lambda_k}))`.

    Parameters
    ----------
    name : str
        Name of the deterministic function node.
    inputs : Data or Parameter
        Named input node :math:`X` of shape ``(N, D)`` or ``(N,)``: a fixed
        ``Data`` node or a learnable ``RandomParameter`` / ``DeterministicParameter``.
    kernel : Kernel
        Covariance kernel function :math:`k(x_1, x_2)`.
    latent : RandomParameter, optional
        Standard normal latent parameter :math:`\mathbf{z}` with one value per input
        (per inducing input with ``inducing_points``, per basis function with ``n_basis``).
        If ``None``, it is created automatically.
    mean : MeanFunction, optional
        GP mean function :math:`m(x)`. Default is :class:`~BayesianDLL.GP.ZeroMean`.
    jitter : float, default 1e-6
        Relative diagonal jitter for numerical stability.
    latent_name : str, optional
        Name of the latent parameter created when ``latent`` is not provided.
        Defaults to ``f"{name}_white"``.
    inducing_points : torch.Tensor or Data or Parameter, optional
        Inducing inputs :math:`Z` of shape ``(M, D)`` or ``(M,)``; may be learnable.
    n_basis : int or sequence of int, optional
        Number of basis functions :math:`m_d` per input feature (one number for all features)
        of the Hilbert-space approximation.
    boundary_factor : float, optional
        Boundary factor :math:`c_b \geq 1` of the Hilbert-space approximation (default 1.5):
        :math:`L = c_b S` for the half-range :math:`S` of the inputs (per feature, centred at
        the midpoint :math:`c` of their range when the node is built).
    centred : bool, default False
        Centred parameterisation of the Hilbert-space approximation (needs ``n_basis``):
        the latent parameter (``f"{name}_weights"``) holds the basis weights with prior
        standard deviations :math:`\sqrt{S(\sqrt{\lambda_k})}` (the deterministic node
        ``f"{name}_basis_sd"``), and ``latent`` cannot be given. NUTS mixes better in
        this form when the data determine the function well (many or precise
        observations); the default non-centred form suits weakly informative data.

    Notes
    -----
    Posterior predictive sampling at new inputs (set with ``Data.set_value``
    after fitting) draws the function from its GP conditional given the values
    at the training inputs; with inducing points or basis functions the function
    is simply evaluated at the new inputs. Outside of prediction the inputs of the
    dense variant must keep one row per latent value.

    The Hilbert-space approximation is accurate within roughly
    :math:`[c - S, c + S]` when :math:`m_d` and :math:`c_b` are large enough for the
    kernel's lengthscale (Riutort-Mayol et al. suggest :math:`c_b \geq 1.2` and
    :math:`m_d \gtrsim 1.75 c_b S / l`); the basis functions vanish at :math:`c \pm L`, so
    predictions beyond the training range need a larger ``boundary_factor``. The
    boundary is fixed when the node is built, also for learnable inputs.
    """

    def __new__(cls, *args, **kwargs):
        # LatentGP(...) builds the dense, inducing-point or Hilbert-space variant; all are LatentGP instances.
        if cls is LatentGP:
            arguments = LatentGP.__signature__.bind(*args, **kwargs).arguments
            sparse, basis = arguments.get("inducing_points") is not None, arguments.get("n_basis") is not None
            if sparse and basis:
                raise ValueError("LatentGP takes either inducing_points or n_basis, not both.")
            if arguments.get("boundary_factor") is not None and not basis:
                raise ValueError("LatentGP boundary_factor needs n_basis.")
            if arguments.get("centred") and not basis:
                raise ValueError("LatentGP centred needs n_basis.")
            cls = _SparseLatentGP if sparse else _HilbertLatentGP if basis else _DenseLatentGP
        return super().__new__(cls)

    def __init__(
        self,
        name,
        inputs,
        kernel: Kernel,
        latent=None,
        mean=None,
        jitter=1e-6,
        latent_name=None,
        inducing_points=None,
        n_basis=None,
        boundary_factor=None,
        centred=False,
    ):
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")

        mean = as_mean_function(mean, "LatentGP")
        inducing_points = self._prepare_inducing(inducing_points)
        self._prepare_basis(inputs, kernel, n_basis, boundary_factor, centred)
        n = self._latent_count(inputs, inducing_points)
        if latent is None:
            latent = self._make_latent(name, latent_name, kernel, n)
        elif centred:
            raise ValueError("LatentGP creates the basis weights itself when centred=True; latent cannot be given.")
        if not _is_named(latent) or isinstance(latent, Data):
            raise TypeError("latent must be a named model parameter node.")
        if latent.constrained_value.numel() != n:
            raise ValueError(
                f"latent must contain one value per {self._latent_unit} ({n}); "
                f"got {latent.constrained_value.numel()}."
            )

        self._kernel_parameters = list(kernel.parameters)
        model_inputs = self._model_inputs(inputs, latent, inducing_points, mean)

        def forward(*values):
            return self._batched_function(values, [False] * len(values))[0][0]

        def vjp(cotangent, *values):
            flags = [False] * len(values)
            return _finish(self._batched_vjp(_as_tensor(cotangent).unsqueeze(0), values, flags), False)

        super().__init__(name, forward, self._derivative_function(), model_inputs, vjp_func=vjp)
        self.gp_inputs = inputs
        self.gp_latent = latent
        self.gp_kernel = kernel
        self.gp_mean = mean
        self.gp_jitter = jitter
        self._finish_init(inducing_points)

    @property
    def _in_posterior_predictive(self):
        return self.__dict__.get("_predictive_mode", False)

    @_in_posterior_predictive.setter
    def _in_posterior_predictive(self, value):
        # Set by the posterior predictive functions; a new prediction starts without cached draws.
        self.__dict__["_predictive_mode"] = bool(value)
        self.__dict__.pop("_predictive_cache", None)

    def _capture_training_data(self):
        """Remember the training inputs (see :class:`ExactGP`); nothing to do by default."""

    def _prepare_basis(self, inputs, kernel, n_basis, boundary_factor, centred):
        """Set up the basis functions of the Hilbert-space variant; nothing to do by default."""

    def _make_latent(self, name, latent_name, kernel, n):
        """The standard normal latent parameter created when ``latent`` is not given."""
        return RandomParameter(latent_name or f"{name}_white", Normal(0.0, 1.0), shape=n)

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


def _same_tensors(first, second):
    return len(first) == len(second) and all(
        a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b) for a, b in zip(first, second)
    )


class _DenseLatentGP(LatentGP):
    r""":class:`LatentGP` without inducing points: ``f = m(X) + L z`` with ``L Lᵀ = K(X, X) + jitter I``.

    **Gradients** (vector-Jacobian product, hand-derived). For a cotangent
    ``ḡ`` of ``f``: ``z`` gets ``Lᵀ ḡ``; with ``dL = L Φ(L⁻¹ dK L⁻ᵀ)``
    (``Φ`` takes the lower triangle and halves the diagonal) every kernel
    hyperparameter gets ``⟨G_K, ∂K/∂θ⟩`` with ``G_K = sym(L⁻ᵀ S L⁻¹)``,
    ``S = Φ(Lᵀ ḡ zᵀ)``. Learnable inputs get ``kernel.input_vjp(X, X, G_K + G_Kᵀ)``
    and the mean's input VJP; the mean's parameters get ``⟨ḡ, ∂m(X)/∂θ⟩``.
    """

    _latent_unit = "training input"
    _training_inputs_data = None

    def _prepare_inducing(self, inducing_points):
        return None

    def _latent_count(self, inputs, inducing_points):
        return _feature_count(inputs)

    def _model_inputs(self, inputs, latent, inducing_points, mean):
        return _with_parameters([inputs, latent, *self._kernel_parameters], mean.parameters)

    def _derivative_function(self):
        return lambda *values: self._jacobians(values)

    def _finish_init(self, inducing_points):
        self._capture_training_data()

    def _capture_training_data(self):
        """Remember the fixed inputs that the latent values belong to (prediction conditions on them)."""
        if _is_learnable(self.gp_inputs):
            self._training_inputs_data = None
            return
        x = _as_tensor(_input_value(self.gp_inputs))
        if _feature_count(x) == self.gp_latent.constrained_value.numel():
            self._training_inputs_data = x.detach().clone()

    def _training_input_value(self):
        if self._training_inputs_data is None:
            return _as_tensor(_input_value(self.gp_inputs))
        return self._training_inputs_data

    def _context(self, values, flags):
        return _BatchContext({id(node): bool(flag) for node, flag in zip(self.inputs, flags)})

    def _at_new_inputs(self, x):
        """Whether the batched inputs ``x`` differ from the training inputs during prediction."""
        if not self._in_posterior_predictive or self._training_inputs_data is None:
            return False
        train = self._training_inputs_data.to(dtype=x.dtype, device=x.device)
        train = train.reshape(train.shape[0] if train.ndim else 1, -1)
        return tuple(x.shape[1:]) != tuple(train.shape) or not bool((x == train).all())

    def _conditional_draw(self, values, context, x, z, kernel_values, raw):
        """One draw ``(B, n*)`` of ``f`` at new inputs ``x`` from its GP conditional given ``f(X) = m(X) + L z``.

        The draw is cached for the current input values, so every reader of
        the node sees the same function during one predictive draw.
        """
        key = [_as_tensor(value) for value in values]
        cache = self.__dict__.get("_predictive_cache")
        if cache is not None and _same_tensors(cache[0], key):
            return cache[1]
        dtype, device = z.dtype, z.device
        x_train = self._training_inputs_data.to(dtype=dtype, device=device)
        x_train = x_train.reshape(x_train.shape[0] if x_train.ndim else 1, -1).unsqueeze(0)
        factor, _ = batched_stable_cholesky(self.gp_kernel.batch_evaluate(x_train, x_train, kernel_values), jitter=self.gp_jitter)
        # K⁻¹ (f(X) - m(X)) = L⁻ᵀ z
        alpha = torch.linalg.solve_triangular(factor.transpose(-2, -1), z.unsqueeze(-1), upper=True)
        cross = self.gp_kernel.batch_evaluate(x_train, x, kernel_values)
        mean = _evaluate_mean(context, self.gp_mean, x, dtype, device, raw)
        predictive_mean = mean + (cross.transpose(-2, -1) @ alpha).squeeze(-1)
        projected = torch.linalg.solve_triangular(factor, cross, upper=False)
        covariance = self.gp_kernel.batch_evaluate(x, x, kernel_values) - projected.transpose(-2, -1) @ projected
        prediction_factor, _ = batched_stable_cholesky(covariance, jitter=self.gp_jitter)
        noise = torch.randn_like(predictive_mean)
        f = predictive_mean + (prediction_factor @ noise.unsqueeze(-1)).squeeze(-1)
        self.__dict__["_predictive_cache"] = (key, f)
        return f

    def _batched_function(self, values, flags):
        """``(f, pieces)`` in batched form for input ``values`` with chain flags ``flags``."""
        context = self._context(values, flags)
        z = context.value(self.inputs[1], raw=values[1])
        z = z.reshape(z.shape[0], -1)
        dtype, device = z.dtype, z.device
        x = context.inputs(self.inputs[0], dtype, device, raw=values[0])
        raw = _raw_values(self.inputs, values)
        kernel_values = context.kernel_values(self.gp_kernel, dtype, device, raw)
        if x.shape[1] != z.shape[1] or self._at_new_inputs(x):
            if not self._in_posterior_predictive or self._training_inputs_data is None:
                raise ValueError(
                    f"LatentGP '{self.name}' has {z.shape[1]} latent values but its inputs have {x.shape[1]} rows. "
                    "Inputs other than the training inputs are only supported for prediction "
                    "(posterior_predicative or predict)."
                )
            return self._conditional_draw(values, context, x, z, kernel_values, raw), None
        covariance = self.gp_kernel.batch_evaluate(x, x, kernel_values)
        size = max(covariance.shape[0], z.shape[0])
        factor, jitter_info = batched_stable_cholesky(_expand_batch(covariance, size), jitter=self.gp_jitter)
        mean = _evaluate_mean(context, self.gp_mean, x, dtype, device, raw)
        f = mean + (factor @ z.unsqueeze(-1)).squeeze(-1)
        return f, (context, x, z, kernel_values, factor, jitter_info, raw)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, x, z, kernel_values, factor, jitter_info, raw) = self._batched_function(values, flags)
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
        def shape_of(node):
            value = _as_tensor(values[self.inputs.index(node)])
            return tuple(value.shape[1:]) if context.is_batched(node) else tuple(value.shape)

        for name, derivative in self.gp_kernel.batch_derivatives(x, x, kernel_values).items():
            parameter = next(p for p in self._kernel_parameters if p.name == name)
            _accumulate(grads, name, _to_parameter_shape(_contract(covariance_cotangent, derivative), shape_of(parameter)))
        inputs_node = self.inputs[0]
        learnable_inputs = not isinstance(inputs_node, Data) and isinstance(inputs_node, (RandomParameter, DeterministicParameter))
        if learnable_inputs:
            input_gradient = self.gp_kernel.batch_input_vjp_symmetric(x, covariance_cotangent, kernel_values)
            _accumulate(grads, inputs_node.name, input_gradient.reshape(input_gradient.shape[0], *shape_of(inputs_node)))
        # f = m(X) + L z: the mean parameters and the inputs get the mean's VJP of ḡ.
        _add_mean_gradients(
            grads, self.gp_mean, context, x, cotangent, shape_of, z.dtype, z.device, raw,
            input_node=inputs_node if learnable_inputs else None, input_shape=shape_of(inputs_node),
        )
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


# ----------------------------------------------------------------------------
# Exact GP
# ----------------------------------------------------------------------------


class _GPDistribution(Distribution):
    """Shared plumbing of the GP likelihood nodes.

    The kernel's parameter nodes are stored in ``_kernel_parameter_nodes`` so
    that the model's batching machinery sees them as arguments of the term
    (and marks the batched ones).

    Posterior predictive draws condition on the *training data*: the
    observations and the fixed inputs at the time the model was built or last
    fitted (:meth:`_capture_training_data`, called by the inference
    functions). Inputs replaced with ``Data.set_value`` afterwards are the
    prediction inputs. While ``_in_posterior_predictive`` is set (by the
    posterior predictive functions), :meth:`sample` draws from the GP
    posterior at the current inputs instead of the prior.
    """

    _in_posterior_predictive = False
    # The predictive functions record the latent function draws of these
    # likelihoods (``_last_latent_samples``) under the GP's name.
    _records_latent = True
    # Without plates, the shape of a generated data set follows the current
    # inputs (``event_shape``), not the shape of the training observations.
    _shape_follows_inputs = True
    _observed_node = None
    _training_observed_data = None
    _training_inputs_data = None
    _training_noise_data = None

    def _bind_observed(self, observed_node, observed_values):
        self._observed_node = observed_node
        self._capture_training_data()

    def _capture_training_data(self):
        """Remember the current observations, fixed inputs and a fixed (``Data``) noise variance as the training data."""
        if self._observed_node is None:
            return
        self._training_observed_data = torch.as_tensor(self._observed_node.observed_values).detach().clone()
        self._capture_training_inputs()
        noise = self.noise_variance
        fixed = isinstance(noise, Data)
        self._training_noise_data = _as_tensor(_input_value(noise)).detach().clone() if fixed else None

    def _training_noise_value(self, dtype=None, device=None):
        """The noise variance of the training data: the captured value of a ``Data`` node, else the current value."""
        if self._training_noise_data is not None:
            value = self._training_noise_data
        else:
            value = _as_tensor(_input_value(self.noise_variance) if _is_named(self.noise_variance) else resolve(self.noise_variance))
        return value if dtype is None else value.to(dtype=dtype, device=device)

    def _capture_training_inputs(self):
        inputs = self.inputs
        self._training_inputs_data = None if _is_learnable(inputs) else _as_tensor(_input_value(inputs)).detach().clone()

    def _training_input_value(self):
        """The training inputs: the captured fixed inputs, or the current value of learnable inputs."""
        if self._training_inputs_data is None:
            return _as_tensor(_input_value(self.inputs))
        return self._training_inputs_data

    def _predicting(self):
        return self._in_posterior_predictive and self._training_observed_data is not None

    def sample(self, n_samples=1):
        """Draws of the observations, latent function plus noise.

        The latent function is drawn from the prior, or from the GP posterior
        given the training data during posterior predictive sampling.
        """
        latent = self._posterior_latent(n_samples) if self._predicting() else self._prior_latent(n_samples)
        self._last_latent_samples = latent
        noise = _resolved_value(self.noise_variance, dtype=latent.dtype, device=latent.device)
        try:
            noise = _noise_view(noise, tuple(latent.shape[1:]), type(self).__name__)
        except ValueError:
            raise ValueError(
                f"The noise variance of '{self.name}' has shape {tuple(noise.shape)}, which does not match the "
                f"{tuple(latent.shape[1:])} inputs at which the GP is evaluated. Per-observation noise must be given "
                "at the prediction inputs as well: pass it as a Data node and update it with Data.set_value "
                "together with the inputs."
            ) from None
        return latent + noise.sqrt() * torch.randn_like(latent)

    def _register_kernel(self, kernel):
        self._kernel_parameter_nodes = list(kernel.parameters)
        for parameter in kernel.parameters:
            self.add_dependency(parameter)

    def _register_mean(self, mean, what):
        """Convert ``mean`` to a :class:`MeanFunction` and register its parameter nodes like the kernel's."""
        self.mean = as_mean_function(mean, what)
        self._mean_parameter_nodes = list(self.mean.parameters)
        for parameter in self._mean_parameter_nodes:
            self.add_dependency(parameter)

    def _mean_at(self, x, parameter_values=None):
        """Unbatched mean ``(n,)`` at inputs ``x`` (current node values, or ``parameter_values``)."""
        values = dict(parameter_values or {})
        for parameter in self.mean.parameters:
            values.setdefault(parameter.name, resolve(parameter))
        return self.mean(x, parameter_values=values)

    @property
    def batch_shape(self):
        return torch.Size()

    def pdf(self, x):
        return torch.exp(self.log_pdf(x))


class ExactGP(_GPDistribution):
    r"""Gaussian Process marginal likelihood with the latent function integrated out.

    Given latent function values :math:`\mathbf{f} \sim \mathcal{N}(\mathbf{m}(X), \mathbf{K})`
    and Gaussian observations :math:`\mathbf{y} \mid \mathbf{f} \sim \mathcal{N}(\mathbf{f}, \sigma^2 \mathbf{I})`,
    this distribution evaluates the exact marginal likelihood:

    .. math::

        \mathbf{y} \sim \mathcal{N}\left(\mathbf{m}(X), \mathbf{K}(X, X) + \sigma^2 \mathbf{I}\right)

    With ``inducing_points`` the marginal likelihood is replaced by an
    inducing-point approximation with :math:`\mathbf{Q}_{nn} = \mathbf{K}_{nm}
    \mathbf{K}_{mm}^{-1} \mathbf{K}_{mn}` that only factorises :math:`M \times M`
    matrices (:math:`\mathcal{O}(N M^2)` time): ``"vfe"`` (Titsias' collapsed
    bound, a lower bound on the exact marginal likelihood), ``"dtc"``
    (:math:`\mathcal{N}(\mathbf{m}, \mathbf{Q}_{nn} + \sigma^2 \mathbf{I})`) or
    ``"fitc"`` (with the diagonal correction :math:`\mathrm{diag}(\mathbf{K}_{nn} - \mathbf{Q}_{nn})`).

    Parameters
    ----------
    name : str
        Name of the GP. Posterior predictive results store the latent function draws under this name.
    inputs : Data or Parameter
        Named input node :math:`X` of shape ``(N, D)`` or ``(N,)``: a fixed
        ``Data`` node or a learnable ``RandomParameter`` / ``DeterministicParameter``.
    kernel : Kernel
        Covariance kernel function :math:`k(x_1, x_2)`.
    noise_variance : float or torch.Tensor or Data or Parameter, default 1e-6
        Observation noise variance :math:`\sigma^2 > 0`: a scalar, or one value per
        observation (heteroscedastic noise, :math:`\mathrm{diag}(\boldsymbol{\sigma}^2)`
        instead of :math:`\sigma^2 \mathbf{I}`), i.e. any shape that broadcasts to the
        observations' shape ``(N,)`` (``(n_outputs, N)`` with ``n_outputs``).
    mean : MeanFunction, optional
        GP mean function :math:`m(x)`. Default is :class:`~BayesianDLL.GP.ZeroMean`.
        A constant or scalar node is automatically wrapped in :class:`~BayesianDLL.GP.ConstantMean`.
    jitter : float, default 1e-6
        Relative diagonal jitter added to :math:`\mathbf{K} + \sigma^2 \mathbf{I}`
        (to :math:`\mathbf{K}_{mm}` with inducing points).
    inducing_points : torch.Tensor or Data or Parameter, optional
        Inducing inputs :math:`Z` of shape ``(M, D)`` or ``(M,)``; may be learnable.
    method : {"vfe", "dtc", "fitc"}, optional
        Approximation used with ``inducing_points`` (default ``"vfe"``).
    n_outputs : int, optional
        Number of independent outputs sharing :math:`X`, :math:`Z`, the kernel, mean and
        noise (with ``inducing_points``); the observations then have shape ``(n_outputs, N)``.

    Notes
    -----
    Learnable inducing inputs should use ``"vfe"``: as a lower bound it can only
    approach the exact marginal likelihood as :math:`Z` improves, whereas DTC and
    FITC can overfit :math:`Z` (FITC tends to underestimate the noise).

    Posterior predictive sampling conditions on the training data (the inputs and
    observations when the model was built or last fitted) and predicts at the
    current inputs, so set new inputs with ``Data.set_value`` after fitting.
    Per-observation noise given as a ``Data`` node is captured with the training
    data; set its value at the new inputs too, so that the generated observations
    get noise there.
    """

    def __new__(cls, *args, **kwargs):
        # ExactGP(...) builds the dense or the inducing-point variant; both are ExactGP instances.
        if cls is ExactGP:
            arguments = ExactGP.__signature__.bind(*args, **kwargs).arguments
            cls = _DenseExactGP if arguments.get("inducing_points") is None else _SparseExactGP
        return super().__new__(cls)

    def __init__(
        self,
        name,
        inputs,
        kernel: Kernel,
        noise_variance=1e-6,
        mean=None,
        jitter=1e-6,
        inducing_points=None,
        method=None,
        n_outputs=None,
    ):
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        if not _is_named(inputs):
            raise TypeError("inputs must be a named Data or model parameter node.")
        self.name = name
        self.inputs = inputs
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.jitter = jitter

        self.add_dependency(inputs)
        self._init_variant(inducing_points, method, n_outputs)
        self._register_kernel(kernel)
        self.add_dependency(noise_variance)
        self._register_mean(mean, self._description)
        _noise_view(self._training_noise_value(), tuple(self.event_shape), self._description)

    def _init_variant(self, inducing_points, method, n_outputs):
        raise NotImplementedError

    def _named_parameter_values(self):
        values = {}
        for parameter in [*self.kernel.parameters, self.noise_variance, *self.mean.parameters]:
            if _is_named(parameter):
                values[parameter.name] = resolve(parameter)
        return values

    def _current_trace(self):
        """A one-draw trace ``(1, 1, *shape)`` of the current values of the nodes the predictions read."""
        nodes = [*self.kernel.parameters, self.noise_variance, *self.mean.parameters]
        nodes += [node for node in (self.inputs, getattr(self, "inducing_points", None)) if _is_learnable(node)]
        trace = {}
        for node in nodes:
            if _is_named(node) and node.name not in trace:
                value = _as_tensor(_input_value(node))
                trace[node.name] = value.reshape(1, 1, *value.shape)
        if _is_named(self.noise_variance):
            # Predictions condition on the noise of the training observations.
            value = self._training_noise_value()
            trace[self.noise_variance.name] = value.reshape(1, 1, *value.shape)
        return trace

    def _posterior_latent(self, n_samples):
        """Posterior draws of the latent function at the current inputs given the training data."""
        return self._predict(self._current_trace(), self._training_observed_data, _input_value(self.inputs), n_samples)[0]


class _DenseExactGP(ExactGP):
    """:class:`ExactGP` without inducing points: the exact marginal likelihood with a dense Cholesky factor."""

    _description = "Exact GP"

    def _init_variant(self, inducing_points, method, n_outputs):
        if method is not None or n_outputs is not None:
            raise ValueError("ExactGP method and n_outputs need inducing_points.")

    @property
    def event_shape(self):
        return (_feature_count(self.inputs),)

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
        # (B, 1) for a scalar noise variance, (B, n) for one value per observation.
        noise, noise_shape = _batched_noise(context, self.noise_variance, dtype, device, (n,), "Exact GP")
        mean = _evaluate_mean(context, self.mean, x, dtype, device)
        size = max(covariance.shape[0], noise.shape[0], mean.shape[0])
        covariance = _expand_batch(covariance, size) + torch.diag_embed(noise.expand(noise.shape[0], n))
        factor, jitter_info = batched_stable_cholesky(covariance, jitter=self.jitter)
        residual = observed - mean
        alpha = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
        alpha = _expand_batch(alpha, size)
        log_det = 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
        log_probability = -0.5 * ((residual * alpha).sum(-1) + log_det + n * math.log(2 * math.pi))
        batched = context.any([self.inputs, self.noise_variance, *self.mean.parameters, *self.kernel.parameters])
        return {
            "context": context, "x": x, "kernel_values": kernel_values, "factor": factor, "jitter_info": jitter_info,
            "alpha": alpha, "log_probability": log_probability, "batched": batched, "noise_shape": noise_shape,
        }

    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        return _finish(-parts["alpha"], parts["batched"])

    def log_pdf_param_grads(self, x):
        r"""``d log p = ⟨½(ααᵀ - Σ⁻¹), dΣ⟩`` for ``Σ = K + diag(σ²)`` (+ jitter), batched over chains.

        The noise variance of observation ``i`` gets the diagonal entry ``i``
        of the sensitivity (summed for a scalar noise variance).
        """
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
        learnable_inputs = _is_learnable(inputs)
        if learnable_inputs:
            input_gradient = self.kernel.batch_input_vjp_symmetric(x_values, sensitivity, parts["kernel_values"])
            shape = _parameter_shape(context, inputs)
            gradients[inputs.name] = input_gradient.reshape(input_gradient.shape[0], *shape)
        if _is_named(self.noise_variance):
            # d(K + diag(σ²) + jitter I) / dσ²_i = e_i e_iᵀ (+ c e_i* e_i*ᵀ, folded into the sensitivity).
            noise_gradient = torch.diagonal(sensitivity, dim1=-2, dim2=-1)
            _accumulate(gradients, self.noise_variance.name, _sum_to_shape(noise_gradient, parts["noise_shape"]))
        # The residual is y - m(X), so ∂ log p / ∂m(X) = α (and learnable inputs get the mean's VJP of α).
        _add_mean_gradients(
            gradients, self.mean, context, x_values, alpha, lambda node: _parameter_shape(context, node),
            alpha.dtype, alpha.device, input_node=inputs if learnable_inputs else None,
            input_shape=_parameter_shape(context, inputs) if learnable_inputs else None,
        )
        return _finish(gradients, parts["batched"])

    def _prior_latent(self, n_samples):
        """Prior draws ``(n_samples, n)`` of the latent function at the current inputs."""
        x = _as_tensor(_input_value(self.inputs))
        if x.ndim == 0:
            x = x.reshape(1)
        dtype, device = x.dtype, x.device
        mean_values = {parameter.name: resolve(parameter) for parameter in self.mean.parameters if _is_named(parameter)}
        factor = stable_cholesky(self.kernel(x, x, parameter_values=self._named_parameter_values()), jitter=self.jitter)
        mean = self.mean(x, parameter_values=mean_values)
        return mean + torch.randn((n_samples, _feature_count(x)), dtype=dtype, device=device) @ factor.transpose(-2, -1)

    def _predict(self, trace, observed_values, inputs, n_samples, observation_noise=None):
        return _dense_exact_gp_predictive(self, trace, observed_values, inputs, n_samples, observation_noise)


# ----------------------------------------------------------------------------
# Sparse (inducing-point) GP
# ----------------------------------------------------------------------------


_SPARSE_METHODS = ("vfe", "dtc", "fitc")


class _SparseExactGP(ExactGP):
    """:class:`ExactGP` with inducing points: the VFE, DTC or FITC approximation.

    Everything is evaluated with Cholesky factors of ``m x m`` matrices, the
    Woodbury identity and the matrix determinant lemma (``O(n m²)`` time,
    ``O(n m)`` memory). The training inputs ``X``, the inducing inputs ``Z``,
    the kernel and mean parameters and the noise variance get hand-derived
    gradients (see :meth:`log_pdf_param_grads`).
    """

    _description = "Sparse GP"

    def _init_variant(self, inducing_points, method, n_outputs):
        method = "vfe" if method is None else str(method).lower()
        if method not in _SPARSE_METHODS:
            raise ValueError(f"ExactGP method must be one of {_SPARSE_METHODS}; got {method!r}.")
        if n_outputs is not None and (isinstance(n_outputs, bool) or not isinstance(n_outputs, int) or n_outputs < 1):
            raise ValueError(f"ExactGP n_outputs must be None or a positive integer; got {n_outputs!r}.")
        self.inducing_points = inducing_points
        self.method = method
        self.n_outputs = n_outputs
        self.add_dependency(inducing_points)

    @property
    def event_shape(self):
        n = _feature_count(self.inputs)
        return (n,) if self.n_outputs is None else (self.n_outputs, n)

    def _output_count(self):
        return 1 if self.n_outputs is None else self.n_outputs

    def _observations(self, observed, n, dtype, device):
        """Observations as ``(R, n)`` (``R = 1`` without ``n_outputs``)."""
        observed = torch.as_tensor(observed, dtype=dtype, device=device)
        if self.n_outputs is None:
            if observed.numel() != n:
                raise ValueError(
                    f"Sparse GP observations must contain one value per input ({n}); got {observed.numel()} values."
                )
            return observed.reshape(1, n)
        if observed.numel() != self.n_outputs * n:
            raise ValueError(
                f"Sparse GP observations with n_outputs={self.n_outputs} must have shape ({self.n_outputs}, {n}); "
                f"got {tuple(observed.shape)}."
            )
        return observed.reshape(self.n_outputs, n)

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
        outputs = self._output_count()
        observed = self._observations(observed, n, dtype, device)
        # (B, G, 1 or n): G = 1 when the outputs share the noise, G = R with one noise per output.
        noise, noise_shape = _batched_noise(context, self.noise_variance, dtype, device, (outputs, n), "Sparse GP")
        groups = noise.shape[1]
        mean = _evaluate_mean(context, self.mean, x, dtype, device)
        kernel_values = context.kernel_values(self.kernel, dtype, device)

        k_mm = self.kernel.batch_evaluate(z, z, kernel_values)
        k_mn = self.kernel.batch_evaluate(z, x, kernel_values)
        k_diag = self.kernel.batch_diag(x, kernel_values)
        size = max(k_mm.shape[0], k_mn.shape[0], k_diag.shape[0], noise.shape[0], mean.shape[0])
        k_mm, k_mn = _expand_batch(k_mm, size), _expand_batch(k_mn, size)
        k_diag = _expand_batch(k_diag, size)
        noise = _expand_batch(noise, size).expand(size, groups, n)
        factor, jitter_info = batched_stable_cholesky(k_mm, jitter=self.jitter)
        # V = L⁻¹ K_mn, so Q_nn = Vᵀ V and diag(Q_nn) = colsum(V ⊙ V).
        v = torch.linalg.solve_triangular(factor, k_mn, upper=False)
        gap = k_diag - v.square().sum(dim=-2)  # diag(K_nn - Q_nn), (B, n)
        lam = gap.clamp_min(0.0).unsqueeze(1) + noise if self.method == "fitc" else noise  # (B, G, n)
        # (B, G, R / G, n): the residuals of the outputs that share a Λ (R = 1 without n_outputs).
        residual = _expand_batch(observed - mean.unsqueeze(-2), size).reshape(size, groups, outputs // groups, n)
        eye_m = torch.eye(v.shape[-2], dtype=dtype, device=device)
        v_group = v.unsqueeze(1)
        v_scaled = v_group / lam.unsqueeze(-2)  # (B, G, m, n)
        # Σ = Vᵀ V + Λ and B = I + V Λ⁻¹ Vᵀ = L_B L_Bᵀ, so |Σ| = |Λ| |B| (one factor per noise group).
        factor_b = torch.linalg.cholesky(eye_m + v_scaled @ v_group.transpose(-2, -1))
        # Woodbury: Σ⁻¹ r = Λ⁻¹ r - Λ⁻¹ Vᵀ L_B⁻ᵀ c with c = L_B⁻¹ V Λ⁻¹ r (one column per output).
        c = torch.linalg.solve_triangular(factor_b, v_scaled @ residual.transpose(-2, -1), upper=False)  # (B, G, m, R/G)
        correction = torch.linalg.solve_triangular(factor_b.transpose(-2, -1), c, upper=True)
        alpha = (residual - (v_group.transpose(-2, -1) @ correction).transpose(-2, -1)) / lam.unsqueeze(-2)
        alpha = alpha.reshape(size, outputs, n)
        log_det = (outputs // groups) * (
            torch.log(lam).sum(dim=(-2, -1)) + 2 * torch.log(torch.diagonal(factor_b, dim1=-2, dim2=-1)).sum(dim=(-2, -1))
        )
        quadratic = (residual.square() / lam.unsqueeze(-2)).sum(dim=(-3, -2, -1)) - c.square().sum(dim=(-3, -2, -1))
        log_probability = -0.5 * (quadratic + log_det + outputs * n * math.log(2 * math.pi))
        # s_i = Σ_r 1 / σ²_ri weighs the VFE trace term -½ Σ_r Σ_i (K_nn - Q_nn)_ii / σ²_ri.
        inverse_noise = (outputs // groups) * (1.0 / noise).sum(1)  # (B, n)
        if self.method == "vfe":
            log_probability = log_probability - 0.5 * (gap * inverse_noise).sum(-1)
        batched = context.any([self.inputs, self.inducing_points, self.noise_variance, *self.mean.parameters, *self.kernel.parameters])
        return {
            "context": context, "x": x, "z": z, "noise": noise, "noise_shape": noise_shape, "factor": factor,
            "jitter_info": jitter_info, "v": v, "lam": lam, "factor_b": factor_b, "alpha": alpha, "gap": gap,
            "inverse_noise": inverse_noise, "kernel_values": kernel_values, "log_probability": log_probability,
            "batched": batched,
        }

    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        alpha = parts["alpha"] if self.n_outputs is not None else parts["alpha"][:, 0]
        return _finish(-alpha, parts["batched"])

    def _cotangents(self, parts):
        r"""Cotangents of the log density w.r.t. ``K_mn``, ``K_mm`` and ``diag(K_nn)`` (batched).

        Every gradient below is linear in the covariance differentials, so it
        is ``⟨G_mn, dK_mn⟩ + ⟨G_mm, dK_mm⟩ + ⟨g_nn, d diag(K_nn)⟩``. For output
        ``r`` (one without ``n_outputs``) let ``Σ_r = Q_nn + Λ_r``, ``α_r = Σ_r⁻¹ r_r``,
        ``W = K_mm⁻¹ K_mn``, ``P_r = W Σ_r⁻¹`` and ``ω_r = ½ (α_r² - diag(Σ_r⁻¹))``;
        ``Λ_r`` (and so ``P_r``) is shared by the outputs unless the noise variance
        differs between them. Every term sums over the outputs:

        * Gaussian term (all methods), from ``dQ = dK_nm W + Wᵀ dK_mn - Wᵀ dK_mm W``
          and ``½ αᵀ dΣ α - ½ tr(Σ⁻¹ dΣ)``:
          ``G_mn = Σ_r (Wα_r) α_rᵀ - P_r`` and ``G_mm = Σ_r -½ (Wα_r)(Wα_r)ᵀ + ½ P_r Wᵀ``;
        * FITC (``dΛ_r = d diag(K_nn) - diag(dQ)``) with ``ω = Σ_r ω_r``:
          ``G_mn -= 2 W diag(ω)``, ``G_mm += W diag(ω) Wᵀ``, ``g_nn = ω``;
        * VFE trace term ``-½ Σ_i s_i (K_nn - Q_nn)_ii`` with ``s_i = Σ_r 1 / σ²_ri``:
          ``G_mn += W diag(s)``, ``G_mm -= ½ W diag(s) Wᵀ``, ``g_nn = -½ s``.

        ``G_mm`` is finally corrected for the scale-dependent jitter
        ``c max(1, max diag K_mm)``: ``G_mm[i*, i*] += c tr(G_mm)``. The last
        return value is ``ω_r`` per output ``(B, R, n)``, the gradient of the
        Gaussian term w.r.t. the noise variances ``σ²_ri``.
        """
        factor, factor_b, v, lam, alpha = parts["factor"], parts["factor_b"], parts["v"], parts["lam"], parts["alpha"]
        size, outputs, n = alpha.shape
        groups = lam.shape[1]
        w = torch.linalg.solve_triangular(factor.transpose(-2, -1), v, upper=True)  # K_mm⁻¹ K_mn, (B, m, n)
        w_alpha = w @ alpha.transpose(-2, -1)  # (B, m, R)
        # Σ_r⁻¹ = Λ_r⁻¹ - Gᵀ G with G = L_B⁻¹ V Λ_r⁻¹ (one per noise group).
        g = torch.linalg.solve_triangular(factor_b, v.unsqueeze(1) / lam.unsqueeze(-2), upper=False)  # (B, G, m, n)
        sigma_inverse_diag = 1.0 / lam - g.square().sum(dim=-2)  # (B, G, n)
        w_group = w.unsqueeze(1)
        p = w_group / lam.unsqueeze(-2) - (w_group @ g.transpose(-2, -1)) @ g  # W Σ⁻¹, (B, G, m, n)
        p_sum = (outputs // groups) * p.sum(1)  # Σ_r P_r
        omega = 0.5 * (alpha.reshape(size, groups, -1, n).square() - sigma_inverse_diag.unsqueeze(2))
        omega = omega.reshape(size, outputs, n)
        diag_weight = omega.sum(1)
        g_mn = w_alpha @ alpha - p_sum
        g_mm = -0.5 * w_alpha @ w_alpha.transpose(-2, -1) + 0.5 * p_sum @ w.transpose(-2, -1)
        g_nn = None
        if self.method == "fitc":
            g_mn = g_mn - 2.0 * w * diag_weight.unsqueeze(-2)
            g_mm = g_mm + (w * diag_weight.unsqueeze(-2)) @ w.transpose(-2, -1)
            g_nn = diag_weight
        elif self.method == "vfe":
            weights = parts["inverse_noise"].unsqueeze(-2)  # s, (B, 1, n)
            g_mn = g_mn + w * weights
            g_mm = g_mm - 0.5 * (w * weights) @ w.transpose(-2, -1)
            g_nn = -0.5 * parts["inverse_noise"]
        g_mm = _jitter_cotangent(parts["jitter_info"], g_mm)
        return g_mn, g_mm, g_nn, omega

    def log_pdf_param_grads(self, x):
        r"""Hand-derived gradients of the approximate log marginal likelihood.

        The cotangents ``G_mn``, ``G_mm`` and ``g_nn`` of :meth:`_cotangents`
        are contracted with the kernel derivatives:

        * a kernel hyperparameter ``θ`` gets
          ``Σ(G_mn ⊙ ∂K_mn/∂θ) + Σ(G_mm ⊙ ∂K_mm/∂θ) + g_nn · ∂diag(K_nn)/∂θ``;
        * the noise variance ``σ²_ri`` of output ``r`` and observation ``i`` gets
          ``ω_ri`` (plus ``(K_nn - Q_nn)_ii / (2σ⁴_ri)`` for VFE), summed over the
          entries that share a value (all of them for a scalar noise variance);
        * learnable inducing inputs ``Z`` get
          ``kernel.input_vjp(Z, X, G_mn) + kernel.input_vjp(Z, Z, G_mm + G_mmᵀ)``
          (``K_nn`` does not depend on ``Z``, and the mean is evaluated at
          ``X``, so ``Z`` gets no mean term);
        * learnable training inputs ``X`` get
          ``kernel.input_vjp(X, Z, G_mnᵀ)`` (``K_mn = K(Z, X)``), plus
          ``kernel.diag_input_vjp(X, g_nn)`` for VFE and FITC (zero for
          stationary kernels, whose diagonal is constant) and the mean's input
          VJP ``∂⟨α, m(X)⟩/∂X``;
        * a parameter ``θ`` of the mean function gets ``⟨α, ∂m(X)/∂θ⟩``.

        With ``n_outputs`` the mean terms use ``Σ_r α_r``.
        """
        parts = self._components(x)
        context, x_values, z = parts["context"], parts["x"], parts["z"]
        noise, alpha, kernel_values = parts["noise"], parts["alpha"], parts["kernel_values"]
        g_mn, g_mm, g_nn, omega = self._cotangents(parts)

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
            # The mean is evaluated at X, not Z, so Z gets kernel terms only.
            z_gradient = self.kernel.batch_input_vjp(z, x_values, g_mn, kernel_values)
            z_gradient = z_gradient + self.kernel.batch_input_vjp_symmetric(z, g_mm, kernel_values)
            shape = _parameter_shape(context, self.inducing_points)
            gradients[self.inducing_points.name] = z_gradient.reshape(z_gradient.shape[0], *shape)

        learnable_inputs = _is_learnable(self.inputs)
        input_shape = _parameter_shape(context, self.inputs) if learnable_inputs else None
        if learnable_inputs:
            # K_mn = K(Z, X): ⟨G_mn, K(Z, X)⟩ = ⟨G_mnᵀ, K(X, Z)⟩; diag(K_nn) enters VFE and FITC.
            x_gradient = self.kernel.batch_input_vjp(x_values, z, g_mn.transpose(-2, -1), kernel_values)
            if g_nn is not None:
                x_gradient = x_gradient + self.kernel.batch_diag_input_vjp(x_values, g_nn, kernel_values)
            _accumulate(gradients, self.inputs.name, x_gradient.reshape(x_gradient.shape[0], *input_shape))

        if _is_named(self.noise_variance):
            noise_gradient = omega  # (B, R, n)
            if self.method == "vfe":
                # (B, G, n) broadcasts over the outputs of a noise group.
                noise_gradient = noise_gradient + 0.5 * parts["gap"].unsqueeze(1) / noise.square()
            _accumulate(gradients, self.noise_variance.name, _sum_to_shape(noise_gradient, parts["noise_shape"]))

        # The residual is y - m(X) and the trace term does not depend on the mean: ∂/∂m(X) = Σ_r α_r.
        _add_mean_gradients(
            gradients, self.mean, context, x_values, alpha.sum(-2), lambda node: _parameter_shape(context, node),
            alpha.dtype, alpha.device, input_node=self.inputs if learnable_inputs else None, input_shape=input_shape,
        )

        return _finish(gradients, parts["batched"])

    def _prior_latent(self, n_samples):
        """Draws from the approximate prior ``Normal(m(X), Q_nn + Λ - σ² I)`` of the latent function.

        Uses ``Vᵀ ε₁ + sqrt(Λ - σ²) ε₂`` with ``V = L⁻¹ K_mn``, so no ``n x n``
        covariance is formed. The draws have shape ``(n_samples, n)``, or
        ``(n_samples, n_outputs, n)`` with independent outputs.
        """
        x = self._training_inputs()
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        parts = self._components(torch.zeros(self.event_shape, dtype=dtype, device=x.device))
        v, gap = parts["v"][0], parts["gap"][0]
        mean = self._mean_at(x)
        shape = (n_samples,) if self.n_outputs is None else (n_samples, self.n_outputs)
        first = torch.randn((*shape, v.shape[0]), dtype=dtype, device=x.device) @ v
        if self.method != "fitc":
            return mean + first
        return mean + first + torch.randn((*shape, gap.numel()), dtype=dtype, device=x.device) * gap.clamp_min(0.0).sqrt()

    def _predict(self, trace, observed_values, inputs, n_samples, observation_noise=None):
        return _sparse_gp_predictive(self, trace, observed_values, inputs, n_samples, observation_noise)


def _draw_count(trace):
    """Number of draws in ``trace`` (values ``(n_chains, n_draws, ...)``); an empty trace is one draw."""
    first = next(iter(trace.values()), None)
    if first is None:
        return 1
    first = torch.as_tensor(first)
    return first.shape[0] * first.shape[1] if first.ndim >= 2 else first.shape[0]


def _gp_predictive(gp, trace, inputs, n_samples=1, observation_noise=None):
    """Sample latent GP values at new inputs from posterior latent traces.

    ``gp`` is a :class:`LatentGP`. ``trace`` may be a ``SamplingResult`` or
    a mapping with values of shape ``(n_chains, n_draws, *shape)``. Kernel
    parameters, the parameters of the mean function and learnable training
    inputs are read from the trace draw by draw; a node missing from the
    trace keeps its current value. Without inducing points the trace must
    contain the GP's values at the training inputs (``trace[gp.name]``), and
    for every draw ``f`` the function at ``inputs`` is drawn ``n_samples``
    times from the noise-free GP conditional
    ``N(m(X*) + K_*X K⁻¹ (f - m(X)), K_** - K_*X K⁻¹ K_X*)``. With inducing
    points the function is deterministic given the whitened values (which the
    trace must contain), ``f* = m(X*) + K_*Z L⁻ᵀ v``, and with basis functions
    (``n_basis``) it is the basis expansion ``f* = m(X*) + Φ(X*) (sqrt(S) ⊙ z)``. A given
    ``observation_noise`` variance (a scalar or one value per new input) is
    added. The result has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, LatentGP):
        raise TypeError("gp must be a LatentGP function returned by BayesianDLL.GP.LatentGP.")
    if not hasattr(trace, "__getitem__"):
        raise TypeError("trace must be a SamplingResult or mapping-like object.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if isinstance(gp, (_SparseLatentGP, _HilbertLatentGP)):
        predict = _sparse_latent_gp_predictive if isinstance(gp, _SparseLatentGP) else _hilbert_latent_gp_predictive
        samples = predict(gp, trace, inputs, n_samples=n_samples)
        if observation_noise is not None:
            noise = _resolved_value(observation_noise, dtype=samples.dtype, device=samples.device)
            samples = samples + _noise_view(noise, (samples.shape[-1],), "observation_noise").sqrt() * torch.randn_like(samples)
        return samples

    function_trace = torch.as_tensor(trace[gp.name])
    latent_draws = function_trace.reshape(-1, function_trace.shape[-1])
    dtype, device = latent_draws.dtype, latent_draws.device
    total_draws = latent_draws.shape[0]
    training_shape = gp._training_input_value().shape
    if _is_learnable(gp.gp_inputs):
        training_draws = _draws_of(trace, gp.gp_inputs, total_draws, dtype, device)
    else:
        training_draws = gp._training_input_value().to(dtype=dtype, device=device).unsqueeze(0)
    new_values = _input_value(inputs).to(dtype=dtype, device=device)
    kernel_draws = _parameter_draws(trace, gp.gp_kernel.parameters, total_draws, dtype, device, _draws_of)
    mean_draws = _parameter_draws(trace, gp.gp_mean.parameters, total_draws, dtype, device, _draws_of)

    outputs = []
    for draw_index in range(total_draws):
        parameter_values = _draw(kernel_draws, draw_index)
        mean_values = _draw(mean_draws, draw_index)
        training_values = training_draws[draw_index if training_draws.shape[0] > 1 else 0].reshape(training_shape)

        covariance = gp.gp_kernel(training_values, training_values, parameter_values=parameter_values)
        cross_covariance = gp.gp_kernel(training_values, new_values, parameter_values=parameter_values)
        new_covariance = gp.gp_kernel(new_values, new_values, parameter_values=parameter_values)
        factor = stable_cholesky(covariance, jitter=gp.gp_jitter)
        residual = latent_draws[draw_index] - gp.gp_mean(training_values, parameter_values=mean_values)
        alpha = torch.cholesky_solve(residual.reshape(-1, 1), factor).reshape(-1)
        predictive_mean = gp.gp_mean(new_values, parameter_values=mean_values) + cross_covariance.transpose(-2, -1) @ alpha
        conditional_covariance = new_covariance - cross_covariance.transpose(-2, -1) @ torch.cholesky_solve(cross_covariance, factor)
        conditional_covariance = (conditional_covariance + conditional_covariance.transpose(-2, -1)) / 2
        if observation_noise is not None:
            extra = _input_value(observation_noise).to(dtype=dtype, device=device)
            conditional_covariance = conditional_covariance + torch.diag_embed(
                _noise_diagonal(extra, (conditional_covariance.shape[-1],), "observation_noise")
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


def _noise_draws(gp, trace, n_draws, dtype, device):
    """``(n_draws or 1, *shape)`` draws of the training noise variance of ``gp``.

    A traced noise node is read from ``trace``; a ``Data`` node or a constant
    keeps the value of the training data. A learnable noise variance missing
    from the trace raises ``KeyError``.
    """
    node = gp.noise_variance
    if _is_named(node) and node.name in trace:
        values = torch.as_tensor(trace[node.name])
        values = values.reshape(-1, *values.shape[2:]) if values.ndim >= 2 else values.reshape(-1, 1)
        return values.to(dtype=dtype, device=device)
    if _is_learnable(node):
        raise KeyError(f"Trace does not contain the model parameter '{node.name}'.")
    return gp._training_noise_value(dtype, device).unsqueeze(0)


def _noise_diagonal(value, target, what="GP"):
    """A noise variance (scalar or per observation) expanded to the observations' shape ``target``."""
    value = _as_tensor(value)
    return _noise_view(value, target, what).expand(target)


def _parameter_draws(trace, parameters, n_draws, dtype, device, draws_of=_trace_draws):
    """``{name: (n_draws or 1, *shape)}`` draws of the named ``parameters`` (kernel or mean parameters)."""
    return {p.name: draws_of(trace, p, n_draws, dtype, device) for p in parameters if _is_named(p)}


def _draw(draws, index):
    """The values of draw ``index`` from :func:`_parameter_draws` (an untraced node has one draw)."""
    return {name: values[index if values.shape[0] > 1 else 0] for name, values in draws.items()}


def _exact_gp_predictive(
    gp,
    trace,
    observed_values,
    inputs,
    n_samples=1,
    observation_noise=None,
):
    """Sample exact-GP predictions conditional on observed training values.

    ``gp`` is an :class:`ExactGP`, ``observed_values`` the training
    observations ``y`` (``(N,)``) and ``inputs`` the new inputs. ``trace``
    (values of shape ``(n_chains, n_draws, *shape)``) contains only
    hyperparameter draws; the latent function is integrated out during
    inference and is sampled here from the GP posterior
    ``mean = m(X*) + K_*X (K + Σ_ε)⁻¹ (y - m(X))`` and
    ``cov = K_** - K_*X (K + Σ_ε)⁻¹ K_X*`` with ``Σ_ε = diag(σ²)``, the mean
    function ``m`` evaluated per draw. Every named kernel parameter, the noise
    variance (unless it is a ``Data`` node, whose training value is used) and
    every named parameter of the mean function must be in the trace;
    learnable training inputs are read from the trace draw by draw (or keep
    their current value when they are not traced). With inducing points the
    approximate posterior of the chosen method is used instead (all solves
    with ``m x m`` factors), learnable inducing inputs are read from the
    trace, and with ``n_outputs = R`` the observations have shape ``(R, N)``
    and the result ``(n_parameter_draws, n_samples, R, n_new)``. A given
    ``observation_noise`` variance (a scalar or one value per new input) is
    added to the diagonal to predict noisy observations. The returned tensor
    has shape ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, ExactGP):
        raise TypeError("gp must be an ExactGP distribution returned by BayesianDLL.GP.ExactGP.")
    if not hasattr(trace, "__getitem__") or not hasattr(trace, "keys"):
        raise TypeError("trace must be a mapping-like sampling result.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    return gp._predict(trace, observed_values, inputs, n_samples, observation_noise)


def _dense_exact_gp_predictive(gp, trace, observed_values, inputs, n_samples=1, observation_noise=None):
    """:func:`_exact_gp_predictive` for an ExactGP without inducing points."""
    training_values = gp._training_input_value()
    new_values = _as_tensor(_input_value(inputs))
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

    n_draws = _draw_count(trace)
    dtype = training_values.dtype
    device = training_values.device
    new_values = new_values.to(dtype=dtype, device=device)
    n_new = _feature_count(new_values)
    parameter_draws = _parameter_draws(trace, gp.kernel.parameters, n_draws, dtype, device)
    noise_draws = _noise_draws(gp, trace, n_draws, dtype, device)
    mean_draws = _parameter_draws(trace, gp.mean.parameters, n_draws, dtype, device)
    extra = None if observation_noise is None else _noise_diagonal(
        _resolved_value(observation_noise, dtype=dtype, device=device), (n_new,), "observation_noise"
    )
    training_shape = training_values.shape
    training_draws = _draws_of(trace, gp.inputs, n_draws, dtype, device) if _is_learnable(gp.inputs) else None

    outputs = []
    for draw_index in range(n_draws):
        kernel_values = _draw(parameter_draws, draw_index)
        mean_values = _draw(mean_draws, draw_index)
        if training_draws is not None:
            training_values = training_draws[draw_index if training_draws.shape[0] > 1 else 0].reshape(training_shape)
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
        noise = _noise_diagonal(noise_draws[draw_index if noise_draws.shape[0] > 1 else 0], (n_training,), "Exact GP")
        factor = stable_cholesky(covariance + torch.diag_embed(noise), jitter=gp.jitter)
        mean_train = gp.mean(training_values, parameter_values=mean_values)
        mean_new = gp.mean(new_values, parameter_values=mean_values)

        alpha = torch.cholesky_solve(
            (observed_values - mean_train).reshape(-1, 1),
            factor,
        ).reshape(-1)
        predictive_mean = mean_new + cross_covariance.transpose(-2, -1) @ alpha
        predictive_covariance = new_covariance - cross_covariance.transpose(-2, -1) @ torch.cholesky_solve(
            cross_covariance,
            factor,
        )
        if extra is not None:
            predictive_covariance = predictive_covariance + torch.diag_embed(extra)
        predictive_covariance = (predictive_covariance + predictive_covariance.transpose(-2, -1)) / 2
        prediction_factor = stable_cholesky(predictive_covariance, jitter=gp.jitter)
        outputs.append(
            predictive_mean
            + torch.randn((n_samples, n_new), dtype=dtype, device=device)
            @ prediction_factor.transpose(-2, -1)
        )

    return torch.stack(outputs)


def _sparse_gp_predictive(gp, trace, observed_values, inputs, n_samples=1, observation_noise=None):
    r""":func:`_exact_gp_predictive` for an ExactGP with inducing points.

    For every hyperparameter draw in ``trace`` the inducing-point posterior
    ``q(u) = N(K_mm Σ_u K_mn Λ⁻¹ r, K_mm Σ_u K_mm)`` with
    ``Σ_u = (K_mm + K_mn Λ⁻¹ K_nm)⁻¹`` (Titsias' optimal ``q(u)`` for VFE,
    the DTC posterior, and for FITC the same with its diagonal ``Λ``) is
    projected to the new inputs:

    ``mean = m(X*) + K_*m Σ_u K_mn Λ⁻¹ r`` and ``cov = K_** - Q_** + K_*m Σ_u K_m*``.

    Here ``r = y - m(X)`` with ``y = observed_values`` and the mean function
    ``m`` evaluated per draw. All solves use ``m x m`` Cholesky factors, so
    the cost is linear in the number of training points. Kernel parameters,
    the noise variance and the named parameters of the mean function must
    be in ``trace``; learnable inducing inputs and learnable training inputs
    are read from the trace draw by draw (a learnable training-input node
    missing from the trace keeps its current value). A given
    ``observation_noise`` variance is added to the diagonal. The result has
    shape ``(n_parameter_draws, n_samples, n_new)``; with ``n_outputs = R``
    independent outputs, ``observed_values`` has shape ``(R, n)`` and the
    result ``(n_parameter_draws, n_samples, R, n_new)`` (the outputs share
    the predictive covariance unless their noise variances differ).
    """
    training_values = gp._training_input_value()
    if training_values.ndim == 0:
        training_values = training_values.reshape(1)
    if not training_values.is_floating_point():
        training_values = training_values.to(torch.get_default_dtype())
    dtype = training_values.dtype
    device = training_values.device
    inducing_values = gp._inducing_inputs(dtype, device)
    new_values = torch.as_tensor(_input_value(inputs)).to(dtype=dtype, device=device)
    if new_values.ndim == 0:
        new_values = new_values.reshape(1)
    n_training = _feature_count(training_values)
    n_new = _feature_count(new_values)
    observed_values = torch.as_tensor(_input_value(observed_values), dtype=dtype, device=device)
    if gp.n_outputs is None and observed_values.numel() != n_training:
        raise ValueError(
            f"observed_values must contain one value per training input ({n_training}); "
            f"got {observed_values.numel()} values."
        )
    observed_values = gp._observations(observed_values, n_training, dtype, device)  # (R, n)

    n_draws = _draw_count(trace)
    outputs = gp._output_count()
    parameter_draws = _parameter_draws(trace, gp.kernel.parameters, n_draws, dtype, device)
    noise_draws = _noise_draws(gp, trace, n_draws, dtype, device)
    mean_draws = _parameter_draws(trace, gp.mean.parameters, n_draws, dtype, device)
    extra = None if observation_noise is None else _noise_view(
        _resolved_value(observation_noise, dtype=dtype, device=device), (outputs, n_new), "observation_noise"
    ).expand(-1, n_new)  # (1 or R, n*)
    eye_m = torch.eye(_feature_count(inducing_values), dtype=dtype, device=device)
    learnable_inducing = _is_named(gp.inducing_points) and not isinstance(gp.inducing_points, Data)
    if learnable_inducing:
        inducing_draws = _trace_draws(trace, gp.inducing_points, n_draws, dtype, device)
    training_shape = training_values.shape
    training_draws = _draws_of(trace, gp.inputs, n_draws, dtype, device) if _is_learnable(gp.inputs) else None

    outputs_list = []
    for draw_index in range(n_draws):
        kernel_values = _draw(parameter_draws, draw_index)
        mean_values = _draw(mean_draws, draw_index)
        if learnable_inducing:
            inducing_values = inducing_draws[draw_index].reshape(gp._inducing_inputs(dtype, device).shape)
        if training_draws is not None:
            training_values = training_draws[draw_index if training_draws.shape[0] > 1 else 0].reshape(training_shape)
        # (G, n): G = 1 when the outputs share the noise, G = R with one noise per output.
        noise = _noise_view(noise_draws[draw_index if noise_draws.shape[0] > 1 else 0], (outputs, n_training), "Sparse GP")
        noise = noise.expand(-1, n_training)
        groups = noise.shape[0]
        mean_train = gp.mean(training_values, parameter_values=mean_values)
        mean_new = gp.mean(new_values, parameter_values=mean_values)

        k_mm = gp.kernel(inducing_values, inducing_values, parameter_values=kernel_values)
        k_mn = gp.kernel(inducing_values, training_values, parameter_values=kernel_values)
        k_ms = gp.kernel(inducing_values, new_values, parameter_values=kernel_values)
        k_ss = gp.kernel(new_values, new_values, parameter_values=kernel_values)
        factor = stable_cholesky(k_mm, jitter=gp.jitter)
        v = torch.linalg.solve_triangular(factor, k_mn, upper=False)
        v_new = torch.linalg.solve_triangular(factor, k_ms, upper=False)
        if gp.method == "fitc":
            k_diag = gp.kernel.diag(training_values, parameter_values=kernel_values)
            lam = (k_diag - v.square().sum(dim=0)).clamp_min(0.0) + noise
        else:
            lam = noise
        # Σ_u = L⁻ᵀ B⁻¹ L⁻¹ with B = I + V Λ⁻¹ Vᵀ = L_B L_Bᵀ (one per noise group).
        v_scaled = v / lam.unsqueeze(-2)  # (G, m, n)
        factor_b = torch.linalg.cholesky(eye_m + v_scaled @ v.transpose(-2, -1))
        residual = (observed_values - mean_train).reshape(groups, outputs // groups, n_training)
        c = torch.linalg.solve_triangular(factor_b, v_scaled @ residual.transpose(-2, -1), upper=False)  # (G, m, R/G)
        projected = torch.linalg.solve_triangular(factor_b, v_new.expand(groups, *v_new.shape), upper=False)  # L_B⁻¹ L⁻¹ K_m*
        predictive_mean = mean_new + (projected.transpose(-2, -1) @ c).transpose(-2, -1).reshape(outputs, n_new)
        predictive_covariance = (
            k_ss
            - v_new.transpose(-2, -1) @ v_new
            + projected.transpose(-2, -1) @ projected
        )  # (G, n*, n*)
        if extra is not None:
            predictive_covariance = predictive_covariance + torch.diag_embed(extra)
        predictive_covariance = (predictive_covariance + predictive_covariance.transpose(-2, -1)) / 2
        prediction_factor = stable_cholesky(predictive_covariance, jitter=gp.jitter)
        prediction_factor = prediction_factor.expand(outputs, n_new, n_new)
        noise_draw = torch.randn((n_samples, outputs, n_new), dtype=dtype, device=device)
        draws = predictive_mean + (prediction_factor @ noise_draw.unsqueeze(-1)).squeeze(-1)
        outputs_list.append(draws if gp.n_outputs is not None else draws[:, 0])

    return torch.stack(outputs_list)


# ----------------------------------------------------------------------------
# Sparse latent (whitened inducing-point) GP
# ----------------------------------------------------------------------------


class _SparseLatentGP(LatentGP):
    r""":class:`LatentGP` with inducing points: ``f(X) = m(X) + K_XZ L⁻ᵀ v`` with ``L Lᵀ = K_ZZ + jitter I``.

    Equivalently ``u = L v ~ Normal(0, K_ZZ)`` are the function values at
    ``Z`` and ``f`` is their kernel interpolation, so the prior covariance of
    ``f`` is the Nyström / DTC approximation ``Q_XX = K_XZ K_ZZ⁻¹ K_ZX``. Only
    ``m x m`` matrices are factorised and ``m`` latent values are sampled;
    with ``Z = X`` the node equals the dense latent GP up to the jitter. No
    FITC-style correction ``diag(K_XX - Q_XX)`` is added (it would need ``n``
    further latent values).

    **Gradients** (hand-derived VJP). For a cotangent ``g`` of ``f`` let
    ``a = L⁻ᵀ v``: ``v`` gets ``c = L⁻¹ K_ZX g``, ``K_ZX`` gets ``G_ZX = a gᵀ``,
    ``L`` gets ``-a cᵀ`` (mapped to ``G_ZZ`` by the Cholesky VJP), a kernel
    hyperparameter ``θ`` gets ``⟨G_ZX, ∂K_ZX/∂θ⟩ + ⟨G_ZZ, ∂K_ZZ/∂θ⟩``, learnable
    ``Z`` and ``X`` get the kernel input VJPs, and the mean's parameters (and
    learnable ``X``) get the mean's VJP of ``g``.
    """

    _latent_unit = "inducing input"

    def _prepare_inducing(self, inducing_points):
        if not _is_named(inducing_points):
            inducing_points = _as_tensor(inducing_points)
            if not inducing_points.is_floating_point():
                inducing_points = inducing_points.to(torch.get_default_dtype())
        return inducing_points

    def _latent_count(self, inputs, inducing_points):
        return _feature_count(inducing_points)

    def _model_inputs(self, inputs, latent, inducing_points, mean):
        model_inputs = [inputs, latent]
        self._inducing_index = None
        if _is_named(inducing_points):
            self._inducing_index = len(model_inputs)
            model_inputs.append(inducing_points)
        model_inputs.extend(self._kernel_parameters)
        self._fixed_inducing = None if _is_named(inducing_points) else inducing_points
        return _with_parameters(model_inputs, mean.parameters)

    def _derivative_function(self):
        return None

    def _finish_init(self, inducing_points):
        self.gp_inducing_points = inducing_points

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
        raw = _raw_values(self.inputs, values)
        kernel_values = context.kernel_values(self.gp_kernel, dtype, device, raw)
        k_mm = self.gp_kernel.batch_evaluate(z, z, kernel_values)
        k_mn = self.gp_kernel.batch_evaluate(z, x, kernel_values)
        size = max(k_mm.shape[0], k_mn.shape[0], v.shape[0])
        factor, jitter_info = batched_stable_cholesky(_expand_batch(k_mm, size), jitter=self.gp_jitter)
        # a = L⁻ᵀ v, so that f = m(X) + K_nm a.
        a = torch.linalg.solve_triangular(factor.transpose(-2, -1), v.unsqueeze(-1), upper=True)
        mean = _evaluate_mean(context, self.gp_mean, x, dtype, device, raw)
        f = mean + (k_mn.transpose(-2, -1) @ a).squeeze(-1)
        return f, (context, x, z, v, kernel_values, k_mn, factor, jitter_info, a, raw)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, x, z, v, kernel_values, k_mn, factor, jitter_info, a, raw) = self._batched_function(values, flags)
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
            _accumulate(grads, name, _to_parameter_shape(gradient, shape_of(parameter, value)))

        if self._inducing_index is not None and self._learnable(self.inputs[self._inducing_index]):
            # The mean is evaluated at X, not Z, so Z gets kernel terms only.
            node = self.inputs[self._inducing_index]
            z_full = _expand_batch(z, size)
            z_gradient = self.gp_kernel.batch_input_vjp(z_full, x, g_mn, kernel_values)
            z_gradient = z_gradient + self.gp_kernel.batch_input_vjp_symmetric(z_full, g_mm, kernel_values)
            grads[node.name] = z_gradient.reshape(size, *shape_of(node, values[self._inducing_index]))
        learnable_inputs = self._learnable(self.inputs[0])
        if learnable_inputs:
            node = self.inputs[0]
            x_gradient = self.gp_kernel.batch_input_vjp(_expand_batch(x, size), z, g_mn.transpose(-2, -1), kernel_values)
            _accumulate(grads, node.name, x_gradient.reshape(size, *shape_of(node, values[0])))
        # f = m(X) + K_nm a: the mean parameters and X get the mean's VJP of g.
        _add_mean_gradients(
            grads, self.gp_mean, context, x, g, lambda node: shape_of(node, values[self.inputs.index(node)]),
            v.dtype, v.device, raw, input_node=self.inputs[0] if learnable_inputs else None,
            input_shape=shape_of(self.inputs[0], values[0]),
        )
        return grads


# ----------------------------------------------------------------------------
# Hilbert-space (reduced-rank) latent GP
# ----------------------------------------------------------------------------


def _accumulate_spectral_gradients(grads, kernel, parameters, frequencies, kernel_values, root, cotangent, shape_of):
    """Add the hyperparameter gradients of ``sqrt(S(ω))`` ``(B, M)`` for its cotangent ``(B, M)`` to ``grads``.

    ``d sqrt(S) = dS / (2 sqrt(S))``; a spectral density that underflows to zero contributes nothing.
    """
    positive = root > 0
    safe = torch.where(positive, root, torch.ones_like(root))
    spectral_cotangent = torch.where(positive, cotangent / (2.0 * safe), torch.zeros_like(cotangent))
    for name, derivative in kernel.batch_spectral_density_derivatives(frequencies, kernel_values).items():
        parameter = next(p for p in parameters if p.name == name)
        _accumulate(grads, name, _to_parameter_shape(_contract_diagonal(spectral_cotangent, derivative), shape_of(parameter)))


class _HilbertBasisScale(DeterministicParameter):
    """``sqrt(S(ω_k))`` ``(M,)``: the prior standard deviations of the weights of a centred Hilbert-space GP.

    A function of the kernel's hyperparameters only; its VJP is the spectral
    part of the non-centred :class:`_HilbertLatentGP` gradient.
    """

    def __init__(self, name, kernel, frequencies):
        self.gp_kernel = kernel
        self._basis_frequencies = frequencies
        self._kernel_parameters = list(kernel.parameters)

        def forward(*values):
            return self._batched_function(values, [False] * len(values))[0][0]

        def vjp(cotangent, *values):
            flags = [False] * len(values)
            return _finish(self._batched_vjp(_as_tensor(cotangent).unsqueeze(0), values, flags), False)

        super().__init__(name, forward, None, list(self._kernel_parameters), vjp_func=vjp)

    def _batched_function(self, values, flags):
        context = _BatchContext({id(node): bool(flag) for node, flag in zip(self.inputs, flags)})
        frequencies = self._basis_frequencies
        raw = _raw_values(self.inputs, values)
        kernel_values = context.kernel_values(self.gp_kernel, frequencies.dtype, frequencies.device, raw)
        root = self.gp_kernel.batch_spectral_density(frequencies, kernel_values).clamp_min(0.0).sqrt()  # (B, M)
        return root, (context, kernel_values, frequencies, root)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, kernel_values, frequencies, root) = self._batched_function(values, flags)

        def shape_of(node):
            value = _as_tensor(values[self.inputs.index(node)])
            return tuple(value.shape[1:]) if context.is_batched(node) else tuple(value.shape)

        grads = {}
        cotangent = cotangent.reshape(cotangent.shape[0], -1).to(dtype=root.dtype)
        _accumulate_spectral_gradients(grads, self.gp_kernel, self._kernel_parameters, frequencies, kernel_values,
                                       root, cotangent, shape_of)
        return grads

    def batched_forward(self, input_values):
        root, _ = self._batched_function(input_values, self._batched_flags())
        return _expand_batch(root, self.owner_model.n_chains)

    def batched_vjp(self, cotangent, input_values):
        flags = self._batched_flags()
        grads = self._batched_vjp(cotangent, input_values, flags)
        result = {}
        for node, flag in zip(self.inputs, flags):
            if flag and node.name in grads:
                result[node.name] = _expand_batch(grads[node.name], self.owner_model.n_chains)
        return result


class _HilbertLatentGP(LatentGP):
    r""":class:`LatentGP` with ``n_basis``: ``f = m(X) + Φ(X) (sqrt(S(ω)) ⊙ z)`` (Hilbert-space approximation).

    ``Φ_ik = Π_d φ_{d, j_d(k)}(x_id)`` with ``φ_{d,j}(x) = L_d^{-1/2} sin(ω_dj (x - c_d + L_d))``
    and ``ω_dj = π j / (2 L_d)`` (``j = 1, …, m_d``); ``S`` is the kernel's spectral
    density at ``ω_k = (ω_{1 j_1(k)}, …, ω_{D j_D(k)})``, so the kernel's
    hyperparameters enter only through ``S`` and no matrix is factorised.

    **Gradients** (hand-derived VJP). For a cotangent ``g`` of ``f`` let
    ``h = Φᵀ g`` and ``a = sqrt(S) ⊙ z``: ``z`` gets ``sqrt(S) ⊙ h``; ``S`` gets
    ``h ⊙ z / (2 sqrt(S))``, contracted with the kernel's spectral-density
    derivatives for every hyperparameter (centred: ``a = z`` gets ``h``, and the
    hyperparameters get theirs through :class:`_HilbertBasisScale`); learnable inputs get
    ``g_i Σ_k ∂Φ_ik/∂x_id a_k`` (one factor of the tensor product replaced by
    its derivative ``L_d^{-1/2} ω_dj cos(…)``) plus the mean's input VJP; the
    mean's parameters get the mean's VJP of ``g``.
    """

    _latent_unit = "basis function"

    def _prepare_inducing(self, inducing_points):
        return None

    def _prepare_basis(self, inputs, kernel, n_basis, boundary_factor, centred):
        self.gp_centred = bool(centred)
        x = _as_tensor(_input_value(inputs))
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        x = x.reshape(-1, 1) if x.ndim <= 1 else x
        if x.ndim != 2:
            raise ValueError(f"LatentGP inputs must have shape (N,) or (N, D); got {tuple(x.shape)}.")
        features = x.shape[1]
        counts = (n_basis,) * features if isinstance(n_basis, int) and not isinstance(n_basis, bool) else tuple(n_basis)
        if len(counts) != features or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in counts):
            raise ValueError(
                f"LatentGP n_basis must be a positive integer or one per input feature ({features}); got {n_basis!r}."
            )
        factor = 1.5 if boundary_factor is None else float(boundary_factor)
        if not factor >= 1.0:
            raise ValueError(f"LatentGP boundary_factor must be at least 1 (the inputs must lie inside the domain); got {factor}.")
        lower, upper = x.min(dim=0).values, x.max(dim=0).values
        half_range = (upper - lower) / 2.0
        if not bool((half_range > 0).all()):
            raise ValueError("Every input feature of a LatentGP with n_basis must vary over the inputs (they set its domain).")
        self.gp_n_basis = counts
        self.gp_boundary_factor = factor
        self._basis_centre = ((upper + lower) / 2.0).detach().clone()
        self._basis_boundary = (factor * half_range).detach().clone()
        # Square roots of the Laplacian eigenvalues per feature, and the tensor-product frequencies (M, D).
        self._axis_frequencies = [
            torch.pi * torch.arange(1, m + 1, dtype=x.dtype, device=x.device) / (2.0 * boundary)
            for m, boundary in zip(counts, self._basis_boundary)
        ]
        self._basis_frequencies = torch.cartesian_prod(*self._axis_frequencies).reshape(-1, features)
        try:
            kernel.spectral_density(self._basis_frequencies[:1])
        except NotImplementedError as error:
            raise TypeError(
                "LatentGP with n_basis needs a stationary kernel with a closed-form spectral density: RBF, Matern "
                f"(any nu), SpectralMixture, their sums and their products with Constant ({error})"
            ) from None

    def _latent_count(self, inputs, inducing_points):
        return math.prod(self.gp_n_basis)

    def _make_latent(self, name, latent_name, kernel, n):
        if not self.gp_centred:
            return super()._make_latent(name, latent_name, kernel, n)
        if kernel.parameters:
            scale = _HilbertBasisScale(f"{name}_basis_sd", kernel, self._basis_frequencies)
        else:
            scale = kernel.spectral_density(self._basis_frequencies).clamp_min(0.0).sqrt()
        return RandomParameter(latent_name or f"{name}_weights", Normal(0.0, scale), shape=n)

    def _model_inputs(self, inputs, latent, inducing_points, mean):
        # Centred: the hyperparameters enter through the prior of the weights, not through f.
        kernel_parameters = [] if self.gp_centred else self._kernel_parameters
        return _with_parameters([inputs, latent, *kernel_parameters], mean.parameters)

    def _derivative_function(self):
        return None

    def _finish_init(self, inducing_points):
        pass

    def _basis(self, x, feature=None):
        """``Φ(X)`` ``(B, n, M)`` for inputs ``(B, n, D)``; with ``feature = d`` its derivative w.r.t. ``x_d``.

        ``Φ(X)`` of the last inputs is cached: with fixed inputs every log
        density and gradient evaluation reuses it, and the comparison of the
        inputs costs ``O(n D)`` instead of the ``O(n M)`` sines.
        """
        if feature is None:
            cached = getattr(self, "_basis_cache", None)
            if cached is not None and _same_tensors((cached[0],), (x,)):
                return cached[1]
        result = None
        for d, (frequencies, centre, boundary) in enumerate(zip(self._axis_frequencies, self._basis_centre, self._basis_boundary)):
            frequencies = frequencies.to(dtype=x.dtype, device=x.device)
            centre, boundary = centre.to(dtype=x.dtype, device=x.device), boundary.to(dtype=x.dtype, device=x.device)
            angle = (x[..., d:d + 1] - centre + boundary) * frequencies  # (B, n, m_d)
            if d == feature:
                values = frequencies * torch.cos(angle) / boundary.sqrt()
            else:
                values = torch.sin(angle) / boundary.sqrt()
            result = values if result is None else (result.unsqueeze(-1) * values.unsqueeze(-2)).flatten(-2)
        if feature is None:
            self._basis_cache = (x.detach().clone(), result.detach())
        return result

    def _batched_function(self, values, flags):
        """``(f, pieces)`` in batched form for input ``values`` with chain flags ``flags``."""
        context = _BatchContext({id(node): bool(flag) for node, flag in zip(self.inputs, flags)})
        z = context.value(self.inputs[1], raw=values[1])
        z = z.reshape(z.shape[0], -1)
        dtype, device = z.dtype, z.device
        x = context.inputs(self.inputs[0], dtype, device, raw=values[0])
        if x.shape[-1] != len(self.gp_n_basis):
            raise ValueError(f"LatentGP '{self.name}' has basis functions for {len(self.gp_n_basis)} input features; got {x.shape[-1]}.")
        raw = _raw_values(self.inputs, values)
        if self.gp_centred:
            kernel_values = frequencies = root = None
            weights = z
        else:
            kernel_values = context.kernel_values(self.gp_kernel, dtype, device, raw)
            frequencies = self._basis_frequencies.to(dtype=dtype, device=device)
            root = self.gp_kernel.batch_spectral_density(frequencies, kernel_values).clamp_min(0.0).sqrt()  # (B, M)
            weights = root * z
        basis = self._basis(x)
        mean = _evaluate_mean(context, self.gp_mean, x, dtype, device, raw)
        f = mean + (basis @ weights.unsqueeze(-1)).squeeze(-1)
        return f, (context, x, z, kernel_values, frequencies, root, weights, basis, raw)

    def _batched_vjp(self, cotangent, values, flags):
        _, (context, x, z, kernel_values, frequencies, root, weights, basis, raw) = self._batched_function(values, flags)
        g = cotangent.reshape(cotangent.shape[0], -1).to(dtype=z.dtype)

        def shape_of(node):
            value = _as_tensor(values[self.inputs.index(node)])
            return tuple(value.shape[1:]) if context.is_batched(node) else tuple(value.shape)

        projected = (basis.transpose(-2, -1) @ g.unsqueeze(-1)).squeeze(-1)  # h = Φᵀ g, (B, M)
        latent = self.inputs[1]
        latent_gradient = projected if self.gp_centred else root * projected
        grads = {latent.name: latent_gradient.reshape(latent_gradient.shape[0], *shape_of(latent))}
        if not self.gp_centred:
            _accumulate_spectral_gradients(grads, self.gp_kernel, self._kernel_parameters, frequencies, kernel_values,
                                           root, projected * z, shape_of)
        inputs_node = self.inputs[0]
        learnable_inputs = _is_learnable(inputs_node)
        if learnable_inputs:
            columns = [g * (self._basis(x, feature=d) @ weights.unsqueeze(-1)).squeeze(-1) for d in range(x.shape[-1])]
            input_gradient = torch.stack(columns, dim=-1)  # (B, n, D)
            _accumulate(grads, inputs_node.name, input_gradient.reshape(input_gradient.shape[0], *shape_of(inputs_node)))
        # f = m(X) + Φ a: the mean parameters and the inputs get the mean's VJP of g.
        _add_mean_gradients(
            grads, self.gp_mean, context, x, g, shape_of, z.dtype, z.device, raw,
            input_node=inputs_node if learnable_inputs else None, input_shape=shape_of(inputs_node),
        )
        return grads


def _hilbert_latent_gp_predictive(gp, trace, new_inputs, n_samples=1, chunk_size=256):
    """Values ``(n_parameter_draws, n_samples, n_new)`` of a Hilbert-space latent GP at new inputs.

    The function is deterministic given the basis weights, so it is the basis
    evaluated at ``new_inputs`` for every draw of ``trace`` (the samples per
    draw are identical); draws are evaluated in batches of ``chunk_size``.
    """
    latent = torch.as_tensor(trace[gp.gp_latent.name])
    dtype, device = latent.dtype, latent.device
    z = latent.reshape(-1, latent.shape[-1])
    n_draws = z.shape[0]
    new = torch.as_tensor(_input_value(new_inputs)).to(dtype=dtype, device=device)
    draws = {node.name: _draws_of(trace, node, n_draws, dtype, device) for node in gp.inputs[2:]}
    outputs = []
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        values, flags = [new, z[start:stop]], [False, True]
        for node in gp.inputs[2:]:
            node_draws = draws[node.name]
            batched = node_draws.shape[0] > 1
            values.append(node_draws[start:stop] if batched else node_draws[0])
            flags.append(batched)
        f, _ = gp._batched_function(values, flags)
        f = _expand_batch(f, stop - start)
        outputs.append(f.unsqueeze(1).expand(stop - start, n_samples, f.shape[-1]))
    return torch.cat(outputs)


def _constructor_signature(init):
    """The signature of ``init`` without ``self`` (the public constructor signature)."""
    signature = inspect.signature(init)
    return signature.replace(parameters=list(signature.parameters.values())[1:])


# ``ExactGP(...)`` and ``LatentGP(...)`` dispatch on these signatures in ``__new__``
# (which unknown arguments fail to bind), and the documentation shows them.
ExactGP.__signature__ = _constructor_signature(ExactGP.__init__)
LatentGP.__signature__ = _constructor_signature(LatentGP.__init__)


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


def _sparse_latent_gp_predictive(gp, trace, new_inputs, n_samples=1, conditional_variance=False, chunk_size=64):
    r"""Values of a latent GP with inducing points at new inputs for every draw of ``trace``.

    Under the DTC approximation the function is deterministic given the
    whitened values ``v``, the inducing inputs and the hyperparameters:
    ``f* = m(X*) + K_*m L⁻ᵀ v`` at ``new_inputs``, so the ``n_samples``
    samples per draw are identical by default. With
    ``conditional_variance=True`` (default ``False``)
    independent Normal noise with the FITC-style variance
    ``diag(K_** - Q_**)`` (the prior variance that the inducing values do not
    explain) is added to every sample.

    ``trace`` is a ``SamplingResult`` (from NUTS or ``guide.sample``) or a
    mapping with values of shape ``(n_chains, n_draws, *shape)``. It must
    contain the whitened values; learnable inducing inputs, kernel
    parameters and the parameters of the mean function ``m`` are read from
    it as well (a node missing from the trace keeps its current value).
    Draws are evaluated in batches of ``chunk_size`` (default ``64``) with
    the batched kernel and mean-function API. The result has shape
    ``(n_parameter_draws, n_samples, n_new)``.
    """
    if not isinstance(gp, _SparseLatentGP):
        raise TypeError("gp must be a LatentGP with inducing points.")
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
    kernel_draws = _parameter_draws(trace, gp.gp_kernel.parameters, n_draws, dtype, device, _draws_of)
    mean_draws = _parameter_draws(trace, gp.gp_mean.parameters, n_draws, dtype, device, _draws_of)

    def take(tensor, start, stop):
        return tensor if tensor.shape[0] == 1 else tensor[start:stop]

    outputs = []
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        size = stop - start
        values = {name: take(draws, start, stop) for name, draws in kernel_draws.items()}
        mean_values = {name: take(draws, start, stop) for name, draws in mean_draws.items()}
        z_chunk = take(z, start, stop)
        k_mm = gp.gp_kernel.batch_evaluate(z_chunk, z_chunk, values)
        k_ms = gp.gp_kernel.batch_evaluate(z_chunk, new, values)
        factor, _ = batched_stable_cholesky(_expand_batch(k_mm, size), jitter=gp.gp_jitter)
        a = torch.linalg.solve_triangular(factor.transpose(-2, -1), v[start:stop].unsqueeze(-1), upper=True)
        f = gp.gp_mean.batch_evaluate(new, mean_values) + (k_ms.transpose(-2, -1) @ a).squeeze(-1)  # (size, n_new)
        f = f.unsqueeze(1).expand(size, n_samples, f.shape[-1])
        if conditional_variance:
            projected = torch.linalg.solve_triangular(factor, _expand_batch(k_ms, size), upper=False)
            k_diag = _expand_batch(gp.gp_kernel.batch_diag(new, values), size)
            variance = (k_diag - projected.square().sum(-2)).clamp_min(0.0)
            f = f + variance.sqrt().unsqueeze(1) * torch.randn(f.shape, dtype=dtype, device=device)
        outputs.append(f)
    return torch.cat(outputs)
