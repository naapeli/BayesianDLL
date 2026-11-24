"""Structured exact Gaussian processes: Kronecker (grid) and Toeplitz (regular 1-D).

Both classes compute the *exact* GP marginal likelihood
``log N(y | mean, K + σ² I)`` and its hand-derived gradients, but exploit the
structure of ``K`` instead of a dense Cholesky factorization:

* :class:`KroneckerGP`: inputs on a Cartesian grid and a product kernel, so
  ``K = K_1 ⊗ … ⊗ K_P``. Cost ``O(Σ_p n_p³ + N Σ_p n_p)`` for
  ``N = Π_p n_p`` instead of ``O(N³)``.
* :class:`ToeplitzGP`: equally spaced 1-D inputs and a stationary kernel, so
  ``K`` is symmetric Toeplitz. Levinson recursion gives the solve and the
  log determinant in ``O(n²)`` time and ``O(n)`` memory instead of
  ``O(n³)`` / ``O(n²)``.

Both are evaluated for all vectorized chains at once (leading chain
dimension; see :mod:`BayesianDLL.GP._process`).
"""

from __future__ import annotations

import math

import torch

from ..Distributions._resolve import resolve
from ._kernels import (
    Constant,
    Kernel,
    Matern,
    Periodic,
    ProductKernel,
    RationalQuadratic,
    RBF,
    SumKernel,
    _feature_matrix,
    _is_named,
)
from ._means import _is_zero
from ._process import (
    _BatchContext,
    _GPDistribution,
    _accumulate,
    _add_mean_gradients,
    _contract,
    _contract_diagonal,
    _draw,
    _draw_count,
    _evaluate_mean,
    _expand_batch,
    _finish,
    _draws_of,
    _input_value,
    _is_learnable,
    _parameter_draws,
    _parameter_shape,
    _resolved_value,
    _to_parameter_shape,
    _trace_draws,
    stable_cholesky,
)


def _mode_product(tensor, matrix, mode):
    """``tensor ×_mode matrix``: contract axis ``mode`` of ``tensor`` with the columns of ``matrix``."""
    result = torch.tensordot(matrix, tensor, dims=([1], [mode]))
    return torch.movedim(result, 0, mode)


def _kronecker_eigenvalues(eigenvalues):
    """Outer product ``λ_1 ⊗ … ⊗ λ_P`` as a tensor of shape ``(n_1, …, n_P)``."""
    result = eigenvalues[0]
    for values in eigenvalues[1:]:
        result = result.unsqueeze(-1) * values
    return result


def _batched_mode_product(tensor, matrix, mode):
    """Batched ``tensor ×_mode matrix``: ``tensor`` is ``(B, n_1, …, n_P)``, ``matrix`` ``(B', n, n_mode)``."""
    moved = torch.movedim(tensor, mode + 1, -1).unsqueeze(-2)  # (B, ..., 1, n_mode)
    rest = moved.ndim - 3
    view = matrix.transpose(-2, -1).reshape(matrix.shape[0], *([1] * rest), matrix.shape[-1], matrix.shape[-2])
    return torch.movedim((moved @ view).squeeze(-2), -1, mode + 1)


def _grid_points(xs):
    """All grid points ``(B, N, Σ_p D_p)`` in row-major order from batched factors ``(B_p, n_p, D_p)``."""
    size = max(x.shape[0] for x in xs)
    indices = torch.cartesian_prod(*[torch.arange(x.shape[1], device=x.device) for x in xs])
    if indices.ndim == 1:
        indices = indices.reshape(-1, 1)
    return torch.cat([_expand_batch(x, size)[:, indices[:, p]] for p, x in enumerate(xs)], dim=-1)


def _batched_kronecker_eigenvalues(eigenvalues):
    """``λ_1 ⊗ … ⊗ λ_P`` per batch element: ``(B, n_1, …, n_P)`` from ``(B_p, n_p)``."""
    result = eigenvalues[0]
    for values in eigenvalues[1:]:
        result = result.unsqueeze(-1) * values.reshape(values.shape[0], *([1] * (result.ndim - 1)), values.shape[-1])
    return result


# ----------------------------------------------------------------------------
# Kronecker-structured GP
# ----------------------------------------------------------------------------


class KroneckerGP(_GPDistribution):
    r"""Exact Gaussian Process marginal likelihood for inputs on a Cartesian product grid.

    Exploits the Kronecker structure of the covariance matrix across grid axes:

    .. math::

        \mathbf{K} = \mathbf{K}_1 \otimes \mathbf{K}_2 \otimes \dots \otimes \mathbf{K}_P

    The kernel is the product :math:`k(\mathbf{x}, \mathbf{x}') = \prod_p k_p(\mathbf{x}_p, \mathbf{x}'_p)`
    of one kernel per axis, and eigendecompositions of the factors give the likelihood in
    :math:`\mathcal{O}\left(\sum_{p} n_p^3 + N \sum_p n_p\right)` time
    instead of :math:`\mathcal{O}(N^3)` where :math:`N = \prod_{p=1}^P n_p`.

    Parameters
    ----------
    name : str
        Name of the GP. Posterior predictive results store the latent function draws under this name.
    grid_inputs : list of torch.Tensor or Data or Parameter
        Coordinates along each grid axis :math:`p = 1, \dots, P`, shape ``(n_p,)`` or
        ``(n_p, D_p)``; an axis may be learnable.
    kernels : list of Kernel
        Kernel function for each corresponding grid axis.
    noise_variance : float or torch.Tensor or Parameter, default 1e-6
        Scalar observation noise variance :math:`\sigma^2 > 0`.
    mean : MeanFunction, optional
        GP mean function :math:`m(X)`. Default is :class:`~BayesianDLL.GP.ZeroMean`.
    jitter : float, default 1e-8
        Additive jitter on the noise variance for numerical stability.

    Notes
    -----
    The observations have shape ``(n_1, ..., n_P)`` or are flattened in row-major
    order (the order of ``torch.cartesian_prod``). The grid must be complete:
    missing observations break the Kronecker structure.
    """

    def __init__(self, name, grid_inputs, kernels, noise_variance=1e-6, mean=None, jitter=1e-8):
        super().__init__()
        grid_inputs = list(grid_inputs)
        kernels = list(kernels)
        if len(grid_inputs) == 0 or len(grid_inputs) != len(kernels):
            raise ValueError("KroneckerGP needs one kernel per grid factor (and at least one factor).")
        for kernel in kernels:
            if not isinstance(kernel, Kernel):
                raise TypeError("kernels must be instances of BayesianDLL.GP.Kernel.")
        self.name = name
        self.grid_inputs = grid_inputs
        self.kernels = kernels
        self.noise_variance = noise_variance
        self.jitter = jitter
        for grid in grid_inputs:
            self.add_dependency(grid)
        self._kernel_parameter_nodes = [p for kernel in kernels for p in kernel.parameters]
        for parameter in self._kernel_parameter_nodes:
            self.add_dependency(parameter)
        self.add_dependency(noise_variance)
        self._register_mean(mean, "KroneckerGP")

    # -- helpers ------------------------------------------------------------
    def _factor_inputs(self):
        factors = [_feature_matrix(torch.as_tensor(_input_value(grid))) for grid in self.grid_inputs]
        dtype = factors[0].dtype if factors[0].is_floating_point() else torch.get_default_dtype()
        device = factors[0].device
        return [factor.to(dtype=dtype, device=device) for factor in factors]

    def _capture_training_inputs(self):
        self._training_inputs_data = [
            None if _is_learnable(grid) else _feature_matrix(torch.as_tensor(_input_value(grid))).detach().clone()
            for grid in self.grid_inputs
        ]

    def _training_axes(self):
        """The training grid axes: the captured fixed axes, or the current value of learnable axes."""
        current = self._factor_inputs()
        stored = self._training_inputs_data or [None] * len(current)
        return [axis if fixed is None else fixed.to(dtype=axis.dtype, device=axis.device) for axis, fixed in zip(current, stored)]

    @property
    def grid_shape(self):
        return tuple(factor.shape[0] for factor in self._factor_inputs())

    @property
    def event_shape(self):
        return self.grid_shape

    def _named_parameter_values(self):
        values = {}
        for kernel in self.kernels:
            for parameter in kernel.parameters:
                values[parameter.name] = resolve(parameter)
        return values

    def _decompose(self, xs, parameter_values):
        """Unbatched eigendecompositions of the factors (used for prediction and sampling)."""
        matrices, eigenvalues, eigenvectors = [], [], []
        for kernel, x in zip(self.kernels, xs):
            matrix = kernel(x, x, parameter_values=parameter_values)
            matrix = (matrix + matrix.transpose(-2, -1)) / 2
            values, vectors = torch.linalg.eigh(matrix)
            matrices.append(matrix)
            eigenvalues.append(values)
            eigenvectors.append(vectors)
        return matrices, eigenvalues, eigenvectors

    def _components(self, observed):
        context = _BatchContext()
        reference = self._factor_inputs()[0]
        dtype, device = reference.dtype, reference.device
        xs = [context.inputs(grid, dtype, device) for grid in self.grid_inputs]
        shape = tuple(x.shape[1] for x in xs)
        n = math.prod(shape)
        observed = torch.as_tensor(observed, dtype=dtype, device=device)
        if observed.numel() != n:
            raise ValueError(f"KroneckerGP observations must contain one value per grid point ({n}); got {observed.numel()}.")
        noise = context.value(self.noise_variance, dtype, device)
        noise = noise.reshape(noise.shape[0], -1)
        if noise.shape[1] != 1:
            raise ValueError("KroneckerGP noise_variance must be scalar.")
        noise = noise[:, 0]
        # A ZeroMean needs no grid points (and adds nothing to the gradients).
        points = None if _is_zero(self.mean) else _grid_points(xs)
        mean = torch.zeros((1, n), dtype=dtype, device=device) if points is None else _evaluate_mean(context, self.mean, points, dtype, device)
        kernel_values, matrices, eigenvalues, eigenvectors = [], [], [], []
        for kernel, x in zip(self.kernels, xs):
            values = context.kernel_values(kernel, dtype, device)
            matrix = kernel.batch_evaluate(x, x, values)
            matrix = (matrix + matrix.transpose(-2, -1)) / 2
            lam, vectors = torch.linalg.eigh(matrix)
            kernel_values.append(values)
            matrices.append(matrix)
            eigenvalues.append(lam)
            eigenvectors.append(vectors)
        size = max(noise.shape[0], mean.shape[0], *(m.shape[0] for m in matrices))
        residual = _expand_batch((observed.reshape(1, n) - mean).reshape(-1, *shape), size)
        # Round-off can make tiny eigenvalues of a PSD factor slightly negative.
        noise_view = (noise + self.jitter).reshape(-1, *([1] * len(shape)))
        denominator = _expand_batch(_batched_kronecker_eigenvalues(eigenvalues).clamp_min(0.0) + noise_view, size)
        rotated = residual
        for mode, vectors in enumerate(eigenvectors):
            rotated = _batched_mode_product(rotated, vectors.transpose(-2, -1), mode)
        scaled = rotated / denominator
        alpha = scaled
        for mode, vectors in enumerate(eigenvectors):
            alpha = _batched_mode_product(alpha, vectors, mode)
        alpha = _expand_batch(alpha, size)
        reduce = tuple(range(1, 1 + len(shape)))
        log_probability = -0.5 * (
            (rotated * scaled).sum(reduce) + torch.log(denominator).sum(reduce) + n * math.log(2 * math.pi)
        )
        batched = context.any([*self.grid_inputs, self.noise_variance, *self.mean.parameters, *self._kernel_parameter_nodes])
        return {
            "context": context, "xs": xs, "points": points, "shape": shape, "noise": noise, "matrices": matrices,
            "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "denominator": denominator, "alpha": alpha,
            "kernel_values": kernel_values, "log_probability": log_probability, "batched": batched,
        }

    # -- Distribution interface ----------------------------------------------
    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        alpha = parts["alpha"].reshape(parts["alpha"].shape[0], *torch.as_tensor(x).shape)
        return _finish(-alpha, parts["batched"])

    def factor_cotangents(self, parts):
        """The matrices ``G_p`` ``(B, n_p, n_p)`` with ``d log p = Σ_p ⟨G_p, dK_p⟩`` (see the class docstring)."""
        alpha, eigenvalues, eigenvectors = parts["alpha"], parts["eigenvalues"], parts["eigenvectors"]
        matrices, inverse = parts["matrices"], 1.0 / parts["denominator"]
        n_modes = len(matrices)
        size = alpha.shape[0]
        cotangents = []
        for p in range(n_modes):
            projected = alpha
            weights = inverse
            for q in range(n_modes):
                if q == p:
                    continue
                projected = _batched_mode_product(projected, matrices[q], q)
                view = [eigenvalues[q].shape[0]] + [1] * n_modes
                view[q + 1] = -1
                weights = weights * eigenvalues[q].reshape(view)
            n_p = alpha.shape[p + 1]
            unfolded_alpha = torch.movedim(alpha, p + 1, 1).reshape(size, n_p, -1)
            unfolded_projected = _expand_batch(torch.movedim(projected, p + 1, 1), size).reshape(size, n_p, -1)
            trace_weights = _expand_batch(torch.movedim(weights, p + 1, 1), size).reshape(size, n_p, -1).sum(dim=-1)
            vectors = eigenvectors[p]
            cotangents.append(
                0.5 * unfolded_alpha @ unfolded_projected.transpose(-2, -1)
                - 0.5 * (vectors * trace_weights.unsqueeze(-2)) @ vectors.transpose(-2, -1)
            )
        return cotangents

    def log_pdf_param_grads(self, x):
        parts = self._components(x)
        context = parts["context"]
        cotangents = self.factor_cotangents(parts)
        gradients = {}
        for kernel, x_p, values, cotangent in zip(self.kernels, parts["xs"], parts["kernel_values"], cotangents):
            for name, derivative in kernel.batch_derivatives(x_p, x_p, values).items():
                gradient = _contract(cotangent, derivative)
                gradients[name] = gradients[name] + gradient if name in gradients else gradient
        for name in list(gradients):
            parameter = next(p for p in self._kernel_parameter_nodes if p.name == name)
            gradients[name] = _to_parameter_shape(gradients[name], _parameter_shape(context, parameter))
        alpha = parts["alpha"]
        flat_alpha = alpha.reshape(alpha.shape[0], -1)
        learnable = [_is_learnable(grid) for grid in self.grid_inputs]
        axis_gradients = [None] * len(self.grid_inputs)
        for p, (kernel, x_p, values, cotangent) in enumerate(zip(self.kernels, parts["xs"], parts["kernel_values"], cotangents)):
            if learnable[p]:
                # ⟨G_p, K_p(X_p, X_p)⟩ with both arguments moving.
                axis_gradients[p] = kernel.batch_input_vjp_symmetric(x_p, cotangent, values)
        if parts["points"] is not None and any(learnable):
            # ∂⟨α, m(points)⟩/∂points, split into the factors' columns and summed over the other axes.
            mean_values = context.kernel_values(self.mean, alpha.dtype, alpha.device)
            point_vjp = self.mean.batch_input_vjp(parts["points"], flat_alpha, mean_values)
            widths = [x_p.shape[-1] for x_p in parts["xs"]]
            shape = parts["shape"]
            for p, columns in enumerate(torch.split(point_vjp, widths, dim=-1)):
                if not learnable[p]:
                    continue
                grid = columns.reshape(columns.shape[0], *shape, widths[p])
                others = tuple(1 + q for q in range(len(shape)) if q != p)
                axis = grid.sum(dim=others) if others else grid
                axis_gradients[p] = axis_gradients[p] + axis
        for grid, gradient in zip(self.grid_inputs, axis_gradients):
            if gradient is not None:
                _accumulate(gradients, grid.name, gradient.reshape(gradient.shape[0], *_parameter_shape(context, grid)))
        if _is_named(self.noise_variance):
            inverse = (1.0 / parts["denominator"]).reshape(alpha.shape[0], -1)
            noise_gradient = 0.5 * (flat_alpha.square().sum(-1) - inverse.sum(-1))
            _accumulate(gradients, self.noise_variance.name, _to_parameter_shape(
                noise_gradient.unsqueeze(-1), _parameter_shape(context, self.noise_variance)
            ))
        if parts["points"] is not None:
            # The residual is y - m(X) at the grid points (row-major), so ∂ log p / ∂m(X) = α.
            _add_mean_gradients(
                gradients, self.mean, context, parts["points"], flat_alpha,
                lambda node: _parameter_shape(context, node), alpha.dtype, alpha.device,
            )
        return _finish(gradients, parts["batched"])

    def _prior_draws(self, axes, eigenvalues, eigenvectors, n_samples):
        """Zero-mean prior draws ``(n_samples, n_1, …, n_P)`` on a grid: ``Q (sqrt(λ) ⊙ ε)``."""
        reference = axes[0]
        shape = tuple(axis.shape[0] for axis in axes)
        scale = _kronecker_eigenvalues(eigenvalues).clamp_min(0.0).sqrt()
        draws = torch.randn((n_samples, *shape), dtype=reference.dtype, device=reference.device) * scale
        for mode, vectors in enumerate(eigenvectors):
            draws = _mode_product(draws, vectors, mode + 1)
        return draws

    def _grid_mean(self, axes, parameter_values=None):
        points = _grid_points([axis.unsqueeze(0) for axis in axes])[0]
        return self._mean_at(points, parameter_values).reshape(tuple(axis.shape[0] for axis in axes))

    def _prior_latent(self, n_samples):
        """Prior draws ``(n_samples, n_1, …, n_P)`` of the latent function on the current grid."""
        xs = self._factor_inputs()
        _, eigenvalues, eigenvectors = self._decompose(xs, self._named_parameter_values())
        return self._grid_mean(xs) + self._prior_draws(xs, eigenvalues, eigenvectors, n_samples)

    def _posterior_latent(self, n_samples):
        r"""Posterior draws of the latent function on the current grid given the training data.

        Matheron's rule: with joint prior draws ``(f*, f)`` on the union of the
        new and training grids and ``ε ~ N(0, σ² I)``,
        ``f* + K_*X (K + σ² I)⁻¹ (y - m(X) - f - ε)`` (plus ``m(X*)``) is an
        exact posterior draw. The union grid is again a Cartesian product, so
        every step uses the factors' eigendecompositions and mode products and
        no ``N x N`` matrix is formed.
        """
        values = self._named_parameter_values()
        train = self._training_axes()
        dtype, device = train[0].dtype, train[0].device
        new = [axis.to(dtype=dtype, device=device) for axis in self._factor_inputs()]
        shape = tuple(axis.shape[0] for axis in train)
        new_shape = tuple(axis.shape[0] for axis in new)
        noise = _resolved_value(self.noise_variance, dtype, device).reshape(-1)[0] + self.jitter
        observed = torch.as_tensor(self._training_observed_data, dtype=dtype, device=device).reshape(shape)

        same_grid = all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(new, train))
        union = train if same_grid else [torch.cat([a, b]) for a, b in zip(new, train)]
        _, eigenvalues, eigenvectors = self._decompose(union, values)
        draws = self._prior_draws(union, eigenvalues, eigenvectors, n_samples)
        if same_grid:
            prior_new = prior_train = draws
        else:
            prior_new = draws[(slice(None), *(slice(0, n) for n in new_shape))]
            prior_train = draws[(slice(None), *(slice(n, None) for n in new_shape))]
            _, eigenvalues, eigenvectors = self._decompose(train, values)

        residual = observed - self._grid_mean(train) - prior_train - noise.sqrt() * torch.randn_like(prior_train)
        for mode, vectors in enumerate(eigenvectors):
            residual = _mode_product(residual, vectors.transpose(-2, -1), mode + 1)
        residual = residual / (_kronecker_eigenvalues(eigenvalues).clamp_min(0.0) + noise)
        for mode, vectors in enumerate(eigenvectors):
            residual = _mode_product(residual, vectors, mode + 1)  # (K + σ² I)⁻¹ r
        for mode, (kernel, a, b) in enumerate(zip(self.kernels, new, train)):
            residual = _mode_product(residual, kernel(a, b, parameter_values=values), mode + 1)  # K_*X
        return self._grid_mean(new) + prior_new + residual


def _split_new_inputs(gp, inputs, xs):
    """Return per-factor feature matrices ``(n*, D_p)`` of the new points."""
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != len(xs):
            raise ValueError("A new grid needs one input array per grid factor.")
        factors = [_feature_matrix(torch.as_tensor(_input_value(v))).to(dtype=xs[0].dtype, device=xs[0].device) for v in inputs]
        # Row-major Cartesian product of the factor indices.
        sizes = [f.shape[0] for f in factors]
        indices = torch.cartesian_prod(*[torch.arange(s, device=xs[0].device) for s in sizes])
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)
        return [factor[indices[:, p]] for p, factor in enumerate(factors)]
    points = _feature_matrix(torch.as_tensor(_input_value(inputs))).to(dtype=xs[0].dtype, device=xs[0].device)
    widths = [x.shape[1] for x in xs]
    if points.shape[1] != sum(widths):
        raise ValueError(f"New points need {sum(widths)} columns (the grid factors' features); got {points.shape[1]}.")
    return list(torch.split(points, widths, dim=1))


def _rowwise_contract(rows, tensor):
    """``out[s] = Σ_i Π_p rows[p][s, i_p] tensor[i_1, …, i_P]`` in ``O(n* N)``."""
    n_new = rows[0].shape[0]
    current = rows[0] @ tensor.reshape(tensor.shape[0], -1)  # (n*, n_2 ⋯ n_P)
    for p in range(1, len(rows)):
        current = current.reshape(n_new, rows[p].shape[1], -1)
        current = (rows[p].unsqueeze(-1) * current).sum(dim=1)
    return current.reshape(n_new)


def kronecker_gp_predictive(gp, trace, observed_values, inputs, n_samples=1, observation_noise=None, diagonal=False):
    r"""Sample predictions of a :class:`KroneckerGP` at new points.

    ``inputs`` is either a list with one array per grid factor (a new grid,
    flattened in row-major order) or an ``(n*, Σ_p D_p)`` matrix of
    arbitrary points whose columns are split between the factors. For every
    hyperparameter draw, with ``U_p = K_p(x*_p, X_p) Q_p`` and the row-wise
    Kronecker product ``U``:

    ``mean = m(X*) + K_*X α`` and ``cov = K_** - U diag(1/(λ+σ²)) Uᵀ``,

    both evaluated without forming an ``N x N`` matrix (the mean in
    ``O(n* N)``; the full covariance needs an ``n* x N`` matrix ``U``).
    With ``diagonal=True`` only the marginal variances are computed (in
    ``O(n* N)``) and the draws are independent across points, which is
    enough for pointwise credible bands on large prediction grids. Here
    ``α = (K + σ² I)⁻¹ (y - m(X))`` with ``y = observed_values`` (grid
    shaped or flattened) and the mean function ``m`` evaluated per draw at
    the training grid points and the new points (the factors' features
    concatenated); the kernel parameters, the noise variance and the named
    parameters of the mean function are read from ``trace``, and a given
    ``observation_noise`` variance is added to the diagonal. The result has
    shape ``(n_parameter_draws, n_samples, n*)``.
    """
    if not isinstance(gp, KroneckerGP):
        raise TypeError("gp must be a KroneckerGP.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    xs = gp._training_axes()
    dtype, device = xs[0].dtype, xs[0].device
    new = _split_new_inputs(gp, inputs, xs)
    n_new = new[0].shape[0]
    observed_values = torch.as_tensor(_input_value(observed_values), dtype=dtype, device=device)
    n_draws = _draw_count(trace)
    parameter_draws = _parameter_draws(trace, [p for kernel in gp.kernels for p in kernel.parameters], n_draws, dtype, device)
    noise_draws = _trace_draws(trace, gp.noise_variance, n_draws, dtype, device)
    mean_draws = _parameter_draws(trace, gp.mean.parameters, n_draws, dtype, device)
    shape = tuple(x.shape[0] for x in xs)
    new_points = torch.cat(new, dim=1)
    # Learnable grid axes are read from the trace draw by draw.
    axis_draws = [_draws_of(trace, grid, n_draws, dtype, device) if _is_learnable(grid) else None for grid in gp.grid_inputs]
    fixed_xs = xs

    outputs = []
    for draw_index in range(n_draws):
        xs = [
            fixed if draws is None else draws[draw_index if draws.shape[0] > 1 else 0].reshape(fixed.shape)
            for fixed, draws in zip(fixed_xs, axis_draws)
        ]
        training_points = _grid_points([x.unsqueeze(0) for x in xs])[0]
        values = _draw(parameter_draws, draw_index)
        mean_values = _draw(mean_draws, draw_index)
        noise = noise_draws[draw_index].reshape(-1)[0]
        residual = observed_values.reshape(shape) - gp.mean(training_points, parameter_values=mean_values).reshape(shape)
        _, eigenvalues, eigenvectors = gp._decompose(xs, values)
        inverse = 1.0 / (_kronecker_eigenvalues(eigenvalues).clamp_min(0.0) + noise + gp.jitter)
        rotated = residual
        for mode, vectors in enumerate(eigenvectors):
            rotated = _mode_product(rotated, vectors.transpose(-2, -1), mode)
        scaled = rotated * inverse  # Qᵀ α
        cross = [kernel(x_new, x, parameter_values=values) for kernel, x_new, x in zip(gp.kernels, new, xs)]
        projected = [k @ q for k, q in zip(cross, eigenvectors)]  # U_p
        predictive_mean = gp.mean(new_points, parameter_values=mean_values) + _rowwise_contract(projected, scaled)
        extra = 0.0
        if observation_noise is not None:
            extra = _resolved_value(observation_noise, dtype, device).reshape(-1)[0]
        if diagonal:
            prior_variance = torch.ones(n_new, dtype=dtype, device=device)
            for kernel, x_new in zip(gp.kernels, new):
                prior_variance = prior_variance * kernel.diag(x_new, parameter_values=values)
            variance = prior_variance - _rowwise_contract([u.square() for u in projected], inverse) + extra
            noise_draw = torch.randn((n_samples, n_new), dtype=dtype, device=device)
            outputs.append(predictive_mean + noise_draw * variance.clamp_min(0.0).sqrt())
            continue
        prior = torch.ones((n_new, n_new), dtype=dtype, device=device)
        for kernel, x_new in zip(gp.kernels, new):
            prior = prior * kernel(x_new, x_new, parameter_values=values)
        rows = projected[0]
        for u in projected[1:]:
            rows = (rows.unsqueeze(-1) * u.unsqueeze(1)).reshape(n_new, -1)
        covariance = prior - (rows * inverse.reshape(1, -1)) @ rows.transpose(-2, -1)
        covariance = covariance + extra * torch.eye(n_new, dtype=dtype, device=device)
        covariance = (covariance + covariance.transpose(-2, -1)) / 2
        from ._process import stable_cholesky
        factor = stable_cholesky(covariance, jitter=max(gp.jitter, 1e-10))
        outputs.append(predictive_mean + torch.randn((n_samples, n_new), dtype=dtype, device=device) @ factor.transpose(-2, -1))
    return torch.stack(outputs)


# ----------------------------------------------------------------------------
# Toeplitz (regular 1-D grid) GP
# ----------------------------------------------------------------------------


def levinson_solve(first_column, rhs):
    r"""Solve ``T X = B`` for symmetric positive-definite Toeplitz matrices ``T``.

    ``first_column`` is ``t`` with ``T_ij = t_|i-j|``; ``rhs`` has shape
    ``(n,)`` or ``(n, k)``. Returns ``(X, log det T)``. A batch of systems is
    solved at once when ``first_column`` has shape ``(B, n)``; ``rhs`` then
    has shape ``(B, n)`` or ``(B, n, k)`` and ``log det T`` shape ``(B,)``.

    This is Levinson's recursion (Golub & Van Loan, Alg. 4.7.2) on
    ``T / t_0``: the Durbin vector solves the Yule-Walker systems and the
    prediction-error variances ``β_k`` give
    ``log det T = n log t_0 + Σ_k log β_k``. ``O(n²)`` time per right-hand
    side and ``O(n)`` memory. The recursion is sequential in ``n``: one
    Python step per row, each step operating on the whole batch. Raises
    ``RuntimeError`` if a ``T`` is not positive definite.
    """
    if first_column.ndim == 1:
        vector = rhs.ndim == 1
        solution, log_det = _levinson(first_column.unsqueeze(0), rhs.reshape(1, rhs.shape[0], -1))
        return (solution[0, :, 0] if vector else solution[0]), log_det[0]
    vector = rhs.ndim == 2
    solution, log_det = _levinson(first_column, rhs.reshape(rhs.shape[0], rhs.shape[1], -1))
    return (solution[..., 0] if vector else solution), log_det


def _levinson(t, rhs):
    """Batched Levinson recursion: ``t`` ``(B, n)``, ``rhs`` ``(B', n, k)`` -> ``((B, n, k), (B,))``."""
    size, n = t.shape
    if size == 1:
        solution, log_det = _levinson_single(t[0], rhs[0])
        return solution.unsqueeze(0), log_det.reshape(1)
    t0 = t[:, 0]
    b = (rhs / t0[:, None, None]).expand(size, n, rhs.shape[-1])
    x = torch.zeros(b.shape, dtype=b.dtype, device=b.device)
    x[:, 0] = b[:, 0]
    log_det = n * torch.log(t0)
    if n == 1:
        return x, log_det
    r = t[:, 1:] / t0[:, None]
    # Reversed copies avoid a flip (and an allocation) in every step:
    # r[:k] reversed is reversed_r[n-1-k:], and y[:k] reversed is
    # reversed_y[n-1-k:].
    reversed_r = r.flip(-1).contiguous()
    y = torch.zeros_like(r)
    reversed_y = torch.zeros_like(r)
    alpha = -r[:, 0]
    y[:, 0] = alpha
    reversed_y[:, n - 2] = alpha
    beta = torch.ones_like(t0)
    betas = torch.empty((size, n - 1), dtype=t.dtype, device=t.device)
    for k in range(1, n):
        beta = (1.0 - alpha * alpha) * beta
        betas[:, k - 1] = beta
        start = n - 1 - k
        r_k = reversed_r[:, start:]
        mu = (b[:, k] - (r_k.unsqueeze(-2) @ x[:, :k]).squeeze(-2)) / beta[:, None]
        x[:, :k] += reversed_y[:, start:, None] * mu[:, None, :]
        x[:, k] = mu
        if k < n - 1:
            alpha = -(r[:, k] + (r_k * y[:, :k]).sum(-1)) / beta
            forward = torch.addcmul(y[:, :k], alpha[:, None], reversed_y[:, start:])
            reversed_y[:, start:].addcmul_(alpha[:, None], y[:, :k])
            y[:, :k] = forward
            y[:, k] = alpha
            reversed_y[:, start - 1] = alpha
    if not bool((betas > 0).all()):
        raise RuntimeError("Toeplitz covariance is not positive definite (Levinson recursion broke down).")
    return x, log_det + torch.log(betas).sum(-1)


def _levinson_single(t, rhs):
    """Unbatched Levinson recursion with Python-float scalars (fewer tensor operations per step)."""
    n = t.shape[0]
    b = rhs / t[0]
    x = torch.zeros(b.shape, dtype=b.dtype, device=b.device)
    x[0] = b[0]
    log_det = n * torch.log(t[0])
    if n == 1:
        return x, log_det
    r = t[1:] / t[0]
    reversed_r = r.flip(0).contiguous()
    y = torch.zeros(n - 1, dtype=t.dtype, device=t.device)
    reversed_y = torch.zeros_like(y)
    alpha = -float(r[0])
    y[0] = alpha
    reversed_y[n - 2] = alpha
    beta = 1.0
    log_beta = 0.0
    for k in range(1, n):
        beta = (1.0 - alpha * alpha) * beta
        if not beta > 0.0:
            raise RuntimeError("Toeplitz covariance is not positive definite (Levinson recursion broke down).")
        log_beta += math.log(beta)
        start = n - 1 - k
        r_k = reversed_r[start:]
        mu = (b[k] - r_k @ x[:k]) / beta
        x[:k].addr_(reversed_y[start:], mu)
        x[k] = mu
        if k < n - 1:
            alpha = -(float(r[k]) + float(r_k @ y[:k])) / beta
            forward = y[:k] + alpha * reversed_y[start:]
            reversed_y[start:].add_(y[:k], alpha=alpha)
            y[:k] = forward
            y[k] = alpha
            reversed_y[start - 1] = alpha
    return x, log_det + log_beta


def _correlation(f, g):
    """``c_k = Σ_u f_u g_{u+k}`` for ``k = 0..n-1`` along the last axis, with an FFT (``O(n log n)``)."""
    n = f.shape[-1]
    size = 2 * n
    spectrum = torch.conj(torch.fft.rfft(f, size, dim=-1)) * torch.fft.rfft(g, size, dim=-1)
    return torch.fft.irfft(spectrum, size, dim=-1)[..., :n]


def _lower_toeplitz_diagonal_sums(a):
    r"""Sums of the diagonals of ``L(a) L(a)ᵀ`` (``L(a)``: lower Toeplitz, first column ``a``), batched.

    ``s_k = Σ_u (n - k - u) a_u a_{u+k}`` for ``k = 0..n-1``.
    """
    n = a.shape[-1]
    lags = torch.arange(n, dtype=a.dtype, device=a.device)
    return (n - lags) * _correlation(a, a) - _correlation(lags * a, a)


def _toeplitz_inverse_diagonal_sums(inverse_first_column):
    r"""``D_k = Σ_i (T⁻¹)_{i+k, i}`` from ``x = T⁻¹ e_0`` (Gohberg-Semencul), batched over leading axes.

    For symmetric positive-definite Toeplitz ``T``,
    ``T⁻¹ = (L(x) L(x)ᵀ - L(c) L(c)ᵀ) / x_0`` with ``c = (0, x_{n-1}, …, x_1)``.
    """
    x = inverse_first_column
    c = torch.cat([torch.zeros_like(x[..., :1]), x[..., 1:].flip(-1)], dim=-1)
    return (_lower_toeplitz_diagonal_sums(x) - _lower_toeplitz_diagonal_sums(c)) / x[..., :1]


_TOEPLITZ_KERNELS = (RBF, Matern, RationalQuadratic, Periodic, Constant)


def _check_stationary(kernel):
    if isinstance(kernel, (SumKernel, ProductKernel)):
        for part in kernel.kernels:
            _check_stationary(part)
        return
    if not isinstance(kernel, _TOEPLITZ_KERNELS):
        raise TypeError(
            "ToeplitzGP needs a stationary kernel built from RBF, Matern, RationalQuadratic, Periodic and "
            f"Constant kernels; got {type(kernel).__name__}. Use noise_variance instead of WhiteNoise."
        )


class ToeplitzGP(_GPDistribution):
    r"""Exact Gaussian Process marginal likelihood for equally spaced 1-D inputs.

    For a stationary kernel on regularly spaced 1-D inputs :math:`x_i = x_0 + i h`,
    the covariance matrix :math:`\mathbf{K} + \sigma^2 \mathbf{I}` is symmetric Toeplitz.
    The marginal likelihood and solve are evaluated via Levinson-Durbin recursion in :math:`\mathcal{O}(N^2)` time
    and :math:`\mathcal{O}(N)` memory.

    Parameters
    ----------
    name : str
        Name of the GP. Posterior predictive results store the latent function draws under this name.
    inputs : torch.Tensor or Data
        Fixed, equally spaced 1-D coordinates of shape ``(N,)`` or ``(N, 1)``.
    kernel : Kernel
        Stationary kernel built from :class:`~BayesianDLL.GP.RBF`, :class:`~BayesianDLL.GP.Matern`,
        :class:`~BayesianDLL.GP.RationalQuadratic`, :class:`~BayesianDLL.GP.Periodic` and
        :class:`~BayesianDLL.GP.Constant` kernels.
    noise_variance : float or torch.Tensor or Parameter, default 1e-6
        Scalar observation noise variance :math:`\sigma^2 > 0`.
    mean : MeanFunction, optional
        GP mean function :math:`m(x)`. Default is :class:`~BayesianDLL.GP.ZeroMean`.
    jitter : float, default 1e-6
        Relative diagonal jitter: ``jitter * max(1, k(0) + σ²)`` is added to the diagonal.
    spacing_tolerance : float, default 1e-6
        Relative tolerance for verifying equal grid spacing.

    Notes
    -----
    Learnable inputs are not supported (a learnable spacing would be redundant
    with the lengthscale); use :class:`ExactGP` for them. The recursion is
    sequential in :math:`N`, so a dense Cholesky factorisation stays
    competitive up to one or two thousand points; beyond that, and for memory,
    the recursion wins. Predictive sampling forms dense matrices over the new
    inputs, so predict on modest windows.
    """

    def __init__(self, name, inputs, kernel: Kernel, noise_variance=1e-6, mean=None, jitter=1e-6, spacing_tolerance=1e-6):
        super().__init__()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of BayesianDLL.GP.Kernel.")
        _check_stationary(kernel)
        if _is_learnable(inputs):
            raise TypeError(
                "ToeplitzGP needs fixed, equally spaced inputs (a Data node or a tensor); a learnable spacing would be "
                "redundant with the kernel's lengthscale, so inputs cannot be a RandomParameter or "
                "DeterministicParameter. Use ExactGP for learnable inputs."
            )
        self.name = name
        self.inputs = inputs
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.jitter = jitter
        x = self._training_inputs()
        if x.shape[1] != 1:
            raise ValueError("ToeplitzGP inputs must be one-dimensional.")
        if x.shape[0] > 1:
            steps = x[1:, 0] - x[:-1, 0]
            if not bool((steps - steps[0]).abs().max() <= spacing_tolerance * steps[0].abs().clamp_min(1e-300)) or not bool(steps[0] != 0):
                raise ValueError("ToeplitzGP inputs must be equally spaced.")
        self.add_dependency(inputs)
        self._register_kernel(kernel)
        self.add_dependency(noise_variance)
        self._register_mean(mean, "ToeplitzGP")

    def _training_x(self):
        """The captured training inputs as an ``(n, 1)`` float matrix."""
        x = _feature_matrix(torch.as_tensor(self._training_input_value()))
        return x if x.is_floating_point() else x.to(torch.get_default_dtype())

    def _training_inputs(self):
        x = _feature_matrix(torch.as_tensor(_input_value(self.inputs)))
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        return x

    @property
    def event_shape(self):
        new = _feature_matrix(torch.as_tensor(_input_value(self.inputs)))
        return (new.shape[0],)

    def _named_parameter_values(self):
        return {parameter.name: resolve(parameter) for parameter in self.kernel.parameters}

    def _batched_first_column(self, x, kernel_values, noise):
        """First columns ``(B, n)`` of ``K + σ² I + jitter`` and ``d t_0 / d (k(0) + σ²)`` ``(B,)``."""
        column = self.kernel.batch_evaluate(x[:, :1], x, kernel_values)[:, 0, :]
        size = max(column.shape[0], noise.shape[0])
        column = _expand_batch(column, size).clone()
        diagonal = column[:, 0] + noise
        scale = diagonal.abs().clamp_min(1.0)
        jitter = max(self.jitter, torch.finfo(x.dtype).eps)
        column[:, 0] = diagonal + jitter * scale
        diagonal_scale = 1.0 + jitter * (diagonal.abs() > 1.0).to(x.dtype)
        return column, diagonal_scale

    def _first_column(self, x, parameter_values, noise):
        """Unbatched first column (used by :func:`toeplitz_gp_predictive`)."""
        values = {name: torch.as_tensor(value).unsqueeze(0) for name, value in parameter_values.items()}
        column, diagonal_scale = self._batched_first_column(
            _feature_matrix(x).unsqueeze(0), values, torch.as_tensor(noise).reshape(1)
        )
        return column[0], diagonal_scale[0]

    def _components(self, observed, need_inverse=False):
        context = _BatchContext()
        reference = self._training_inputs()
        dtype, device = reference.dtype, reference.device
        x = context.inputs(self.inputs, dtype, device)
        n = x.shape[1]
        observed = torch.as_tensor(observed, dtype=dtype, device=device).reshape(-1)
        if observed.numel() != n:
            raise ValueError(f"ToeplitzGP observations must contain one value per input ({n}); got {observed.numel()}.")
        noise = context.value(self.noise_variance, dtype, device)
        noise = noise.reshape(noise.shape[0], -1)
        if noise.shape[1] != 1:
            raise ValueError("ToeplitzGP noise_variance must be scalar.")
        kernel_values = context.kernel_values(self.kernel, dtype, device)
        column, diagonal_scale = self._batched_first_column(x, kernel_values, noise[:, 0])
        mean = _evaluate_mean(context, self.mean, x, dtype, device)
        size = max(column.shape[0], mean.shape[0])
        column, diagonal_scale = _expand_batch(column, size), _expand_batch(diagonal_scale, size)
        residual = _expand_batch(observed - mean, size)
        if need_inverse:
            unit = torch.zeros_like(residual)
            unit[:, 0] = 1.0
            solution, log_det = _levinson(column, torch.stack([residual, unit], dim=-1))
            alpha, inverse_column = solution[..., 0], solution[..., 1]
        else:
            solution, log_det = _levinson(column, residual.unsqueeze(-1))
            alpha, inverse_column = solution[..., 0], None
        log_probability = -0.5 * ((residual * alpha).sum(-1) + log_det + n * math.log(2 * math.pi))
        batched = context.any([self.inputs, self.noise_variance, *self.mean.parameters, *self.kernel.parameters])
        return {
            "context": context, "x": x, "alpha": alpha, "inverse_column": inverse_column,
            "diagonal_scale": diagonal_scale, "kernel_values": kernel_values,
            "log_probability": log_probability, "batched": batched,
        }

    def log_pdf(self, x):
        parts = self._components(x)
        return _finish(parts["log_probability"], parts["batched"])

    def log_pdf_grad(self, x):
        parts = self._components(x)
        return _finish(-parts["alpha"], parts["batched"])

    def column_cotangent(self, parts):
        """``g`` ``(B, n)`` with ``d log p = g · dt`` for a change ``dt`` of the first column."""
        alpha = parts["alpha"]
        weights = torch.full_like(alpha[:1], 2.0)
        weights[:, 0] = 1.0
        quadratic = _correlation(alpha, alpha)
        trace = _toeplitz_inverse_diagonal_sums(parts["inverse_column"])
        return 0.5 * weights * (quadratic - trace)

    def log_pdf_param_grads(self, x):
        parts = self._components(x, need_inverse=True)
        context = parts["context"]
        cotangent = self.column_cotangent(parts)
        # The scale-dependent jitter multiplies the derivative of t_0.
        cotangent_kernel = cotangent.clone()
        cotangent_kernel[:, 0] = cotangent[:, 0] * parts["diagonal_scale"]
        x_values = parts["x"]
        gradients = {}
        for name, derivative in self.kernel.batch_derivatives(x_values[:, :1], x_values, parts["kernel_values"]).items():
            parameter = next(p for p in self.kernel.parameters if p.name == name)
            gradient = _contract_diagonal(cotangent_kernel, derivative[..., 0, :])
            gradients[name] = _to_parameter_shape(gradient, _parameter_shape(context, parameter))
        if _is_named(self.noise_variance):
            noise_gradient = (cotangent[:, 0] * parts["diagonal_scale"]).unsqueeze(-1)
            _accumulate(gradients, self.noise_variance.name,
                        _to_parameter_shape(noise_gradient, _parameter_shape(context, self.noise_variance)))
        # The residual is y - m(X), so ∂ log p / ∂m(X) = α.
        _add_mean_gradients(
            gradients, self.mean, context, x_values, parts["alpha"], lambda node: _parameter_shape(context, node),
            x_values.dtype, x_values.device,
        )
        return _finish(gradients, parts["batched"])

    def _prior_latent(self, n_samples):
        """Prior draws of the latent function at the current inputs (dense Cholesky; intended for small ``n``)."""
        x = self._training_inputs()
        factor = stable_cholesky(self.kernel(x, x, parameter_values=self._named_parameter_values()), jitter=self.jitter)
        return self._mean_at(x) + torch.randn((n_samples, x.shape[0]), dtype=x.dtype, device=x.device) @ factor.transpose(-2, -1)

    def _posterior_latent(self, n_samples):
        """Posterior draws at the current inputs given the training data (Levinson solves, dense over the new inputs)."""
        x = self._training_x()
        dtype, device = x.dtype, x.device
        n = x.shape[0]
        new = _feature_matrix(torch.as_tensor(_input_value(self.inputs))).to(dtype=dtype, device=device)
        observed_values = torch.as_tensor(self._training_observed_data, dtype=dtype, device=device).reshape(-1)
        values = self._named_parameter_values()
        mean_values = {p.name: resolve(p) for p in self.mean.parameters if _is_named(p)}
        noise = _resolved_value(self.noise_variance, dtype, device).reshape(-1)[0]
        column, _ = self._first_column(x, values, noise)
        cross = self.kernel(x, new, parameter_values=values)
        rhs = torch.cat([(observed_values - self.mean(x, parameter_values=mean_values)).reshape(n, 1), cross], dim=1)
        solution, _ = levinson_solve(column, rhs)
        predictive_mean = self.mean(new, parameter_values=mean_values) + cross.transpose(-2, -1) @ solution[:, 0]
        covariance = self.kernel(new, new, parameter_values=values) - cross.transpose(-2, -1) @ solution[:, 1:]
        covariance = (covariance + covariance.transpose(-2, -1)) / 2
        factor = stable_cholesky(covariance, jitter=max(self.jitter, 1e-10))
        return predictive_mean + torch.randn((n_samples, new.shape[0]), dtype=dtype, device=device) @ factor.transpose(-2, -1)


def toeplitz_gp_predictive(gp, trace, observed_values, inputs, n_samples=1, observation_noise=None):
    r"""Sample predictions of a :class:`ToeplitzGP` at arbitrary new 1-D inputs.

    ``mean = m(X*) + K_*X α`` and ``cov = K_** - K_*X T⁻¹ K_X*``, where the
    solves use :func:`levinson_solve` with ``n*`` right-hand sides
    (``O(n² n*)``; no ``n x n`` matrix is formed), ``T = K + σ² I`` plus
    jitter and ``α = T⁻¹ (y - m(X))`` with ``y = observed_values`` and the
    mean function ``m`` evaluated per draw. The kernel parameters, noise
    variance and the named parameters of the mean function are read from
    ``trace``; a given ``observation_noise`` variance is added to the
    diagonal. The result has shape ``(n_parameter_draws, n_samples, n*)``.
    """
    if not isinstance(gp, ToeplitzGP):
        raise TypeError("gp must be a ToeplitzGP.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    x = gp._training_x()
    dtype, device = x.dtype, x.device
    n = x.shape[0]
    new = _feature_matrix(torch.as_tensor(_input_value(inputs))).to(dtype=dtype, device=device)
    n_new = new.shape[0]
    observed_values = torch.as_tensor(_input_value(observed_values), dtype=dtype, device=device).reshape(-1)
    n_draws = _draw_count(trace)
    parameter_draws = _parameter_draws(trace, gp.kernel.parameters, n_draws, dtype, device)
    noise_draws = _trace_draws(trace, gp.noise_variance, n_draws, dtype, device)
    mean_draws = _parameter_draws(trace, gp.mean.parameters, n_draws, dtype, device)
    outputs = []
    for draw_index in range(n_draws):
        values = _draw(parameter_draws, draw_index)
        mean_values = _draw(mean_draws, draw_index)
        noise = noise_draws[draw_index].reshape(-1)[0]
        column, _ = gp._first_column(x, values, noise)
        cross = gp.kernel(x, new, parameter_values=values)  # (n, n*)
        rhs = torch.cat([(observed_values - gp.mean(x, parameter_values=mean_values)).reshape(n, 1), cross], dim=1)
        solution, _ = levinson_solve(column, rhs)
        predictive_mean = gp.mean(new, parameter_values=mean_values) + cross.transpose(-2, -1) @ solution[:, 0]
        covariance = gp.kernel(new, new, parameter_values=values) - cross.transpose(-2, -1) @ solution[:, 1:]
        if observation_noise is not None:
            covariance = covariance + _resolved_value(observation_noise, dtype, device).reshape(-1)[0] * torch.eye(n_new, dtype=dtype, device=device)
        covariance = (covariance + covariance.transpose(-2, -1)) / 2
        factor = stable_cholesky(covariance, jitter=max(gp.jitter, 1e-10))
        outputs.append(predictive_mean + torch.randn((n_samples, n_new), dtype=dtype, device=device) @ factor.transpose(-2, -1))
    return torch.stack(outputs)

