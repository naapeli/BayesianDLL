"""Generalised Gauss-Newton curvature of the unconstrained log joint (see :class:`GaussNewtonCurvature`)."""

from contextlib import contextmanager

import networkx as nx
import torch

from .._model import _distribution_arguments


def finite_difference_columns(gradient, flat, step=None, columns=None):
    """Central differences of ``gradient(flat)`` for the given columns (all by default), symmetrised.

    Column ``j`` is ``(g(x + h_j e_j) - g(x - h_j e_j)) / (2 h_j)`` with
    ``h_j = step * max(1, |x_j|)``. With ``columns`` the other columns are
    taken from the transposed rows (entries between two uncomputed columns
    are zero).
    """
    if step is None:
        step = torch.finfo(flat.dtype).eps ** (1 / 3)
    dim = flat.numel()
    hessian = torch.zeros(dim, dim, dtype=flat.dtype, device=flat.device)
    for j in range(dim) if columns is None else columns:
        h = step * max(1.0, abs(float(flat[j])))
        shift = torch.zeros_like(flat)
        shift[j] = h
        hessian[:, j] = (gradient(flat + shift) - gradient(flat - shift)) / (2 * h)
    if columns is not None:
        mask = torch.zeros(dim, dtype=torch.bool, device=flat.device)
        mask[list(columns)] = True
        hessian = torch.where(mask.unsqueeze(0), hessian, hessian.T)
    return 0.5 * (hessian + hessian.T)


def project_psd(matrix):
    """``matrix`` if it is positive semi-definite, else the nearest one (negative eigenvalues set to zero)."""
    scale = float(matrix.abs().max())
    if scale == 0.0:
        return matrix
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    if int(torch.linalg.cholesky_ex(matrix + 1e-12 * scale * identity)[1]) == 0:
        return matrix
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp(min=0)) @ vectors.T


class NodeJacobians:
    """Jacobians of model nodes w.r.t. the flat unconstrained latents of ``layout`` (at the current values)."""

    def __init__(self, model, layout):
        self.model = model
        self.layout = layout
        self.latent = set(layout.names)
        self._ancestors = {}
        self._jacobians = {}

    # ------------------------------------------------------------------ graph
    def ancestors(self, node):
        if node not in self._ancestors:
            self._ancestors[node] = nx.ancestors(self.model.graph, node) | {node}
        return self._ancestors[node]

    def depends_on_latents(self, node):
        return bool(self.ancestors(node) & self.latent)

    def node_value(self, node):
        model = self.model
        parameter = model.params[node] if node in model.params else model.deterministic_params[node]
        return torch.as_tensor(parameter.constrained_value)

    def jacobian(self, node):
        """``d node / d theta`` as ``(node.numel(), D)``: one seeded reverse pass per element of the node."""
        if node not in self._jacobians:
            model, layout = self.model, self.layout
            value = self.node_value(node)
            m = value.numel()
            nodes = self.ancestors(node)
            rows = []
            with self._frozen(nodes):
                for i in range(m):
                    seed = torch.zeros(m, dtype=value.dtype, device=value.device)
                    seed[i] = 1
                    grads = model._reverse_pass(layout.names, nodes=nodes, active_terms=(), seeds={node: seed.reshape(value.shape)})
                    rows.append(layout.pack(grads))
            self._jacobians[node] = torch.stack(rows) if rows else torch.zeros(0, layout.dim, dtype=layout.dtype, device=value.device)
        return self._jacobians[node]

    @contextmanager
    def _frozen(self, nodes):
        """Compute the deterministic nodes among ``nodes`` once and reuse the values (they are recomputed on every access otherwise)."""
        model = self.model
        frozen = []
        try:
            for name in model._topo_order:
                if name in nodes and name in model.deterministic_params and model.deterministic_params[name]._value_override is None:
                    parameter = model.deterministic_params[name]
                    parameter._value_override = torch.as_tensor(parameter.constrained_value)
                    frozen.append(parameter)
            yield
        finally:
            for parameter in frozen:
                parameter._value_override = None

class GaussNewtonCurvature(NodeJacobians):
    r"""Generalised Gauss-Newton curvature :math:`-H_{GGN}` of the unconstrained log joint at the current values.

    The log joint is a sum of log-density terms :math:`\log p(v \mid \eta(\theta))`
    (the likelihood terms and the prior terms of the latents, the latter with
    the :math:`\log|\det J|` of their transforms), where :math:`\eta` are the
    arguments of the distribution and :math:`\theta` the flat unconstrained
    latents of ``layout``. The curvature is assembled term by term:

    * An observed term whose distribution has a parameter curvature
      (:meth:`~BayesianDLL.Distributions.Distribution.log_pdf_param_curvature`,
      the Fisher information, positive semi-definite) contributes
      :math:`J^\top F J` with :math:`J = \partial\eta/\partial\theta` taken from the
      hand-written VJPs (one seeded reverse pass per element of an argument node).
    * The prior terms contribute their exact negative Hessian, obtained by
      central differences of the hand-written prior gradient (only for the
      latents that are coupled to other terms through distribution
      arguments; independent priors are differentiated one by one). If it
      is not positive semi-definite (e.g. a latent scale of a hierarchical
      prior) it is projected onto the positive semi-definite matrices.
    * Terms without a parameter curvature and the summed-out terms of
      enumerated latents contribute the positive semi-definite part of their
      exact negative Hessian by finite differences of the gradient.

    ``matrix`` is the sum, a positive semi-definite ``(D, D)`` matrix; it
    equals the exact negative Hessian when every observed term is Gaussian
    with a fixed scale and arguments linear in the latents.
    """

    def __init__(self, model, layout, fd_step=None):
        super().__init__(model, layout)
        self.fd_step = fd_step
        self.fallback_terms = []
        self.matrix = self._assemble()

    # ------------------------------------------------------------ assembling
    def _dependencies(self, distribution):
        model = self.model
        return [n for n in distribution.parameters
                if (n in model.params or n in model.deterministic_params) and self.depends_on_latents(n)]

    def _assemble(self):
        model, layout = self.model, self.layout
        flat = layout.current(model)
        grouped = model._enumeration_group_of
        total = torch.zeros(layout.dim, layout.dim, dtype=layout.dtype, device=flat.device)
        for name, observed in model.observed_params.items():
            if name in grouped:
                continue
            dependencies = self._dependencies(observed.distribution)
            if not dependencies:
                continue
            try:
                total += self._observed_term(observed, dependencies)
            except NotImplementedError:
                self.fallback_terms.append(name)
                total += self._negative_hessian(lambda x, name=name: self._gradient(x, {name}.__contains__), flat)
        total += self._prior_curvature(flat)
        for group in model._enumeration_groups:
            total += self._negative_hessian(lambda x, group=group: self._group_gradient(x, group), flat)
        return 0.5 * (total + total.T)

    def _observed_term(self, observed, dependencies):
        distribution, values = observed.distribution, observed.observed_values
        curvature = distribution.log_pdf_param_curvature(values)
        used = {name for pair in curvature for name in pair}
        if any(name not in used for name in dependencies):
            raise NotImplementedError("a latent argument has no curvature")
        events = {getattr(arg, "name", None): ndim for arg, ndim in _distribution_arguments(distribution)}
        batch = tuple(distribution.log_pdf(values).shape)
        dim = self.layout.dim
        total = torch.zeros(dim, dim, dtype=self.layout.dtype, device=self.layout.device)
        size = 1
        for extent in batch:
            size *= extent

        def expanded(node):
            shape = tuple(self.node_value(node).shape)
            event = shape[len(shape) - events.get(node, 0):]
            jac = self.jacobian(node).reshape(*shape, dim)
            jac = torch.broadcast_to(jac, (*batch, *event, dim))
            return jac.reshape(size, -1, dim), event

        for (a, b), block in curvature.items():
            if a not in dependencies or b not in dependencies:
                continue
            jac_a, event_a = expanded(a)
            jac_b, event_b = (jac_a, event_a) if a == b else expanded(b)
            block = torch.broadcast_to(block, (*batch, *event_a, *event_b)).reshape(size, jac_a.shape[1], jac_b.shape[1])
            weighted = torch.einsum("nab,nbd->nad", block, jac_b)
            part = jac_a.reshape(-1, dim).T @ weighted.reshape(-1, dim)
            total += part if a == b else part + part.T
        return total

    # ------------------------------------------------------ exact fall-backs
    def _gradient(self, flat, active):
        model, layout = self.model, self.layout
        with model.temporarily_set_many(layout.unpack(flat)):
            return layout.pack(model._reverse_pass(layout.names, active_terms=active))

    def _group_gradient(self, flat, group):
        model, layout = self.model, self.layout
        with model.temporarily_set_many(layout.unpack(flat)):
            grads = group.grad(model, layout.names)
            return layout.pack({name: grads[name] for name in layout.names})

    def _negative_hessian(self, gradient, flat, columns=None):
        return project_psd(-finite_difference_columns(gradient, flat, self.fd_step, columns))

    def _prior_curvature(self, flat):
        """Exact negative Hessian of the prior terms (with the transforms' log-determinants), made positive semi-definite."""
        model, layout = self.model, self.layout
        grouped = model._enumeration_group_of
        priors = {name for name in model.params if name not in grouped}
        coupled = set()
        for name in priors:
            dependencies = self._dependencies(model.params[name].distribution)
            for node in dependencies:
                coupled |= self.ancestors(node) & self.latent
            if dependencies and name in self.latent:
                coupled.add(name)
        total = torch.zeros(layout.dim, layout.dim, dtype=layout.dtype, device=flat.device)
        if coupled:
            columns = [j for name in layout.names if name in coupled for j in range(*layout.slices[name])]
            total += self._negative_hessian(lambda x: self._gradient(x, priors.__contains__), flat, columns)
        for name in layout.names:
            if name in coupled or name not in priors:
                continue
            start, end = layout.slices[name]
            distribution, shape = model.params[name].distribution, layout.shapes[name]

            def own_gradient(x, distribution=distribution, shape=shape):
                return distribution._log_prob_grad_unconstrained(x.reshape(shape)).reshape(-1)

            total[start:end, start:end] += self._negative_hessian(own_gradient, flat[start:end])
        return total
