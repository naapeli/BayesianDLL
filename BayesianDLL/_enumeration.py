"""Exact marginalisation of finite-support discrete latent variables.

A :class:`RandomParameter` created with ``enumerate=True`` (or moved with
:func:`BayesianDLL.marginalize`) is summed out of the joint density. The
continuous parameters are then sampled from the marginal posterior and the
discrete values are drawn exactly from their conditional posterior afterwards.

Every enumerated parameter ``z`` defines a *factor group*: the prior of ``z``
and every term of its Markov blanket that depends on ``z`` (random and
observed children, reached through deterministic nodes). Enumerated
parameters whose groups share a term are merged into one group.

A group is evaluated in one of two ways.

``"factorised"`` (one enumerated parameter): the group's terms are evaluated
once per support value ``k`` with every coordinate of ``z`` set to ``k``. The
elementwise log densities of each term are summed onto the shape of ``z`` by
broadcasting (``l_k[i]``: everything that coordinate ``i`` affects), and the
marginal is ``sum_i logsumexp_k l_k[i]``. This costs ``K`` evaluations instead
of ``K ** n``. It is exact when element ``e`` of every term only depends on the
coordinate of ``z`` that broadcasts onto ``e`` -- e.g. ``z`` and the
observations share a plate and deterministic nodes act elementwise along it.
This is checked once, numerically: the terms are re-evaluated at random mixed
assignments of ``z`` and compared with the values predicted by the
factorisation.

``"brute_force"`` (several interacting enumerated parameters, or a group that
does not factorise): the sum runs over all joint assignments of all elements,
which is only allowed for at most ``MAX_BRUTE_FORCE_CONFIGURATIONS``
assignments.

Gradients use ``d/dtheta log sum_k exp(l_k) = sum_k w_k dl_k/dtheta`` with
``w = softmax_k(l_k)``: the model's reverse pass is run once per support value
(or assignment) with the group's term cotangents weighted elementwise by
``w_k``.
"""

import itertools

import torch


MAX_BRUTE_FORCE_CONFIGURATIONS = 4096
_PROBE_SEED = 20261116
_N_PROBES = 2


def _sum_to_shape(tensor, shape, batched=False):
    if batched:
        from ._batching import sum_to_batched
        return sum_to_batched(tensor, shape)
    from ._model import _sum_to_shape as sum_to_shape
    return sum_to_shape(tensor, shape)


def _batched(model):
    return getattr(model, "n_chains", None) is not None


def _total(model, value):
    """Sum to a scalar, or to one value per chain for a batched model."""
    if _batched(model) and value.ndim > 0:
        return value.reshape(value.shape[0], -1).sum(-1)
    return value.sum()


def _align_weight(model, weight, shape):
    """Broadcast per-coordinate weights onto an elementwise term shape."""
    if _batched(model):
        from ._batching import pad_to
        weight = pad_to(weight, len(shape))
    return torch.broadcast_to(weight, shape)


def _support(parameter):
    value = parameter.constrained_value
    values = parameter.distribution.state_space.values
    return values.to(dtype=value.dtype, device=value.device)


def weight_gradient(gradient, weight):
    """Multiply an elementwise gradient by per-element weights.

    ``weight`` has the shape of the term's elementwise log density. Gradients
    with extra trailing (event) dimensions are weighted per batch element.
    """
    if weight is None:
        return gradient
    if not isinstance(gradient, torch.Tensor):
        gradient = torch.as_tensor(gradient, dtype=weight.dtype, device=weight.device)
    if weight.ndim == 0 or gradient.ndim <= weight.ndim:
        return gradient * weight
    return gradient * weight.reshape(*weight.shape, *([1] * (gradient.ndim - weight.ndim)))


def _elementwise_log_prob(model, name):
    """Elementwise log density of the term ``name`` (without Jacobian terms)."""
    if hasattr(model, "_elementwise_term"):
        return model._elementwise_term(name)
    if name in model.observed_params:
        observed = model.observed_params[name]
        return observed.distribution.log_pdf(observed.observed_values)
    parameter = model.params[name] if name in model.params else model.enumerated_params[name]
    return parameter.distribution.log_pdf(parameter.constrained_value)


class EnumerationGroup:
    """Enumerated parameters together with the log density terms they affect."""

    def __init__(self, enumerated_names, random_terms, observed_terms):
        self.enumerated_names = list(enumerated_names)
        # Non-enumerated random parameters whose priors depend on the group.
        self.random_terms = list(random_terms)
        self.observed_terms = list(observed_terms)
        self.mode = None
        self._signature = None

    @property
    def terms(self):
        return self.enumerated_names + self.random_terms + self.observed_terms

    def __repr__(self):
        return f"EnumerationGroup(enumerated={self.enumerated_names}, terms={self.terms}, mode={self.mode})"

    # ------------------------------------------------------------ helpers
    def _parameters(self, model):
        return [model.enumerated_params[name] for name in self.enumerated_names]

    def _saved_values(self, model):
        return [p.constrained_value for p in self._parameters(model)]

    def _restore(self, model, values):
        for parameter, value in zip(self._parameters(model), values):
            if _batched(model) and value.ndim < parameter.constrained_value.ndim:
                value = value.unsqueeze(0).expand(model.n_chains, *value.shape)
            parameter.set_constrained_value(value)

    @staticmethod
    def _value_shape(model, parameter):
        """Shape of the parameter's value without the chain dimension."""
        shape = parameter.constrained_value.shape
        return shape[1:] if _batched(model) else shape

    def _term_values(self, model):
        return [(name, _elementwise_log_prob(model, name)) for name in self.terms]

    def _constant(self, model):
        """Jacobian terms of the continuous random children (independent of the enumerated values)."""
        from ._batching import batched_evaluation
        constant = 0.0
        with batched_evaluation(getattr(model, "n_chains", None)):
            for name in self.random_terms:
                parameter = model.params[name]
                constant = constant + parameter.distribution.transform.log_abs_det_jacobian(parameter.unconstrained_value)
        return constant

    def _shape_signature(self, model):
        return tuple(tuple(self._value_shape(model, p)) for p in self._parameters(model))

    # ---------------------------------------------------------- detection
    def prepare(self, model):
        """Decide (once per shape signature) how this group is summed out."""
        signature = self._shape_signature(model)
        if self.mode is not None and signature == self._signature:
            return
        self.mode = self._detect(model)
        self._signature = signature

    def _detect(self, model):
        reason = None
        if len(self.enumerated_names) == 1:
            reason = self._factorisation_failure(model)
            if reason is None:
                return "factorised"
        else:
            reason = f"the enumerated parameters {self.enumerated_names} share log density terms"

        n_configurations = self._n_configurations(model)
        if n_configurations <= MAX_BRUTE_FORCE_CONFIGURATIONS:
            return "brute_force"
        raise ValueError(
            f"Cannot sum out {self.enumerated_names}: {reason}, so the sum does not factorise "
            f"over their coordinates, and summing over all {n_configurations} joint assignments "
            f"exceeds the limit of {MAX_BRUTE_FORCE_CONFIGURATIONS}. Enumerate a single parameter "
            "whose coordinates only affect the matching (broadcast) elements of its children, e.g. "
            "a latent and its observations in the same plate, or sample the parameter with Gibbs instead."
        )

    def _n_configurations(self, model):
        total = 1
        for parameter in self._parameters(model):
            total *= len(parameter.distribution.state_space) ** self._value_shape(model, parameter).numel()
            if total > MAX_BRUTE_FORCE_CONFIGURATIONS:
                return total
        return total

    def _factorisation_failure(self, model):
        """Return why the group does not factorise over coordinates, or ``None``."""
        (name,) = self.enumerated_names
        parameter = model.enumerated_params[name]
        saved = self._saved_values(model)
        shape = saved[0].shape
        if saved[0].numel() == 1:
            return None
        support = _support(parameter)
        try:
            stacks = None
            for k in support:
                parameter.set_constrained_value(torch.full_like(saved[0], k))
                values = self._term_values(model)
                if stacks is None:
                    stacks = [[value] for _, value in values]
                else:
                    for stack, (_, value) in zip(stacks, values):
                        stack.append(value)
            for (term, _), stack in zip(self._term_values(model), stacks):
                term_shape = stack[0].shape
                try:
                    broadcast = torch.broadcast_shapes(shape, term_shape)
                except RuntimeError:
                    broadcast = None
                if broadcast != term_shape:
                    return (
                        f"the elementwise log density of '{term}' has shape {tuple(term_shape)}, "
                        f"onto which '{name}' of shape {tuple(shape)} does not broadcast"
                    )
            stacks = [torch.stack(stack, dim=0) for stack in stacks]

            generator = torch.Generator().manual_seed(_PROBE_SEED)
            for _ in range(_N_PROBES):
                index = torch.randint(0, len(support), tuple(shape), generator=generator).to(device=support.device)
                parameter.set_constrained_value(support[index].to(dtype=saved[0].dtype))
                for (term, value), stack in zip(self._term_values(model), stacks):
                    gather_index = torch.broadcast_to(index, value.shape).unsqueeze(0)
                    expected = torch.gather(stack, 0, gather_index).squeeze(0)
                    if not bool(torch.isclose(value, expected, rtol=1e-5, atol=1e-8, equal_nan=True).all()):
                        return (
                            f"element i of '{term}' depends on other coordinates of '{name}' than "
                            "the one broadcast onto it"
                        )
        finally:
            self._restore(model, saved)
        return None

    # --------------------------------------------------------- evaluation
    def _configurations(self, model):
        """All joint assignments (brute force) as lists of tensors, one per enumerated parameter."""
        parameters = self._parameters(model)
        supports = [_support(p) for p in parameters]
        shapes = [self._value_shape(model, p) for p in parameters]
        sizes = [shape.numel() for shape in shapes]
        per_element = [range(len(s)) for s, n in zip(supports, sizes) for _ in range(n)]
        configurations = []
        for indices in itertools.product(*per_element):
            offset = 0
            values = []
            for support, shape, n in zip(supports, shapes, sizes):
                index = torch.tensor(indices[offset:offset + n], dtype=torch.long, device=support.device)
                values.append(support[index].reshape(shape))
                offset += n
            configurations.append(values)
        return configurations

    def evaluate(self, model):
        """Per-assignment log densities of the group at the current parameter values.

        Returns a dict with ``"log_terms"``: for the factorised mode a tensor
        of shape ``(K, *z_shape)`` for the ``K`` support values, for brute
        force a tensor of shape ``(n_configurations,)``, in a batched model
        with the chain dimension after the first one (``z_shape`` then
        includes it); ``"configurations"``: the assignments (support values
        or lists of tensors); ``"term_shapes"``: elementwise term shapes; and
        ``"log_marginal"``: ``sum logsumexp`` of the log terms over the
        assignments plus the log-Jacobians of the group's continuous random
        children (a scalar, or one value per chain).
        """
        self.prepare(model)
        saved = self._saved_values(model)
        try:
            log_terms = []
            term_shapes = None
            if self.mode == "factorised":
                shape = saved[0].shape
                batched = _batched(model)
                configurations = list(_support(self._parameters(model)[0]))
                for k in configurations:
                    self._parameters(model)[0].set_constrained_value(torch.full_like(saved[0], k))
                    values = self._term_values(model)
                    term_shapes = {term: value.shape for term, value in values}
                    total = 0.0
                    for _, value in values:
                        total = total + _sum_to_shape(value, shape, batched)
                    log_terms.append(torch.broadcast_to(torch.as_tensor(total, dtype=saved[0].dtype, device=saved[0].device), shape))
            else:
                configurations = self._configurations(model)
                for values in configurations:
                    self._restore(model, values)
                    terms = self._term_values(model)
                    term_shapes = {term: value.shape for term, value in terms}
                    log_terms.append(sum(_total(model, value) for _, value in terms))
            log_terms = torch.stack(log_terms, dim=0)
        finally:
            self._restore(model, saved)
        log_marginal = _total(model, torch.logsumexp(log_terms, dim=0)) + self._constant(model)
        return {
            "log_terms": log_terms,
            "configurations": configurations,
            "term_shapes": term_shapes,
            "log_marginal": log_marginal,
        }

    def log_marginal(self, model):
        return self.evaluate(model)["log_marginal"]

    @staticmethod
    def _weights(log_terms):
        return torch.exp(log_terms - torch.logsumexp(log_terms, dim=0, keepdim=True))

    def grad(self, model, param_names, nodes=None):
        """Gradient of the group's log marginal w.r.t. the unconstrained ``param_names``."""
        evaluation = self.evaluate(model)
        weights = self._weights(evaluation["log_terms"])
        term_shapes = evaluation["term_shapes"]
        active = set(self.terms)
        saved = self._saved_values(model)
        result = {}
        try:
            for index, configuration in enumerate(evaluation["configurations"]):
                weight = weights[index]
                if self.mode == "factorised":
                    self._parameters(model)[0].set_constrained_value(torch.full_like(saved[0], configuration))
                    if weight.numel() == 1:
                        term_weights = {term: weight.reshape(()) for term in active}
                    else:
                        term_weights = {term: _align_weight(model, weight, term_shapes[term]) for term in active}
                else:
                    if bool((weight == 0.0).all()):
                        continue
                    self._restore(model, configuration)
                    term_weights = {term: weight for term in active}
                grads = model._reverse_pass(param_names, nodes=nodes, active_terms=active, term_weights=term_weights)
                for name, value in grads.items():
                    result[name] = result[name] + value if name in result else value
        finally:
            self._restore(model, saved)
        return result

    def sample(self, model):
        """Draw the enumerated parameters from their conditional posterior.

        The drawn values are set on the parameters. Returns
        ``{name: (value, probabilities)}`` where ``probabilities`` has shape
        ``(*value_shape, K)`` and holds the marginal posterior probability of
        every support value per coordinate.
        """
        evaluation = self.evaluate(model)
        weights = self._weights(evaluation["log_terms"])
        parameters = self._parameters(model)
        if self.mode == "factorised":
            parameter = parameters[0]
            support = _support(parameter)
            probabilities = torch.movedim(weights, 0, -1)
            uniform = torch.rand(probabilities.shape[:-1], dtype=probabilities.dtype, device=probabilities.device)
            index = (torch.cumsum(probabilities, dim=-1) < uniform.unsqueeze(-1)).sum(dim=-1).clamp(max=len(support) - 1)
            value = support[index].to(dtype=parameter.constrained_value.dtype)
            parameter.set_constrained_value(value)
            return {parameter.name: (value, probabilities)}

        if _batched(model):
            return self._sample_brute_force_batched(model, evaluation, weights, parameters)
        uniform = torch.rand((), dtype=weights.dtype, device=weights.device)
        index = int((torch.cumsum(weights, dim=0) < uniform).sum().clamp(max=len(weights) - 1).item())
        configuration = evaluation["configurations"][index]
        self._restore(model, configuration)
        draws = {}
        for position, parameter in enumerate(parameters):
            support = _support(parameter)
            stacked = torch.stack([c[position] for c in evaluation["configurations"]], dim=0)  # (C, *shape)
            one_hot = (stacked.unsqueeze(-1) == support).to(dtype=weights.dtype)
            probabilities = (weights.reshape(-1, *([1] * (one_hot.ndim - 1))) * one_hot).sum(dim=0)
            draws[parameter.name] = (configuration[position].to(dtype=parameter.constrained_value.dtype), probabilities)
        return draws


def _sample_brute_force_batched(self, model, evaluation, weights, parameters):
    """Per-chain draws of all joint assignments (``weights`` has shape ``(n_configurations, n_chains)``)."""
    configurations = evaluation["configurations"]
    uniform = torch.rand(weights.shape[1:], dtype=weights.dtype, device=weights.device)
    index = (torch.cumsum(weights, dim=0) < uniform).sum(dim=0).clamp(max=len(configurations) - 1)
    draws = {}
    for position, parameter in enumerate(parameters):
        support = _support(parameter)
        stacked = torch.stack([c[position] for c in configurations], dim=0)  # (K, *shape)
        value = stacked[index].to(dtype=parameter.constrained_value.dtype)  # (n_chains, *shape)
        one_hot = (stacked.unsqueeze(-1) == support).to(dtype=weights.dtype)  # (K, *shape, S)
        probabilities = torch.einsum("kc,k...->c...", weights, one_hot)
        parameter.set_constrained_value(value)
        draws[parameter.name] = (value, probabilities)
    return draws


EnumerationGroup._sample_brute_force_batched = _sample_brute_force_batched


def build_groups(model):
    """Group the enumerated parameters of ``model`` by shared log density terms."""
    names = list(model.enumerated_params)
    if not names:
        return []
    blankets = {name: model._markov_blanket([name]) for name in names}
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    owner = {}
    for name in names:
        random_terms, observed_terms, _ = blankets[name]
        for term in random_terms + observed_terms:
            if term in owner:
                parent[find(owner[term])] = find(name)
            else:
                owner[term] = name

    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)

    result = []
    for members in groups.values():
        random_terms, observed_terms = [], []
        for member in members:
            member_random, member_observed, _ = blankets[member]
            for term in member_random:
                if term not in model.enumerated_params and term not in random_terms:
                    random_terms.append(term)
            for term in member_observed:
                if term not in observed_terms:
                    observed_terms.append(term)
        result.append(EnumerationGroup(members, random_terms, observed_terms))
    return result
