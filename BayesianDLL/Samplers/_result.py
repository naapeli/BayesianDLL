import torch


def _normalize_slice(s):
    if isinstance(s, bool):
        raise TypeError("Thinning step must be an integer or slice, got bool.")
    if isinstance(s, int):
        if s < 1:
            raise ValueError(f"Thinning step must be an integer >= 1, got {s}.")
        return slice(None, None, s)
    elif isinstance(s, slice):
        return s
    else:
        raise TypeError(f"Expected int or slice for thinning, got {type(s).__name__}.")


class SampleStats:
    """Per-iteration sampler statistics of the post-warmup draws, one entry per sampling block.

    Each block entry is a dict with the block's ``"params"`` (parameter
    names), its ``"sampler"`` (``"nuts"``, ``"metropolis"`` or ``"gibbs"``)
    and ``"stats"``, a dict mapping a statistic name to a tensor of shape
    ``(n_chains, n_samples)``. NUTS records ``step_size``, ``accept_stat``,
    ``tree_depth``, ``n_leapfrog``, ``diverging``, ``energy`` and ``lp``;
    Metropolis records ``accept_stat``, ``accepted``, ``step_size`` (the
    proposal scale) and ``lp``; the finite Gibbs sampler ``accept_stat``,
    ``step_size`` and ``lp``.

    ``stats[i]`` returns the statistics of block ``i`` and ``stats["energy"]``
    the statistic of the only block that records it.
    """

    def __init__(self, blocks=None):
        self.blocks = [] if blocks is None else list(blocks)

    @property
    def params(self):
        return [block["params"] for block in self.blocks]

    @property
    def samplers(self):
        return [block["sampler"] for block in self.blocks]

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return (block["stats"] for block in self.blocks)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.blocks[key]["stats"]
        matches = [block["stats"][key] for block in self.blocks if key in block["stats"]]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"No sampling block records the statistic '{key}'.")
        raise KeyError(f"Several sampling blocks record '{key}'; index the block first, e.g. sample_stats[0]['{key}'].")

    def __contains__(self, key):
        return any(key in block["stats"] for block in self.blocks)

    def thin(self, step=1):
        s = _normalize_slice(step)
        return SampleStats([
            {**block, "stats": {name: value[:, s] for name, value in block["stats"].items()}}
            for block in self.blocks
        ])

    def __repr__(self):
        summary = [(block["sampler"], block["params"], list(block["stats"])) for block in self.blocks]
        return f"SampleStats({summary})"


class SamplingResult:
    def __init__(self, trace, divergences, acceptance_probabilities, step_sizes, deterministic_trace=None, sample_stats=None):
        self.trace = trace
        self.divergences = divergences
        self.acceptance_probabilities = acceptance_probabilities
        self.step_sizes = step_sizes
        self.deterministic_trace = {} if deterministic_trace is None else deterministic_trace
        self.sample_stats = SampleStats() if sample_stats is None else sample_stats

    def __getitem__(self, key):
        if key in self.trace:
            return self.trace[key]
        return self.deterministic_trace[key]

    def __contains__(self, key):
        return key in self.trace or key in self.deterministic_trace

    def keys(self):
        return {**self.trace, **self.deterministic_trace}.keys()

    def values(self):
        return {**self.trace, **self.deterministic_trace}.values()

    def items(self):
        return {**self.trace, **self.deterministic_trace}.items()

    def thin(self, step: int | slice = 1) -> "SamplingResult":
        s = _normalize_slice(step)
        thinned_trace = {name: val[:, s] for name, val in self.trace.items()}
        thinned_det = {name: val[:, s] for name, val in self.deterministic_trace.items()}
        return SamplingResult(
            trace=thinned_trace,
            divergences=list(self.divergences),
            acceptance_probabilities=[list(ap) for ap in self.acceptance_probabilities],
            step_sizes=[list(ss) for ss in self.step_sizes],
            deterministic_trace=thinned_det,
            sample_stats=self.sample_stats.thin(step),
        )

    def summary(self, hdi_prob=0.94, ci_kind="eti", round_to=3, include_deterministic=False):
        from ..Evaluation._summary import summary as _summary
        return _summary(self, hdi_prob=hdi_prob, ci_kind=ci_kind, round_to=round_to, include_deterministic=include_deterministic)

    def __repr__(self):
        params = list(self.trace.keys()) + list(self.deterministic_trace.keys())
        return f"SamplingResult(params={params}, divergences={self.divergences})"


class PredicativeResult:
    """Predictive draws of shape ``(n_samples, samples_per_step, *predictive_shape)`` per observed node.

    ``prior`` holds the prior draws of the model parameters (a
    :class:`SamplingResult` with one chain) for results of
    ``sample_prior_predicative``, and is ``None`` otherwise.
    """

    def __init__(self, samples: dict[str, torch.Tensor], prior=None):
        self.samples = samples
        self.prior = prior

    def __getitem__(self, key):
        return self.samples[key]

    def __contains__(self, key):
        return key in self.samples

    def keys(self):
        return self.samples.keys()

    def values(self):
        return self.samples.values()

    def items(self):
        return self.samples.items()

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def thin(self, step: int | slice = 1, dim: int | tuple[int, ...] = 1) -> "PredicativeResult":
        s = _normalize_slice(step)
        if dim == 1:
            thinned = {name: val[:, s] for name, val in self.samples.items()}
        elif dim == 0:
            thinned = {name: val[s] for name, val in self.samples.items()}
        elif dim in ((0, 1), (1, 0)):
            thinned = {name: val[s, s] for name, val in self.samples.items()}
        else:
            raise ValueError(f"dim must be 0, 1, or (0, 1), got {dim}.")
        prior = self.prior
        if prior is not None and dim != 1:
            prior = prior.thin(step)
        return PredicativeResult(thinned, prior=prior)

    def __repr__(self):
        return f"PredicativeResult(samples={list(self.samples.keys())})"


def thin(result, step: int | slice = 1, **kwargs):
    if hasattr(result, "thin") and callable(result.thin):
        return result.thin(step, **kwargs)
    raise TypeError(f"Object of type '{type(result).__name__}' does not support thinning.")
