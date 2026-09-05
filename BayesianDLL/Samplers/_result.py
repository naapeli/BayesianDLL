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


class SamplingResult:
    def __init__(self, trace, divergences, acceptance_probabilities, step_sizes, deterministic_trace=None):
        self.trace = trace
        self.divergences = divergences
        self.acceptance_probabilities = acceptance_probabilities
        self.step_sizes = step_sizes
        self.deterministic_trace = {} if deterministic_trace is None else deterministic_trace

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
        )

    def summary(self, hdi_prob=0.94, ci_kind="eti", round_to=3, include_deterministic=False):
        from ..Evaluation._summary import summary as _summary
        return _summary(self, hdi_prob=hdi_prob, ci_kind=ci_kind, round_to=round_to, include_deterministic=include_deterministic)

    def __repr__(self):
        params = list(self.trace.keys()) + list(self.deterministic_trace.keys())
        return f"SamplingResult(params={params}, divergences={self.divergences})"


class PredicativeResult:
    def __init__(self, samples: dict[str, torch.Tensor]):
        self.samples = samples

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
        return PredicativeResult(thinned)

    def __repr__(self):
        return f"PredicativeResult(samples={list(self.samples.keys())})"


def thin(result, step: int | slice = 1, **kwargs):
    if hasattr(result, "thin") and callable(result.thin):
        return result.thin(step, **kwargs)
    raise TypeError(f"Object of type '{type(result).__name__}' does not support thinning.")
