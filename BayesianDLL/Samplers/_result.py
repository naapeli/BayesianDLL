import torch


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

    def __repr__(self):
        return f"PredicativeResult(samples={list(self.samples.keys())})"
