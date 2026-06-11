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
