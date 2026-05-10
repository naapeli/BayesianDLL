class SamplingResult:
    def __init__(self, trace, divergences, acceptance_probabilities, step_sizes):
        self.trace = trace
        self.divergences = divergences
        self.acceptance_probabilities = acceptance_probabilities
        self.step_sizes = step_sizes

    def __getitem__(self, key):
        return self.trace[key]

    def __contains__(self, key):
        return key in self.trace

    def keys(self):
        return self.trace.keys()

    def values(self):
        return self.trace.values()

    def items(self):
        return self.trace.items()

    def __repr__(self):
        return f"SamplingResult(params={list(self.trace.keys())}, divergences={self.divergences})"
