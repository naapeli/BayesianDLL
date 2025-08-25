import torch


def gelman_rubin(trace):
    r_hats = {}
    for name, samples in trace.items():
        if samples.ndim != 3:
            raise NotImplementedError()
        
        n_chains, n_samples, _ = samples.shape
        chain_means = samples.mean(dim=1)
        grand_mean = chain_means.mean(dim=0)
        B = n_samples / (n_chains - 1) * ((chain_means - grand_mean) ** 2).sum(dim=0)
        chain_vars = samples.var(dim=1, unbiased=True)
        W = chain_vars.mean(dim=0)
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        r_hat = torch.sqrt(var_hat / W)

        r_hats[name] = r_hat
    return r_hats
