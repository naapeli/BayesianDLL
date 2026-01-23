import torch


def _auto_correlation(chains, lag):
    n_chains, n_samples, n_features = chains.shape
    V_t = 1 / (n_chains * (n_samples - lag)) * torch.sum(torch.sum((chains[:, lag:] - chains[:, :n_samples - lag]) ** 2, dim=1, keepdim=True), dim=0, keepdim=True).squeeze()
    chain_means = chains.mean(dim=1)
    grand_mean = chain_means.mean(dim=0)
    B = n_samples / (n_chains - 1) * ((chain_means - grand_mean) ** 2).sum(dim=0)
    chain_vars = chains.var(dim=1, unbiased=True)
    W = chain_vars.mean(dim=0)
    var = (n_samples - 1) / n_samples * W + B / n_samples
    rho_t = 1 - V_t / (2 * var)
    return rho_t

def effective_sample_size(samples):
    results = {}
    for name, chains in samples.items():
        if chains.ndim != 3:
            raise NotImplementedError("Each parameter tensor must be (n_chains, n_samples, n_features)")
        
        n_chains, n_samples, n_features = chains.shape
        tau = torch.ones(n_features)
        rho_t_prev = None
        for k in range(1, n_samples):
            rho_k = _auto_correlation(chains, k)
            if (rho_t_prev is not None) and torch.all(rho_t_prev + rho_k < 0):
                tau -= 2 * rho_t_prev
                break
            rho_t_prev = rho_k
            tau += 2 * rho_k
        results[name] = n_chains * n_samples / tau
    return results
