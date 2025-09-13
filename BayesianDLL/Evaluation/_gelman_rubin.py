import torch
from scipy.stats import rankdata, norm


# def gelman_rubin(trace):
#     r_hats = {}
#     for name, samples in trace.items():
#         if samples.ndim != 3:
#             raise NotImplementedError()
        
#         n_chains, n_samples, _ = samples.shape
#         chain_means = samples.mean(dim=1)
#         grand_mean = chain_means.mean(dim=0)
#         B = n_samples / (n_chains - 1) * ((chain_means - grand_mean) ** 2).sum(dim=0)
#         chain_vars = samples.var(dim=1, unbiased=True)
#         W = chain_vars.mean(dim=0)
#         var_hat = (n_samples - 1) / n_samples * W + B / n_samples
#         r_hat = torch.sqrt(var_hat / W)

#         r_hats[name] = r_hat
#     return r_hats


def _rhat_core(chains):
    n_chains, n_samples = chains.shape
    chain_means = chains.mean(dim=1)
    grand_mean = chain_means.mean(dim=0)
    B = n_samples / (n_chains - 1) * ((chain_means - grand_mean) ** 2).sum(dim=0)
    chain_vars = chains.var(dim=1, unbiased=True)
    W = chain_vars.mean(dim=0)
    Var_hat = (n_samples - 1) / n_samples * W + B / n_samples
    return torch.clamp(torch.sqrt(Var_hat / W), min=1)


def _split_rhat_core(chains):
    n_chains, n_samples = chains.shape
    if n_samples % 2 != 0:
        chains = chains[:, :-1]
        n_samples -= 1
    split_chains = chains.reshape(n_chains * 2, n_samples // 2)
    return _rhat_core(split_chains)


def _rank_normalize(chains):
    flat = chains.flatten().cpu().numpy()
    ranks = rankdata(flat, method="average")
    u = ranks / (len(flat) + 1.0)  # TODO: look at aki vehtari talk "some of my favourite evaluation diagnostics" => (rank - 3/8) / (S - 1/4)
    z = norm.ppf(u)
    return torch.tensor(z, dtype=chains.dtype, device=chains.device).reshape(chains.shape)


def _rank_normalized_rhat_core(chains):
    norm_chains = _rank_normalize(chains)
    return _split_rhat_core(norm_chains)



def gelman_rubin(samples, method="rank"):
    results = {}
    for name, chains in samples.items():
        if chains.ndim != 3:
            raise NotImplementedError("Each parameter tensor must be (n_chains, n_samples, n_features)")
        
        _, _, n_features = chains.shape
        rhat_vals = torch.zeros(n_features, dtype=chains.dtype, device=chains.device)
        for f in range(n_features):
            feature_chains = chains[:, :, f]
            if method == "classical":
                rhat_vals[f] = _rhat_core(feature_chains)
            elif method == "split":
                rhat_vals[f] = _split_rhat_core(feature_chains)
            elif method == "rank":
                rhat_vals[f] = _rank_normalized_rhat_core(feature_chains)
            else:
                raise ValueError("method must be 'classical', 'split', or 'rank'")
        results[name] = rhat_vals
    return results
