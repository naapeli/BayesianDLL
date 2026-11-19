import torch


def _energy_from(result, block):
    if isinstance(result, torch.Tensor):
        return result
    stats = getattr(result, "sample_stats", result)
    if block is not None:
        return stats[block]["energy"]
    energy_blocks = [index for index, block_stats in enumerate(stats) if "energy" in block_stats]
    if not energy_blocks:
        raise ValueError("The result has no energy statistics; E-BFMI needs draws from a Hamiltonian sampler (NUTS).")
    if len(energy_blocks) > 1:
        raise ValueError(f"Several sampling blocks record energies ({energy_blocks}); choose one with block=<index>.")
    return stats[energy_blocks[0]]["energy"]


def bfmi(result, block=None):
    """Estimated Bayesian fraction of missing information (E-BFMI) of each chain.

    Following Betancourt (2016), with ``E_n`` the Hamiltonian at draw ``n``::

        E-BFMI = mean_{n>=2} (E_n - E_{n-1})^2 / mean_n (E_n - mean(E))^2

    (the numerator averages the ``N - 1`` differences, the denominator is the
    biased variance over the ``N`` draws). Values below about 0.3
    indicate that the momentum resampling explores the energy distribution
    poorly, which usually points to heavy tails or a poor parametrisation.

    Args:
        result: A :class:`SamplingResult` (or its ``sample_stats``) of a run
            with a NUTS block, or an energy tensor of shape
            ``(n_chains, n_draws)`` (a 1-d tensor is read as a single chain).
        block: Index of the sampling block whose ``"energy"`` statistics to
            use. Default ``None`` picks the only block that records energies
            and raises a ``ValueError`` if there are none or several.

    Returns:
        A tensor of shape ``(n_chains,)``.
    """
    energy = _energy_from(result, block)
    energy = torch.as_tensor(energy)
    if energy.ndim == 1:
        energy = energy.unsqueeze(0)
    if energy.ndim != 2:
        raise ValueError(f"energy must have shape (n_chains, n_draws), got {tuple(energy.shape)}.")
    if energy.shape[1] < 2:
        raise ValueError("E-BFMI needs at least two draws per chain.")
    numerator = torch.diff(energy, dim=1).square().mean(dim=1)
    denominator = energy.var(dim=1, unbiased=False)
    return numerator / denominator
