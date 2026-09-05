import torch
import numpy as np
from itertools import product
import pandas as pd

from ._effective_sample_size import effective_sample_size
from ._gelman_rubin import gelman_rubin
from ._mcse import mcse


def _hdi(ary: np.ndarray, hdi_prob: float = 0.94) -> tuple[float, float]:
    flat = np.asarray(ary).flatten()
    n = len(flat)
    if n < 2:
        return float(flat[0]), float(flat[0])
    
    sorted_draws = np.sort(flat)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_widths = sorted_draws[interval_idx_inc:] - sorted_draws[:n_intervals]
    min_idx = int(np.argmin(interval_widths))
    hdi_min = float(sorted_draws[min_idx])
    hdi_max = float(sorted_draws[min_idx + interval_idx_inc])
    return hdi_min, hdi_max


def summary(
    result,
    hdi_prob: float = 0.94,
    ci_kind: str = "eti",
    round_to: int = 3,
    include_deterministic: bool = True
):
    """
    Create a summary DataFrame of MCMC sampling results.

    Parameters:
    -----------
    result : SamplingResult or dict[str, Tensor]
        MCMC sampling output from Model.sample().
    hdi_prob : float, default 0.94
        Probability mass for credible interval (e.g. 0.94 or 0.95).
    ci_kind : {"eti", "hdi"}, default "eti"
        Type of credible interval to compute ("eti" = equal-tail interval, "hdi" = highest density interval).
    round_to : int or None, default 3
        Decimal places for display rounding (use None for raw numbers).
    include_deterministic : bool, default True
        Whether to include deterministic parameter traces in the summary table.

    Returns:
    --------
    pandas.DataFrame
        Summary statistics table:
        mean, sd, ci_lower, ci_upper, ess_bulk, ess_tail, r_hat, mcse_mean, mcse_sd.
    """
    # Extract parameter trace dictionary
    traces = {}
    if hasattr(result, "trace"):
        traces.update(result.trace)
        if include_deterministic and hasattr(result, "deterministic_trace"):
            traces.update(result.deterministic_trace)
    elif isinstance(result, dict):
        traces = result
    else:
        raise TypeError("result must be a SamplingResult instance or a dictionary of tensors.")

    # Compute diagnostics using public evaluation functions
    ess_bulk_dict = effective_sample_size(traces, method="bulk")
    ess_tail_dict = effective_sample_size(traces, method="tail")
    rhat_dict = gelman_rubin(traces, method="rank")
    mcse_mean_dict = mcse(traces, method="mean")
    mcse_sd_dict = mcse(traces, method="sd")

    p_low = round((1.0 - hdi_prob) / 2.0 * 100, 1)
    p_high = round((1.0 - (1.0 - hdi_prob) / 2.0) * 100, 1)
    p_low_str = f"{p_low:g}%"
    p_high_str = f"{p_high:g}%"
    prefix = "hdi" if ci_kind == "hdi" else "eti"
    col_ci_low = f"{prefix}_{p_low_str}"
    col_ci_high = f"{prefix}_{p_high_str}"

    columns = [
        "mean",
        "sd",
        col_ci_low,
        col_ci_high,
        "ess_bulk",
        "ess_tail",
        "r_hat",
        "mcse_mean",
        "mcse_sd",
    ]

    row_names = []
    rows = []

    for name, tensor in traces.items():
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if tensor.ndim < 2:
            continue

        n_chains, n_samples = tensor.shape[:2]
        extra_shape = tensor.shape[2:]
        ary_np = tensor.detach().cpu().numpy()

        if len(extra_shape) == 0:
            indices_list = [()]
        else:
            ranges = [range(dim) for dim in extra_shape]
            indices_list = list(product(*ranges))

        for idx in indices_list:
            if len(idx) == 0:
                row_label = name
                feat_chains = ary_np
                ess_b = float(ess_bulk_dict[name].item())
                ess_t = float(ess_tail_dict[name].item())
                r_hat_val = float(rhat_dict[name].item())
                mcse_m = float(mcse_mean_dict[name].item())
                mcse_s = float(mcse_sd_dict[name].item())
            else:
                idx_str = ",".join(str(i) for i in idx)
                row_label = f"{name}[{idx_str}]"
                slicing = (slice(None), slice(None)) + idx
                feat_chains = ary_np[slicing]
                ess_b = float(ess_bulk_dict[name][idx].item())
                ess_t = float(ess_tail_dict[name][idx].item())
                r_hat_val = float(rhat_dict[name][idx].item())
                mcse_m = float(mcse_mean_dict[name][idx].item())
                mcse_s = float(mcse_sd_dict[name][idx].item())

            flat_draws = feat_chains.flatten()
            mean_val = float(np.mean(flat_draws))
            sd_val = float(np.std(flat_draws, ddof=1))  # Sample standard deviation (Arviz uses ddof=0 by default).

            if ci_kind == "hdi":
                ci_min, ci_max = _hdi(flat_draws, hdi_prob=hdi_prob)
            else:
                p_l = (1.0 - hdi_prob) / 2.0
                p_h = 1.0 - p_l
                ci_min = float(np.quantile(flat_draws, p_l))
                ci_max = float(np.quantile(flat_draws, p_h))

            row_data = [
                mean_val,
                sd_val,
                ci_min,
                ci_max,
                ess_b,
                ess_t,
                r_hat_val,
                mcse_m,
                mcse_s,
            ]

            if round_to is not None:
                row_data = [
                    round(mean_val, round_to),
                    round(sd_val, round_to),
                    round(ci_min, round_to),
                    round(ci_max, round_to),
                    int(np.floor(ess_b)),
                    int(np.floor(ess_t)),
                    round(r_hat_val, max(2, round_to)),
                    round(mcse_m, round_to + 1),
                    round(mcse_s, round_to + 1),
                ]

            row_names.append(row_label)
            rows.append(row_data)

    return pd.DataFrame(rows, index=row_names, columns=columns)
