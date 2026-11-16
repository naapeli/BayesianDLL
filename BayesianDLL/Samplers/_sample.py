import torch
import numpy as np
import inspect
from tqdm import tqdm
from warnings import warn
import os
import sys
import threading
import time
import struct
from multiprocessing import shared_memory
from loky import get_reusable_executor

from ._NUTS import NUTS
from ._Metropolis import Metropolis, FiniteGibbs, has_finite_support
from ._result import SamplingResult, PredicativeResult, SampleStats
from .._active_model._active_model import get_active_model
from ..Evaluation import gelman_rubin
from ..Evaluation._bfmi import bfmi
from ..Evaluation._pointwise import pointwise_log_likelihood_plan, pointwise_log_likelihood
from ..Distributions._state_space import JointStateSpace


_SAMPLER_CLASSES = {"nuts": NUTS, "metropolis": Metropolis, "gibbs": FiniteGibbs}
_SAMPLER_ALIASES = {"finite_gibbs": "gibbs", "finitegibbs": "gibbs"}
# Constructor arguments supplied by the library itself, not by sampler_params.
_INTERNAL_ARGUMENTS = {"self", "log_target", "gradient", "inverse_transformation", "state_space", "elementwise_log_target"}


def _tunable_arguments(cls):
    return {
        name for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name not in _INTERNAL_ARGUMENTS and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def _filter_sampler_params(cls, sampler_params):
    """Keep the settings ``cls`` understands.

    ``sample(..., **sampler_params)`` passes the same settings to every block,
    e.g. ``metric="dense"`` is meant for NUTS and ignored by the discrete
    samplers. Settings that no sampler understands are rejected.
    """
    known = set().union(*(_tunable_arguments(c) for c in _SAMPLER_CLASSES.values()))
    unknown = sorted(set(sampler_params) - known)
    if unknown:
        raise TypeError(f"Unknown sampler setting(s): {unknown}. Known settings: {sorted(known)}.")
    accepted = _tunable_arguments(cls)
    return {name: value for name, value in sampler_params.items() if name in accepted}


def _select_sampler(log_target, state_space, gradient=None, sampler="auto", sampler_params=None, elementwise_log_target=None):
    if sampler_params is None:
        sampler_params = {}

    is_continuous = state_space.is_continuous()
    is_discrete = state_space.is_discrete()
    sampler = _SAMPLER_ALIASES.get(sampler, sampler)

    if sampler == "auto":
        if is_continuous and gradient is not None:
            sampler = "nuts"
        elif is_discrete and has_finite_support(state_space):
            sampler = "gibbs"
        else:
            sampler = "metropolis"

    if sampler == "nuts":
        if not is_continuous:
            raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
        if gradient is None:
            raise RuntimeError("NUTS sampler requires a gradient function.")
        instance = NUTS(log_target, gradient, lambda x: x, **_filter_sampler_params(NUTS, sampler_params))
    elif sampler == "metropolis":
        if not (is_continuous or is_discrete):
            raise RuntimeError("State space must be either continuous or discrete for Metropolis sampler.")
        instance = Metropolis(log_target, state_space, **_filter_sampler_params(Metropolis, sampler_params))
    elif sampler == "gibbs":
        if not (is_discrete and has_finite_support(state_space)):
            raise RuntimeError("The finite Gibbs sampler can only be used with discrete distributions with a finite support.")
        instance = FiniteGibbs(log_target, state_space, elementwise_log_target=elementwise_log_target, **_filter_sampler_params(FiniteGibbs, sampler_params))
    else:
        raise RuntimeError(f"A distribution is incompatable with the chosen sampler: '{sampler}'.")

    instance.init_sampler()
    return instance


def _sampler_name(sampler):
    if isinstance(sampler, NUTS):
        return "nuts"
    if isinstance(sampler, FiniteGibbs):
        return "gibbs"
    if isinstance(sampler, Metropolis):
        return "metropolis"
    return type(sampler).__name__


class SamplingBlock:
    """A group of parameters updated jointly by one sampler in a Gibbs scheme.

    The block's target is the part of the joint log density that depends on
    its parameters (their priors, the priors of their random children and the
    likelihoods of their observed children); the remaining terms are constant
    while the block moves and are not evaluated.
    """

    def __init__(self, model, param_names, sampler_type="auto", sampler_params=None):
        self.model = model
        self.param_names = [param_names] if isinstance(param_names, str) else list(param_names)
        self.sampler_params = sampler_params or {}

        # Check parameter existence
        for name in self.param_names:
            if name in getattr(model, "enumerated_params", {}):
                raise ValueError(f"Parameter '{name}' is enumerated (summed out of the model) and cannot be part of a sampling block.")
            if name not in model.params:
                raise KeyError(f"Parameter '{name}' not found in model parameters.")

        # Disallow mixing continuous and discrete variables in the same block
        is_continuous_list = [
            model.params[name].distribution.transformed_state_space.is_continuous()
            for name in self.param_names
        ]
        if any(is_continuous_list) and not all(is_continuous_list):
            raise ValueError(
                f"Continuous and discrete variables cannot be grouped in the same block: {self.param_names}"
            )

        self.is_continuous = all(is_continuous_list)

        # Slice mapping for flat 1D tensor packing/unpacking
        self.slices = {}
        offset = 0
        for name in self.param_names:
            param = model.params[name]
            shape = param.unconstrained_value.shape
            numel = param.unconstrained_value.numel()
            self.slices[name] = (offset, offset + numel, shape)
            offset += numel
        self.total_dim = offset

        merged_params = {}
        for name in self.param_names:
            merged_params.update(model.params[name].sampler_params)
        merged_params.update(self.sampler_params)

        if len(self.param_names) == 1:
            state_space = model.params[self.param_names[0]].distribution.transformed_state_space
        else:
            spaces_dict = {
                name: model.params[name].distribution.transformed_state_space
                for name in self.param_names
            }
            state_space = JointStateSpace(spaces_dict, self.slices)

        if sampler_type == "auto":
            requested = [_SAMPLER_ALIASES.get(model.params[name].sampler, model.params[name].sampler) for name in self.param_names]
            if self.is_continuous:
                sampler_type = "metropolis" if "metropolis" in requested else "nuts"
            elif "metropolis" in requested or not has_finite_support(state_space):
                sampler_type = "metropolis"
            else:
                sampler_type = "gibbs"

        self.sampler_type = sampler_type

        # Only the terms in the Markov blanket of the block change with it.
        self._random_terms, self._observed_terms, self._grad_nodes = model.block_terms(self.param_names)

        # Proposals far outside the typical set can make a model fail numerically
        # (e.g. an overflowing exp(log_lengthscale) gives a non-finite GP
        # covariance). Such points are treated as having zero density, so the
        # samplers reject them (NUTS records a divergence) instead of crashing.
        # Errors are only caught once the model has been evaluated successfully,
        # so a model that is broken everywhere still raises immediately.
        numerical_errors = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)
        self._evaluated = False

        def log_target(theta_flat):
            self.unpack_and_set(theta_flat)
            if not self._evaluated:
                value = self.block_log_prob()
                self._evaluated = True
                return value
            try:
                return self.block_log_prob()
            except numerical_errors:
                return torch.tensor(-float("inf"), dtype=theta_flat.dtype, device=theta_flat.device)

        gradient = None
        if self.is_continuous:
            def gradient(theta_flat):
                self.unpack_and_set(theta_flat)
                if not self._evaluated:
                    grads_dict = self.model.joint_grad_log_prob(self.param_names, nodes=self._grad_nodes)
                else:
                    try:
                        grads_dict = self.model.joint_grad_log_prob(self.param_names, nodes=self._grad_nodes)
                    except numerical_errors:
                        return torch.full_like(theta_flat, float("nan"))
                grads = [grads_dict[name].reshape(-1) for name in self.param_names]
                return torch.cat(grads, dim=0)

        self.sampler = _select_sampler(
            log_target=log_target,
            state_space=state_space,
            gradient=gradient,
            sampler=self.sampler_type,
            sampler_params=merged_params,
        )
        self.sampler_type = _sampler_name(self.sampler)
        self._last_theta = None
        self._last_version = None
        self.last_stats = {}

    def block_log_prob(self):
        """Log density terms that depend on this block (at the model's current values)."""
        return self.model._log_prob_of_terms(self._random_terms, self._observed_terms)

    def pack(self) -> torch.Tensor:
        tensors = [self.model.params[name].unconstrained_value.reshape(-1) for name in self.param_names]
        return torch.cat(tensors, dim=0)

    def unpack_and_set(self, theta_flat: torch.Tensor):
        for name, (start, end, shape) in self.slices.items():
            self.model.params[name].set_unconstrained_value(theta_flat[start:end].reshape(shape))

    def step(self, warmup: bool):
        # The samplers cache the target at the state they returned. That cache
        # is only valid while no other block has moved since (the shared
        # ``_sampling_version`` counter of the model tracks moves) and the
        # block's own parameters still hold that state.
        version = getattr(self.model, "_sampling_version", 0)
        theta_flat = self.pack()
        if (
            self._last_theta is not None
            and self._last_version == version
            and torch.equal(theta_flat, self._last_theta)
        ):
            theta_flat = self._last_theta
        new_theta_flat, step_size, acc_prob, diverging = self.sampler.step(theta_flat, warmup=warmup)
        self.unpack_and_set(new_theta_flat)
        if not torch.equal(new_theta_flat, theta_flat):
            version += 1
            self.model._sampling_version = version
        self._last_theta = new_theta_flat
        self._last_version = version
        self.last_stats = dict(getattr(self.sampler, "last_stats", {}))
        return step_size, acc_prob, diverging


def _build_blocks(model, blocks_spec=None, sampler_params=None):
    if blocks_spec is not None:
        built_blocks = []
        covered = set()
        for b in blocks_spec:
            if isinstance(b, SamplingBlock):
                block = b
            elif isinstance(b, (list, tuple)):
                block = SamplingBlock(model, list(b), sampler_params=sampler_params)
            elif isinstance(b, str):
                block = SamplingBlock(model, [b], sampler_params=sampler_params)
            else:
                raise TypeError(f"Invalid block specification: {b}")

            built_blocks.append(block)
            for name in block.param_names:
                if name in covered:
                    raise ValueError(f"Parameter '{name}' appears in multiple blocks.")
                covered.add(name)

        all_params = set(model.params.keys())
        missing = all_params - covered
        if missing:
            raise ValueError(f"The following parameters were not included in any block: {missing}")
        return built_blocks

    nuts_names = []
    continuous_metropolis_names = []
    gibbs_names = []
    discrete_metropolis_names = []
    for name, p in model.params.items():
        space = p.distribution.transformed_state_space
        requested = _SAMPLER_ALIASES.get(p.sampler, p.sampler)
        if space.is_continuous():
            if requested in ("auto", "nuts"):
                nuts_names.append(name)
            else:
                continuous_metropolis_names.append(name)
        elif requested in ("auto", "gibbs") and has_finite_support(space):
            gibbs_names.append(name)
        else:
            discrete_metropolis_names.append(name)

    built_blocks = []
    if nuts_names:
        built_blocks.append(SamplingBlock(model, nuts_names, sampler_type="nuts", sampler_params=sampler_params))
    if continuous_metropolis_names:
        built_blocks.append(SamplingBlock(model, continuous_metropolis_names, sampler_type="metropolis", sampler_params=sampler_params))
    if gibbs_names:
        built_blocks.append(SamplingBlock(model, gibbs_names, sampler_type="gibbs", sampler_params=sampler_params))
    if discrete_metropolis_names:
        built_blocks.append(SamplingBlock(model, discrete_metropolis_names, sampler_type="metropolis", sampler_params=sampler_params))

    return built_blocks


def _chain_seeds(seed, n_chains):
    """Independent, reproducible per-chain seeds derived from one integer seed."""
    sequences = np.random.SeedSequence(int(seed)).spawn(n_chains)
    return [int(sequence.generate_state(1, dtype=np.uint64)[0] % (2 ** 63)) for sequence in sequences]


def sample(n_samples, warmup_length, n_chains=4, model=None, progress_bar=True, start_point_variance=1, blocks=None, check_convergence=True, seed=None, log_likelihood=True, **sampler_params):
    """Draw posterior samples with one process per chain.

    ``seed`` makes runs reproducible: every chain gets its own seed derived
    from it. With ``seed=None`` the base seed is drawn from torch's global
    generator, so the chains still differ from each other and a preceding
    ``torch.manual_seed`` also makes the run reproducible.

    ``log_likelihood=True`` records the elementwise log-likelihood
    ``log p(y_i | theta)`` of every observed node for every kept draw in
    ``result.log_likelihood[name]`` (shape ``(n_chains, n_samples,
    *log_density_shape)``), which :func:`~BayesianDLL.Evaluation.loo` and
    :func:`~BayesianDLL.Evaluation.waic` use. Pass ``False`` to save the
    memory and time.

    Remaining keyword arguments are sampler settings for every block (each
    sampler takes the ones it understands), e.g. ``delta=0.9`` or
    ``metric="dense"`` for NUTS.
    """
    model = get_active_model() if model is None else model
    # Decide how the enumerated parameters are summed out before starting the
    # workers, so a model that cannot be enumerated fails here.
    model.prepare_enumeration()
    if log_likelihood:
        log_likelihood_plan, log_likelihood_unavailable = pointwise_log_likelihood_plan(model)
    else:
        log_likelihood_plan, log_likelihood_unavailable = None, {}

    if seed is None:
        seed = int(torch.randint(0, 2 ** 62, (1,)).item())
    chain_seeds = _chain_seeds(seed, n_chains)

    initial_values = {}
    for name, parameter in model.params.items():
        initial_values[name] = parameter.unconstrained_value

    shm = None
    shm_name = None
    updater_thread = None
    stop_event = threading.Event()
    bars = []

    if progress_bar and n_chains > 1:
        shm = shared_memory.SharedMemory(create=True, size=n_chains * 128)
        shm.buf[:n_chains * 128] = b"\x00" * (n_chains * 128)
        shm_name = shm.name

        for chain in range(n_chains):
            bar = tqdm(
                total=n_samples + warmup_length,
                position=chain,
                leave=True,
                bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}"
            )
            bar.set_description(f"Chain {chain + 1}/{n_chains} warmup", refresh=False)
            bars.append(bar)

        def _updater():
            while not stop_event.is_set():
                for chain, bar in enumerate(bars):
                    m, is_warmup, divs, n_b, *floats = struct.unpack_from("=iiii8f8f", shm.buf, chain * 128)
                    if m > bar.n:
                        bar.n = m
                        if is_warmup:
                            bar.set_description(f"Chain {chain + 1}/{n_chains} warmup", refresh=False)
                        else:
                            bar.set_description(f"Chain {chain + 1}/{n_chains} sample", refresh=False)
                        if n_b > 0:
                            s_sizes = floats[:n_b]
                            acc_probs = floats[8:8 + n_b]
                            bar.set_postfix({
                                "avg. acc. probs": [f"{prob / m:.3f}" for prob in acc_probs],
                                "step sizes": [f"{step_size:.3f}" for step_size in s_sizes],
                                "divs": divs
                            }, refresh=False)
                        bar.refresh()
                time.sleep(0.05)

        updater_thread = threading.Thread(target=_updater, daemon=True)
        updater_thread.start()

    max_workers = min(n_chains, os.cpu_count())
    executor = get_reusable_executor(max_workers=max_workers, timeout=300)
    try:
        results = list(executor.map(
            _sample_single_chain,
            range(n_chains),
            [n_chains]*n_chains,
            [model]*n_chains,
            [initial_values]*n_chains,
            [start_point_variance]*n_chains,
            [n_samples]*n_chains,
            [warmup_length]*n_chains,
            [progress_bar]*n_chains,
            [blocks]*n_chains,
            [shm_name]*n_chains,
            [sampler_params]*n_chains,
            chain_seeds,
            [log_likelihood_plan]*n_chains,
        ))
    finally:
        if updater_thread is not None:
            stop_event.set()
            updater_thread.join()

            # Final refresh to 100% on all bars
            for chain, bar in enumerate(bars):
                m, is_warmup, divs, n_b, *floats = struct.unpack_from("=iiii8f8f", shm.buf, chain * 128)
                bar.n = n_samples + warmup_length
                bar.set_description(f"Chain {chain + 1}/{n_chains} sample", refresh=False)
                if n_b > 0:
                    s_sizes = floats[:n_b]
                    acc_probs = floats[8:8 + n_b]
                    total_steps = n_samples + warmup_length
                    bar.set_postfix({
                        "avg. acc. probs": [f"{prob / total_steps:.3f}" for prob in acc_probs],
                        "step sizes": [f"{step_size:.3f}" for step_size in s_sizes],
                        "divs": divs
                    }, refresh=False)
                bar.refresh()
                bar.close = lambda: None
                if hasattr(tqdm, "_instances") and bar in tqdm._instances:
                    tqdm._instances.remove(bar)

            # Clean move below all progress bars
            sys.stderr.write("\n" * n_chains)
            sys.stderr.flush()

            shm.close()
            shm.unlink()

    trace = {name: torch.stack([res[0][name] for res in results], dim=0) for name in model.params.keys()}
    for name in model.enumerated_params:
        trace[name] = torch.stack([res[7][name] for res in results], dim=0)
    enumerated_probs = {name: torch.stack([res[8][name] for res in results], dim=0) for name in model.enumerated_params}
    det_trace = {name: torch.stack([res[1][name] for res in results], dim=0) for name in model.deterministic_params.keys()}
    divergences = [res[2] for res in results]
    acceptance_probabilities = [[prob / (n_samples + warmup_length) for prob in res[3]] for res in results]
    step_sizes = [res[4] for res in results]
    sample_stats = _combine_sample_stats([res[5] for res in results])
    log_lik = None
    if log_likelihood_plan is not None:
        log_lik = {name: torch.stack([res[9][name] for res in results], dim=0) for name in log_likelihood_plan}

    if check_convergence and n_chains > 1:
        if n_samples >= 50:
            r_hats = gelman_rubin({name: trace[name] for name in model.params})
            for name, statistics in r_hats.items():
                if torch.any(statistics > 1.01):
                    warn(f"The gelman-Ruben statistic of {name} is above 1.01 ({torch.round(statistics, decimals=3).tolist()}) and indicates poor convergence. Consider increasing the amount of warmup steps or reparametrizing the model.")
    elif check_convergence and n_chains == 1:
        warn(f"The convergence of the chain is not checked when n_chains == 1. Increase it to at least 2 to enable convergence diagnostics.")

    if sum(divergences) > 0:
        warn(f"There were {sum(divergences)} divergences across all chains after tuning. Increase target acceptance probability or reparameterize the model.")

    _check_hamiltonian_diagnostics(sample_stats, results[0][6] if results else [], n_samples)

    return SamplingResult(trace, divergences, acceptance_probabilities, step_sizes, deterministic_trace=det_trace, sample_stats=sample_stats, enumerated_probs=enumerated_probs, log_likelihood=log_lik, log_likelihood_unavailable=log_likelihood_unavailable)


def _combine_sample_stats(per_chain):
    """Stack per-chain block statistics into tensors of shape (n_chains, n_samples)."""
    if not per_chain:
        return SampleStats()
    blocks = []
    for index, first in enumerate(per_chain[0]):
        stats = {}
        for name in first["stats"]:
            stats[name] = torch.stack([chain[index]["stats"][name] for chain in per_chain], dim=0)
        blocks.append({"params": list(first["params"]), "sampler": first["sampler"], "stats": stats})
    return SampleStats(blocks)


def _check_hamiltonian_diagnostics(sample_stats, max_depths, n_samples, bfmi_threshold=0.3, saturation_threshold=0.05):
    """Warn about low E-BFMI and frequently saturated NUTS trees."""
    for index, block in enumerate(sample_stats.blocks):
        stats = block["stats"]
        if "energy" in stats and n_samples >= 20:
            values = bfmi(stats["energy"])
            low = [chain for chain, value in enumerate(values.tolist()) if value < bfmi_threshold]
            if low:
                warn(f"The E-BFMI of chain(s) {low} is below {bfmi_threshold} ({[round(values[c].item(), 3) for c in low]}) for the parameters {block['params']}. This indicates that the sampler explores the posterior poorly; consider reparametrizing the model.")
        max_depth = max_depths[index] if index < len(max_depths) else None
        if "tree_depth" in stats and max_depth is not None and stats["tree_depth"].numel() > 0:
            saturated = (stats["tree_depth"] >= max_depth).double().mean().item()
            if saturated > saturation_threshold:
                warn(f"{100 * saturated:.1f}% of the NUTS iterations for the parameters {block['params']} reached the maximum tree depth ({max_depth}). Consider increasing max_depth or reparametrizing the model.")


def _sample_single_chain(chain, n_chains, model, initial_values, start_point_variance, n_samples, warmup_length, progress_bar, blocks_spec=None, shm_name=None, sampler_params=None, seed=None, log_likelihood_plan=None):
    torch.set_num_threads(1)
    if seed is not None:
        # Worker processes may be fresh or reused; seeding here makes each chain
        # reproducible and independent of the worker it runs in.
        torch.manual_seed(seed)
    for name, parameter in model.params.items():
        value = initial_values[name].clone()
        if parameter.distribution.state_space.is_continuous():
            value = value + start_point_variance * torch.randn_like(value)
        parameter.set_unconstrained_value(value)

    blocks = _build_blocks(model, blocks_spec, sampler_params)
    for block in blocks:
        if hasattr(block.sampler, "set_warmup_length"):
            # Warmup steps are the iterations m < warmup_length (m starts at 1).
            block.sampler.set_warmup_length(max(warmup_length - 1, 0))

    # Attach to shared memory if multi-chain
    shm = None
    if shm_name is not None:
        shm = shared_memory.SharedMemory(name=shm_name)

    # For single chain execution, use local tqdm directly
    _progress_bar = None
    if progress_bar and shm_name is None:
        _progress_bar = tqdm(
            range(1, n_samples + warmup_length + 1),
            bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}"
        )

    acceptance_probabilities = [1.0 for _ in range(len(blocks))]
    step_sizes = [1.0 for _ in range(len(blocks))]
    divergences_count = 0
    block_stats = [{} for _ in blocks]

    # Pre-allocate trace storage: shape (n_samples, *constrained_shape)
    chain_trace = {}
    for name, parameter in model.params.items():
        chain_trace[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape), dtype=parameter.unconstrained_value.dtype)

    # Enumerated parameters: exact draws from p(z | other parameters, data) and
    # the per-coordinate probabilities of every support value.
    chain_enum_trace = {}
    chain_enum_probs = {}
    for name, parameter in model.enumerated_params.items():
        n_values = len(parameter.distribution.state_space)
        chain_enum_trace[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape), dtype=parameter.constrained_value.dtype)
        chain_enum_probs[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape, n_values), dtype=parameter.constrained_value.dtype)

    chain_det_trace = {}
    for name, parameter in model.deterministic_params.items():
        chain_det_trace[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape), dtype=parameter.constrained_value.dtype)

    # Elementwise log-likelihood of the observed nodes per kept draw.
    chain_log_lik = {}
    if log_likelihood_plan is not None:
        for name, values in pointwise_log_likelihood(model, log_likelihood_plan).items():
            chain_log_lik[name] = torch.empty(size=(n_samples, *values.shape), dtype=values.dtype, device=values.device)

    loop_range = _progress_bar if _progress_bar is not None else range(1, n_samples + warmup_length + 1)
    for m in loop_range:
        if _progress_bar is not None:
            if m < warmup_length: _progress_bar.set_description(f"Chain warmup", refresh=False)
            else: _progress_bar.set_description(f"Chain sample", refresh=False)
            _progress_bar.set_postfix({
                "avg. acc. probs": [f"{prob / m:.3f}" for prob in acceptance_probabilities],
                "step sizes": [f"{step_size:.3f}" for step_size in step_sizes],
                "divs": divergences_count
            }, refresh=False)

        for i, block in enumerate(blocks):
            step_size, acceptance_probability, diverging = block.step(warmup=(m < warmup_length))
            if diverging and m >= warmup_length:
                divergences_count += 1
            step_sizes[i] = step_size
            acceptance_probabilities[i] += acceptance_probability
            if m > warmup_length:
                for stat_name, value in block.last_stats.items():
                    block_stats[i].setdefault(stat_name, []).append(value)

        if m > warmup_length:
            if chain_enum_trace:
                # Sets the drawn values on the parameters, so deterministic
                # nodes below are recorded with them.
                for name, (value, probabilities) in model.sample_enumerated().items():
                    chain_enum_trace[name][m - warmup_length - 1] = value
                    chain_enum_probs[name][m - warmup_length - 1] = probabilities
            for name, parameter in model.params.items():
                chain_trace[name][m - warmup_length - 1] = parameter.constrained_value
            for name, parameter in model.deterministic_params.items():
                chain_det_trace[name][m - warmup_length - 1] = parameter.constrained_value
            if chain_log_lik:
                for name, values in pointwise_log_likelihood(model, log_likelihood_plan).items():
                    chain_log_lik[name][m - warmup_length - 1] = values

        if shm is not None:
            n_b = min(len(blocks), 8)
            s_padded = (step_sizes + [0.0] * 8)[:8]
            a_padded = (acceptance_probabilities + [0.0] * 8)[:8]
            struct.pack_into("=iiii8f8f", shm.buf, chain * 128, m, int(m < warmup_length), divergences_count, n_b, *s_padded, *a_padded)

    if shm is not None:
        shm.close()

    stats = [
        {
            "params": list(block.param_names),
            "sampler": block.sampler_type,
            "stats": {name: torch.tensor(values, dtype=torch.float64) for name, values in block_stats[i].items()},
        }
        for i, block in enumerate(blocks)
    ]
    max_depths = [getattr(block.sampler, "max_depth", None) for block in blocks]
    return chain_trace, chain_det_trace, divergences_count, acceptance_probabilities, step_sizes, stats, max_depths, chain_enum_trace, chain_enum_probs, chain_log_lik

def sample_posterior_predicative(n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = get_active_model() if model is None else model
    trace = sample(n_samples, warmup_length, 4, model, progress_bar, check_convergence=False)
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def posterior_predicative(trace, n_samples=20, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = get_active_model() if model is None else model
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def sample_prior_predicative(n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = get_active_model() if model is None else model
    old_observed = model.observed_params
    model.observed_params = {}  # with prior distributions, one should sample from the priors without the likelihood terms
    model.compile()  # the factor groups of enumerated parameters must not contain the likelihood terms
    try:
        trace = sample(n_samples, warmup_length, 4, model, progress_bar)
    finally:
        model.observed_params = old_observed
        model.compile()
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def sample_predicative(trace, n_samples=None, samples_per_step=20, model=None, progress_bar=True, warmup_per_sample=20):
    model = get_active_model() if model is None else model

    samplers = {}
    state_spaces = {}
    for name, parameter in model.observed_params.items():
        samplers[name] = _decide_predicative_step(parameter)
        state_spaces[name] = parameter.distribution.transformed_state_space

    n_chains, trace_length = next(iter(trace.values())).shape[:2]
    flattened_trace = {name: values.flatten(0, 1) for name, values in trace.items()}
    total_samples = n_chains * trace_length

    if n_samples is None:
        n_samples = total_samples
        indices = torch.arange(total_samples)
    else:
        if total_samples < n_samples:
            raise RuntimeError(f"n_samples ({n_samples}) must be less than or equal to the total trace length ({total_samples}).")
        indices = torch.linspace(0, total_samples - 1, steps=n_samples).long()

    predicative_samples = {name: torch.empty(size=(n_samples, samples_per_step, *parameter.predictive_shape), dtype=parameter.observed_values.dtype) for name, parameter in model.observed_params.items()}

    _progress_bar = tqdm(
        range(n_samples),
        desc="Predicative sampling",
        bar_format=r"{desc}: {percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}"
    ) if progress_bar else range(n_samples)

    for i in _progress_bar:
        prior_values = {
            name: parameter.distribution.transform.forward(flattened_trace[name][indices[i]])
            for name, parameter in {**model.params, **model.enumerated_params}.items()
            if name in flattened_trace
        }
        with model.temporarily_set_many(prior_values):
            for name, sampler in samplers.items():
                sampler.reset()
                if hasattr(sampler, "set_warmup_length"):
                    sampler.set_warmup_length(warmup_per_sample)
                parameter = model.observed_params[name]
                init_value = parameter.observed_values
                theta = _init_theta(state_spaces[name], parameter.predictive_shape, init_value.dtype)
                for m in range(warmup_per_sample + samples_per_step):
                    theta, step_size, acceptance_probability, diverging = sampler.step(theta, m < warmup_per_sample)
                    if m >= warmup_per_sample:
                        predicative_samples[name][i, m - warmup_per_sample] = parameter.distribution.transform.inverse(theta)

    return PredicativeResult(predicative_samples)

def _init_theta(state_space, shape, dtype):
    if state_space.is_continuous():
        return torch.randn(shape, dtype=dtype)
    elif state_space.is_discrete():
        first_value = next(iter(state_space))
        return first_value * torch.ones(shape, dtype=dtype)

def _decide_predicative_step(parameter):
    state_space = parameter.distribution.transformed_state_space

    def log_target(x):
        return parameter.distribution._log_prob_unconstrained(x)

    gradient = None
    if state_space.is_continuous():
        def gradient(x):
            return parameter.distribution._log_prob_grad_unconstrained(x)

    elementwise_log_target = None
    if state_space.is_discrete():
        # Generated observations are independent given the parameters, so a
        # finite discrete observation can be drawn exactly coordinate-wise when
        # log_pdf returns one log probability per element.
        def elementwise_log_target(x):
            return parameter.distribution.log_pdf(parameter.distribution.transform.inverse(x))

    return _select_sampler(
        log_target=log_target,
        state_space=state_space,
        gradient=gradient,
        sampler=parameter.sampler,
        sampler_params=parameter.sampler_params,
        elementwise_log_target=elementwise_log_target,
    )
