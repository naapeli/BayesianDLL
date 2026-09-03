import torch
from tqdm import tqdm
from warnings import warn
import os
import sys
import threading
import time
import struct
from multiprocessing import shared_memory

from . import NUTS, Metropolis
from ._result import SamplingResult, PredicativeResult
from loky import get_reusable_executor
from .._active_model import _active_model
from ..Evaluation import gelman_rubin
from ..Distributions._state_space import JointStateSpace


def _select_sampler(log_target, state_space, gradient=None, sampler="auto", sampler_params=None):
    if sampler_params is None:
        sampler_params = {}

    is_continuous = state_space.is_continuous()
    is_discrete = state_space.is_discrete()

    if sampler == "auto":
        if is_continuous and gradient is not None:
            sampler = "nuts"
        else:
            sampler = "metropolis"

    if sampler == "nuts":
        if not is_continuous:
            raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
        if gradient is None:
            raise RuntimeError("NUTS sampler requires a gradient function.")
        instance = NUTS(log_target, gradient, lambda x: x, **sampler_params)
    elif sampler == "metropolis":
        if not (is_continuous or is_discrete):
            raise RuntimeError("State space must be either continuous or discrete for Metropolis sampler.")
        instance = Metropolis(log_target, state_space, **sampler_params)
    else:
        raise RuntimeError(f"A distribution is incompatable with the chosen sampler: '{sampler}'.")

    instance.init_sampler()
    return instance


class SamplingBlock:
    def __init__(self, model, param_names, sampler_type="auto", sampler_params=None):
        self.model = model
        self.param_names = [param_names] if isinstance(param_names, str) else list(param_names)
        self.sampler_params = sampler_params or {}

        # Check parameter existence
        for name in self.param_names:
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

        if sampler_type == "auto":
            requested = [model.params[name].sampler for name in self.param_names]
            if not self.is_continuous or "metropolis" in requested:
                sampler_type = "metropolis"
            else:
                sampler_type = "nuts"

        self.sampler_type = sampler_type

        def log_target(theta_flat):
            self.unpack_and_set(theta_flat)
            return self.model.model_log_prob()

        gradient = None
        if self.is_continuous:
            def gradient(theta_flat):
                self.unpack_and_set(theta_flat)
                grads_dict = self.model.joint_grad_log_prob(self.param_names)
                grads = [grads_dict[name].reshape(-1) for name in self.param_names]
                return torch.cat(grads, dim=0)

        if len(self.param_names) == 1:
            state_space = model.params[self.param_names[0]].distribution.transformed_state_space
        else:
            spaces_dict = {
                name: model.params[name].distribution.transformed_state_space
                for name in self.param_names
            }
            state_space = JointStateSpace(spaces_dict, self.slices)

        self.sampler = _select_sampler(
            log_target=log_target,
            state_space=state_space,
            gradient=gradient,
            sampler=self.sampler_type,
            sampler_params=merged_params,
        )

    def pack(self) -> torch.Tensor:
        tensors = [self.model.params[name].unconstrained_value.reshape(-1) for name in self.param_names]
        return torch.cat(tensors, dim=0)

    def unpack_and_set(self, theta_flat: torch.Tensor):
        for name, (start, end, shape) in self.slices.items():
            self.model.params[name].set_unconstrained_value(theta_flat[start:end].reshape(shape))

    def step(self, warmup: bool):
        theta_flat = self.pack()
        new_theta_flat, step_size, acc_prob, diverging = self.sampler.step(theta_flat, warmup=warmup)
        self.unpack_and_set(new_theta_flat)
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

    nuts_names = [
        name for name, p in model.params.items()
        if p.distribution.transformed_state_space.is_continuous() and p.sampler in ("auto", "nuts")
    ]
    metropolis_names = [
        name for name in model.params if name not in nuts_names
    ]

    built_blocks = []
    if nuts_names:
        built_blocks.append(SamplingBlock(model, nuts_names, sampler_type="nuts", sampler_params=sampler_params))
    if metropolis_names:
        built_blocks.append(SamplingBlock(model, metropolis_names, sampler_type="metropolis", sampler_params=sampler_params))

    return built_blocks


def sample(n_samples, warmup_length, n_chains=4, model=None, progress_bar=True, start_point_variance=1, blocks=None, **sampler_params):
    model = _active_model._active_model if model is None else model
    
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
    executor = get_reusable_executor(max_workers=max_workers)
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
            [sampler_params]*n_chains
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
    det_trace = {name: torch.stack([res[1][name] for res in results], dim=0) for name in model.deterministic_params.keys()}
    divergences = [res[2] for res in results]
    acceptance_probabilities = [[prob / (n_samples + warmup_length) for prob in res[3]] for res in results]
    step_sizes = [res[4] for res in results]

    if n_chains > 1:
        r_hats = gelman_rubin(trace)
        for name, statistics in r_hats.items():
            if torch.any(statistics > 1.1):  # 1.01
                warn(f"The gelman-Ruben statistic of {name} is above 1.1 ({torch.round(statistics, decimals=3).tolist()}) and indicates poor convergence. Consider increasing the amount of warmup steps or reparametrizing the model.")
    else:
        warn(f"The The convergence of the chain is not checked when n_chains 1. Increase it to atleast 2 to enable convergence diagnostics.")

    if sum(divergences) > 0:
        warn(f"There were {sum(divergences)} divergences across all chains after tuning. Increase target acceptance probability or reparameterize the model.")

    return SamplingResult(trace, divergences, acceptance_probabilities, step_sizes, deterministic_trace=det_trace)

def _sample_single_chain(chain, n_chains, model, initial_values, start_point_variance, n_samples, warmup_length, progress_bar, blocks_spec=None, shm_name=None, sampler_params=None):
    torch.set_num_threads(1)
    for name, parameter in model.params.items():
        value = initial_values[name].clone()
        if parameter.distribution.state_space.is_continuous():
            value = value + start_point_variance * torch.randn_like(value)
        parameter.set_unconstrained_value(value)
    
    blocks = _build_blocks(model, blocks_spec, sampler_params)

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

    # Pre-allocate trace storage: shape (n_samples, *constrained_shape)
    chain_trace = {}
    for name, parameter in model.params.items():
        chain_trace[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape), dtype=parameter.unconstrained_value.dtype)

    chain_det_trace = {}
    for name, parameter in model.deterministic_params.items():
        chain_det_trace[name] = torch.empty(size=(n_samples, *parameter.constrained_value.shape), dtype=parameter.constrained_value.dtype)

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
            for name, parameter in model.params.items():
                chain_trace[name][m - warmup_length - 1] = parameter.constrained_value
            for name, parameter in model.deterministic_params.items():
                chain_det_trace[name][m - warmup_length - 1] = parameter.constrained_value

        if shm is not None:
            n_b = min(len(blocks), 8)
            s_padded = (step_sizes + [0.0] * 8)[:8]
            a_padded = (acceptance_probabilities + [0.0] * 8)[:8]
            struct.pack_into("=iiii8f8f", shm.buf, chain * 128, m, int(m < warmup_length), divergences_count, n_b, *s_padded, *a_padded)

    if shm is not None:
        shm.close()

    return chain_trace, chain_det_trace, divergences_count, acceptance_probabilities, step_sizes

def sample_posterior_predicative(n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = _active_model._active_model if model is None else model
    trace = sample(n_samples, warmup_length, 4, model, progress_bar)
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def posterior_predicative(trace, n_samples=20, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = _active_model._active_model if model is None else model
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def sample_prior_predicative(n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    model = _active_model._active_model if model is None else model
    old_observed = model.observed_params
    model.observed_params = {}  # with prior distributions, one should sample from the priors without the likelihood terms
    trace = sample(n_samples, warmup_length, 4, model, progress_bar)
    model.observed_params = old_observed
    return sample_predicative(trace, n_samples, samples_per_step, model, progress_bar, warmup_per_sample)

def sample_predicative(trace, n_samples=None, samples_per_step=20, model=None, progress_bar=True, warmup_per_sample=20):
    model = _active_model._active_model if model is None else model

    old_prior_values = {}
    for name, parameter in model.params.items():
        old_prior_values[name] = parameter.constrained_value
    
    samplers = {}
    state_spaces = {}
    for name, parameter in model.observed_params.items():
        samplers[name] = _decide_predicative_step(parameter)
        state_spaces[name] = parameter.distribution.transformed_state_space

    predicative_samples = {name: torch.empty(size=(n_samples, samples_per_step, *parameter.observed_values.shape), dtype=parameter.observed_values.dtype) for name, parameter in model.observed_params.items()}

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

    _progress_bar = tqdm(
        range(n_samples),
        desc="Predicative sampling",
        bar_format=r"{desc}: {percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}"
    ) if progress_bar else range(n_samples)

    for i in _progress_bar:
        prior_values = {}
        for name, values in flattened_trace.items():
            prior_values[name] = values[indices[i]]
        
        for name, parameter in model.params.items():
            parameter.set_constrained_value(prior_values[name])

        for name, sampler in samplers.items():
            sampler.reset()
            parameter = model.observed_params[name]
            init_value = parameter.observed_values
            theta = _init_theta(state_spaces[name], init_value.shape, init_value.dtype)
            for m in range(warmup_per_sample + samples_per_step):
                theta, step_size, acceptance_probability, diverging = sampler.step(theta, m < warmup_per_sample)
                if m >= warmup_per_sample:
                    predicative_samples[name][i, m - warmup_per_sample] = theta

    for name, parameter in model.params.items():
        unconstrained_value = parameter.distribution.transform.forward(prior_values[name])
        parameter.set_unconstrained_value(unconstrained_value)
    
    predicative_samples = {name: model.observed_params[name].distribution.transform.inverse(samples.reshape(-1, *samples.shape[2:])).reshape(samples.shape) for name, samples in predicative_samples.items()}
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

    return _select_sampler(
        log_target=log_target,
        state_space=state_space,
        gradient=gradient,
        sampler=parameter.sampler,
        sampler_params=parameter.sampler_params,
    )
