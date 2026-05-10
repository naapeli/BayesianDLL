import torch
from tqdm import tqdm
from functools import partial
from warnings import warn
import os

from . import NUTS, Metropolis
from ._result import SamplingResult
from loky import get_reusable_executor
from .._active_model import _active_model
from ..Evaluation import gelman_rubin


def sample(n_samples, warmup_length, n_chains=4, model=None, progress_bar=True, start_point_variance=1):
    model = _active_model._active_model if model is None else model
    
    initial_values = {}
    for name, parameter in model.params.items():
        initial_values[name] = parameter.unconstrained_value
        
    max_workers = min(n_chains, os.cpu_count())
    executor = get_reusable_executor(max_workers=max_workers)
    results = list(executor.map(
        _sample_single_chain,
        range(n_chains),
        [n_chains]*n_chains,
        [model]*n_chains,
        [initial_values]*n_chains,
        [start_point_variance]*n_chains,
        [n_samples]*n_chains,
        [warmup_length]*n_chains,
        [progress_bar]*n_chains
    ))

    trace = {name: torch.stack([res[0][name] for res in results], dim=0) for name in model.params.keys()}
    divergences = [res[1] for res in results]
    acceptance_probabilities = [[prob / (n_samples + warmup_length) for prob in res[2]] for res in results]
    step_sizes = [res[3] for res in results]

    if n_chains > 1:
        r_hats = gelman_rubin(trace)
        for name, statistics in r_hats.items():
            if torch.any(statistics > 1.1):  # 1.01
                warn(f"The gelman-Ruben statistic of {name} is above 1.1 ({torch.round(statistics, decimals=3).tolist()}) and indicates poor convergence. Consider increasing the amount of warmup steps or reparametrizing the model.")
    else:
        warn(f"The The convergence of the chain is not checked when n_chains 1. Increase it to atleast 2 to enable convergence diagnostics.")

    if sum(divergences) > 0:
        warn(f"There were {sum(divergences)} divergences across all chains after tuning. Increase target acceptance probability or reparameterize the model.")

    return SamplingResult(trace, divergences, acceptance_probabilities, step_sizes)

def _sample_single_chain(chain, n_chains, model, initial_values, start_point_variance, n_samples, warmup_length, progress_bar):
    torch.set_num_threads(1)
    for name, parameter in model.params.items():
        value = initial_values[name].clone()
        if parameter.distribution.state_space.is_continuous():
            value = value + start_point_variance * torch.randn_like(value)
        parameter.set_unconstrained_value(value)
    
    samplers = {}
    for name, parameter in model.params.items():
        samplers[name] = _decide_step(model, parameter)

    _progress_bar = tqdm(range(1, n_samples + warmup_length + 1), position=chain, leave=False, bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}") if progress_bar else range(1, n_samples + warmup_length + 1)
    acceptance_probabilities = [1.0 for _ in range(len(samplers))]
    step_sizes = [1.0 for _ in range(len(samplers))]
    divergences_count = 0
    chain_trace = {name: torch.empty(size=(n_samples, parameter.constrained_value.size(1)), dtype=parameter.unconstrained_value.dtype) for name, parameter in model.params.items()}

    for m in _progress_bar:
        if progress_bar:
            if m < warmup_length: _progress_bar.set_description(f"Chain {chain + 1}/{n_chains} warmup", refresh=False)
            else: _progress_bar.set_description(f"Chain {chain + 1}/{n_chains} sample", refresh=False)
            _progress_bar.set_postfix({
                "avg. acc. probs": [f"{prob / m:.3f}" for prob in acceptance_probabilities],
                "step sizes": [f"{step_size:.3f}" for step_size in step_sizes],
                "divs": divergences_count
            }, refresh=False)

        for i, (name, sampler) in enumerate(samplers.items()):
            theta = model.params[name].unconstrained_value
            new_theta, step_size, acceptance_probability, diverging = sampler.step(theta, m < warmup_length)
            if diverging and m >= warmup_length:
                divergences_count += 1
            step_sizes[i] = step_size
            acceptance_probabilities[i] += acceptance_probability
            model.params[name].set_unconstrained_value(new_theta)
            if m > warmup_length: chain_trace[name][m - warmup_length - 1] = model.params[name].constrained_value

    return chain_trace, divergences_count, acceptance_probabilities, step_sizes

def _decide_step(model, parameter):
    _log_prob_func = partial(model.log_prob, parameter.name)

    state_space = parameter.distribution.transformed_state_space

    if state_space.is_continuous() and (parameter.sampler == "auto" or parameter.sampler == "nuts"):
        sampler = NUTS(_log_prob_func, partial(model.grad_log_prob, parameter.name), lambda x: x, **parameter.sampler_params)
    elif (state_space.is_discrete() or state_space.is_continuous()) and (parameter.sampler == "auto" or parameter.sampler == "metropolis"):
        sampler = Metropolis(_log_prob_func, state_space, **parameter.sampler_params)
    else:
        raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
    
    sampler.init_sampler()
    return sampler

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
        # shape = parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value).shape  # TODO: get the shape (if shape is not (1,) need to modify and change the sampling)
        samplers[name] = _decide_predicative_step(parameter)
        state_spaces[name] = parameter.distribution.transformed_state_space

    predicative_samples = {name: torch.empty(size=(n_samples, samples_per_step, len(parameter.observed_values[0])), dtype=parameter.observed_values[0].dtype) for name, parameter in model.observed_params.items()}

    n_chains, trace_length, _ = next(iter(trace.values())).shape
    flattened_trace = {name: values.flatten(0, 1) for name, values in trace.items()}
    total_samples = n_chains * trace_length

    if n_samples is None:
        n_samples = total_samples
        indices = torch.arange(total_samples)
    else:
        if total_samples < n_samples:
            raise RuntimeError(f"n_samples ({n_samples}) must be less than or equal to the total trace length ({total_samples}).")
        indices = torch.linspace(0, total_samples - 1, steps=n_samples).long()

    # _progress_bar = tqdm(range(n_samples), desc="Predicative sample") if progress_bar else range(n_samples)
    # for i in _progress_bar:
    for i in range(n_samples):
        # print(f"Predicative sample {i + 1}")
        prior_values = {}
        for name, values in flattened_trace.items():
            prior_values[name] = values[indices[i]].unsqueeze(0)
        
        for name, parameter in model.params.items():
            # unconstrained_value = parameter.distribution.transform.forward(prior_values[name])
            # parameter.set_unconstrained_value(unconstrained_value)
            parameter.set_constrained_value(prior_values[name])

        for name, sampler in samplers.items():
            sampler.reset()
            parameter = model.observed_params[name]
            init_value = parameter.observed_values[0].unsqueeze(0)
            theta = _init_theta(state_spaces[name], init_value.shape, init_value.dtype)
            _progress_bar = tqdm(range(1, warmup_per_sample + samples_per_step + 1), bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}") if progress_bar else range(1, warmup_per_sample + samples_per_step + 1)
            acceptance_probabilities = 0
            step_size = 1
            # for m in range(warmup_per_sample + samples_per_step):
            for m in _progress_bar:
                if progress_bar:
                    if m < warmup_per_sample: _progress_bar.set_description(f"{name} predicative sample {i + 1} warmup")
                    else: _progress_bar.set_description(f"{name} predicative sample {i + 1}")
                    _progress_bar.set_postfix({
                        "avg. acc. probs": f"{acceptance_probabilities / m:.3f}",
                        "step sizes": f"{step_size:.3f}"
                    })
                m = m - 1  # shift back to range(0, end) for following logic
                theta, step_size, acceptance_probability, diverging = sampler.step(theta, m < warmup_per_sample)
                acceptance_probabilities += acceptance_probability
                if m >= warmup_per_sample:
                    predicative_samples[name][i, m - warmup_per_sample] = theta

    for name, parameter in model.params.items():
        unconstrained_value = parameter.distribution.transform.forward(prior_values[name])
        parameter.set_unconstrained_value(unconstrained_value)
    
    predicative_samples = {name: model.observed_params[name].distribution.transform.inverse(samples.reshape(n_samples * samples_per_step, -1)).reshape(n_samples, samples_per_step, -1) for name, samples in predicative_samples.items()}
    return predicative_samples

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
    if state_space.is_continuous() and (parameter.sampler == "auto" or parameter.sampler == "nuts"):
        def gradient(x):
            return parameter.distribution._log_prob_grad_unconstrained(x)
        sampler = NUTS(log_target, gradient, lambda x: x, **parameter.sampler_params)
    elif (state_space.is_discrete() or state_space.is_continuous()) and (parameter.sampler == "auto" or parameter.sampler == "metropolis"):
        sampler = Metropolis(log_target, state_space, **parameter.sampler_params)
    else:
        raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
    
    sampler.init_sampler()
    return sampler
