import torch
from tqdm import tqdm
from functools import partial

from . import NUTS, Metropolis
from .._active_model import _active_model


def sample(n_samples, warmup_length, model=None, progress_bar=True):
    model = _active_model._active_model if model is None else model

    samplers = {}
    for name, parameter in model.params.items():
        samplers[name] = _decide_step(model, parameter)
    
    trace = {name: [] for name in model.params.keys()}
    _progress_bar = tqdm(range(1, n_samples + warmup_length + 1), bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}") if progress_bar else range(1, n_samples + warmup_length + 1)
    acceptance_probabilities = [1.0 for _ in range(len(samplers))]
    step_sizes = [1.0 for _ in range(len(samplers))]
    for m in _progress_bar:
        if progress_bar:
            if m < warmup_length: _progress_bar.set_description(f"Warmup")
            else: _progress_bar.set_description(f"Sample")
            _progress_bar.set_postfix({
                "avg. acc. probs": [f"{prob / m:.3f}" for prob in acceptance_probabilities],
                "step sizes": [f"{step_size:.3f}" for step_size in step_sizes]
            })

        for i, (name, sampler) in enumerate(samplers.items()):
            theta = model.params[name].unconstrained_value
            new_theta, step_size, acceptance_probability = sampler.step(theta, m < warmup_length)
            step_sizes[i] = step_size
            acceptance_probabilities[i] += acceptance_probability
            model.params[name].set_unconstrained_value(new_theta)
            trace[name].append(model.params[name].constrained_value)

    trace = {name: torch.stack(samples[warmup_length:]) for name, samples in trace.items()}
    return trace

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

def sample_posterior_predicative(n_samples, warmup_length, model=None, progress_bar=True, warmup_per_sample=20):
    model = _active_model._active_model if model is None else model
    trace = sample(n_samples, warmup_length, model, progress_bar)
    return sample_predicative(trace, n_samples, model, progress_bar, warmup_per_sample)

def sample_prior_predicative(n_samples, warmup_length, model=None, progress_bar=True, warmup_per_sample=20):
    model = _active_model._active_model if model is None else model
    old_observed = model.observed_params
    model.observed_params = {}  # with prior distributions, one should sample from the priors without the likelihood terms
    trace = sample(n_samples, warmup_length, model, progress_bar)
    model.observed_params = old_observed
    return sample_predicative(trace, n_samples, model, progress_bar, warmup_per_sample)

def sample_predicative(trace, n_samples=None, model=None, progress_bar=True, warmup_per_sample=20):
    model = _active_model._active_model if model is None else model

    old_prior_values = {}
    for name, parameter in model.params.items():
        old_prior_values[name] = parameter.constrained_value
    
    samplers = {}
    state_spaces = {}
    for name, parameter in model.observed_params.items():
        samplers[name] = _decide_predicative_step(parameter)
        state_spaces[name] = parameter.distribution.transformed_state_space

    predicative_samples = {name: [] for name in model.observed_params.keys()}

    trace_length = len(next(iter(trace.values())))
    n_samples = trace_length if n_samples is None else n_samples
    if trace_length < n_samples:
        raise RuntimeError("n_samples must be less than the length of the trace or None.")

    _progress_bar = tqdm(range(n_samples), desc="Predicative sample") if progress_bar else range(n_samples)
    for i in _progress_bar:
        prior_values = {}
        for name, values in trace.items():
            prior_values[name] = values[i]
        
        for name, parameter in model.params.items():
            parameter.set_constrained_value(prior_values[name])

        for name, sampler in samplers.items():
            init_value = model.observed_params[name].observed_values[0]
            theta = _init_theta(state_spaces[name], init_value.shape, init_value.dtype)
            no_squeeze = theta.ndim == 1
            theta = theta if no_squeeze else theta.unsqueeze(0)
            for m in range(warmup_per_sample + 1):
                theta, _, _ = sampler.step(theta, m < warmup_per_sample)
            theta = theta if no_squeeze else theta.squeeze(0)
            predicative_samples[name].append(theta)

    for name, parameter in model.params.items():
        parameter.set_constrained_value(prior_values[name])
    
    predicative_samples = {name: model.observed_params[name].distribution.transform.inverse(torch.stack(samples)) for name, samples in predicative_samples.items()}
    return predicative_samples

def _init_theta(state_space, shape, dtype):
    if state_space.is_continuous():
        return torch.randn(shape, dtype=dtype)
    elif state_space.is_discrete():
        if hasattr(state_space, "values"):
            return state_space.values[0]
        else:
            return torch.ones(shape, dtype=dtype)

def _decide_predicative_step(parameter):
    state_space = parameter.distribution.transformed_state_space

    if state_space.is_continuous() and (parameter.sampler == "auto" or parameter.sampler == "nuts"):
        sampler = NUTS(parameter.distribution._log_prob_unconstrained, parameter.distribution._log_prob_grad_unconstrained, lambda x: x, **parameter.sampler_params)
    elif (state_space.is_discrete() or state_space.is_continuous()) and (parameter.sampler == "auto" or parameter.sampler == "metropolis"):
        sampler = Metropolis(parameter.distribution._log_prob_unconstrained, state_space, **parameter.sampler_params)
    else:
        raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
    
    sampler.init_sampler()
    return sampler
