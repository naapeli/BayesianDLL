import torch
from tqdm import tqdm
from functools import partial

from . import NUTS, Metropolis
from .._active_model import _active_model


def sample(n_samples, warmup_length, model=None):
    model = _active_model._active_model if model is None else model

    samplers = {}
    for name, parameter in model.params.items():
        samplers[name] = _decide_step(model, parameter)
    
    trace = {name: [] for name in model.params.keys()}
    
    progress_bar = tqdm(range(1, n_samples + warmup_length + 1))
    for m in progress_bar:
        if m < warmup_length: progress_bar.set_description(f"Warmup")
        else: progress_bar.set_description(f"Sample")

        for name, sampler in samplers.items():
            theta = model.params[name].unconstrained_value
            new_theta = sampler.step(theta, m < warmup_length)
            model.params[name].set_unconstrained_value(new_theta)
            trace[name].append(model.params[name].constrained_value)

    trace = {name: torch.stack(samples[warmup_length:]) for name, samples in trace.items()}
    return trace

def _decide_step(model, parameter):
    _log_prob_func = partial(model.log_prob, parameter.name)

    state_space = parameter.distribution.transformed_state_space

    if state_space.is_continuous() and (parameter.sampler == "auto" or parameter.sampler == "nuts"):
        sampler = NUTS(_log_prob_func, partial(model.grad_log_prob, parameter.name), lambda x: x)
    elif (state_space.is_discrete() or state_space.is_continuous()) and (parameter.sampler == "auto" or parameter.sampler == "metropolis"):
        sampler = Metropolis(_log_prob_func, state_space)
    else:
        raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
    
    sampler.init_sampler()
    return sampler

def sample_posterior_predicative(trace, n_samples=None, model=None):
    pass  # see https://chatgpt.com/c/6813ce89-1794-8011-9641-cdc807f30241
    # model = _active_model._active_model if model is None else model
    # trace_length = len(next(iter(trace.items()))[1])
    # n_samples = trace_length if n_samples is None else n_samples

    # if n_samples > trace_length: raise RuntimeError("The trace should be at least as long as the amount of posterior draws.")

    # posterior_predicative = {}
    # for 
