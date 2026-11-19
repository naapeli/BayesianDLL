"""``sample(..., chain_method="vectorized")``: all chains in one process as one batched computation."""

import contextlib
import math

import torch
from tqdm import tqdm

from ._vectorized_samplers import VectorizedFiniteGibbs, VectorizedMetropolis, VectorizedNUTS, VectorizedSlice
from ._NUTS import NUTS
from ._Metropolis import Metropolis, FiniteGibbs
from ._Slice import Slice
from ._result import SampleStats


_NUMERICAL_ERRORS = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)


def _filtered(cls, sampler_params):
    from ._sample import _filter_sampler_params
    return _filter_sampler_params(cls, sampler_params)


class VectorizedBlock:
    """A Gibbs block of parameters moved for all chains of a batched model at once.

    The flat state of the block has shape ``(n_chains, total_dim)``. The
    block's target is its Markov blanket, as for :class:`SamplingBlock`.
    """

    def __init__(self, model, param_names, sampler_type="auto", sampler_params=None):
        from ._sample import _block_sampler_type, _block_state_space, _check_block_parameters
        self.model = model
        self.param_names = [param_names] if isinstance(param_names, str) else list(param_names)
        self.is_continuous = _check_block_parameters(model, self.param_names)
        self.n_chains = model.n_chains
        state_space, self.slices, self.total_dim = _block_state_space(model, self.param_names)
        self.state_space = state_space
        merged = {}
        for name in self.param_names:
            merged.update(model.params[name].sampler_params)
        merged.update(sampler_params or {})
        self.sampler_type = _block_sampler_type(model, self.param_names, sampler_type, state_space, self.is_continuous)
        self._random_terms, self._observed_terms, self._grad_nodes = model.block_terms(self.param_names)
        self._evaluated = False

        kind = self.sampler_type
        if kind == "nuts":
            if not self.is_continuous:
                raise RuntimeError("A distribution is incompatable with the chosen sampler. NUTS can only be used with continuous distributions.")
            self.sampler = VectorizedNUTS(self.log_target, self.gradient, self.n_chains, **_filtered(NUTS, merged))
            self.sampler.value_and_gradient = self.value_and_gradient
        elif kind == "metropolis":
            self.sampler = VectorizedMetropolis(self.log_target, state_space, self.n_chains, **_filtered(Metropolis, merged))
        elif kind == "slice":
            if not self.is_continuous:
                raise RuntimeError("The slice sampler can only be used with continuous distributions.")
            self.sampler = VectorizedSlice(self.log_target, state_space, self.n_chains, **_filtered(Slice, merged))
        elif kind == "gibbs":
            self.sampler = VectorizedFiniteGibbs(self.log_target, state_space, self.n_chains, **_filtered(FiniteGibbs, merged))
        else:
            raise RuntimeError(f"A distribution is incompatable with the chosen sampler: '{kind}'.")
        self._last_theta = None
        self._last_version = None
        self.last_stats = {}

    # ------------------------------------------------------------ packing
    def pack(self):
        values = [self.model.params[name].unconstrained_value.reshape(self.n_chains, -1) for name in self.param_names]
        return torch.cat(values, dim=1)

    def unpack_and_set(self, theta):
        for name, (start, end, shape) in self.slices.items():
            self.model.params[name].set_unconstrained_value(theta[:, start:end].reshape(self.n_chains, *shape))

    # ------------------------------------------------------------ targets
    def _guarded(self, function, chain_function, failure):
        """Evaluate batched; when that fails numerically, evaluate chain by chain (failing chains get ``failure``)."""
        if not self._evaluated:
            value = function()
            self._evaluated = True
            return value
        try:
            return function()
        except _NUMERICAL_ERRORS:
            def guarded():
                try:
                    return chain_function()
                except _NUMERICAL_ERRORS:
                    return failure()
            mode = self.model._batch_mode
            self.model._batch_mode = "loop"
            try:
                return self.model._loop_over_chains(guarded)
            finally:
                self.model._batch_mode = mode

    def value_and_gradient(self, theta):
        """Log target ``(C,)`` and gradient ``(C, d)`` with a single unpacking of ``theta``."""
        self.unpack_and_set(theta)
        return self.log_target(theta, unpack=False), self.gradient(theta, unpack=False)

    def log_target(self, theta, unpack=True):
        if unpack:
            self.unpack_and_set(theta)
        model = self.model

        def evaluate():
            return model._log_prob_of_terms(self._random_terms, self._observed_terms)

        def failure():
            return torch.tensor(-math.inf, dtype=theta.dtype, device=theta.device)

        value = self._guarded(evaluate, evaluate, failure)
        return torch.as_tensor(value, dtype=theta.dtype, device=theta.device).expand(self.n_chains)

    def gradient(self, theta, unpack=True):
        if unpack:
            self.unpack_and_set(theta)
        model = self.model

        def evaluate():
            grads = model.joint_grad_log_prob(self.param_names, nodes=self._grad_nodes)
            leading = grads[self.param_names[0]].shape[0] if model.n_chains is not None else None
            if leading is None:
                return torch.cat([grads[name].reshape(-1) for name in self.param_names])
            return torch.cat([grads[name].reshape(model.n_chains, -1) for name in self.param_names], dim=1)

        def failure():
            return torch.full((self.total_dim,), math.nan, dtype=theta.dtype, device=theta.device)

        return self._guarded(evaluate, evaluate, failure)

    # ------------------------------------------------------------ stepping
    def step(self, warmup):
        version = getattr(self.model, "_sampling_version", 0)
        theta = self.pack()
        if self._last_theta is not None and self._last_version == version and torch.equal(theta, self._last_theta):
            theta = self._last_theta
        new_theta, step_size, accept, diverging = self.sampler.step(theta, warmup=warmup)
        self.unpack_and_set(new_theta)
        if new_theta is not theta:
            version += 1
            self.model._sampling_version = version
        self._last_theta = new_theta
        self._last_version = version
        self.last_stats = self.sampler.last_stats
        return step_size, accept, diverging


def _build_vectorized_blocks(model, blocks_spec, sampler_params):
    from ._sample import SamplingBlock, _default_block_specs
    if blocks_spec is None:
        return [VectorizedBlock(model, names, kind, sampler_params) for names, kind in _default_block_specs(model)]
    built, covered = [], set()
    for spec in blocks_spec:
        if isinstance(spec, SamplingBlock):
            block = VectorizedBlock(model, spec.param_names, spec.sampler_type, {**(sampler_params or {}), **spec.sampler_params})
        elif isinstance(spec, (list, tuple)):
            block = VectorizedBlock(model, list(spec), sampler_params=sampler_params)
        elif isinstance(spec, str):
            block = VectorizedBlock(model, [spec], sampler_params=sampler_params)
        else:
            raise TypeError(f"Invalid block specification: {spec}")
        for name in block.param_names:
            if name in covered:
                raise ValueError(f"Parameter '{name}' appears in multiple blocks.")
            covered.add(name)
        built.append(block)
    missing = set(model.params) - covered
    if missing:
        raise ValueError(f"The following parameters were not included in any block: {missing}")
    return built


def _rng_context(seed, device):
    if seed is None:
        return contextlib.nullcontext()
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devices)


def _chain_value(value, n_chains):
    """A recorded value with a leading chain dimension."""
    value = torch.as_tensor(value)
    if value.ndim == 0 or value.shape[0] != n_chains:
        return value.unsqueeze(0).expand(n_chains, *value.shape)
    return value


def sample_vectorized(n_samples, warmup_length, n_chains, model, progress_bar, init_scale, blocks, seed,
                      log_likelihood_plan, sampler_params, trace_device="cpu"):
    """Run ``n_chains`` chains of the batched model in lock step.

    Returns the pieces :func:`~BayesianDLL.sample` assembles into a
    :class:`SamplingResult`: ``(trace, det_trace, enum_probs, log_lik,
    divergences, acceptance, step_sizes, sample_stats, max_depths)``, where
    ``trace`` also holds the draws of the enumerated parameters, the tensors
    have shape ``(n_chains, n_samples, ...)``, ``log_lik`` is ``None`` without
    a plan, and ``divergences``, ``acceptance`` and ``step_sizes`` are lists
    per chain (the latter two with one entry per block).
    """
    from ..Evaluation._pointwise import pointwise_log_likelihood

    device = model.device
    with _rng_context(seed, device):
        if seed is not None:
            torch.manual_seed(int(seed))
        initial = {}
        for name, parameter in model.params.items():
            value = parameter.unconstrained_value.unsqueeze(0).expand(n_chains, *parameter.unconstrained_value.shape).clone()
            if parameter.distribution.state_space.is_continuous():
                value = value + init_scale * torch.randn_like(value)
            initial[name] = value

        with model.batched(n_chains, values=initial):
            blocks = _build_vectorized_blocks(model, blocks, sampler_params)
            for block in blocks:
                if hasattr(block.sampler, "set_warmup_length"):
                    block.sampler.set_warmup_length(warmup_length)
            return _run(model, blocks, n_samples, warmup_length, n_chains, progress_bar, log_likelihood_plan,
                        trace_device, pointwise_log_likelihood)


def _run(model, blocks, n_samples, warmup_length, n_chains, progress_bar, log_likelihood_plan, trace_device, pointwise_log_likelihood):
    device = model.device
    dtype = model.dtype
    total = n_samples + warmup_length
    trace = {
        name: torch.empty((n_chains, n_samples, *p.constrained_value.shape[1:]), dtype=p.constrained_value.dtype, device=device)
        for name, p in model.params.items()
    }
    det_trace, enum_trace, enum_probs, log_lik = {}, {}, {}, {}
    acceptance = torch.zeros((n_chains, len(blocks)), dtype=dtype, device=device)
    step_sizes = torch.ones((n_chains, len(blocks)), dtype=dtype, device=device)
    divergences = torch.zeros(n_chains, dtype=torch.long, device=device)
    stats = [{} for _ in blocks]

    def store(target, name, index, value):
        value = _chain_value(value, n_chains)
        if name not in target:
            target[name] = torch.empty((n_chains, n_samples, *value.shape[1:]), dtype=value.dtype, device=value.device)
        target[name][:, index] = value

    bar = tqdm(total=total, disable=not progress_bar, desc=f"{n_chains} chains (vectorized) {'warmup' if warmup_length > 0 else 'sample'}",
               bar_format=r"{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}{postfix}")
    try:
        for m in range(1, total + 1):
            warmup = m <= warmup_length
            for i, block in enumerate(blocks):
                step_size, accept, diverging = block.step(warmup=warmup)
                if not warmup:
                    divergences = divergences + torch.as_tensor(diverging, device=device).to(torch.long)
                step_sizes[:, i] = torch.as_tensor(step_size, dtype=dtype, device=device)
                acceptance[:, i] = acceptance[:, i] + torch.as_tensor(accept, dtype=dtype, device=device)
                if m > warmup_length:
                    for stat, value in block.last_stats.items():
                        stats[i].setdefault(stat, []).append(torch.as_tensor(value, device=device).to(dtype).expand(n_chains))
            if m > warmup_length:
                index = m - warmup_length - 1
                if model.enumerated_params:
                    for name, (value, probabilities) in model.sample_enumerated().items():
                        store(enum_trace, name, index, value)
                        store(enum_probs, name, index, probabilities)
                for name, parameter in model.params.items():
                    trace[name][:, index] = parameter.constrained_value
                for name, parameter in model.deterministic_params.items():
                    store(det_trace, name, index, parameter.constrained_value)
                if log_likelihood_plan is not None:
                    for name, values in pointwise_log_likelihood(model, log_likelihood_plan).items():
                        store(log_lik, name, index, values)
            if progress_bar and (m % 10 == 0 or m == total):
                bar.set_description(f"{n_chains} chains (vectorized) {'warmup' if warmup else 'sample'}", refresh=False)
                bar.set_postfix({
                    "avg. acc. probs": [f"{float(a):.3f}" for a in (acceptance.mean(0) / m).tolist()],
                    "step sizes": [f"{float(s):.3f}" for s in step_sizes.mean(0).tolist()],
                    "divs": int(divergences.sum()),
                }, refresh=False)
            bar.update(1)
    finally:
        bar.close()

    def out(tensor):
        return tensor if trace_device is None else tensor.to(trace_device)

    trace = {name: out(value) for name, value in trace.items()}
    trace.update({name: out(value) for name, value in enum_trace.items()})
    sample_stats = SampleStats([
        {
            "params": list(block.param_names),
            "sampler": block.sampler_type,
            "stats": {name: torch.stack(values, dim=1).to(device="cpu", dtype=torch.float64) for name, values in stats[i].items()},
        }
        for i, block in enumerate(blocks)
    ])
    acceptance_list = (acceptance / total).cpu().tolist()
    step_size_list = step_sizes.cpu().tolist()
    max_depths = [getattr(block.sampler, "max_depth", None) for block in blocks]
    return (
        trace,
        {name: out(value) for name, value in det_trace.items()},
        {name: out(value) for name, value in enum_probs.items()},
        {name: out(value) for name, value in log_lik.items()} if log_likelihood_plan is not None else None,
        [int(d) for d in divergences.cpu().tolist()],
        acceptance_list,
        step_size_list,
        sample_stats,
        max_depths,
    )
