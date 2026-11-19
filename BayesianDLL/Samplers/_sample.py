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
from ._Slice import Slice
from ._result import SamplingResult, PredicativeResult, SampleStats
from .._active_model._active_model import get_active_model
from ..Evaluation import gelman_rubin
from ..Evaluation._bfmi import bfmi
from ..Evaluation._pointwise import pointwise_log_likelihood_plan, pointwise_log_likelihood
from ..Distributions._state_space import JointStateSpace


_SAMPLER_CLASSES = {"nuts": NUTS, "metropolis": Metropolis, "gibbs": FiniteGibbs, "slice": Slice}
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
    elif sampler == "slice":
        if not is_continuous:
            raise RuntimeError("A distribution is incompatable with the chosen sampler. The slice sampler can only be used with continuous distributions.")
        instance = Slice(log_target, state_space, **_filter_sampler_params(Slice, sampler_params))
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
    if isinstance(sampler, Slice):
        return "slice"
    return type(sampler).__name__


def _block_sampler_type(model, param_names, sampler_type, state_space, is_continuous):
    """The sampler of a block: ``sampler_type`` or, for ``"auto"``, the one its parameters request."""
    if sampler_type != "auto":
        return sampler_type
    requested = [model.params[name].sampler for name in param_names]
    if is_continuous:
        if "metropolis" in requested:
            return "metropolis"
        if "slice" in requested:
            return "slice"
        return "nuts"
    if "slice" in requested:
        raise ValueError(f"The slice sampler can only be used with continuous parameters: {param_names}")
    if "metropolis" in requested or not has_finite_support(state_space):
        return "metropolis"
    return "gibbs"


def _block_state_space(model, param_names):
    """The (joint) unconstrained state space of a block and the flat slices of its parameters."""
    slices = {}
    offset = 0
    for name in param_names:
        value = model.params[name].unconstrained_value
        shape = value.shape[1:] if getattr(model, "n_chains", None) is not None else value.shape
        numel = shape.numel()
        slices[name] = (offset, offset + numel, shape)
        offset += numel
    if len(param_names) == 1:
        space = model.params[param_names[0]].distribution.transformed_state_space
    else:
        spaces = {name: model.params[name].distribution.transformed_state_space for name in param_names}
        space = JointStateSpace(spaces, slices)
    return space, slices, offset


def _check_block_parameters(model, param_names):
    for name in param_names:
        if name in getattr(model, "enumerated_params", {}):
            raise ValueError(f"Parameter '{name}' is enumerated (summed out of the model) and cannot be part of a sampling block.")
        if name not in model.params:
            raise KeyError(f"Parameter '{name}' not found in model parameters.")
    is_continuous_list = [model.params[name].distribution.transformed_state_space.is_continuous() for name in param_names]
    if any(is_continuous_list) and not all(is_continuous_list):
        raise ValueError(f"Continuous and discrete variables cannot be grouped in the same block: {param_names}")
    return all(is_continuous_list)


def _default_block_specs(model):
    """Default Gibbs blocks: ``[(param_names, sampler_type), ...]``."""
    nuts_names = []
    slice_names = []
    continuous_metropolis_names = []
    gibbs_names = []
    discrete_metropolis_names = []
    for name, p in model.params.items():
        space = p.distribution.transformed_state_space
        requested = p.sampler
        if space.is_continuous():
            if requested in ("auto", "nuts"):
                nuts_names.append(name)
            elif requested == "slice":
                slice_names.append(name)
            else:
                continuous_metropolis_names.append(name)
        elif requested == "slice":
            raise ValueError(f"The slice sampler can only be used with continuous parameters, but '{name}' is discrete.")
        elif requested in ("auto", "gibbs") and has_finite_support(space):
            gibbs_names.append(name)
        else:
            discrete_metropolis_names.append(name)
    specs = [
        (nuts_names, "nuts"), (slice_names, "slice"), (continuous_metropolis_names, "metropolis"),
        (gibbs_names, "gibbs"), (discrete_metropolis_names, "metropolis"),
    ]
    return [(names, kind) for names, kind in specs if names]


class SamplingBlock:
    """A group of parameters updated jointly by one sampler in a Gibbs scheme.

    The block's target is the part of the joint log density that depends on
    its parameters (their priors, the priors of their random children and the
    likelihoods of their observed children); the remaining terms are constant
    while the block moves and are not evaluated. Points where the model fails
    numerically (after its first successful evaluation) get density zero.

    Parameters
    ----------
    model : Model
        The compiled model.
    param_names : str or list of str
        The random parameters of the block, all continuous or all discrete.
        Enumerated parameters cannot be sampled.
    sampler_type : {"auto", "nuts", "metropolis", "slice", "gibbs"}, default "auto"
        The block's sampler. ``"auto"`` takes the sampler the parameters
        request (``"metropolis"`` before ``"slice"``), else NUTS for continuous
        blocks and the finite Gibbs sampler for discrete blocks with finite
        support (Metropolis otherwise).
    sampler_params : dict, optional
        Sampler settings; they override the parameters' own ``sampler_params``.
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

        self.sampler_type = _block_sampler_type(model, self.param_names, sampler_type, state_space, self.is_continuous)
        sampler_type = self.sampler_type

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

    return [
        SamplingBlock(model, names, sampler_type=kind, sampler_params=sampler_params)
        for names, kind in _default_block_specs(model)
    ]


def _chain_seeds(seed, n_chains):
    """Independent, reproducible per-chain seeds derived from one integer seed."""
    sequences = np.random.SeedSequence(int(seed)).spawn(n_chains)
    return [int(sequence.generate_state(1, dtype=np.uint64)[0] % (2 ** 63)) for sequence in sequences]


def sample(n_samples, warmup_length, n_chains=4, model=None, progress_bar=True, init_scale=1, blocks=None, check_convergence=True, seed=None, log_likelihood=True, chain_method=None, device=None, trace_device="cpu", **sampler_params):
    """Draw posterior samples from ``n_chains`` MCMC chains.

    The random parameters are updated in Gibbs blocks (by default one NUTS
    block for the continuous parameters, one finite Gibbs block for the
    discrete parameters with finite support, and Metropolis or slice blocks
    for the parameters that request them; see :class:`SamplingBlock`), each
    block's sampler targeting the part of the log density that depends on it.
    Every iteration ``m = 1, ..., warmup_length + n_samples`` steps every
    block once. The ``warmup_length`` iterations ``m <= warmup_length`` adapt
    the samplers and are not recorded; the ``n_samples`` iterations
    ``m > warmup_length`` do not adapt and are recorded. Divergences and the
    sample statistics cover exactly the recorded iterations.

    Parameters
    ----------
    n_samples : int
        Number of recorded draws per chain.
    warmup_length : int
        Number of warmup (adaptation) iterations per chain, see above.
    n_chains : int, default 4
        Number of chains.
    model : Model, optional
        The model; the active model (inside ``with model:``) by default.
    progress_bar : bool, default True
        Show progress bars (one per chain in parallel mode).
    init_scale : float, default 1
        Standard deviation of the Gaussian jitter added to the initial
        unconstrained values of each chain (the model's current values), for
        the continuous parameters; discrete parameters start at their current
        values. The ``init_scale`` of a Gaussian guide group
        (:class:`~BayesianDLL.Guide`, :class:`~BayesianDLL.Group`) is likewise
        a standard deviation around the current unconstrained values: the
        initial scale of the variational distribution.
    blocks : list, optional
        The Gibbs blocks: each entry is a parameter name, a list of names, or
        a :class:`SamplingBlock`. Every random parameter must be in exactly
        one block. Default: the blocks described above.
    check_convergence : bool, default True
        Warn when the rank-normalised R-hat of a parameter exceeds 1.01
        (needs ``n_chains > 1`` and ``n_samples >= 50``).
    seed : int, optional
        See below.
    log_likelihood : bool, default True
        See below.
    chain_method : {"parallel", "vectorized"}, optional
        See below.
    device : str or torch.device, optional
        See below.
    trace_device : str or torch.device or None, default "cpu"
        See below.
    **sampler_params
        Sampler settings for every block, see below.

    Returns
    -------
    SamplingResult
        Traces of shape ``(n_chains, n_samples, *parameter_shape)`` (constrained
        values) of the random, enumerated and deterministic parameters, the
        sampler statistics, divergences and the pointwise log-likelihood.

    After sampling, warnings are issued for R-hat above 1.01, for divergences
    after warmup, for an E-BFMI below 0.3 and when more than 5% of the NUTS
    iterations reach the maximal tree depth.

    ``chain_method`` selects how the chains run (the names follow NumPyro):

    * ``"parallel"``: one worker process per chain. Every process evaluates
      the model for its own chain. This is the default for a model on the CPU.
    * ``"vectorized"``: all chains run in this process in lock step. The
      model is batched (see :meth:`~BayesianDLL.Model.batch`): every latent
      parameter gets a leading chain dimension and one log density or
      gradient evaluation covers all chains. NUTS, Metropolis, the finite
      Gibbs sampler and the slice sampler have vectorized versions with
      per-chain step sizes, metrics, adaptation and accept/reject decisions,
      and Gibbs blocks work as in the parallel mode. This is much faster per
      chain for many chains, and it is the only mode on a GPU, because worker
      processes and CUDA do not mix. The GP nodes are batched over the
      chains as well; nodes that cannot be batched (user functions that do
      not broadcast over a leading chain dimension) are evaluated chain by
      chain inside the batched computation.
    * ``None`` (default): ``"vectorized"`` when the model lives on a CUDA (or
      ROCm) device, ``"parallel"`` otherwise.

    ``device`` moves the model to that device first (``Model.to(device)``,
    in place); e.g. ``device="cuda"`` samples on the GPU. On AMD GPUs the
    ROCm build of PyTorch exposes the GPU as device ``"cuda"``. In
    vectorized mode the traces are kept on the model's device during
    sampling and moved to ``trace_device`` at the end (``None`` keeps them on
    the model's device). Both modes return the same :class:`SamplingResult`.

    ``seed`` makes runs reproducible. In parallel mode every chain gets its
    own seed derived from it; in vectorized mode it seeds torch's generator
    of the model's device for the run (the global random state is restored
    afterwards). With ``seed=None`` the parallel base seed is drawn from
    torch's global generator and the vectorized run uses the global
    generator, so a preceding ``torch.manual_seed`` also makes runs
    reproducible.

    ``log_likelihood=True`` records the elementwise log-likelihood
    ``log p(y_i | theta)`` of every observed node for every kept draw in
    ``result.log_likelihood[name]`` (shape ``(n_chains, n_samples,
    *log_density_shape)``), which :func:`~BayesianDLL.Evaluation.loo` and
    :func:`~BayesianDLL.Evaluation.waic` use. Pass ``False`` to save the
    memory and time.

    Remaining keyword arguments are sampler settings for every block (each
    sampler takes the ones it understands), e.g. ``delta=0.9`` or
    ``metric="dense"`` for NUTS, ``target_accept_rate=0.44`` for Metropolis
    and ``width=0.5`` for the slice sampler; a setting no sampler understands
    raises a ``TypeError``. Settings given to a parameter
    (``sampler_params``) apply to its block and are overridden by these.
    """
    model = get_active_model() if model is None else model
    if device is not None:
        model.to(device)
    on_gpu = model.device.type != "cpu"
    if chain_method is None:
        chain_method = "vectorized" if on_gpu else "parallel"
    if chain_method not in ("parallel", "vectorized"):
        raise ValueError(f"chain_method must be 'parallel' or 'vectorized', got {chain_method!r}.")
    if chain_method == "parallel" and on_gpu:
        raise ValueError(
            f"chain_method='parallel' runs every chain in a worker process, which does not work with a model on "
            f"{model.device}. Use chain_method='vectorized'."
        )
    # Decide how the enumerated parameters are summed out before starting the
    # workers, so a model that cannot be enumerated fails here.
    model.prepare_enumeration()
    if log_likelihood:
        log_likelihood_plan, log_likelihood_unavailable = pointwise_log_likelihood_plan(model)
    else:
        log_likelihood_plan, log_likelihood_unavailable = None, {}

    if chain_method == "vectorized":
        from ._vectorized import sample_vectorized
        (trace, det_trace, enumerated_probs, log_lik, divergences, acceptance_probabilities, step_sizes,
         sample_stats, max_depths) = sample_vectorized(
            n_samples, warmup_length, n_chains, model, progress_bar, init_scale, blocks, seed,
            log_likelihood_plan, sampler_params, trace_device=trace_device,
        )
        return _finish_sampling(model, n_samples, n_chains, check_convergence, trace, det_trace, enumerated_probs,
                                log_lik, divergences, acceptance_probabilities, step_sizes, sample_stats, max_depths,
                                log_likelihood_unavailable)

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
            bar.set_description(f"Chain {chain + 1}/{n_chains} {'warmup' if warmup_length > 0 else 'sample'}", refresh=False)
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
            [init_scale]*n_chains,
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
    max_depths = results[0][6] if results else []
    return _finish_sampling(model, n_samples, n_chains, check_convergence, trace, det_trace, enumerated_probs,
                            log_lik, divergences, acceptance_probabilities, step_sizes, sample_stats, max_depths,
                            log_likelihood_unavailable)


def _finish_sampling(model, n_samples, n_chains, check_convergence, trace, det_trace, enumerated_probs, log_lik,
                     divergences, acceptance_probabilities, step_sizes, sample_stats, max_depths, log_likelihood_unavailable):
    """Convergence diagnostics and the :class:`SamplingResult` of both chain methods."""
    if check_convergence and n_chains > 1:
        if n_samples >= 50:
            r_hats = gelman_rubin({name: trace[name] for name in model.params})
            for name, statistics in r_hats.items():
                # NaN (every chain the same constant) is not a convergence
                # failure; inf (chains stuck at different values) is.
                if torch.any(statistics > 1.01):
                    stuck = " An infinite R-hat means that chains are stuck at different values." if torch.any(torch.isinf(statistics)) else ""
                    warn(f"The Gelman–Rubin R-hat of {name} is above 1.01 ({torch.round(statistics, decimals=3).tolist()}) and indicates poor convergence.{stuck} Consider increasing the amount of warmup steps or reparametrizing the model.")
    elif check_convergence and n_chains == 1:
        warn(f"The convergence of the chain is not checked when n_chains == 1. Increase it to at least 2 to enable convergence diagnostics.")

    if sum(divergences) > 0:
        warn(f"There were {sum(divergences)} divergences across all chains after tuning. Increase target acceptance probability or reparameterize the model.")

    _check_hamiltonian_diagnostics(sample_stats, max_depths, n_samples)

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


def _sample_single_chain(chain, n_chains, model, initial_values, init_scale, n_samples, warmup_length, progress_bar, blocks_spec=None, shm_name=None, sampler_params=None, seed=None, log_likelihood_plan=None):
    torch.set_num_threads(1)
    if seed is not None:
        # Worker processes may be fresh or reused; seeding here makes each chain
        # reproducible and independent of the worker it runs in.
        torch.manual_seed(seed)
    for name, parameter in model.params.items():
        value = initial_values[name].clone()
        if parameter.distribution.state_space.is_continuous():
            value = value + init_scale * torch.randn_like(value)
        parameter.set_unconstrained_value(value)

    blocks = _build_blocks(model, blocks_spec, sampler_params)
    for block in blocks:
        if hasattr(block.sampler, "set_warmup_length"):
            # Warmup steps are the iterations m <= warmup_length (m starts at 1).
            block.sampler.set_warmup_length(warmup_length)

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

    acceptance_probabilities = [0.0 for _ in range(len(blocks))]
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
        warmup = m <= warmup_length
        if _progress_bar is not None:
            if warmup: _progress_bar.set_description(f"Chain warmup", refresh=False)
            else: _progress_bar.set_description(f"Chain sample", refresh=False)
            # The accumulators hold the m - 1 iterations done so far.
            _progress_bar.set_postfix({
                "avg. acc. probs": [f"{prob / max(m - 1, 1):.3f}" for prob in acceptance_probabilities],
                "step sizes": [f"{step_size:.3f}" for step_size in step_sizes],
                "divs": divergences_count
            }, refresh=False)

        for i, block in enumerate(blocks):
            step_size, acceptance_probability, diverging = block.step(warmup=warmup)
            if diverging and not warmup:
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

def sample_prior_predicative(n_samples=20, samples_per_step=500, warmup_per_sample=100, model=None, progress_bar=True):
    """Draw from the prior predictive distribution by ancestral sampling.

    For each of the ``n_samples`` prior draws, the random parameters are
    sampled from their priors in topological order (no MCMC) and
    ``samples_per_step`` data sets are generated from the likelihoods given
    those parameters.

    Returns a :class:`PredicativeResult` with samples of shape
    ``(n_samples, samples_per_step, *predictive_shape)``. Its ``prior``
    attribute is a :class:`SamplingResult` (one chain of ``n_samples`` draws)
    with the prior draws of the random and deterministic parameters, so
    ``summary`` and the posterior plots work on it.

    ``warmup_per_sample`` is only used by the MCMC fallback for distributions
    that cannot be sampled directly. ``model`` defaults to the active model;
    ``progress_bar`` shows a progress bar over the draws. A warning is issued
    when the prior of a parameter depends on an observed node, because its
    draws then condition on the observed data.
    """
    model = get_active_model() if model is None else model
    return _ancestral_predictive(model, n_samples, samples_per_step, None, progress_bar, warmup_per_sample)

def posterior_predicative(trace, n_samples=None, samples_per_step=20, warmup_per_sample=20, model=None, progress_bar=True):
    """Posterior predictive samples by exact ancestral sampling.

    ``trace`` is a :class:`SamplingResult` (or a dict of constrained traces of
    shape ``(n_chains, n_samples, ...)``) holding every random parameter of
    ``model`` (the active model by default). ``n_samples`` evenly spaced
    draws are taken from the chains concatenated (all draws when ``None``;
    more than the trace holds raises a ``RuntimeError``). For each draw the model parameters are set to it, the graph is
    walked in topological order and every observed node is sampled
    ``samples_per_step`` times with its distribution's ``sample`` method,
    shaped like the node's ``predictive_shape`` (plate sizes, including
    plates sized by mutable ``Data``, are resolved at call time). When an
    observed node is a parent of another observed node, the child conditions
    on the *replicated* value of the parent.

    A likelihood without a sampler (``sample`` raising ``NotImplementedError``
    or missing) falls back to MCMC for that node only, with a warning; the
    fallback uses ``warmup_per_sample`` warmup steps.

    Returns a :class:`PredicativeResult` of tensors with shape
    ``(n_samples, samples_per_step, *predictive_shape)``.
    """
    model = get_active_model() if model is None else model

    n_chains, trace_length = next(iter(trace.values())).shape[:2]
    flattened_trace = {name: values.flatten(0, 1) for name, values in trace.items()}
    total_samples = n_chains * trace_length

    if n_samples is None:
        n_samples = total_samples
        indices = torch.arange(total_samples)
    else:
        if total_samples < n_samples:
            raise RuntimeError(f"n_samples ({n_samples}) must be less than or equal to the total trace length ({total_samples}).")
        # Integer positions computed on the host (the traces live on the CPU by default).
        indices = torch.linspace(0, total_samples - 1, steps=n_samples, dtype=torch.float64, device="cpu").long()

    missing = [name for name in model.params if name not in flattened_trace]
    if missing:
        raise KeyError(f"The trace has no draws of the random parameter(s) {missing}.")

    device = model.device

    def parameter_values(i):
        return {
            name: parameter.distribution.transform.forward(flattened_trace[name][indices[i]].to(device))
            for name, parameter in {**model.params, **model.enumerated_params}.items()
            if name in flattened_trace
        }

    return _ancestral_predictive(model, n_samples, samples_per_step, parameter_values, progress_bar, warmup_per_sample)


def _observed_parents(model):
    """For each observed node, the observed nodes whose value it reads (directly or through deterministic nodes)."""
    parents = {}
    for name in model.observed_params:
        found = set()
        seen = set()
        stack = list(model.graph.predecessors(name))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in model.observed_params:
                found.add(node)
            elif node in model.deterministic_params:
                stack.extend(model.graph.predecessors(node))
        parents[name] = found
    return parents


class _MCMCFallback:
    """MCMC draws for a node whose distribution cannot be sampled directly."""

    def __init__(self, parameter, warmup):
        self.parameter = parameter
        self.warmup = warmup
        self.sampler = _decide_predicative_step(parameter)
        self.state_space = parameter.distribution.transformed_state_space

    def draw(self, n, shape, dtype):
        sampler = self.sampler
        sampler.reset()
        if hasattr(sampler, "set_warmup_length"):
            sampler.set_warmup_length(self.warmup)
        transform = self.parameter.distribution.transform
        device = torch.as_tensor(self.parameter.observed_values).device
        theta = _init_theta(self.state_space, transform.unconstrained_shape(tuple(shape)), dtype, device)
        draws = []
        for m in range(self.warmup + n):
            theta, _, _, _ = sampler.step(theta, m < self.warmup)
            if m >= self.warmup:
                draws.append(transform.inverse(theta))
        return torch.stack(draws, dim=0)


def _ancestral_predictive(model, n_draws, samples_per_step, parameter_values, progress_bar, warmup_per_sample):
    """Prior (``parameter_values is None``) or posterior predictive draws by ancestral sampling.

    ``parameter_values(i)`` returns the unconstrained parameter values of the
    ``i``-th posterior draw. In prior mode the random parameters are drawn
    from their priors in topological order instead.
    """
    import networkx as nx
    from ._ancestral import draw_shaped, can_sample

    if not model._compiled:
        model.compile()
    prior = parameter_values is None
    latents = {**model.params, **getattr(model, "enumerated_params", {})}
    random_order = [latents[n] for n in model._topo_order if n in latents]
    observed_order = [model.observed_params[n] for n in model._topo_order if n in model.observed_params]
    observed_parents = _observed_parents(model)
    replicated_parents = set().union(set(), *observed_parents.values())
    chained = bool(replicated_parents)

    if prior:
        for parameter in random_order:
            if any(node in model.observed_params for node in nx.ancestors(model.graph, parameter.name)):
                warn(f"The prior of '{parameter.name}' depends on an observed node; its prior predictive draws condition on the observed data.")

    fallbacks = {}

    def draw(parameter, n, shape, dtype):
        if parameter.name not in fallbacks and can_sample(parameter.distribution):
            try:
                return draw_shaped(parameter.distribution, n, shape).to(dtype)
            except NotImplementedError:
                pass
        if parameter.name not in fallbacks:
            warn(
                f"{type(parameter.distribution).__name__} of '{parameter.name}' cannot be sampled directly; "
                f"falling back to MCMC for this node ({warmup_per_sample} warmup steps per draw)."
            )
            fallbacks[parameter.name] = _MCMCFallback(parameter, warmup_per_sample)
        return fallbacks[parameter.name].draw(n, shape, dtype).to(dtype)

    predicative_samples = {
        # Predictive draws are collected on the CPU, like the traces of sample().
        parameter.name: torch.empty(size=(n_draws, samples_per_step, *parameter.predictive_shape), dtype=parameter.observed_values.dtype, device="cpu")
        for parameter in observed_order
    }
    prior_trace = {parameter.name: [] for parameter in random_order}
    prior_det_trace = {name: [] for name in model.deterministic_params}

    saved_values = {name: parameter.unconstrained_value for name, parameter in latents.items()}
    saved_observed = {parameter.name: parameter._observed_values for parameter in observed_order}

    _progress_bar = tqdm(
        range(n_draws),
        desc="Prior predicative sampling" if prior else "Predicative sampling",
        bar_format=r"{desc}: {percentage:3.0f}% | {bar} | {n_fmt}/{total} | {elapsed}<{remaining}> | {rate_fmt}"
    ) if progress_bar else range(n_draws)

    try:
        for i in _progress_bar:
            if prior:
                for parameter in random_order:
                    value = draw(parameter, 1, parameter.constrained_value.shape, parameter.constrained_value.dtype)[0]
                    parameter.set_constrained_value(value)
                    prior_trace[parameter.name].append(parameter.constrained_value.clone())
                for name, parameter in model.deterministic_params.items():
                    prior_det_trace[name].append(torch.as_tensor(parameter.constrained_value).clone())
            else:
                for name, value in parameter_values(i).items():
                    latents[name].set_unconstrained_value(value)

            if not chained:
                for parameter in observed_order:
                    samples = predicative_samples[parameter.name]
                    samples[i] = draw(parameter, samples_per_step, parameter.predictive_shape, samples.dtype)
                continue

            # An observed node feeds another node: draw the replicated data
            # sets one at a time, so that each child conditions on the
            # replicated value of its observed parents.
            try:
                for j in range(samples_per_step):
                    for parameter in observed_order:
                        samples = predicative_samples[parameter.name]
                        value = draw(parameter, 1, parameter.predictive_shape, samples.dtype)[0]
                        samples[i, j] = value
                        if parameter.name in replicated_parents:
                            parameter.observed_values = value
            finally:
                for parameter in observed_order:
                    parameter._observed_values = saved_observed[parameter.name]
    finally:
        for name, value in saved_values.items():
            latents[name].set_unconstrained_value(value)
        for parameter in observed_order:
            parameter._observed_values = saved_observed[parameter.name]

    result = PredicativeResult(predicative_samples)
    if prior:
        trace = {name: torch.stack(values, dim=0).unsqueeze(0) for name, values in prior_trace.items() if values}
        det_trace = {name: torch.stack(values, dim=0).unsqueeze(0) for name, values in prior_det_trace.items() if values}
        result.prior = SamplingResult(trace, [0], [[]], [[]], deterministic_trace=det_trace)
    return result

def _init_theta(state_space, shape, dtype, device=None):
    if state_space.is_continuous():
        return torch.randn(shape, dtype=dtype, device=device)
    elif state_space.is_discrete():
        first_value = next(iter(state_space))
        return first_value * torch.ones(shape, dtype=dtype, device=device)

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
