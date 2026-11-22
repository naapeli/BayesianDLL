import torch


def _normalize_slice(s):
    if isinstance(s, bool):
        raise TypeError("Thinning step must be an integer or slice, got bool.")
    if isinstance(s, int):
        if s < 1:
            raise ValueError(f"Thinning step must be an integer >= 1, got {s}.")
        return slice(None, None, s)
    elif isinstance(s, slice):
        return s
    else:
        raise TypeError(f"Expected int or slice for thinning, got {type(s).__name__}.")


class SampleStats:
    """Per-iteration sampler diagnostic statistics for post-warmup draws.

    Parameters
    ----------
    blocks : list of dict, optional
        List of block diagnostics dictionaries. Each dictionary contains:
        - ``"params"``: list of parameter names updated by this block.
        - ``"sampler"``: name of the sampler (e.g. ``"nuts"``, ``"metropolis"``,
          ``"slice"``, or ``"gibbs"``).
        - ``"stats"``: mapping from statistic name (e.g. ``"step_size"``,
          ``"accept_stat"``, ``"diverging"``, ``"energy"``, ``"lp"``) to
          a tensor of shape ``(n_chains, n_samples)``.

    Attributes
    ----------
    blocks : list of dict
        Internal list of sampling block dictionaries.
    params : list of list of str
        Parameter names updated by each block.
    samplers : list of str
        The sampler type for each block.

    Notes
    -----
    Indexing behavior (``sample_stats[key]``):

    * **Integer index** ``sample_stats[i]``:
      Returns the ``dict`` of statistics for block ``i``, where each value is
      a tensor of shape ``(n_chains, n_samples)``.
    * **String key** ``sample_stats["stat_name"]``:
      If a statistic is unique across blocks (such as ``"energy"`` or
      ``"tree_depth"`` in a NUTS block, or when there is only a single sampling
      block), indexing directly by its name returns the statistic tensor of
      shape ``(n_chains, n_samples)``.
      If multiple blocks record the same statistic (for instance ``"step_size"``
      or ``"accept_stat"`` in a multi-block model), direct string indexing raises
      a :class:`KeyError` instructing to index the desired block first, e.g.:
      ``sample_stats[0]["step_size"]``.
    """

    def __init__(self, blocks=None):
        self.blocks = [] if blocks is None else list(blocks)

    @property
    def params(self):
        return [block["params"] for block in self.blocks]

    @property
    def samplers(self):
        return [block["sampler"] for block in self.blocks]

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return (block["stats"] for block in self.blocks)

    def __getitem__(self, key):
        """Access sampler statistics by block index or unique statistic name.

        Parameters
        ----------
        key : int or str
            - If an integer ``i``: returns the statistics dictionary
              ``dict[str, Tensor]`` for the ``i``-th sampling block.
            - If a string ``name``: returns the tensor of shape
              ``(n_chains, n_samples)`` for that statistic, provided exactly
              one sampling block recorded it.

        Returns
        -------
        dict[str, Tensor] or Tensor
            The block dictionary if ``key`` is an ``int``, or the statistic
            tensor if ``key`` is a unique ``str``.

        Raises
        ------
        IndexError
            If an integer block index is out of bounds.
        KeyError
            If a string key is not recorded by any block, or if it is recorded
            by multiple blocks (in which case index by block first, e.g.
            ``sample_stats[0][key]``).
        """
        if isinstance(key, int):
            return self.blocks[key]["stats"]
        matches = [block["stats"][key] for block in self.blocks if key in block["stats"]]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"No sampling block records the statistic '{key}'.")
        raise KeyError(f"Several sampling blocks record '{key}'; index the block first, e.g. sample_stats[0]['{key}'].")

    def __contains__(self, key):
        return any(key in block["stats"] for block in self.blocks)

    def thin(self, step=1):
        """Statistics of every ``step``-th draw (an int >= 1 or a slice) of every chain."""
        s = _normalize_slice(step)
        return SampleStats([
            {**block, "stats": {name: value[:, s] for name, value in block["stats"].items()}}
            for block in self.blocks
        ])

    def __repr__(self):
        summary = [(block["sampler"], block["params"], list(block["stats"])) for block in self.blocks]
        return f"SampleStats({summary})"


class SamplingResult:
    """Draws and diagnostics of :func:`~BayesianDLL.sample` or :func:`~BayesianDLL.sample_smc`.

    Indexing (``result[name]``), ``keys``, ``values`` and ``items`` cover the
    random (and enumerated) parameters and the deterministic nodes.

    Attributes
    ----------
    trace : dict[str, Tensor]
        Constrained draws of the random and enumerated parameters, of shape
        ``(n_chains, n_samples, *parameter_shape)``.
    deterministic_trace : dict[str, Tensor]
        Draws of the deterministic nodes, of the same layout.
    divergences : list of int
        Number of divergent NUTS transitions after warmup, per chain.
    acceptance_probabilities : list of list of float
        Per chain and sampling block, the mean acceptance statistic over all
        iterations (warmup included).
    step_sizes : list of list of float
        Per chain and sampling block, the final step size (proposal scale or
        mean slice width for the other samplers).
    sample_stats : SampleStats
        Per-iteration sampler statistics of the recorded draws.
    enumerated_probs : dict[str, Tensor]
        For every enumerated (summed out) parameter, the posterior probability
        of each support value per coordinate and draw, of shape
        ``(n_chains, n_samples, *value_shape, K)``.
    log_likelihood : dict[str, Tensor]
        Elementwise ``log p(y_i | theta)`` of every observed node per draw, of
        shape ``(n_chains, n_samples, *log_density_shape)``.
    log_likelihood_unavailable : dict[str, str]
        Observed nodes whose pointwise log-likelihood could not be recorded,
        mapped to the reason.
    log_marginal_likelihood : float or None
        :func:`~BayesianDLL.sample_smc` only (``None`` otherwise): the
        estimate of ``log p(y)``.
    beta_schedule : list of float or None
        :func:`~BayesianDLL.sample_smc` only: the inverse temperatures,
        starting with 0.
    stage_ess : list of float or None
        :func:`~BayesianDLL.sample_smc` only: the effective sample size of
        every stage's incremental weights.
    stage_acceptance : list of float or None
        :func:`~BayesianDLL.sample_smc` only: the mean acceptance of every
        stage's mutation.
    kernel : str or None
        :func:`~BayesianDLL.sample_smc` only: the mutation kernel.
    """

    def __init__(self, trace, divergences, acceptance_probabilities, step_sizes, deterministic_trace=None, sample_stats=None, enumerated_probs=None, log_likelihood=None, log_likelihood_unavailable=None,
                 log_marginal_likelihood=None, beta_schedule=None, stage_ess=None, stage_acceptance=None, kernel=None):
        self.trace = trace
        self.divergences = divergences
        self.acceptance_probabilities = acceptance_probabilities
        self.step_sizes = step_sizes
        self.deterministic_trace = {} if deterministic_trace is None else deterministic_trace
        self.sample_stats = SampleStats() if sample_stats is None else sample_stats
        # For every enumerated (summed out) parameter: the posterior probability
        # of each support value per coordinate and draw, with shape
        # (n_chains, n_samples, *value_shape, K).
        self.enumerated_probs = {} if enumerated_probs is None else enumerated_probs
        # Elementwise log-likelihood log p(y_i | theta) of every observed node
        # per draw, with shape (n_chains, n_samples, *log_density_shape).
        self.log_likelihood = {} if log_likelihood is None else log_likelihood
        # Observed nodes whose pointwise log-likelihood could not be recorded,
        # mapped to the reason.
        self.log_likelihood_unavailable = {} if log_likelihood_unavailable is None else dict(log_likelihood_unavailable)
        # Sequential Monte Carlo results (None for MCMC).
        self.log_marginal_likelihood = log_marginal_likelihood
        self.beta_schedule = beta_schedule
        self.stage_ess = stage_ess
        self.stage_acceptance = stage_acceptance
        self.kernel = kernel

    def __getitem__(self, key):
        if key in self.trace:
            return self.trace[key]
        return self.deterministic_trace[key]

    def __contains__(self, key):
        return key in self.trace or key in self.deterministic_trace

    def keys(self):
        return {**self.trace, **self.deterministic_trace}.keys()

    def values(self):
        return {**self.trace, **self.deterministic_trace}.values()

    def items(self):
        return {**self.trace, **self.deterministic_trace}.items()

    def thin(self, step: int | slice = 1) -> "SamplingResult":
        """A new result keeping every ``step``-th draw (or the draws selected by a slice) of every chain.

        The traces, sample statistics, enumerated probabilities and
        log-likelihoods are thinned along the draw dimension; the
        divergences, acceptance probabilities and step sizes are copied
        unchanged, as are the attributes of :func:`~BayesianDLL.sample_smc`
        (``log_marginal_likelihood``, ``beta_schedule``, ``stage_ess``,
        ``stage_acceptance``, ``kernel``), which describe the run, not the draws.
        """
        s = _normalize_slice(step)
        thinned_trace = {name: val[:, s] for name, val in self.trace.items()}
        thinned_det = {name: val[:, s] for name, val in self.deterministic_trace.items()}
        return SamplingResult(
            trace=thinned_trace,
            divergences=list(self.divergences),
            acceptance_probabilities=[list(ap) for ap in self.acceptance_probabilities],
            step_sizes=[list(ss) for ss in self.step_sizes],
            deterministic_trace=thinned_det,
            sample_stats=self.sample_stats.thin(step),
            enumerated_probs={name: val[:, s] for name, val in self.enumerated_probs.items()},
            log_likelihood={name: val[:, s] for name, val in self.log_likelihood.items()},
            log_likelihood_unavailable=self.log_likelihood_unavailable,
            log_marginal_likelihood=self.log_marginal_likelihood,
            beta_schedule=None if self.beta_schedule is None else list(self.beta_schedule),
            stage_ess=None if self.stage_ess is None else list(self.stage_ess),
            stage_acceptance=None if self.stage_acceptance is None else list(self.stage_acceptance),
            kernel=self.kernel,
        )

    def summary(self, hdi_prob=0.94, ci_kind="eti", round_to=3, include_deterministic=False):
        """Summary table of the draws; see :func:`~BayesianDLL.Evaluation.summary`."""
        from ..Evaluation._summary import summary as _summary
        return _summary(self, hdi_prob=hdi_prob, ci_kind=ci_kind, round_to=round_to, include_deterministic=include_deterministic)

    @property
    def lp(self) -> torch.Tensor | None:
        """The log posterior density tensor of shape ``(n_chains, n_samples)`` for each draw, if recorded."""
        if self.sample_stats is None:
            return None
        try:
            return self.sample_stats["lp"]
        except (KeyError, ValueError, IndexError):
            return None

    def mode(self, include_deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Return the parameter values of the draw with the highest log posterior density (MAP sample).

        If log posterior densities are not available, falls back to coordinate-wise mode/median.

        Parameters
        ----------
        include_deterministic : bool, default=False
            Whether to include deterministic parameters in the returned dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to tensors of shape ``(*param_shape)``.
        """
        traces = {**self.trace, **(self.deterministic_trace if include_deterministic else {})}
        lp = self.lp
        if lp is not None and lp.numel() > 0:
            flat_lp = lp.flatten(0, 1)
            best_idx = torch.argmax(flat_lp).item()
            chain_idx = best_idx // lp.shape[1]
            sample_idx = best_idx % lp.shape[1]
            return {name: val[chain_idx, sample_idx].clone() for name, val in traces.items()}

        res = {}
        for name, val in traces.items():
            flat = val.flatten(0, 1)
            if not flat.is_floating_point():
                res[name] = torch.mode(flat, dim=0).values.clone()
            else:
                res[name] = torch.median(flat, dim=0).values.clone()
        return res

    def median(self, include_deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Return the coordinate-wise posterior median for each parameter.

        Parameters
        ----------
        include_deterministic : bool, default=False
            Whether to include deterministic parameters in the returned dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to tensors of shape ``(*param_shape)``.
        """
        traces = {**self.trace, **(self.deterministic_trace if include_deterministic else {})}
        res = {}
        for name, val in traces.items():
            flat = val.flatten(0, 1)
            res[name] = torch.median(flat, dim=0).values.clone()
        return res

    def mean(self, include_deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Return the coordinate-wise posterior mean for each parameter.

        Parameters
        ----------
        include_deterministic : bool, default=False
            Whether to include deterministic parameters in the returned dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to tensors of shape ``(*param_shape)``.
        """
        traces = {**self.trace, **(self.deterministic_trace if include_deterministic else {})}
        res = {}
        for name, val in traces.items():
            flat = val.flatten(0, 1)
            if flat.is_floating_point():
                res[name] = torch.mean(flat, dim=0).clone()
            else:
                res[name] = torch.mean(flat.double(), dim=0).to(val.dtype).clone()
        return res

    def quantile(self, q: float | torch.Tensor, include_deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Return the coordinate-wise posterior quantile for each parameter.

        Parameters
        ----------
        q : float or Tensor
            Quantile to compute, e.g. 0.05 or 0.95.
        include_deterministic : bool, default=False
            Whether to include deterministic parameters in the returned dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to tensors of shape ``(*param_shape)``.
        """
        traces = {**self.trace, **(self.deterministic_trace if include_deterministic else {})}
        res = {}
        q_tensor = torch.as_tensor(q, dtype=torch.float64)
        for name, val in traces.items():
            flat = val.flatten(0, 1)
            if flat.is_floating_point():
                res[name] = torch.quantile(flat, q=q_tensor, dim=0).clone()
            else:
                res[name] = torch.quantile(flat.double(), q=q_tensor, dim=0).to(val.dtype).clone()
        return res

    def __repr__(self):
        params = list(self.trace.keys()) + list(self.deterministic_trace.keys())
        return f"SamplingResult(params={params}, divergences={self.divergences})"


class PredicativeResult:
    """Predictive draws of shape ``(n_samples, samples_per_step, *predictive_shape)`` per observed node.

    ``prior`` holds the prior draws of the model parameters (a
    :class:`SamplingResult` with one chain) for results of
    ``sample_prior_predicative``, and is ``None`` otherwise.
    """

    def __init__(self, samples: dict[str, torch.Tensor], prior=None, deterministic_samples: dict[str, torch.Tensor] = None):
        self.samples = samples
        self.prior = prior
        self.deterministic = {} if deterministic_samples is None else deterministic_samples

    def __getitem__(self, key):
        if key in self.samples:
            return self.samples[key]
        if key in self.deterministic:
            return self.deterministic[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.samples or key in self.deterministic

    def keys(self):
        return self.samples.keys()

    def values(self):
        return self.samples.values()

    def items(self):
        return self.samples.items()

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def thin(self, step: int | slice = 1, dim: int | tuple[int, ...] = 1) -> "PredicativeResult":
        """Keep every ``step``-th entry (an int >= 1 or a slice) along ``dim``.

        ``dim=1`` thins the replications per parameter draw, ``dim=0`` the
        parameter draws (and the prior draws in ``prior``), ``(0, 1)`` both
        with the same step.
        """
        s = _normalize_slice(step)
        if dim == 1:
            thinned = {name: val[:, s] for name, val in self.samples.items()}
        elif dim == 0:
            thinned = {name: val[s] for name, val in self.samples.items()}
        elif dim in ((0, 1), (1, 0)):
            thinned = {name: val[s, s] for name, val in self.samples.items()}
        else:
            raise ValueError(f"dim must be 0, 1, or (0, 1), got {dim}.")
        prior = self.prior
        if prior is not None and dim != 1:
            prior = prior.thin(step)
        thinned_det = {}
        if self.deterministic and dim != 1:
            thinned_det = {name: val[s] for name, val in self.deterministic.items()}
        elif self.deterministic:
            thinned_det = dict(self.deterministic)
        return PredicativeResult(thinned, prior=prior, deterministic_samples=thinned_det)

    def __repr__(self):
        return f"PredicativeResult(samples={list(self.samples.keys())})"
