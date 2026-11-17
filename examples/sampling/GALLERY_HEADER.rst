Samplers
========

Every sampler of the library, each checked against exact answers. A
parameter chooses its sampler with ``sampler=`` (``"auto"`` by default), and
``sample(..., blocks=[...])`` groups parameters into blocks that are updated
in turn.

* **NUTS** (``"nuts"``, the automatic choice for continuous parameters) uses
  gradients and adapts its step size and mass matrix. It is the first choice
  for continuous posteriors. Use ``metric="dense"`` when the posterior has
  strong linear correlations that no reparameterisation removes.
* **Slice** (``"slice"``) updates one coordinate at a time without gradients
  and needs no tuning. It suits a few continuous parameters whose gradient
  is unavailable or unreliable.
* **Metropolis** (``"metropolis"``) makes random-walk proposals for
  continuous or discrete parameters. It is simple but mixes slowly: use it as
  a baseline, or for discrete parameters with an infinite support.
* **Gibbs** (``"gibbs"``, the automatic choice for discrete parameters with a
  finite support) draws each coordinate exactly from its full conditional
  distribution, with no tuning and no rejections.
* **Blocked sampling** combines these samplers in one model; each block is
  updated conditional on the others.
* **Sequential Monte Carlo** (``sample_smc``) tempers a population of
  particles from the prior to the posterior. It handles multimodal
  posteriors and estimates the marginal likelihood for Bayes factors.
* **Vectorized chains** (``chain_method="vectorized"``) run all chains in one
  process with batched model evaluations. This is faster for many chains and
  is the way to sample on a GPU.

A discrete parameter with a finite support can also be summed out of the
model exactly instead of being sampled (``RandomParameter(...,
enumerate=True)``); the Gaussian mixture examples show this enumeration.
