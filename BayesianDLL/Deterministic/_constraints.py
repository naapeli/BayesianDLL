"""Parameterisations of stationary autoregressive and invertible moving-average coefficients."""
import torch

from ._blocks import _Block


def pacf_to_ar(pacf):
    """AR coefficients ``(..., p)`` of a stationary process from partial autocorrelations in ``(-1, 1)`` (Durbin-Levinson).

    The AR polynomial is :math:`1 - \\sum_j \\phi_j z^j`. Also returns the list of the intermediate coefficient
    vectors, needed by :func:`pacf_to_ar_vjp`.
    """
    p = pacf.shape[-1]
    phi = pacf[..., :1]
    history = [phi]
    for k in range(1, p):
        r = pacf[..., k:k + 1]
        phi = torch.cat([phi - r * phi.flip(-1), r], dim=-1)
        history.append(phi)
    return phi, history


def pacf_to_ar_vjp(grad, pacf, history):
    """Gradient of a scalar with respect to ``pacf`` given its gradient ``grad`` with respect to the AR coefficients."""
    p = pacf.shape[-1]
    out = [None] * p
    g = grad
    for k in range(p - 1, 0, -1):
        r = pacf[..., k:k + 1]
        prev = history[k - 1]
        g_head, g_last = g[..., :k], g[..., k:k + 1]
        out[k] = g_last - (g_head * prev.flip(-1)).sum(-1, keepdim=True)
        g = g_head - r * g_head.flip(-1)
    out[0] = g
    return torch.cat(out, dim=-1)


class StationaryAR(_Block):
    r"""Turn unconstrained numbers in (-1, 1) into autoregressive (AR) coefficients that always give a stationary process.

    **The problem.** An AR model :math:`y_t = \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \text{noise}` only describes a
    well-behaved (*stationary*) process, one that does not explode or drift away, when the coefficients lie in a
    small, oddly shaped region of :math:`\mathbb{R}^p`. For :math:`p = 1` it is :math:`|\phi_1| < 1`, but for larger
    :math:`p` it is a curved region (all roots of :math:`1 - \sum_j \phi_j z^j` outside the unit circle) that is
    not a box. Giving the coefficients an ordinary prior (say Normal) and sampling them directly makes the
    sampler keep proposing points outside this region, where the likelihood is zero, so sampling is slow and
    unreliable.

    **The solution.** A stationary AR process can be described by :math:`p` *partial autocorrelations*
    :math:`r_1, \dots, r_p`, and every stationary process corresponds to exactly one set with all
    :math:`r_k \in (-1, 1)`, and the other way around: *any* values in :math:`(-1, 1)` give a stationary process.
    The region is therefore a simple box. This block converts the partial autocorrelations into AR coefficients
    with the Durbin-Levinson recursion, so you put the prior on the partial autocorrelations (for example
    ``Uniform(-1, 1)`` for each) and use the output as the coefficients. The conversion is differentiable (the
    gradient is written by hand) and works with vectorized chains.

    Parameters
    ----------
    name : str
        Node name.
    pacf : Parameter or torch.Tensor
        Partial autocorrelations of shape ``(..., p)``, each in :math:`(-1, 1)`.

    Returns
    -------
    The deterministic node holds the AR coefficients :math:`\phi_1, \dots, \phi_p` of shape ``(..., p)``; with
    ``p = 1`` they equal the partial autocorrelation.

    Examples
    --------
    >>> pacf = RandomParameter("pacf", Uniform(-torch.ones(3), torch.ones(3)))
    >>> phi = StationaryAR("phi", pacf)                    # always a stationary AR(3)
    >>> y = ObservedParameter("y", AR(phi, sigma, n_steps=200), data)
    """

    def __init__(self, name, pacf):
        super().__init__(name, [pacf])

    def _arg_event_ndims(self, unbatched_ndims):
        return [1]

    def _compute(self, values):
        return pacf_to_ar(values[0])[0]

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None]
        pacf = values[0]
        phi, history = pacf_to_ar(pacf)
        return [pacf_to_ar_vjp(g, pacf, history)]


class InvertibleMA(_Block):
    r"""Turn unconstrained numbers in (-1, 1) into moving-average (MA) coefficients that are always invertible.

    **The problem.** An MA model :math:`y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q
    \varepsilon_{t-q}` is only identifiable when it is *invertible*: the coefficients must lie in a curved region
    (all roots of :math:`1 + \sum_j \theta_j z^j` outside the unit circle). Otherwise a different set of
    coefficients gives exactly the same likelihood, so the posterior has several equivalent modes that a sampler
    handles poorly. Sampling the coefficients directly from an ordinary prior does not respect this region.

    **The solution.** Partial autocorrelations :math:`r_1, \dots, r_q \in (-1, 1)` describe the allowed region as a
    simple box: *any* values in :math:`(-1, 1)` give an invertible MA process, and every invertible process is
    obtained exactly once. This block maps them to MA coefficients (the Durbin-Levinson recursion of
    :class:`StationaryAR` with the sign flipped, :math:`\theta = -\phi(r)`), so you put the prior on the partial
    autocorrelations (for example ``Uniform(-1, 1)``) and use the output as the MA coefficients.

    Parameters
    ----------
    name : str
        Node name.
    pacf : Parameter or torch.Tensor
        Partial autocorrelations of shape ``(..., q)``, each in :math:`(-1, 1)`.

    Returns
    -------
    The deterministic node holds the MA coefficients :math:`\theta_1, \dots, \theta_q` of shape ``(..., q)``.

    Examples
    --------
    >>> pacf = RandomParameter("pacf", Uniform(-torch.ones(2), torch.ones(2)))
    >>> theta = InvertibleMA("theta", pacf)                # always an invertible MA(2)
    >>> y = ObservedParameter("y", SARIMAX(ma=theta, sigma=sigma), data)
    """

    def __init__(self, name, pacf):
        super().__init__(name, [pacf])

    def _arg_event_ndims(self, unbatched_ndims):
        return [1]

    def _compute(self, values):
        return -pacf_to_ar(values[0])[0]

    def _arg_grads(self, g, values, out, needs):
        if not needs[0]:
            return [None]
        pacf = values[0]
        phi, history = pacf_to_ar(pacf)
        return [-pacf_to_ar_vjp(g, pacf, history)]
