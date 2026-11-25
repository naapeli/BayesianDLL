"""Parameterisations of stationary autoregressive and invertible moving-average coefficients."""
import torch

from ..Deterministic._blocks import _Block


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
    r"""Stationary AR coefficients from unconstrained partial autocorrelations.

    The Durbin-Levinson recursion maps partial autocorrelations :math:`r_1, \dots, r_p \in (-1, 1)` one-to-one onto
    the coefficients :math:`\phi_1, \dots, \phi_p` whose polynomial :math:`1 - \sum_j \phi_j z^j` has all roots
    outside the unit circle, i.e. onto the stationary region. A prior on the partial autocorrelations (for example
    :math:`r_j = 2 b_j - 1` with :math:`b_j \sim \mathrm{Beta}`, or a ``Uniform(-1, 1)``) is therefore a prior on
    stationary coefficients; the likelihoods in :mod:`BayesianDLL.TimeSeries` need no further constraint. Works
    with vectorized chains.

    Parameters
    ----------
    name : str
        Node name.
    pacf : Parameter or torch.Tensor
        Partial autocorrelations of shape ``(..., p)`` with values in ``(-1, 1)``.
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
    r"""Invertible moving-average coefficients from unconstrained partial autocorrelations.

    The MA polynomial :math:`1 + \sum_j \theta_j z^j` has all roots outside the unit circle exactly when
    :math:`-\theta` are stationary AR coefficients, so :math:`\theta = -\phi(r)` with :math:`\phi(r)` the
    Durbin-Levinson map of :class:`StationaryAR`.

    Parameters
    ----------
    name : str
        Node name.
    pacf : Parameter or torch.Tensor
        Partial autocorrelations of shape ``(..., q)`` with values in ``(-1, 1)``.
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
