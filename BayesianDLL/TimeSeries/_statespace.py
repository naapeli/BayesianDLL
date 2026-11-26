"""Linear Gaussian state-space model with an exact Kalman-filter likelihood."""
import math
import torch

from ..Distributions._distributions import Distribution, _as_float, _sample_shape
from ..Distributions._resolve import resolve
from ..Distributions._transforms import IdentityTransform
from ..Distributions._state_space import ContinuousReal
from . import _kalman
from ._kalman import _mv, _t
from ._lyapunov import _solve_lyapunov


def _psd_sqrt(A):
    """A factor ``L`` with ``L L' = A`` for a symmetric positive semi-definite matrix (eigendecomposition)."""
    w, V = torch.linalg.eigh(0.5 * (A + _t(A)))
    return V * torch.sqrt(w.clamp(min=0)).unsqueeze(-2)


def _detach(x):
    return x.detach()


def _tt(x, like):
    return torch.as_tensor(x).to(device=like.device)


def spectral_radius(T):
    """Largest absolute eigenvalue of square matrices ``(..., m, m)``."""
    return torch.linalg.eigvals(T).abs().amax(-1)


class StateSpace(Distribution):
    r"""Linear Gaussian state-space model; the likelihood of the observed series integrates the hidden states out exactly.

    With :math:`m` hidden states and :math:`k` observed components:

    .. math::

        y_t = Z \alpha_t + d_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H), \qquad
        \alpha_{t+1} = T \alpha_t + R \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q), \qquad
        \alpha_1 \sim \mathcal{N}(a_1, P_1)

    The value of the distribution is the whole observed series of shape ``(..., N, k)``; its log density is the
    Kalman-filter prediction-error decomposition :math:`\sum_t \log \mathcal{N}(y_t \mid Z a_t + d_t, F_t)`. The
    system matrices are learnable: each may be a number, a tensor or a model node (a random or deterministic
    parameter), and the gradients with respect to all of them and to the series itself are computed by a
    hand-written backward (adjoint) recursion through the filter. System matrices are time-invariant; the
    observation offset :math:`d_t` (for example a regression term :math:`x_t^\top \beta`) varies with time.

    Missing observations are ``nan`` entries of the series (a component may be missing while the others at the
    same time are observed). They are integrated out exactly: the filter predicts through them.

    Matrices must be given as matrices (``(m, m)``, ...), any leading dimensions are batch dimensions (chains or
    posterior draws). :math:`H`, :math:`Q` and :math:`P_1` must be symmetric. If ``init_cov`` is omitted,
    :math:`P_1` is the stationary covariance solving :math:`P = T P T' + R Q R'`, which requires a stable
    transition (all eigenvalues inside the unit circle; the log density is ``-inf`` otherwise).

    Parameters
    ----------
    transition : torch.Tensor or Parameter
        Transition matrix :math:`T` of shape ``(..., m, m)``.
    design : torch.Tensor or Parameter
        Design matrix :math:`Z` of shape ``(..., k, m)``.
    state_cov : torch.Tensor or Parameter
        State innovation covariance :math:`Q` of shape ``(..., r, r)``.
    observation_cov : torch.Tensor or Parameter, optional
        Observation noise covariance :math:`H` of shape ``(..., k, k)`` (default: zero).
    selection : torch.Tensor or Parameter, optional
        Selection matrix :math:`R` of shape ``(..., m, r)`` (default: identity, so ``r = m``).
    init_mean : torch.Tensor or Parameter, optional
        Initial state mean :math:`a_1` of shape ``(..., m)`` (default: zero).
    init_cov : torch.Tensor or Parameter, optional
        Initial state covariance :math:`P_1` of shape ``(..., m, m)`` (default: stationary covariance).
    observation_offset : torch.Tensor or Parameter, optional
        Time-varying observation offset :math:`d_t` of shape ``(..., N, k)``.
    n_steps : int, optional
        Series length. Needed for sampling and to give a latent series its shape.
    """

    _param_event_ndims = {"transition": 2, "design": 2, "state_cov": 2, "observation_cov": 2, "selection": 2,
                          "init_mean": 1, "init_cov": 2, "observation_offset": 2}

    _PARAMS = ("transition", "design", "state_cov", "observation_cov", "selection", "init_mean", "init_cov", "observation_offset")

    def __init__(self, transition, design, state_cov, observation_cov=None, selection=None, init_mean=None, init_cov=None,
                 observation_offset=None, n_steps=None):
        super().__init__(IdentityTransform(), ContinuousReal(), ContinuousReal())
        self.transition = transition
        self.design = design
        self.state_cov = state_cov
        self.observation_cov = observation_cov
        self.selection = selection
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.observation_offset = observation_offset
        self.n_steps = None if n_steps is None else int(n_steps)
        for name in self._PARAMS:
            value = getattr(self, name)
            if value is not None:
                self.add_dependency(value)

    # ------------------------------------------------------------ shapes
    @property
    def n_states(self):
        return resolve(self.transition).shape[-1]

    @property
    def n_obs_dims(self):
        return resolve(self.design).shape[-2]

    @property
    def event_shape(self):
        return () if self.n_steps is None else (self.n_steps, self.n_obs_dims)

    @property
    def batch_shape(self):
        shapes = []
        for name in self._PARAMS:
            value = getattr(self, name)
            if value is not None:
                shapes.append(tuple(resolve(value).shape[: resolve(value).ndim - self._param_event_ndims[name]]))
        return torch.broadcast_shapes(*shapes)

    def _length(self):
        if self.n_steps is None:
            raise ValueError(f"{type(self).__name__}: give the series length n_steps to sample or to use it as a latent parameter.")
        return self.n_steps

    # ------------------------------------------------------------ system
    def _system(self, batch_extra=()):
        """Full-batch system quantities as a dict (all tensors with the common batch shape ``B``)."""
        names = [n for n in self._PARAMS if getattr(self, n) is not None]
        values = dict(zip(names, _as_float(*[getattr(self, n) for n in names]) if len(names) > 1 else [_as_float(getattr(self, names[0]))]))
        Tm, Z, Q = values["transition"], values["design"], values["state_cov"]
        m, k = Tm.shape[-1], Z.shape[-2]
        dtype, device = Tm.dtype, Tm.device
        shapes = [tuple(v.shape[: v.ndim - self._param_event_ndims[n]]) for n, v in values.items()]
        B = torch.broadcast_shapes(*shapes, tuple(batch_extra))
        ex = lambda v, e: v.expand(*B, *v.shape[v.ndim - e:]) if v is not None else None
        H = values.get("observation_cov")
        H = torch.zeros(k, k, dtype=dtype, device=device) if H is None else H
        R = values.get("selection")
        R = torch.eye(m, dtype=dtype, device=device) if R is None else R
        a1 = values.get("init_mean")
        a1 = torch.zeros(m, dtype=dtype, device=device) if a1 is None else a1
        sysm = {"T": ex(Tm, 2), "Z": ex(Z, 2), "Q": ex(Q, 2), "H": ex(H, 2), "R": ex(R, 2), "a1": ex(a1, 1),
                "P1": ex(values.get("init_cov"), 2), "d": values.get("observation_offset"), "B": B, "values": values}
        return sysm

    def _prepare(self, y):
        """Time-major data and system, with the stationary initial covariance when it is not given."""
        y = torch.as_tensor(y)
        if y.shape[-1] != self.n_obs_dims or y.ndim < 2:
            raise ValueError(f"The series must have shape (..., N, {self.n_obs_dims}); got {tuple(y.shape)}.")
        N = y.shape[-2]
        d_shape = ()
        offset = resolve(self.observation_offset) if self.observation_offset is not None else None
        sysm = self._system(batch_extra=y.shape[:-2])
        B = sysm["B"]
        dtype = sysm["T"].dtype
        y = y.to(dtype)
        observed = ~torch.isnan(y)
        y0 = torch.where(observed, y, torch.zeros_like(y))
        tm = lambda v: v.movedim(-2, 0).reshape(N, *([1] * (len(B) - (v.ndim - 2))), *v.shape[:-2], v.shape[-1]).expand(N, *B, v.shape[-1])
        sysm.update(N=N, y=tm(y0), mask=tm(observed.to(dtype)), nan_free=bool(observed.all()))
        if offset is not None:
            if offset.shape[-2] not in (1, N):
                raise ValueError("observation_offset must have N rows.")
            sysm["d"] = tm(sysm["values"]["observation_offset"].expand(*offset.shape[:-2], N, offset.shape[-1]))
        else:
            sysm["d"] = torch.zeros_like(sysm["y"])
        sysm["RQR"] = sysm["R"] @ sysm["Q"] @ _t(sysm["R"])
        stable = torch.ones(B, dtype=torch.bool, device=y.device)
        if sysm["P1"] is None:
            stable = spectral_radius(sysm["T"]) < 1 - 1e-9
            Tsafe = torch.where(stable[..., None, None], sysm["T"], torch.zeros_like(sysm["T"]))
            sysm["T_safe"] = Tsafe
            sysm["P1"] = 0.5 * (lambda X: X + _t(X))(_solve_lyapunov(Tsafe, sysm["RQR"]))
            sysm["P1_stationary"] = True
        else:
            sysm["T_safe"] = sysm["T"]
            sysm["P1_stationary"] = False
        sysm["stable"] = stable
        return sysm

    def _run(self, y):
        s = self._prepare(y)
        arrays = [_detach(s[n]) for n in ("y", "mask", "d", "T_safe", "Z", "H", "RQR", "a1", "P1")]
        ll, cache, a_end, P_end = _kalman.kalman_forward(*arrays)
        like = s["T"]
        ll, a_end, P_end = _tt(ll, like), _tt(a_end, like), _tt(P_end, like)
        ok = s["stable"] & torch.isfinite(ll)
        s.update(cache=cache, a_end=a_end, P_end=P_end, ok=ok, k_mask=arrays[1], k_T=arrays[3])
        return torch.where(ok, ll, torch.full_like(ll, -math.inf)), s

    # ------------------------------------------------------------ density
    def log_pdf(self, y):
        return self._run(y)[0]

    def _gradients(self, y):
        ll, s = self._run(y)
        g = {k: _tt(v, s["T"]) for k, v in _kalman.kalman_backward(s["cache"], s["k_mask"], s["k_T"]).items()}
        T, R, Q = s["T"], s["R"], s["Q"]
        RQRbar, Tbar, Pbar = g["RQR"], g["T"], g["P"]
        if s["P1_stationary"]:
            Lam = _solve_lyapunov(s["T_safe"], Pbar, transpose=True)
            Tbar = Tbar + 2 * Lam @ s["T_safe"] @ s["P1"]
            RQRbar = RQRbar + Lam
        RQRbar = 0.5 * (RQRbar + _t(RQRbar))
        grads = {
            "transition": Tbar, "design": g["Z"], "observation_cov": g["H"],
            "state_cov": _t(R) @ RQRbar @ R, "selection": 2 * RQRbar @ R @ Q,
            "init_mean": g["a"], "init_cov": Pbar,
        }
        d = g["d"].movedim(0, -2)
        grads["observation_offset"] = d
        ok = s["ok"]
        out = {}
        for name in self._PARAMS:
            node = getattr(self, name)
            if node is None:
                continue
            value = grads[name]
            if name == "observation_offset":
                offset = resolve(self.observation_offset)
                if offset.shape[-2] == 1 and s["N"] != 1:
                    value = value.sum(-2, keepdim=True)
            mask = ok.reshape(ok.shape + (1,) * (value.ndim - ok.ndim))
            value = torch.where(mask, value, torch.zeros_like(value))
            key = self.resolve_name(name, node)
            out[key] = out[key] + value if key in out else value
        return out, g, ok, s

    def log_pdf_param_grads(self, y):
        return self._gradients(y)[0]

    def log_pdf_grad(self, y):
        y = torch.as_tensor(y)
        _, g, ok, s = self._gradients(y)
        grad = g["y"].movedim(0, -2)
        grad = torch.where(ok.reshape(ok.shape + (1, 1)), grad, torch.zeros_like(grad))
        return grad.sum_to_size(y.shape) if grad.shape != y.shape else grad

    # ------------------------------------------------------------ filtering and prediction
    def filter(self, y):
        """Run the Kalman filter on ``y`` of shape ``(..., N, k)``.

        Returns a dict with ``log_likelihood`` ``(...)``, the one-step-ahead predictions ``predicted_mean``
        ``(..., N + 1, m)`` and ``predicted_cov`` ``(..., N + 1, m, m)`` of the states (the last entry
        predicts the state after the series, the start of a forecast), and the ``innovations`` ``(..., N, k)`` with
        covariance ``innovation_cov`` ``(..., N, k, k)`` (entries of missing observations are zero / identity).
        """
        ll, s = self._run(y)
        a_list, P_list, Finv_list, u_list = s["cache"][0], s["cache"][1], s["cache"][4], s["cache"][5]
        like = s["T"]
        mean = torch.stack([_tt(a, like) for a in a_list] + [s["a_end"]], dim=-2)
        cov = torch.stack([_tt(P, like) for P in P_list] + [s["P_end"]], dim=-3)
        Finv = torch.stack([_tt(f, like) for f in Finv_list], dim=-3)
        F = torch.linalg.inv(Finv)
        u = torch.stack([_tt(x, like) for x in u_list], dim=-2)
        return {"log_likelihood": ll, "predicted_mean": mean, "predicted_cov": cov,
                "innovations": (F @ u.unsqueeze(-1)).squeeze(-1), "innovation_cov": F}

    def _simulate_states(self, s, n_samples, steps, a0, P0, offset, sample_shape):
        """Simulate ``steps`` observations starting from ``alpha ~ N(a0, P0)`` (shapes ``(*B, ...)``)."""
        B = s["B"]
        dtype, device = s["T"].dtype, s["T"].device
        m, k = s["T"].shape[-1], s["Z"].shape[-2]
        Lp, Lq, Lh = _psd_sqrt(P0), _psd_sqrt(s["Q"]), _psd_sqrt(s["H"])
        R, Tm, Z = s["R"], s["T"], s["Z"]
        r = Lq.shape[-1]
        alpha = a0 + _mv(Lp, torch.randn(n_samples, *B, m, dtype=dtype, device=device))
        out = []
        for t in range(steps):
            y = _mv(Z, alpha) + _mv(Lh, torch.randn(n_samples, *B, k, dtype=dtype, device=device))
            if offset is not None:
                y = y + offset[..., t, :]
            out.append(y)
            alpha = _mv(Tm, alpha) + _mv(R @ Lq, torch.randn(n_samples, *B, r, dtype=dtype, device=device))
        return torch.stack(out, dim=-2)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Simulate series of shape ``(n_samples, *batch_shape, T, k)`` from the model (unconditionally)."""
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        N = self._length()
        s = self._system()
        if s["P1"] is None:
            if not bool((spectral_radius(s["T"]) < 1).all()):
                raise ValueError("Sampling with the stationary initial distribution needs a stable transition matrix.")
            RQR = s["R"] @ s["Q"] @ _t(s["R"])
            s["P1"] = 0.5 * (lambda X: X + _t(X))(_solve_lyapunov(s["T"], RQR))
        offset = None
        if self.observation_offset is not None:
            offset = s["d"].expand(*s["B"], N, s["d"].shape[-1]) if s["d"].shape[-2] in (1, N) else s["d"]
        return self._simulate_states(s, n_samples, N, s["a1"], s["P1"], offset, None)

    def forecast(self, y, steps, n_samples=1, future_offset=None):
        """Draw future observations given the observed series ``y`` of shape ``(..., N, k)`` (missing values allowed).

        The state after the last observation has the filtered predictive distribution
        :math:`\\mathcal{N}(a_{N+1}, P_{N+1})`; future observations are simulated from it exactly.
        Returns ``(n_samples, *batch_shape, steps, k)``. Give the parameters with leading batch dimensions
        (for example a posterior sample) to forecast for every draw.

        Parameters
        ----------
        y : torch.Tensor
            Observed series.
        steps : int
            Forecast horizon.
        n_samples : int
            Number of simulated futures per batch element.
        future_offset : torch.Tensor, optional
            Observation offsets :math:`d_t` of the forecast period, shape ``(..., steps, k)``
            (required when the model has an ``observation_offset``).
        """
        if self.observation_offset is not None and future_offset is None:
            raise ValueError("The model has an observation_offset: give future_offset for the forecast period.")
        _, s = self._run(y)
        if future_offset is not None:
            future_offset = torch.as_tensor(future_offset, dtype=s["T"].dtype)
        return self._simulate_states(s, n_samples, steps, s["a_end"], s["P_end"], future_offset, None)
