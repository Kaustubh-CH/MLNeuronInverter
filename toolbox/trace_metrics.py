#!/usr/bin/env python
"""Differentiable full-trace distance metrics for voltage traces (torch, GPU).

Two timing-tolerant trace distances used both as (a) an alternative *variation*
measure in sensitivity_variation.py and (b) candidate training losses that give a
usable gradient when a predicted spike is close-but-not-coincident with the
target (raw MSE gives an almost-flat/cliffy landscape there — see
project_ca3_voltage_only_degeneracy).

  multiscale_blurred_mse   van-Rossum-style: low-pass both traces at several
                           Gaussian widths and sum the MSEs.  Large sigma = wide
                           gradient basin (far-apart spikes still pull together),
                           small sigma sharpens timing once aligned; a raw MSE
                           term pins amplitude.  Cheap, O(T) per scale, no
                           alignment step.  This is the recommended default.

  soft_dtw                 soft dynamic-time-warping (Cuturi & Blondel 2017):
                           smooth-min over warping paths of the squared-Euclidean
                           local cost.  Explicit time alignment; O(T^2) so traces
                           are decimated to <= max_t first.  Comparison baseline.

Both take a (B, T) mV tensor.  In the sensitivity sweep they are used as a
distance-to-reference: how far sweeping one channel pushes the soma trace away
from the all-default trace, averaged over the swept samples.
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# multi-scale blurred (van-Rossum) MSE
# ---------------------------------------------------------------------------
def gaussian_kernel(sigma, device, dtype=torch.float32):
    """Normalised 1-D Gaussian conv kernel, shape (1, 1, 2*radius+1)."""
    radius = max(1, int(round(4.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def multiscale_blurred_mse(pred, target, sigmas=(16., 8., 4., 2., 1.),
                           include_raw=True, reduce_batch=False):
    """Multi-scale blurred MSE between (B, T) traces (target may be (1, T)).

    For each sigma, low-pass both traces with a Gaussian and accumulate the
    per-sample mean-squared error; optionally add a raw (unblurred) MSE term so
    amplitude is pinned once traces are aligned.

    Returns (B,) per-sample distance by default (mean over time), or a scalar
    mean over the batch when ``reduce_batch=True`` (the training-loss form).
    """
    if pred.dim() != 2:
        raise ValueError("pred must be (B, T)")
    if target.dim() == 2 and target.shape[0] == 1 and pred.shape[0] != 1:
        target = target.expand_as(pred)
    p = pred.unsqueeze(1)                          # (B, 1, T)
    t = target.unsqueeze(1)
    out = pred.new_zeros(pred.shape[0])
    for s in sigmas:
        k = gaussian_kernel(s, pred.device, pred.dtype)
        pad = k.shape[-1] // 2
        bp = F.conv1d(p, k, padding=pad)
        bt = F.conv1d(t, k, padding=pad)
        # conv1d 'padding' can add one extra sample on even kernels; crop to T.
        bp = bp[..., :p.shape[-1]]
        bt = bt[..., :t.shape[-1]]
        out = out + ((bp - bt) ** 2).mean(dim=(1, 2))
    if include_raw:
        out = out + ((p - t) ** 2).mean(dim=(1, 2))
    return out.mean() if reduce_batch else out


# ---------------------------------------------------------------------------
# soft-DTW  (decimated, batched wavefront)
# ---------------------------------------------------------------------------
def decimate(V, max_t):
    """Strided decimation of (B, T) -> (B, <=max_t).  Decimation (not avg-pool)
    keeps spike peak heights; avg-pool would flatten them."""
    T = V.shape[-1]
    if max_t is None or T <= max_t:
        return V
    stride = (T + max_t - 1) // max_t
    return V[:, ::stride]


def _softmin3(a, b, c, gamma):
    x = torch.stack([a, b, c], dim=0)              # (3, B, L)
    return -gamma * torch.logsumexp(-x / gamma, dim=0)


def _sdtw_raw(X, Y, gamma, vscale, max_batch):
    """Un-normalised soft-DTW R[n, m] between (B, n) and (B, m), batched over the
    anti-diagonal wavefront.  Local cost = squared diff scaled by ``vscale`` mV.
    Returns (B,).  Pass already-``decimate``-d traces (O(n*m) time and memory)."""
    B, n = X.shape
    m = Y.shape[1]
    INF = 1e10
    outs = []
    for lo in range(0, B, max_batch):
        xb = X[lo:lo + max_batch] / vscale
        yb = Y[lo:lo + max_batch] / vscale
        b = xb.shape[0]
        D = (xb[:, :, None] - yb[:, None, :]) ** 2          # (b, n, m)
        R = xb.new_full((b, n + 1, m + 1), INF)
        R[:, 0, 0] = 0.0
        for k in range(2, n + m + 1):
            i = torch.arange(max(1, k - m), min(n, k - 1) + 1, device=X.device)
            j = k - i
            R[:, i, j] = D[:, i - 1, j - 1] + _softmin3(
                R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1], gamma)
        outs.append(R[:, n, m])
    return torch.cat(outs, dim=0)


def soft_dtw(X, Y, gamma=0.1, vscale=30.0, max_batch=128, divergence=True):
    """Soft-DTW (Cuturi 2017) between (B, n) traces, per-step-normalised.

    ``divergence=True`` returns the soft-DTW DIVERGENCE (Blondel 2021)
    ``D(x,y) - 1/2 D(x,x) - 1/2 D(y,y)``, which is >= 0 and 0 iff x == y — the
    plain soft-DTW is biased negative by the soft-min entropy and a bare
    ``identical`` pair does not read 0.  Result is divided by the trace length so
    it is a per-step aligned cost, comparable across stims of different T.
    """
    n = X.shape[1]
    xy = _sdtw_raw(X, Y if Y.shape[0] == X.shape[0] else Y.expand_as(X),
                   gamma, vscale, max_batch)
    if not divergence:
        return xy / float(n)
    xx = _sdtw_raw(X, X, gamma, vscale, max_batch)
    yy = _sdtw_raw(Y, Y, gamma, vscale, max_batch)
    if yy.shape[0] == 1:
        yy = yy.expand_as(xy)
    return (xy - 0.5 * xx - 0.5 * yy) / float(n)


# ---------------------------------------------------------------------------
# ensemble-VARIANCE measures (spread, not distance) — for chaotic sensitivity
# ---------------------------------------------------------------------------
# Distance-to-reference (MSE/blur/DTW) SATURATES for a chaotic trajectory: once
# two traces decorrelate in phase the distance sits at ~the attractor diameter
# regardless of the parameter change, so it cannot grade sensitivity.  The
# visible fan-out is ENSEMBLE VARIANCE, and the useful (identifiable) part is the
# component that is a SMOOTH function of the swept parameter — the chaotic part
# is erratic in the parameter and roughly equal across channels.  Two estimators:
#   smoothed_ensemble_std  — low-pass each trace, then std across the sweep; the
#                            blur removes the fast chaotic jitter (parameter-free).
#   parameter_explained_var— regress the trace on the swept theta; the fitted
#                            (smooth-in-theta) part is the SYSTEMATIC variance, the
#                            residual is chaotic.  Exact split; needs the thetas.
def gaussian_smooth(x, sigma_samp):
    """Gaussian low-pass along time of a (B, T) tensor."""
    k = gaussian_kernel(sigma_samp, x.device, x.dtype)
    pad = k.shape[-1] // 2
    y = F.conv1d(x.unsqueeze(1), k, padding=pad)
    return y[:, :, :x.shape[-1]].squeeze(1)


def smoothed_ensemble_std(V, sigma_samp):
    """Per-timepoint std across the sweep of the low-passed traces.  V (N, T) ->
    (T,) std curve; the caller reduces over time (mean).  Removes the fast chaotic
    spike-timing jitter, leaving the systematic envelope spread."""
    return gaussian_smooth(V, sigma_samp).std(dim=0)


def parameter_explained_var(V, theta, degree=5, ridge=1e-8):
    """Decompose the ensemble variance of V (N, T) into the part explained by a
    smooth (polynomial, ``degree``) function of the swept unit parameter ``theta``
    (N,) — the SYSTEMATIC variance — and the chaotic residual.

    Returns (sys_var_t (T,), tot_var_t (T,)); sys <= tot pointwise, and
    tot - sys is the chaotic residual variance.  R^2(t) = sys/tot is the fraction
    of the fan-out actually driven by the parameter at time t."""
    th = theta.reshape(-1, 1)
    Phi = torch.cat([th ** k for k in range(degree + 1)], dim=1)     # (N, d)
    d = Phi.shape[1]
    A = Phi.T @ Phi + ridge * torch.eye(d, dtype=V.dtype, device=V.device)
    beta = torch.linalg.solve(A, Phi.T @ V)                          # (d, T)
    pred = Phi @ beta                                                # (N, T) = E[V|theta]
    mu = V.mean(0, keepdim=True)
    sys_var = ((pred - mu) ** 2).mean(0)                             # (T,)
    tot_var = ((V - mu) ** 2).mean(0)                                # (T,)
    return sys_var, tot_var


def soft_dtw_to_ref(X, ref, gamma=0.1, vscale=30.0, max_batch=128):
    """Soft-DTW divergence of each (B, T) sample to a single (T,)/(1, T) ref.

    Computes the ref self-term once (not per sample), so cost is ~2x a single
    soft-DTW instead of 3x.  Returns (B,) >= 0."""
    ref2 = ref.view(1, -1)
    n = X.shape[1]
    xy = _sdtw_raw(X, ref2.expand_as(X), gamma, vscale, max_batch)
    xx = _sdtw_raw(X, X, gamma, vscale, max_batch)
    yy = _sdtw_raw(ref2, ref2, gamma, vscale, max_batch)     # (1,)
    return (xy - 0.5 * xx - 0.5 * yy.expand_as(xy)) / float(n)
