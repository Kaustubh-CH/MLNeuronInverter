#!/usr/bin/env python
"""Differentiable (PyTorch) surrogate for 11 eFEL electrophysiology features.

The forward path is pure torch on a ``(B, T)`` voltage tensor (mV), so the
feature-extraction gradients flow back into the voltage trace.  This lets the
soft features replace the non-differentiable ``efel.get_feature_values()`` calls
used by ``ScoreFunctionsHDF5.py`` inside a training loss (voltage -> jaxley -> CNN).

The 11 features (exact eFEL names) are::

    mean_frequency, AP_amplitude, AHP_depth_abs_slow, fast_AHP_change,
    AHP_slow_time, spike_half_width, time_to_first_spike, inv_first_ISI,
    ISI_CV, ISI_values, adaptation_index

Three of them (mean_frequency, time_to_first_spike, AP_amplitude) reuse the
already-validated surrogates from ``scripts/softefel_validate_50sims.py``
(Pearson r vs real eFEL = 0.94 / 0.999 / 0.79).  The remaining eight are built on
top of a *differentiable spike-time extractor* (see :func:`_spike_times`) which
yields per-spike times and a soft presence indicator without any ``.item()``,
boolean indexing or numpy in the forward path.

Run ``python toolbox/soft_efel.py`` for the self-test / validation report.
"""
import numpy as np
import torch

DT_MS = 0.1
SPIKE_THR = -20.0          # mV, eFEL default Threshold for spike detection
_EPS = 1e-7

# 11 eFEL feature names, canonical order (matches ScoreFunctionsHDF5.py FEATURES)
FEATURES = [
    "mean_frequency", "AP_amplitude", "AHP_depth_abs_slow", "fast_AHP_change",
    "AHP_slow_time", "spike_half_width", "time_to_first_spike", "inv_first_ISI",
    "ISI_CV", "ISI_values", "adaptation_index",
]

# The 6 features whose differentiable surrogate tracks real eFEL at Pearson
# r >= 0.79 on the RUN1 50-sim validation (see the r-table below).  This is the
# recommended set to carry the TRAINING gradient in HybridLoss; the other 5 are
# eval-only (their surrogates are too noisy and DEGRADE score ranking when added
# — see docs/perf_research/softefel_chaotic.md).
STRONG_FEATURES = [
    "time_to_first_spike", "mean_frequency", "inv_first_ISI",
    "AHP_depth_abs_slow", "ISI_values", "AP_amplitude",
]

# Sub-threshold / whole-trace features (differentiable) that the spike-centric
# set above is BLIND to.  They are timing-ROBUST and expose the channels whose
# main action is below spike threshold (leak, km, kd -> resting V, steady state,
# ramp-to-threshold, sag) — precisely the information the raw-voltage variation
# sees but count/height/ISI features cannot.  All are real eFEL names except
# "AUC" (mean |V - baseline| over the trace, a whole-trace excursion integral
# with no direct eFEL equivalent).  Windows are taken as fractions of the trace
# (no explicit stim_start/stim_end); with native-tmax sims the trace ends at
# stim end, so the tail window == steady state at stim end.
SUBTHRESHOLD_FEATURES = [
    "voltage_base", "steady_state_voltage_stimend", "voltage_deflection",
    "minimum_voltage", "AUC",
]
# Convenience: the STRONG spike set + the sub-threshold set.  Use this as the
# eFEL feature list in sensitivity_variation.py (--efelFeatures) / HybridLoss to
# cover BOTH the spiking and sub-threshold regimes in one sweep.
STRONG_PLUS_SUBTHRESHOLD = STRONG_FEATURES + SUBTHRESHOLD_FEATURES

# Characteristic per-feature scale (physical units) used to make features
# COMPARABLE inside a training loss: the feature-matching loss divides each
# (sim - data) difference by its scale before the (Huber) penalty, so no single
# feature dominates by unit magnitude.  Fixed (not batch-derived) so the loss
# scale is stable across batches/stims.  Values ~ the real-eFEL std of the
# feature on the RUN1 true traces (sqrt of the eFEL(true) variance).
FEATURE_SCALES = {
    "mean_frequency":       20.0,   # Hz
    "AP_amplitude":         26.0,   # mV
    "AHP_depth_abs_slow":   21.0,   # mV
    "fast_AHP_change":      1.7,    # relative
    "AHP_slow_time":        0.12,   # ISI fraction
    "spike_half_width":     3.3,    # ms
    "time_to_first_spike":  73.0,   # ms
    "inv_first_ISI":        110.0,  # Hz
    "ISI_CV":               0.24,   # unitless
    "ISI_values":           9.6,    # ms
    "adaptation_index":     0.086,  # unitless
    # sub-threshold / whole-trace (provisional scales ~ across-dataset mV spread;
    # recalibrate to the real-trace std like the spike features above once wired
    # into the offline validation).
    "voltage_base":                   6.0,   # mV, resting membrane potential
    "steady_state_voltage_stimend":  12.0,   # mV
    "voltage_deflection":            12.0,   # mV, steady-state minus baseline
    "minimum_voltage":               12.0,   # mV, deepest sag / AHP
    "AUC":                           20.0,   # mV, mean |V - baseline| over trace
}

# Validation on the RUN1 50-sim cached traces (pooled true+pred, 3 stims):
# Pearson r of soft vs real eFEL, and the numpy "ceiling" (exact peak-time
# reconstruction of the eFEL definition) for context.  See _self_test().
#   feature              soft r   ceiling   notes
#   mean_frequency        0.94      -       strong (validated)
#   time_to_first_spike   1.00      -       strong (validated)
#   AP_amplitude          0.79      -       strong (validated)
#   AHP_depth_abs_slow    0.89     0.89     strong (matches ceiling)
#   inv_first_ISI         0.94      -       strong
#   ISI_values (mean)     0.84      -       strong
#   ISI_CV                0.49     0.66     moderate (ISI 2nd-order stat)
#   adaptation_index      0.36     0.49     moderate (ISI 2nd-order stat)
#   spike_half_width      0.07     0.34*    WEAK  (*r=0.34 once eFEL's ~10
#                                           garbage-outlier traces are removed;
#                                           eFEL emits -99/+29 ms half-widths on
#                                           malformed sub-threshold "spikes")
#   AHP_slow_time         0.16     0.55     WEAK  (trough-time fraction is noisy)
#   fast_AHP_change      -0.14     0.00     WEAK  (ill-defined: even an EXACT
#                                           reconstruction fails to correlate)


# ---------------------------------------------------------------------------
# smooth reductions over time
# ---------------------------------------------------------------------------
def _softplus_max(V, beta):
    """smooth max over the last axis: (1/beta) logsumexp(beta V)."""
    return torch.logsumexp(beta * V, dim=-1) / beta


def _softplus_min(V, beta):
    return -torch.logsumexp(-beta * V, dim=-1) / beta


def _soft_select(V, t, logw, beta_min, win_gain=6.0):
    """Softly pick the MINIMUM-voltage sample inside a soft time window.

    logw : (B, S, T) additive log-weight (log of the time window, ~0 inside,
           large-negative outside).  ``win_gain`` sharpens the in/out contrast so
           out-of-window low-voltage baseline samples cannot leak into the softmin.
    Returns (trough_V, trough_t) each (B, S).
    """
    lw = win_gain * logw - beta_min * V.unsqueeze(1)   # (B,S,T) favour low V, hard window
    w = torch.softmax(lw, dim=-1)                      # normalised over time
    trough_v = (w * V.unsqueeze(1)).sum(-1)            # (B,S)
    trough_t = (w * t.view(1, 1, -1)).sum(-1)          # (B,S)
    return trough_v, trough_t


# ---------------------------------------------------------------------------
# DIFFERENTIABLE SPIKE-TIME EXTRACTOR
# ---------------------------------------------------------------------------
def _spike_times(V, k, thr, a, t_edge, nmax):
    """Extract per-spike times differentiably.

    Returns
    -------
    t_spk   : (B, nmax) soft spike time (ms) for each spike slot
    present : (B, nmax) soft presence mass in [0, 1] of each spike slot
    rising  : (B, T-1) soft rising-edge signal (each spike ~ unit mass)
    s       : (B, T)   soft spike gate sigmoid(k (V-thr))
    """
    B = V.shape[0]
    s = torch.sigmoid(k * (V - thr))                       # (B,T)
    rising = torch.relu(s[:, 1:] - s[:, :-1])              # (B,T-1) ~unit mass/spike
    C = torch.cumsum(rising, dim=1)                        # (B,T-1) soft count so far

    idx = torch.arange(1, nmax + 1, dtype=V.dtype, device=V.device)  # (nmax,)
    Cb = C.unsqueeze(1)                                    # (B,1,T-1)
    lo = idx.view(1, -1, 1) - 1.0                          # (1,nmax,1)
    hi = idx.view(1, -1, 1)
    memb = torch.sigmoid(a * (Cb - lo)) - torch.sigmoid(a * (Cb - hi))  # (B,nmax,T-1)
    w = rising.unsqueeze(1) * memb                         # (B,nmax,T-1)
    present = w.sum(-1)                                    # (B,nmax) ~in [0,1]
    t_spk = (w * t_edge.view(1, 1, -1)).sum(-1) / (present + _EPS)      # (B,nmax)
    return t_spk, present, rising, s


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def soft_efel_features(V, k=2.0, thr=SPIKE_THR, dt_ms=DT_MS, beta=0.5,
                       a=8.0, beta_min=1.0, nmax=64, c_win=3.0,
                       hw_sigma_ms=2.0, fast_win_ms=5.0, base_ms=3.0,
                       ss_frac=0.1, only=None):
    """Compute soft eFEL features on a ``(B, T)`` mV tensor.

    Returns a dict ``name -> (B,)`` differentiable tensor, keyed by eFEL names.
    Every per-spike / per-ISI statistic is presence-weighted so that absent
    spike slots (non-spiking traces) do not corrupt the value; on such traces
    the real eFEL feature is undefined (NaN) and is masked out in validation.

    ``only`` : optional iterable of feature names.  When given, the expensive
    per-spike window tensors (``(B, nmax, T)``) for features NOT requested are
    skipped, and those features are returned as zero placeholders.  This cuts
    peak memory / backward-graph size when a training loss needs only a subset
    (e.g. ``STRONG_FEATURES``).  ``None`` computes all 11 (used by the self-test
    and offline validation).
    """
    if V.dim() != 2:
        raise ValueError("V must be (B, T)")
    B, T = V.shape
    want = set(FEATURES) if only is None else set(only)
    _z = V.new_zeros(B)                                                 # placeholder
    need_isi = bool(want & {"ISI_values", "ISI_CV", "inv_first_ISI",
                            "adaptation_index", "AHP_depth_abs_slow", "AHP_slow_time"})
    need_isi_ahp = bool(want & {"AHP_depth_abs_slow", "AHP_slow_time"})
    need_halfwidth = "spike_half_width" in want
    need_fast = ("fast_AHP_change" in want) or need_halfwidth
    dur_s = T * dt_ms / 1000.0
    t_ms = torch.arange(T, dtype=V.dtype, device=V.device) * dt_ms      # (T,)
    t_edge = t_ms[:-1] + 0.5 * dt_ms                                    # (T-1,) edge midpoints

    # -- soft gate / rising edges / spike times --------------------------------
    t_spk, present, rising, s = _spike_times(V, k, thr, a, t_edge, nmax)
    count = rising.sum(-1)                                              # (B,)

    # ========================= validated 3 (reused) ==========================
    mean_frequency = count / dur_s                                     # Hz

    surv = torch.cumprod(1.0 - s + _EPS, dim=1)                        # P(no spike up to t)
    surv_prev = torch.cat([torch.ones(B, 1, dtype=V.dtype, device=V.device),
                           surv[:, :-1]], dim=1)
    p_first = s * surv_prev
    p_norm = p_first / (p_first.sum(-1, keepdim=True) + _EPS)
    time_to_first_spike = (t_ms.unsqueeze(0) * p_norm).sum(-1)         # ms

    soft_peak = _softplus_max(V, beta)                                # (B,)
    AP_amplitude = soft_peak - thr

    # ================= sub-threshold / whole-trace features ==================
    # Timing-ROBUST scalars the spike-centric set is blind to.  Computed only
    # when requested (kept as zero placeholders otherwise), so callers asking for
    # a spike-only subset pay nothing and the default-11 self-test is unchanged.
    _sub = {f: _z for f in SUBTHRESHOLD_FEATURES}
    if want & set(SUBTHRESHOLD_FEATURES):
        _nb = max(1, int(round(base_ms / dt_ms)))          # baseline window (samples)
        _ns = max(1, int(round(T * ss_frac)))              # steady-state tail window
        _v_base = V[:, :_nb].mean(-1)                      # (B,) resting baseline (mV)
        _v_ss = V[:, -_ns:].mean(-1)                       # (B,) steady state at stim end
        # soft global min (sag / AHP envelope).  A softmin over T samples carries
        # a -log(T)/beta offset (a flat -70 mV trace reads ~ -70 - log(T)/beta);
        # it is constant per stim so it cancels in the across-sample std and in a
        # matched-length sim-vs-data loss.  beta=4 keeps that offset ~1-2 mV.
        _v_min = _softplus_min(V, 4.0)                     # (B,) soft global min
        _cand = {
            "voltage_base": _v_base,
            "steady_state_voltage_stimend": _v_ss,
            "voltage_deflection": _v_ss - _v_base,         # SS minus baseline
            "minimum_voltage": _v_min,
            "AUC": (V - _v_base.unsqueeze(-1)).abs().mean(-1),  # mean |V-base| over trace
        }
        for _name, _val in _cand.items():
            if _name in want:
                _sub[_name] = _val

    # ============================ ISI family =================================
    # (built only when a downstream feature needs spike-to-spike intervals)
    ISI_values = inv_first_ISI = ISI_CV = adaptation_index = _z
    AHP_depth_abs_slow = AHP_slow_time = fast_AHP_change = spike_half_width = _z
    if not need_isi and not need_fast:
        return {
            "mean_frequency": mean_frequency, "AP_amplitude": AP_amplitude,
            "AHP_depth_abs_slow": AHP_depth_abs_slow, "fast_AHP_change": fast_AHP_change,
            "AHP_slow_time": AHP_slow_time, "spike_half_width": spike_half_width,
            "time_to_first_spike": time_to_first_spike, "inv_first_ISI": inv_first_ISI,
            "ISI_CV": ISI_CV, "ISI_values": ISI_values, "adaptation_index": adaptation_index,
            **_sub,
        }

    # ISI_j = t_{j+1} - t_j, presence-weighted so absent spikes don't count.
    isi = t_spk[:, 1:] - t_spk[:, :-1]                                 # (B,nmax-1)
    w_isi = present[:, 1:] * present[:, :-1]                           # (B,nmax-1)
    sw = w_isi.sum(-1) + _EPS                                          # (B,)

    # mean ISI (ISI_values scalar reduction)
    isi_mean = (w_isi * isi).sum(-1) / sw
    ISI_values = isi_mean

    # ISI coefficient of variation.  Floor the mean-ISI denominator at 1 ms:
    # a near-zero soft mean-ISI (spurious coincident spike slots) otherwise makes
    # CV explode to ~1e6 while eFEL sits at O(1).  1 ms == 1000 Hz, non-physical
    # as a real ISI, so the floor only clips numerical blow-ups.
    isi_var = (w_isi * (isi - isi_mean.unsqueeze(-1)) ** 2).sum(-1) / sw
    isi_std = torch.sqrt(isi_var + _EPS)
    ISI_CV = isi_std / torch.clamp(isi_mean, min=1.0)

    # first ISI: weight by the first present pair (spike slots fill from 0)
    first_w = w_isi / (w_isi.sum(-1, keepdim=True) + _EPS)
    # bias strongly toward the earliest present pair
    order_bias = torch.arange(isi.shape[1], dtype=V.dtype, device=V.device)
    first_sel = torch.softmax(torch.log(w_isi + _EPS) - 4.0 * order_bias.view(1, -1), dim=-1)
    first_isi = (first_sel * isi).sum(-1)
    # Floor the first ISI at 1 ms (== 1000 Hz cap) so a near-zero soft first-ISI
    # cannot send 1000/first_isi to ~1e18; eFEL's inv_first_ISI variance is O(1e4).
    inv_first_ISI = 1000.0 / torch.clamp(first_isi, min=1.0)           # Hz

    # adaptation index: mean over consecutive ISI pairs of
    # (ISI[j+1]-ISI[j])/(ISI[j+1]+ISI[j]), presence weighted.
    isi_a, isi_b = isi[:, :-1], isi[:, 1:]
    adapt_term = (isi_b - isi_a) / (isi_b + isi_a + _EPS)
    w_pair = w_isi[:, :-1] * w_isi[:, 1:]
    adaptation_index = (w_pair * adapt_term).sum(-1) / (w_pair.sum(-1) + _EPS)

    # ====================== per-ISI AHP (slow) features ======================
    # eFEL's slow AHP is the absolute voltage minimum over the full inter-spike
    # interval (verified against peak-to-peak segment minima).  Window each ISI
    # slot as ~1 for t_j < t < t_{j+1}; the softmin naturally avoids the spike
    # peaks (high V) so no extra masking is needed.
    tt = t_ms.view(1, 1, -1)                                           # (1,1,T)
    if need_isi_ahp:
        tj = t_spk[:, :-1].unsqueeze(-1)                               # (B,nmax-1,1)
        tj1 = t_spk[:, 1:].unsqueeze(-1)
        iv = (torch.sigmoid(c_win * (tt - tj)) - torch.sigmoid(c_win * (tt - tj1)))
        iv = torch.clamp(iv, min=0.0)                                  # (B,nmax-1,T)
        logiv = torch.log(iv + _EPS)

        trough_v, trough_t = _soft_select(V, t_ms, logiv, beta_min)    # (B,nmax-1)
        # AHP_depth_abs_slow: absolute voltage of the (slow) trough, mean over ISIs.
        AHP_depth_abs_slow = (w_isi * trough_v).sum(-1) / sw
        # AHP_slow_time: trough time as a fraction of the ISI.
        slow_frac = (trough_t - t_spk[:, :-1]) / (isi + _EPS)
        slow_frac = torch.clamp(slow_frac, 0.0, 1.0)
        AHP_slow_time = (w_isi * slow_frac).sum(-1) / sw

    if need_fast:
        # ==================== fast AHP (per spike) ===========================
        # trough in a short window (~fast_win_ms) right after each spike.
        tsp = t_spk.unsqueeze(-1)                                      # (B,nmax,1)
        fiv = (torch.sigmoid(c_win * (tt - tsp))
               - torch.sigmoid(c_win * (tt - (tsp + fast_win_ms))))
        fiv = torch.clamp(fiv, min=0.0)                                # (B,nmax,T)
        logfiv = torch.log(fiv + _EPS)
        fast_v, _ = _soft_select(V, t_ms, logfiv, beta_min)           # (B,nmax)
        fast_depth = thr - fast_v                                      # (B,nmax) positive depth
        # fast_AHP_change: relative change of fast-AHP depth between spikes.
        fd_a, fd_b = fast_depth[:, :-1], fast_depth[:, 1:]
        fast_change_term = (fd_b - fd_a) / (fd_a + _EPS)
        fast_AHP_change = (w_isi * fast_change_term).sum(-1) / sw

    if need_halfwidth:
        # ==================== spike half-width (per spike) ===================
        # per-spike peak inside a Gaussian window, then width at half-amplitude.
        gsig = hw_sigma_ms
        gauss = torch.exp(-0.5 * ((tt - tsp) / gsig) ** 2)            # (B,nmax,T)
        wpk = torch.softmax(torch.log(gauss + _EPS) + beta * 4.0 * V.unsqueeze(1), dim=-1)
        peak_i = (wpk * V.unsqueeze(1)).sum(-1)                        # (B,nmax) near-max V in window
        # half level is measured from the AP onset, which sits well below the
        # -20 mV detection threshold; use the post-spike fast trough as a
        # per-spike baseline so the amplitude (hence half level) tracks height.
        onset_i = torch.minimum(fast_v, torch.full_like(fast_v, thr)) # (B,nmax)
        half_level = 0.5 * (peak_i + onset_i)                         # (B,nmax)
        # plateau window (~1 near spike, 0 outside +/- 3*sigma)
        Wplat = 3.0 * gsig
        plat = (torch.sigmoid(4.0 * (tt - (tsp - Wplat)))
                - torch.sigmoid(4.0 * (tt - (tsp + Wplat))))
        above = torch.sigmoid(k * (V.unsqueeze(1) - half_level.unsqueeze(-1)))  # (B,nmax,T)
        width_i = (plat * above).sum(-1) * dt_ms                      # (B,nmax) ms
        spike_half_width = (present * width_i).sum(-1) / (present.sum(-1) + _EPS)

    return {
        "mean_frequency": mean_frequency,
        "AP_amplitude": AP_amplitude,
        "AHP_depth_abs_slow": AHP_depth_abs_slow,
        "fast_AHP_change": fast_AHP_change,
        "AHP_slow_time": AHP_slow_time,
        "spike_half_width": spike_half_width,
        "time_to_first_spike": time_to_first_spike,
        "inv_first_ISI": inv_first_ISI,
        "ISI_CV": ISI_CV,
        "ISI_values": ISI_values,
        "adaptation_index": adaptation_index,
        **_sub,
    }


# ---------------------------------------------------------------------------
# Real eFEL (non-differentiable) -- scalar reduction per feature to match soft
# ---------------------------------------------------------------------------
# reduction convention: 'first' -> a[0]; 'mean' -> nanmean over spikes;
# 'scalar' -> a[0] (single-value features); mean_frequency defaults 0 if None.
_REDUCE = {
    "mean_frequency": "scalar",
    "AP_amplitude": "mean",
    "AHP_depth_abs_slow": "mean",
    "fast_AHP_change": "mean",
    "AHP_slow_time": "mean",
    "spike_half_width": "mean",
    "time_to_first_spike": "first",
    "inv_first_ISI": "first",
    "ISI_CV": "scalar",
    "ISI_values": "mean",
    "adaptation_index": "scalar",
}


def real_efel_features(v_np, dt_ms=DT_MS):
    """Real eFEL feature values reduced to a scalar per feature (matching soft).

    Empty / None features -> NaN (mean_frequency -> 0.0, matching eFEL's
    no-spike convention).  Returns dict name -> float.
    """
    import efel
    v = np.asarray(v_np, dtype=float)
    tr = {"T": [i * dt_ms for i in range(len(v))], "V": list(map(float, v)),
          "stim_start": [0], "stim_end": [len(v) * dt_ms]}
    fv = efel.get_feature_values([tr], FEATURES)[0]
    out = {}
    for f in FEATURES:
        a = fv.get(f)
        if a is None or (hasattr(a, "__len__") and len(a) == 0):
            out[f] = np.nan
        else:
            arr = np.asarray(a, dtype=float)
            red = _REDUCE[f]
            if red == "first" or red == "scalar":
                out[f] = float(arr[0])
            else:
                out[f] = float(np.nanmean(arr))
    if not np.isfinite(out["mean_frequency"]):
        out["mean_frequency"] = 0.0
    return out


# ---------------------------------------------------------------------------
def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def _self_test():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traces = os.path.join(here, "docs/perf_research/run1_50sims_scored_traces.npz")
    out_npz = os.path.join(here, "docs/perf_research/softefel_all11.npz")
    stims = ["5k50kInterChaoticB", "5k0step_500", "5k0chirp"]

    z = np.load(traces, allow_pickle=True)
    dev = "cpu"

    soft = {f: [] for f in FEATURES}
    efl = {f: [] for f in FEATURES}
    is_pred, stim_lab = [], []

    for stim in stims:
        for side in ("true", "pred"):
            v = z[f"{side}_{stim}"].astype(np.float64)                 # (N,T)
            N = v.shape[0]
            with torch.no_grad():
                sf = soft_efel_features(torch.tensor(v, dtype=torch.float64, device=dev))
            for f in FEATURES:
                soft[f].extend(sf[f].cpu().numpy().tolist())
            for i in range(N):
                ef = real_efel_features(v[i])
                for f in FEATURES:
                    efl[f].append(ef[f])
            is_pred.extend([side == "pred"] * N)
            stim_lab.extend([stim] * N)

    # -------- per-feature Pearson r (pooled true+pred, 3 stims) --------------
    print("\n=== soft eFEL vs real eFEL — per-feature Pearson r "
          "(pooled true+pred, 3 stims) ===")
    print(f"{'feature':22s} {'r':>8s} {'n':>6s}")
    rows = {}
    for f in FEATURES:
        r, n = _pearson(soft[f], efl[f])
        rows[f] = (r, n)
        print(f"{f:22s} {r:>8.3f} {n:>6d}")

    # -------- differentiability proof ---------------------------------------
    vg = torch.tensor(z[f"true_{stims[0]}"][:6], dtype=torch.float64, requires_grad=True)
    sf = soft_efel_features(vg)
    # scalar built from all 11 soft features (normalise scales so all contribute)
    score = sum(sf[f].abs().mean() for f in FEATURES)
    score.backward()
    gmax = float(vg.grad.abs().max())
    print(f"\n[differentiability] score built from all 11 soft features; "
          f"max|d score / d V| = {gmax:.3e}  (nonzero -> gradient flows into V)")

    # -------- save arrays for downstream plotting ---------------------------
    save = {}
    for f in FEATURES:
        save[f"soft_{f}"] = np.asarray(soft[f], float)
        save[f"efel_{f}"] = np.asarray(efl[f], float)
    save["is_pred"] = np.asarray(is_pred, bool)
    save["stim"] = np.asarray(stim_lab)
    np.savez(out_npz, **save)
    print(f"\nsaved per-feature soft+eFEL arrays -> {out_npz}")
    return rows, gmax


if __name__ == "__main__":
    _self_test()
