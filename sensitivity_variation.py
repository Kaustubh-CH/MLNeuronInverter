#!/usr/bin/env python3
"""One-at-a-time (OAT) voltage-variation sensitivity analysis for jaxley cells.

WHY
---
To pick the *best stimulus for identifying a given ion channel*, we ask a
simple, model-free question: if I hold every conductance at its default and
sweep ONLY channel c across its full physical range, how much does the soma
voltage move?  A stimulus that makes the voltage swing a lot when channel c
changes is a stimulus that *exposes* channel c — good for recovering it.  A
stimulus under which the voltage barely budges hides that channel.

WHAT it does
------------
For every stimulus CSV in --stimDir and every trainable conductance of the
cell:

  1. Draw N (default 500) unit-parameter samples for the target channel,
     uniformly in [-1, 1] (the log-scaled space the training data lives in),
     with every OTHER channel pinned at its default (unit 0).  The same N
     draws are reused across all stims so cross-stim numbers are comparable.
  2. Convert unit -> physical (phys = center * 10^(unit * log_halfspan)) and
     simulate all P*N cells under the stimulus with the cached jaxley bridge.
  3. Measure the *variation across the dataset*: at each time point take the
     std of the soma voltage across the N samples, then average over time.
     That scalar (mV) is how much channel c moves the trace under this stim.

Outputs (under --outDir, default ./sensitivity_variation/<cell>/):
  variation_matrix.csv              stims x channels, mean temporal std (mV)
  variation_matrix_peak.csv         stims x channels, peak temporal std (mV)
  variation_matrix_normalized.csv   each channel column scaled to its own max
  ranking_per_channel.csv           per channel, stims ranked best->worst
  summary_heatmap.png               stims x channels heatmap (per-column norm)
  ranking_per_channel.png           top-K stims per channel (raw mV)
  per_stim_variation.pdf            one page per stim: bars + mean+-std bands
  summary.yaml                      best stim per channel + metadata

Usage
-----
  # full run (do this inside a GPU salloc / sbatch, JAX_PLATFORMS=cuda)
  python sensitivity_variation.py --cell ca3_pyramidal --nSamples 500

  # quick smoke test on CPU
  python sensitivity_variation.py --cell ca3_pyramidal --nSamples 12 \
         --maxStims 2 --outDir /tmp/sv_smoke
"""

import os, sys, time, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jax
jax.config.update("jax_enable_x64", True)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import phys_par_range_to_arrays, load_stim_csv


# Default CA3 physical ranges (input_meta from a trained ca3 sum_train.yaml).
# Used only as a fallback when --physRange is not resolvable.
_CA3_PHYS_PAR_RANGE = [
    [3.9417e-05, 0.5], [0.04, 0.5], [0.01, 0.5],
    [0.04, 0.5], [0.00052, 0.5], [0.00025, 0.5],
]
_DEFAULT_PHYS_YAML = ("/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_matched/"
                      "ca3_pyramidal_synth/52835681/out/sum_train.yaml")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="ca3_pyramidal",
                   help="jaxley cell registry name (default ca3_pyramidal)")
    p.add_argument("--stimDir",
                   default="/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims",
                   help="directory of stimulus CSVs to sweep")
    p.add_argument("--stims", nargs="*", default=None,
                   help="explicit stim stems (default: every *.csv in --stimDir)")
    p.add_argument("--maxStims", type=int, default=None,
                   help="cap number of stims (for quick tests)")
    p.add_argument("--stimStart", type=int, default=0,
                   help="slice: index of first stim to process (chunked runs)")
    p.add_argument("--stimCount", type=int, default=None,
                   help="slice: number of stims to process from --stimStart")
    p.add_argument("--nSamples", type=int, default=500,
                   help="samples per (channel, stim) sensitivity set")
    p.add_argument("--physRange", default=None,
                   help="sum_train.yaml whose input_meta.phys_par_range to use "
                        "(default: a known ca3 run; else cell hardcoded fallback)")
    p.add_argument("--solver", default="bwd_euler")
    p.add_argument("--simBatch", type=int, default=256,
                   help="jaxley vmap batch size per forward call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--somaProbe", type=int, default=0,
                   help="index of the soma recording among cell.record() sites")
    p.add_argument("--nativeTmax", action="store_true", default=True,
                   help="set t_max per stim = len(stim)*dt_stim (default on)")
    p.add_argument("--fixedTmax", dest="nativeTmax", action="store_false",
                   help="keep the cell's built-in t_max for every stim")
    p.add_argument("--topK", type=int, default=15,
                   help="how many stims to show in the per-channel ranking plot")
    p.add_argument("--efel", action="store_true", default=False,
                   help="also measure across-sample variation of soft-eFEL "
                        "features (scaled by FEATURE_SCALES, as in HybridLoss)")
    p.add_argument("--efelFeatures", nargs="*", default=None,
                   help="eFEL feature subset (default: soft_efel.STRONG_FEATURES)")
    p.add_argument("--efelChunk", type=int, default=128,
                   help="batch size for the soft-eFEL forward (memory bound)")
    # ---- full-trace distance metrics (dist-to-default variation) ------------
    p.add_argument("--blur", action="store_true", default=False,
                   help="also measure multi-scale blurred-MSE (van-Rossum) variation")
    p.add_argument("--dtw", action="store_true", default=False,
                   help="also measure soft-DTW divergence variation (O(T^2), decimated)")
    p.add_argument("--metricSamples", type=int, default=96,
                   help="samples per channel used for the blur/dtw dist-to-ref metrics")
    p.add_argument("--metricMaxT", type=int, default=2000,
                   help="decimate traces to <= this for the blurred-MSE metric")
    p.add_argument("--dtwMaxT", type=int, default=400,
                   help="decimate traces to <= this for soft-DTW (quadratic cost)")
    p.add_argument("--dtwGamma", type=float, default=0.1, help="soft-DTW smoothing")
    p.add_argument("--vscale", type=float, default=30.0,
                   help="mV cost scale keeping soft-DTW gamma unit-stable")
    p.add_argument("--blurSigmasMs", nargs="*", type=float,
                   default=[10.0, 5.0, 2.0, 1.0, 0.5],
                   help="Gaussian widths (ms) for multi-scale blurred MSE; "
                        "largest ~ max spike-timing offset to bridge")
    # ---- ensemble-VARIANCE metrics (spread, chaos-robust; not distance) ------
    p.add_argument("--sysvar", action="store_true", default=False,
                   help="parameter-explained (systematic) std via theta-regression (mV); "
                        "the chaos-robust sensitivity — distance metrics saturate on chaos")
    p.add_argument("--smoothvar", action="store_true", default=False,
                   help="smoothed-ensemble std (mV): low-pass then std across sweep; "
                        "theta-free proxy for the systematic variance")
    p.add_argument("--sysvarDegree", type=int, default=5,
                   help="polynomial degree in theta for the systematic-variance fit")
    p.add_argument("--smoothvarSigmaMs", type=float, default=20.0,
                   help="Gaussian width (ms) for the smoothed-ensemble std")
    p.add_argument("--saveTraces", action="store_true", default=False,
                   help="persist the swept voltage traces (decimated fp16) + thetas so new "
                        "variance measures can be recomputed WITHOUT re-simulating")
    p.add_argument("--saveTracesMaxT", type=int, default=2000,
                   help="decimate saved traces to <= this many timepoints")
    p.add_argument("-o", "--outDir", default=None)
    return p.parse_args()


def compute_efel_variation(V, P, N, features, scales, device, dt_ms, chunk):
    """Across-sample variation of each soft-eFEL feature, per channel.

    V: (P*N, T) mV numpy.  Returns (P, F) scaled std: for channel p, feature f,
    std over the N samples of feature f, divided by its FEATURE_SCALES entry
    (so features are comparable, matching HybridLoss._efel_feat_loss)."""
    import torch as _t
    from toolbox.soft_efel import soft_efel_features
    Bt = P * N
    feat = {f: np.full(Bt, np.nan) for f in features}
    for i in range(0, Bt, chunk):
        vt = _t.tensor(V[i:i + chunk], dtype=_t.float64, device=device)
        with _t.no_grad():
            fd = soft_efel_features(vt, dt_ms=float(dt_ms), only=list(features))
        for f in features:
            feat[f][i:i + chunk] = fd[f].detach().cpu().numpy()
    out = np.full((P, len(features)), np.nan)
    for fi, f in enumerate(features):
        vals = feat[f].reshape(P, N)
        out[:, fi] = np.nanstd(vals, axis=1) / float(scales[f])
    return out


def compute_metric_variation(V, Vref, P, N, M, device, dt_stim,
                             do_blur, do_dtw, blur_sigmas_ms, blur_maxT,
                             dtw_maxT, dtw_gamma, vscale):
    """Distance-to-reference variation of the blurred-MSE and soft-DTW metrics.

    For each channel take the first M swept samples, measure their full-trace
    distance to the all-default reference ``Vref`` (1, T), and average over the
    M samples.  Both metrics are timing-tolerant, so this reflects how far the
    channel moves the trace WITHOUT the spike-timing-jitter inflation that the
    raw per-timepoint std suffers on chaotic stims.  Returns (blur (P,), dtw (P,))
    numpy arrays (either may be None)."""
    import torch as _t
    from toolbox.trace_metrics import (multiscale_blurred_mse, soft_dtw_to_ref,
                                        decimate)
    idx = np.concatenate([np.arange(p * N, p * N + M) for p in range(P)])
    Xs = _t.tensor(V[idx], dtype=_t.float64, device=device)        # (P*M, T)
    ref = _t.tensor(Vref, dtype=_t.float64, device=device).view(1, -1)

    def _reduce(vec):
        a = vec.detach().to(_t.float64).cpu().numpy().reshape(P, M)
        return np.nanmean(a, axis=1)                                # (P,)

    blur = dtw = None
    if do_blur:
        stride = max(1, (Xs.shape[1] + blur_maxT - 1) // blur_maxT)
        dt_dec = dt_stim * stride
        sig = tuple(max(1.0, s / dt_dec) for s in blur_sigmas_ms)
        b = multiscale_blurred_mse(decimate(Xs, blur_maxT),
                                   decimate(ref, blur_maxT), sigmas=sig)
        blur = _reduce(b)
    if do_dtw:
        d = soft_dtw_to_ref(decimate(Xs, dtw_maxT), decimate(ref, dtw_maxT),
                            gamma=dtw_gamma, vscale=vscale)
        dtw = _reduce(d)
    return blur, dtw


def compute_variance_metrics(V, P, N, unit_samples, device, dt_stim,
                             do_sys, do_smooth, sys_degree, smooth_sigma_ms):
    """Ensemble-VARIANCE (spread) measures that separate the SYSTEMATIC (parameter-
    driven, identifiable) spread from chaotic spike-timing jitter — unlike the
    distance metrics (MSE/blur/DTW), which saturate at the attractor diameter on a
    chaotic stim and cannot grade sensitivity.

      sysvar    : sqrt(mean_t Var_t[E[V(t)|theta]]) — the parameter-explained std,
                  from a degree-``sys_degree`` polynomial regression of the trace on
                  the swept unit parameter; the chaotic residual is discarded.
      smoothvar : mean_t std_N(lowpass_sigma V)(t) — a theta-free proxy; the blur
                  removes the fast chaotic jitter, leaving the systematic envelope.
    Returns (sysvar (P,), smoothvar (P,)) in mV (either may be None)."""
    import torch as _t
    from toolbox.trace_metrics import parameter_explained_var, smoothed_ensemble_std
    sysv = np.full(P, np.nan) if do_sys else None
    smv = np.full(P, np.nan) if do_smooth else None
    for p in range(P):
        Vp = _t.tensor(V[p * N:(p + 1) * N], dtype=_t.float64, device=device)
        fin = _t.isfinite(Vp).all(1)
        Vp = Vp[fin]
        if Vp.shape[0] < 8:
            continue
        if do_smooth:
            smv[p] = float(smoothed_ensemble_std(Vp, smooth_sigma_ms / dt_stim).mean().cpu())
        if do_sys:
            th = _t.tensor(unit_samples[p], dtype=_t.float64, device=device)[fin]
            sv, _tv = parameter_explained_var(Vp, th, degree=sys_degree)
            sysv[p] = float(_t.sqrt(sv).mean().cpu())     # mean_t systematic std (mV)
    return sysv, smv


def resolve_phys_range(args, P):
    """Return (centers, logspans, param_names) for the cell."""
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cell}")
    param_names = list(cell_mod.PARAM_KEYS)

    ppr = None
    yaml_path = args.physRange or (_DEFAULT_PHYS_YAML if args.cell == "ca3_pyramidal" else None)
    if yaml_path and os.path.exists(yaml_path):
        md = read_yaml(yaml_path, verb=0)
        ppr = md.get("input_meta", {}).get("phys_par_range")
        names = md.get("input_meta", {}).get("parName")
        if names:
            param_names = list(names)
    if ppr is None and args.cell == "ca3_pyramidal":
        ppr = _CA3_PHYS_PAR_RANGE
    if ppr is None:
        raise SystemExit(
            f"could not resolve phys_par_range for cell {args.cell!r}; "
            f"pass --physRange <sum_train.yaml>")

    centers, logspans = phys_par_range_to_arrays(ppr)
    centers, logspans = centers[:P], logspans[:P]
    return centers.astype(np.float64), logspans.astype(np.float64), param_names[:P]


def discover_stims(args):
    if args.stims:
        stims = list(args.stims)
    else:
        from pathlib import Path
        stims = sorted(p.stem for p in Path(args.stimDir).glob("*.csv"))
    if args.stimStart or args.stimCount is not None:
        end = None if args.stimCount is None else args.stimStart + args.stimCount
        stims = stims[args.stimStart:end]
    if args.maxStims:
        stims = stims[:args.maxStims]
    return stims


def build_unit_samples(P, N, seed):
    """(P, N) unit draws in [-1,1]; row p sweeps channel p, reused across stims."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(P, N))


def simulate_stim(phys, cell_name, stim, solver, sim_bs, device, out_len_hint=None):
    """Simulate a (M, P) phys batch under one stim -> (M, T) soma trace (numpy)."""
    outs = []
    with torch.no_grad():
        for i in range(0, phys.shape[0], sim_bs):
            v = JaxleyBridge.simulate_batch(
                phys[i:i + sim_bs], cell_name, stim, solver=solver)
            outs.append(v[:, 0, :].cpu())     # soma probe = recorded row 0
    return torch.cat(outs, dim=0).numpy()


def main():
    args = get_parser()
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cell + param space.
    spec = jaxley_cells.get(args.cell)
    P = len(spec.param_keys)
    centers, logspans, param_names = resolve_phys_range(args, P)
    print(f"[sv] cell={args.cell}  P={P} channels={param_names}")

    # Point the bridge at the requested stim directory.
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cell}")
    cell_mod._STIM_DIR = __import__("pathlib").Path(args.stimDir)

    stims = discover_stims(args)
    print(f"[sv] {len(stims)} stims from {args.stimDir}")

    N = args.nSamples
    unit_samples = build_unit_samples(P, N, args.seed)   # (P, N)

    # Assemble the P*N unit matrix once: block p rows sweep channel p only.
    U = np.zeros((P * N, P), dtype=np.float64)
    for p in range(P):
        U[p * N:(p + 1) * N, p] = unit_samples[p]
    # The blur/dtw metrics measure distance to the ALL-DEFAULT trace (unit 0);
    # append it as one extra cell so it is simulated under every stim.
    want_ref = args.blur or args.dtw
    if want_ref:
        U = np.vstack([U, np.zeros((1, P), dtype=np.float64)])
    phys_all = centers * np.power(10.0, U * logspans)      # (P*N (+1), P)
    phys_t = torch.tensor(phys_all, dtype=torch.float64, device=device)

    outDir = args.outDir or os.path.join("sensitivity_variation", args.cell)
    os.makedirs(outDir, exist_ok=True)

    dt_stim = spec.dt_stim
    var_mean = np.full((len(stims), P), np.nan)   # mean temporal std (mV)
    var_peak = np.full((len(stims), P), np.nan)   # peak temporal std (mV)
    nan_frac = np.zeros((len(stims), P))
    traces_for_pdf = {}                           # stim -> (P, n_show, T), t axis
    blur_var = np.full((len(stims), P), np.nan) if args.blur else None
    dtw_var = np.full((len(stims), P), np.nan) if args.dtw else None
    sysv_var = np.full((len(stims), P), np.nan) if args.sysvar else None
    smv_var = np.full((len(stims), P), np.nan) if args.smoothvar else None
    trace_store = {}                              # stim -> decimated fp16 traces

    # eFEL feature-variation (optional).
    efel_feats = None
    efel_var = None
    if args.efel:
        from toolbox.soft_efel import STRONG_PLUS_SUBTHRESHOLD, FEATURE_SCALES
        efel_feats = (list(args.efelFeatures) if args.efelFeatures
                      else list(STRONG_PLUS_SUBTHRESHOLD))
        efel_scales = {f: FEATURE_SCALES[f] for f in efel_feats}
        efel_var = np.full((len(stims), P, len(efel_feats)), np.nan)  # scaled std
        print(f"[sv] eFEL features: {efel_feats}")

    ok_stims = []
    from pathlib import Path
    for si, stim in enumerate(stims):
        csv_path = Path(args.stimDir) / f"{stim}.csv"
        try:
            stim_arr = load_stim_csv(csv_path)
            if stim_arr.ndim != 1 or stim_arr.size < 2:
                raise ValueError(f"stim not 1D/too short: shape {stim_arr.shape}")
        except Exception as e:
            print(f"[sv] SKIP {stim}: {e}")
            continue

        # Match sim length to this stim (as the data generators do).
        if args.nativeTmax:
            cell_mod._T_MAX = float(len(stim_arr)) * float(dt_stim)
            JaxleyBridge.clear_cache()

        t0 = time.time()
        try:
            V = simulate_stim(phys_t, args.cell, stim, args.solver,
                              args.simBatch, device)     # (P*N, T)
        except Exception as e:
            print(f"[sv] SKIP {stim}: sim failed: {e}")
            continue

        # Peel the appended all-default reference row off before the P*N reshape.
        Vref = None
        if want_ref:
            Vref = V[P * N:P * N + 1]      # (1, T) all-default trace
            V = V[:P * N]

        T = V.shape[1]
        Vpn = V.reshape(P, N, T)
        finite = np.isfinite(Vpn)
        nan_frac[si] = 1.0 - finite.reshape(P, -1).mean(axis=1)
        per_t_std = np.nanstd(Vpn, axis=1)              # (P, T)
        var_mean[si] = np.nanmean(per_t_std, axis=1)
        var_peak[si] = np.nanmax(np.nan_to_num(per_t_std, nan=0.0), axis=1)

        # eFEL feature variation for this stim.
        if args.efel:
            try:
                efel_var[si] = compute_efel_variation(
                    V, P, N, efel_feats, efel_scales, device,
                    dt_ms=spec.dt, chunk=args.efelChunk)
            except Exception as e:
                print(f"[sv]   eFEL failed for {stim}: {e}")

        # Full-trace distance metrics: how far sweeping each channel pushes the
        # soma trace from the all-default reference (timing-tolerant).
        if want_ref:
            try:
                bvec, dvec = compute_metric_variation(
                    V, Vref, P, N, min(args.metricSamples, N), device,
                    float(dt_stim), args.blur, args.dtw, args.blurSigmasMs,
                    args.metricMaxT, args.dtwMaxT, args.dtwGamma, args.vscale)
                if args.blur:
                    blur_var[si] = bvec
                if args.dtw:
                    dtw_var[si] = dvec
            except Exception as e:
                print(f"[sv]   trace-metric failed for {stim}: {e}")

        # Ensemble-variance metrics (chaos-robust spread).
        if args.sysvar or args.smoothvar:
            try:
                svec, mvec = compute_variance_metrics(
                    V, P, N, unit_samples, device, float(dt_stim),
                    args.sysvar, args.smoothvar, args.sysvarDegree, args.smoothvarSigmaMs)
                if args.sysvar:
                    sysv_var[si] = svec
                if args.smoothvar:
                    smv_var[si] = mvec
            except Exception as e:
                print(f"[sv]   variance-metric failed for {stim}: {e}")

        # Persist the (decimated) swept traces so new measures need no re-sim.
        if args.saveTraces:
            stride = max(1, (T + args.saveTracesMaxT - 1) // args.saveTracesMaxT)
            trace_store[stim] = V[:, ::stride].astype(np.float16)

        # Keep a few traces per channel for the per-stim PDF.
        n_show = min(12, N)
        traces_for_pdf[stim] = (Vpn[:, :n_show, :].copy(),
                                np.arange(T) * dt_stim)
        ok_stims.append(stim)
        print(f"[sv] [{si+1}/{len(stims)}] {stim:<28} T={T} "
              f"var(mV)={np.array2string(var_mean[si], precision=2)} "
              f"nan={nan_frac[si].max():.2f}  ({time.time()-t0:.1f}s)")

    # Drop skipped stims.
    keep = [i for i, s in enumerate(stims) if s in ok_stims]
    stims = [stims[i] for i in keep]
    var_mean = var_mean[keep]; var_peak = var_peak[keep]; nan_frac = nan_frac[keep]
    if efel_var is not None:
        efel_var = efel_var[keep]
    if blur_var is not None:
        blur_var = blur_var[keep]
    if dtw_var is not None:
        dtw_var = dtw_var[keep]
    if sysv_var is not None:
        sysv_var = sysv_var[keep]
    if smv_var is not None:
        smv_var = smv_var[keep]
    S = len(stims)
    if S == 0:
        raise SystemExit("[sv] no stims produced usable simulations")

    if args.saveTraces and trace_store:
        np.savez(os.path.join(outDir, "saved_traces.npz"),
                 theta=unit_samples.astype(np.float32), dt_stim=float(dt_stim),
                 channels=np.array(param_names), N=int(N),
                 **{f"V__{s}": trace_store[s] for s in stims if s in trace_store})
        print(f"[sv] saved decimated traces for {len(trace_store)} stims -> {outDir}/saved_traces.npz")

    write_outputs(outDir, stims, param_names, var_mean, var_peak, nan_frac,
                  traces_for_pdf, dt_stim, args, efel_var, efel_feats,
                  blur_var, dtw_var, sysv_var, smv_var)
    print(f"[sv] DONE  {S} stims x {P} channels  -> {outDir}/  "
          f"(elapsed {time.time()-t_start:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────
# outputs
# ─────────────────────────────────────────────────────────────────────────

def write_outputs(outDir, stims, param_names, var_mean, var_peak, nan_frac,
                  traces_for_pdf, dt_stim, args, efel_var=None, efel_feats=None,
                  blur_var=None, dtw_var=None, sysv_var=None, smv_var=None):
    import csv
    S, P = var_mean.shape

    def dump_matrix(fname, M):
        with open(os.path.join(outDir, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stim"] + list(param_names))
            for i, s in enumerate(stims):
                w.writerow([s] + [f"{M[i, p]:.5f}" for p in range(P)])

    dump_matrix("variation_matrix.csv", var_mean)
    dump_matrix("variation_matrix_peak.csv", var_peak)

    # Per-channel normalization (column / column-max) for cross-channel view.
    col_max = np.nanmax(var_mean, axis=0)
    col_max_safe = np.where(col_max > 0, col_max, 1.0)
    var_norm = var_mean / col_max_safe
    dump_matrix("variation_matrix_normalized.csv", var_norm)

    # Per-channel ranking (best stim first).
    with open(os.path.join(outDir, "ranking_per_channel.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "rank", "stim", "variation_mV", "variation_norm"])
        for p in range(P):
            order = np.argsort(-var_mean[:, p])
            for rank, i in enumerate(order):
                w.writerow([param_names[p], rank + 1, stims[i],
                            f"{var_mean[i, p]:.5f}", f"{var_norm[i, p]:.4f}"])

    # summary.yaml — best stim per channel + top-3.
    summary = {"cell": args.cell, "n_stims": int(S), "n_channels": int(P),
               "n_samples_per_set": int(args.nSamples), "stim_dir": args.stimDir,
               "best_stim_per_channel": {}}
    for p in range(P):
        order = np.argsort(-var_mean[:, p])
        summary["best_stim_per_channel"][param_names[p]] = {
            "best": stims[order[0]],
            "variation_mV": float(var_mean[order[0], p]),
            "top3": [{"stim": stims[i], "variation_mV": float(var_mean[i, p])}
                     for i in order[:3]],
        }
    # ── eFEL feature-variation outputs (optional) ──────────────────────────
    if efel_var is not None and efel_feats is not None:
        write_efel_outputs(outDir, stims, param_names, efel_var, efel_feats,
                           args.topK, summary)
    # ── full-trace distance-metric outputs (optional) ──────────────────────
    if blur_var is not None:
        write_scalar_metric_outputs(
            outDir, stims, param_names, blur_var, "blur", args.topK, summary,
            unit="mean blurred-MSE to default (mV^2)",
            title="Multi-scale blurred-MSE variation (dist to default; per-channel norm)\n"
                  "brighter = sweeping this channel moves the blurred trace more")
    if dtw_var is not None:
        write_scalar_metric_outputs(
            outDir, stims, param_names, dtw_var, "dtw", args.topK, summary,
            unit="mean soft-DTW divergence to default",
            title="Soft-DTW divergence variation (dist to default; per-channel norm)\n"
                  "brighter = sweeping this channel moves the time-aligned trace more")
    if sysv_var is not None:
        write_scalar_metric_outputs(
            outDir, stims, param_names, sysv_var, "sysvar", args.topK, summary,
            unit="parameter-explained (systematic) std (mV)",
            title="Systematic (parameter-explained) variance (per-channel norm)\n"
                  "chaos-robust: brighter = this channel SYSTEMATICALLY moves the trace")
    if smv_var is not None:
        write_scalar_metric_outputs(
            outDir, stims, param_names, smv_var, "smoothvar", args.topK, summary,
            unit="smoothed-ensemble std (mV)",
            title="Smoothed-ensemble variance (per-channel norm)\n"
                  "brighter = this channel moves the low-pass (jitter-free) envelope more")
    write_yaml(summary, os.path.join(outDir, "summary.yaml"))

    _plot_summary_heatmap(outDir, stims, param_names, var_mean, var_norm)
    _plot_ranking_per_channel(outDir, stims, param_names, var_mean, args.topK)
    _plot_per_stim_pdf(outDir, stims, param_names, var_mean, traces_for_pdf)


def write_efel_outputs(outDir, stims, param_names, efel_var, efel_feats, topK, summary):
    """efel_var: (S, P, F) scaled across-sample std.  Aggregate = mean over F."""
    import csv
    S, P, F = efel_var.shape
    agg = np.nanmean(efel_var, axis=2)                    # (S, P)  overall eFEL sensitivity

    # Per-feature detail (all in one npz) + aggregate CSV.
    np.savez(os.path.join(outDir, "efel_variation_per_feature.npz"),
             efel_var=efel_var, stims=np.array(stims), channels=np.array(param_names),
             features=np.array(efel_feats))
    with open(os.path.join(outDir, "efel_variation_aggregate.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stim"] + list(param_names))
        for i, s in enumerate(stims):
            w.writerow([s] + [f"{agg[i, p]:.5f}" for p in range(P)])

    with open(os.path.join(outDir, "efel_ranking_per_channel.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "rank", "stim", "efel_sensitivity", "top_feature"])
        for p in range(P):
            order = np.argsort(-agg[:, p])
            for rank, i in enumerate(order):
                tf = efel_feats[int(np.nanargmax(efel_var[i, p]))] if np.any(np.isfinite(efel_var[i, p])) else "-"
                w.writerow([param_names[p], rank + 1, stims[i], f"{agg[i, p]:.5f}", tf])

    summary["efel_features"] = list(efel_feats)
    summary["efel_best_stim_per_channel"] = {}
    for p in range(P):
        order = np.argsort(-agg[:, p])
        summary["efel_best_stim_per_channel"][param_names[p]] = {
            "best": stims[order[0]], "efel_sensitivity": float(agg[order[0], p]),
            "top3": [{"stim": stims[i], "efel_sensitivity": float(agg[i, p])}
                     for i in order[:3]]}

    col_max = np.nanmax(agg, axis=0)
    agg_norm = agg / np.where(col_max > 0, col_max, 1.0)
    _plot_summary_heatmap(outDir, stims, param_names, agg, agg_norm,
                          fname="efel_summary_heatmap.png",
                          title="eFEL feature-variation sensitivity (per-channel normalized)\n"
                                "brighter = stim makes this channel move eFEL features more")
    _plot_ranking_per_channel(outDir, stims, param_names, agg, topK,
                              fname="efel_ranking_per_channel.png",
                              xlabel="mean scaled eFEL-feature std",
                              suptitle=f"Top-{topK} stimuli per channel by eFEL-feature variation")
    _plot_efel_feature_breakdown(outDir, param_names, efel_var, efel_feats)


def write_scalar_metric_outputs(outDir, stims, param_names, mat, name, topK,
                                summary, unit, title):
    """Emit CSV + ranking + heatmap + summary for a scalar (S, P) variation
    metric (blur / dtw), mirroring the eFEL aggregate outputs."""
    import csv
    S, P = mat.shape
    with open(os.path.join(outDir, f"{name}_variation_aggregate.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh); w.writerow(["stim"] + list(param_names))
        for i, s in enumerate(stims):
            w.writerow([s] + [f"{mat[i, p]:.6f}" for p in range(P)])

    with open(os.path.join(outDir, f"{name}_ranking_per_channel.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "rank", "stim", f"{name}_variation"])
        for p in range(P):
            order = np.argsort(-np.nan_to_num(mat[:, p], nan=-np.inf))
            for rank, i in enumerate(order):
                w.writerow([param_names[p], rank + 1, stims[i], f"{mat[i, p]:.6f}"])

    summary[f"{name}_best_stim_per_channel"] = {}
    for p in range(P):
        order = np.argsort(-np.nan_to_num(mat[:, p], nan=-np.inf))
        summary[f"{name}_best_stim_per_channel"][param_names[p]] = {
            "best": stims[order[0]], "variation": float(mat[order[0], p]),
            "top3": [{"stim": stims[i], "variation": float(mat[i, p])}
                     for i in order[:3]]}

    col_max = np.nanmax(mat, axis=0)
    mat_norm = mat / np.where(col_max > 0, col_max, 1.0)
    _plot_summary_heatmap(outDir, stims, param_names, mat, mat_norm,
                          fname=f"{name}_summary_heatmap.png", title=title)
    _plot_ranking_per_channel(outDir, stims, param_names, mat, topK,
                              fname=f"{name}_ranking_per_channel.png",
                              xlabel=unit,
                              suptitle=f"Top-{topK} stimuli per channel by {name} variation")


def _plot_summary_heatmap(outDir, stims, param_names, var_mean, var_norm,
                          fname="summary_heatmap.png", title=None):
    S, P = var_mean.shape
    if title is None:
        title = ("Voltage-variation sensitivity (per-channel normalized)\n"
                 "brighter = this stim exposes this channel more; stims sorted best→worst")
    # Sort stims by overall (mean over channels of normalized variation).
    order = np.argsort(-var_norm.mean(axis=1))
    M = var_norm[order]
    labels = [stims[i] for i in order]
    fig, ax = plt.subplots(figsize=(1.1 * P + 4, max(4, 0.28 * S + 1.5)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(P)); ax.set_xticklabels(param_names, rotation=35, ha="right")
    ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    for p in range(P):
        top = int(np.nanargmax(var_mean[:, p]))
        top_row = int(np.where(order == top)[0][0])
        ax.add_patch(plt.Rectangle((p - 0.5, top_row - 0.5), 1, 1,
                     fill=False, edgecolor="red", lw=1.5))
    fig.colorbar(im, ax=ax, fraction=0.03, label="normalized variation")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, fname), dpi=130)
    plt.close(fig)


def _plot_ranking_per_channel(outDir, stims, param_names, var_mean, topK,
                              fname="ranking_per_channel.png",
                              xlabel="mean temporal std (mV)",
                              suptitle=None):
    S, P = var_mean.shape
    if suptitle is None:
        suptitle = f"Top-{topK} stimuli per channel by voltage variation"
    ncol = 3; nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for p in range(P):
        ax = axes[p]
        order = np.argsort(-var_mean[:, p])[:topK][::-1]
        y = np.arange(len(order))
        ax.barh(y, var_mean[order, p], color="C0")
        ax.set_yticks(y); ax.set_yticklabels([stims[i] for i in order], fontsize=7)
        ax.set_title(param_names[p], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.3, axis="x")
    for p in range(P, len(axes)):
        axes[p].axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outDir, fname), dpi=130)
    plt.close(fig)


def _plot_efel_feature_breakdown(outDir, param_names, efel_var, efel_feats):
    """Heatmap: features x channels, best-over-stims scaled std (which feature
    each channel is most exposed through, and by how much)."""
    F = len(efel_feats); P = len(param_names)
    best = np.nanmax(efel_var, axis=0)          # (P, F): best stim per (channel, feature)
    M = best.T                                  # (F, P)
    fig, ax = plt.subplots(figsize=(1.1 * P + 3, 0.6 * F + 2))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xticks(range(P)); ax.set_xticklabels(param_names, rotation=35, ha="right")
    ax.set_yticks(range(F)); ax.set_yticklabels(efel_feats, fontsize=8)
    for i in range(F):
        for j in range(P):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black")
    ax.set_title("eFEL feature sensitivity (best stim per cell) — scaled across-sample std")
    fig.colorbar(im, ax=ax, fraction=0.03, label="scaled std")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "efel_feature_breakdown.png"), dpi=130)
    plt.close(fig)


def _plot_per_stim_pdf(outDir, stims, param_names, var_mean, traces_for_pdf):
    S, P = var_mean.shape
    ncol = 3; nrow = int(np.ceil(P / ncol))
    with PdfPages(os.path.join(outDir, "per_stim_variation.pdf")) as pdf:
        for si, stim in enumerate(stims):
            fig = plt.figure(figsize=(13, 3.0 + 2.6 * nrow))
            gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.1] + [1] * nrow)
            # Top: bar of per-channel variation for this stim.
            axb = fig.add_subplot(gs[0, :])
            axb.bar(range(P), var_mean[si], color="C1")
            axb.set_xticks(range(P)); axb.set_xticklabels(param_names, rotation=20, ha="right", fontsize=8)
            axb.set_ylabel("var (mV)")
            axb.set_title(f"{stim}  —  per-channel voltage variation "
                          f"(sweep 1 channel, hold rest at default)", fontsize=11)
            axb.grid(alpha=0.3, axis="y")
            # Mini-panels: mean +- std voltage band per channel.
            if stim in traces_for_pdf:
                Vpn, t = traces_for_pdf[stim]        # (P, n_show, T)
                for p in range(P):
                    ax = fig.add_subplot(gs[1 + p // ncol, p % ncol])
                    tr = Vpn[p]
                    mu = np.nanmean(tr, axis=0); sd = np.nanstd(tr, axis=0)
                    ax.fill_between(t, mu - sd, mu + sd, alpha=0.3, color="C0")
                    for k in range(min(6, tr.shape[0])):
                        ax.plot(t, tr[k], lw=0.4, alpha=0.5, color="C0")
                    ax.plot(t, mu, lw=1.0, color="k")
                    ax.set_title(f"{param_names[p]}  (var={var_mean[si, p]:.1f} mV)", fontsize=8)
                    ax.set_xlabel("t (ms)", fontsize=7); ax.set_ylabel("V (mV)", fontsize=7)
                    ax.tick_params(labelsize=6)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
