#!/usr/bin/env python3
"""Look directly at the ChaoticRamp trace ensemble per channel and test which
VARIANCE measure actually reflects the fan-out you see in the traces.

Motivation: distance-to-reference metrics (raw MSE, blurred MSE, soft-DTW) all
SATURATE for a chaotic trajectory — once two traces decorrelate in phase the
distance sits at ~the attractor diameter regardless of how big the parameter
change was, so they cannot grade sensitivity.  The visible fan-out is ENSEMBLE
VARIANCE, not a pairwise distance, and it must be measured in a representation
where chaotic spike-timing jitter does not dominate.

For one stim, sweep each channel over [-1,1] (others default) and compute, per
channel, several candidate spread measures on the (N, T) trace ensemble:

  raw_std        mean_t std_N V(t)                     (current metric)
  smooth_std@σ   mean_t std_N (lowpass_σ V)(t)         blurred VARIANCE (not dist)
  rate_std       mean_t std_N r(t), r = smoothed spike rate (Hz)   rate-space spread
  pca_totvar     total ensemble variance = Σ singular^2 / N
  pca_dim        participation ratio (effective # of trace-space dims used)

Writes traces + measures to an npz and a per-channel diagnostic PDF.
Run inside a GPU salloc (JAX_PLATFORMS=cuda).
"""
import os, sys, time, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import jax; jax.config.update("jax_enable_x64", True)

from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import load_stim_csv
from toolbox.trace_metrics import (gaussian_kernel, multiscale_blurred_mse, soft_dtw_to_ref,
                                    decimate, parameter_explained_var)
import sensitivity_variation as sv
from types import SimpleNamespace


def smooth(x, sigma_samp):
    """Gaussian low-pass along time for (K,T) torch tensor."""
    k = gaussian_kernel(sigma_samp, x.device, x.dtype); pad = k.shape[-1] // 2
    y = torch.nn.functional.conv1d(x.unsqueeze(1), k, padding=pad)
    return y[:, :, :x.shape[1]].squeeze(1)


def spike_rate(V, dt_ms, thr=-20.0, sigma_ms=15.0):
    """Smoothed instantaneous firing rate (Hz) from threshold up-crossings."""
    up = ((V[:, 1:] > thr) & (V[:, :-1] <= thr)).to(V.dtype)
    up = torch.nn.functional.pad(up, (1, 0))
    return smooth(up, sigma_ms / dt_ms) / (dt_ms / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="ca3_pyramidal")
    ap.add_argument("--stim", default="5kChaoticRamp")
    ap.add_argument("--stimDir",
                    default="/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/stim_interpolated")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--sysDegree", type=int, default=5,
                    help="polynomial degree for the parameter-explained (systematic) variance")
    ap.add_argument("--solver", default="bwd_euler")
    ap.add_argument("--simBatch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outDir", default="/pscratch/sd/k/ktub1999/tmp_neuInv/chaoticramp_probe")
    args = ap.parse_args()
    os.makedirs(args.outDir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = jaxley_cells.get(args.cell)
    P = len(spec.param_keys)
    centers, logspans, names = sv.resolve_phys_range(SimpleNamespace(cell=args.cell, physRange=None), P)
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cell}")
    cell_mod._STIM_DIR = __import__("pathlib").Path(args.stimDir)
    dt_ms = float(spec.dt_stim)

    # build sweep: block p sweeps channel p over [-1,1]; append all-default ref.
    rng = np.random.default_rng(args.seed)
    us = rng.uniform(-1, 1, size=(P, args.N))
    U = np.zeros((P * args.N, P))
    for p in range(P):
        U[p * args.N:(p + 1) * args.N, p] = us[p]
    U = np.vstack([U, np.zeros((1, P))])
    phys = torch.tensor(centers * np.power(10.0, U * logspans), dtype=torch.float64, device=dev)

    stim_arr = load_stim_csv(__import__("pathlib").Path(args.stimDir) / f"{args.stim}.csv")
    cell_mod._T_MAX = float(len(stim_arr)) * dt_ms
    JaxleyBridge.clear_cache()
    print(f"[probe] stim={args.stim} P={P} N={args.N} T={len(stim_arr)} dt={dt_ms}ms", flush=True)
    t0 = time.time()
    V = sv.simulate_stim(phys, args.cell, args.stim, args.solver, args.simBatch, dev)  # (P*N+1, T)
    print(f"[probe] simulated {V.shape} in {time.time()-t0:.1f}s", flush=True)
    Vref = V[P * args.N:P * args.N + 1]
    V = V[:P * args.N]
    T = V.shape[1]
    t_ms = np.arange(T) * dt_ms

    sigmas_ms = [5.0, 10.0, 20.0, 40.0]
    rows = {}
    curves = {}     # channel -> dict of per-t curves
    traces = {}
    for p in range(P):
        Vp = V[p * args.N:(p + 1) * args.N]                    # (N,T) numpy
        finite = np.isfinite(Vp).all(1)
        Vp = Vp[finite]
        thp = us[p][finite]                                    # swept unit theta (N,)
        Vt = torch.tensor(Vp, dtype=torch.float64, device=dev)
        tht = torch.tensor(thp, dtype=torch.float64, device=dev)
        # --- systematic (parameter-explained) variance, raw & smoothed ---
        sys_var, tot_var = parameter_explained_var(Vt, tht, degree=args.sysDegree)
        sys_std = float(torch.sqrt(sys_var).mean().cpu())      # mean_t systematic std (mV)
        tot_std = float(torch.sqrt(tot_var).mean().cpu())
        sys_frac = float((sys_var.mean() / (tot_var.mean() + 1e-9)).cpu())   # variance R^2
        Vsm20 = smooth(Vt, 20.0 / dt_ms)
        sys_var_sm, _ = parameter_explained_var(Vsm20, tht, degree=args.sysDegree)
        sys_std_sm = float(torch.sqrt(sys_var_sm).mean().cpu())
        # raw std curve
        sig_raw = np.nanstd(Vp, axis=0)                        # (T,)
        # smoothed-voltage variance (blurred VARIANCE, not distance)
        sm_std = {}
        for s in sigmas_ms:
            vs = smooth(Vt, s / dt_ms)
            sm_std[s] = vs.std(0).cpu().numpy()
        # firing-rate spread
        r = spike_rate(Vt, dt_ms, sigma_ms=15.0)               # (N,T) Hz
        sig_rate = r.std(0).cpu().numpy()
        mean_rate = r.mean(0).cpu().numpy()
        # PCA of centered ensemble
        Xc = Vp - Vp.mean(0, keepdims=True)
        sv_ = np.linalg.svd(Xc, compute_uv=False)
        ev = sv_ ** 2 / max(1, Vp.shape[0])
        totvar = float(ev.sum())
        pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))   # participation ratio
        # distance-to-ref (to demonstrate saturation): raw & blur mean dist
        refT = torch.tensor(Vref, dtype=torch.float64, device=dev).view(1, -1)
        dist_raw = float(((Vt - refT) ** 2).mean(1).mean())
        dist_blur = float(multiscale_blurred_mse(Vt, refT,
                          sigmas=tuple(max(1.0, s / dt_ms) for s in sigmas_ms)).mean())

        rows[names[p]] = dict(
            raw_std=float(np.nanmean(sig_raw)), raw_peak=float(np.nanmax(sig_raw)),
            **{f"smooth_std@{int(s)}ms": float(np.nanmean(v)) for s, v in sm_std.items()},
            rate_std=float(np.nanmean(sig_rate)), rate_peak=float(np.nanmax(sig_rate)),
            sys_std=sys_std, sys_frac=sys_frac, sys_std_sm=sys_std_sm, tot_std=tot_std,
            pca_totvar=totvar, pca_dim=pr,
            dist_raw=dist_raw, dist_blur=dist_blur, n_ok=int(Vp.shape[0]))
        curves[names[p]] = dict(sig_raw=sig_raw, sm_std=sm_std, sig_rate=sig_rate,
                                mean_rate=mean_rate)
        traces[names[p]] = Vp[:24]
        print(f"[probe] {names[p]:16s} raw_std={rows[names[p]]['raw_std']:.2f} "
              f"sys_std={sys_std:.2f}({100*sys_frac:.0f}%syst) smooth40={rows[names[p]]['smooth_std@40ms']:.2f} "
              f"rate_std={rows[names[p]]['rate_std']:.2f}Hz dist_raw={dist_raw:.1f}", flush=True)

    # SAVE THE FULL VOLTAGES this time: V_all (P*N, T), the swept thetas, and the ref.
    np.savez(os.path.join(args.outDir, f"{args.stim}_probe.npz"),
             V_all=V.astype(np.float32), theta=us.astype(np.float32),
             ref=Vref.astype(np.float32), N=args.N,
             traces={k: v for k, v in traces.items()}, rows=rows, curves=curves,
             t_ms=t_ms, names=np.array(names), stim=args.stim, dt_ms=dt_ms, allow_pickle=True)
    print(f"[probe] saved full voltages V_all={V.shape} + theta + ref to the npz", flush=True)

    # ---- table ----
    keys = ["raw_std", "sys_std", "sys_frac", "sys_std_sm", "smooth_std@40ms",
            "rate_std", "pca_dim", "dist_raw", "dist_blur"]
    print("\n" + "=" * 100)
    print("VARIANCE MEASURES per channel  (stim=" + args.stim + ")")
    print("channel".ljust(16) + "".join(k.replace("smooth_std", "sm").rjust(11) for k in keys))
    for p in range(P):
        r = rows[names[p]]
        print(names[p].ljust(16) + "".join(f"{r[k]:11.2f}" for k in keys))
    print("=" * 100)
    # discrimination: coefficient of variation across channels (higher = better separates channels)
    print("cross-channel spread (max/min across the 6 channels) — how well each measure SEPARATES channels:")
    for k in keys:
        vals = np.array([rows[names[p]][k] for p in range(P)])
        print(f"   {k:16s} max/min = {vals.max()/(vals.min()+1e-9):7.2f}   values={np.round(vals,2)}")

    # ---- plots ----
    with PdfPages(os.path.join(args.outDir, f"{args.stim}_probe.pdf")) as pdf:
        for p in range(P):
            c = curves[names[p]]
            fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
            for k in range(min(20, traces[names[p]].shape[0])):
                ax[0].plot(t_ms, traces[names[p]][k], lw=0.4, alpha=0.5)
            ax[0].set_title(f"{names[p]} — 20 swept traces (raw mV)")
            ax[1].plot(t_ms, c["mean_rate"], "k"); ax[1].fill_between(
                t_ms, c["mean_rate"] - c["sig_rate"], c["mean_rate"] + c["sig_rate"], alpha=0.3)
            ax[1].set_title("mean firing rate r(t) ± std across sweep (Hz)")
            ax[2].plot(t_ms, c["sig_raw"], label="raw std(t)")
            for s, v in c["sm_std"].items():
                ax[2].plot(t_ms, v, label=f"smooth {int(s)}ms std(t)")
            ax[2].legend(fontsize=7); ax[2].set_title("voltage spread std_N(t): raw vs smoothed")
            ax[3].plot(t_ms, c["sig_rate"], "C3"); ax[3].set_title("rate spread std_N r(t) (Hz)")
            ax[3].set_xlabel("t (ms)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[probe] wrote {args.outDir}/{args.stim}_probe.{{npz,pdf}}")


if __name__ == "__main__":
    main()
