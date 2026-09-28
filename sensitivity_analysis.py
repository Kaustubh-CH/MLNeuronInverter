#!/usr/bin/env python3
"""Identifiability / sensitivity analysis of a jaxley-cell inverse problem under a stim battery.

Ported from CNN_Jaxley:sensitivity_analysis.py and generalised: cell, stim battery, stim
scale, solver dt, loss window (prefix skip) and z-space are CLI-selectable, "lin" phys rows
are honoured, and the feature x channel matrix of feature_channel_sensitivity.py is computed
from the SAME perturbed simulations (one sim pass, both views).

All quantities live in the unit-parameter space [-1,1] the CNN predicts.

  1. Jacobian  J = d(z V)/d(unit theta)  by central finite differences (+-eps) at K
     operating points drawn from the pack's test split (--h5) or uniform in the box.
  2. Fisher information  F_s = mean_k J_k^T J_k / T  per stim; information from independent
     stims ADDS:  F_multi = sum_s F_s.
  3. Marginal sensitivity sqrt(diag F) (z units per unit-step) and the raw-mV RMS twin.
  4. Cramér-Rao bound  CRB_p = sigma * sqrt((F^-1)_pp)  and the identifiability index
     CRB / prior_std  (>= 1  =>  the trace cannot beat the prior on that channel).
  5. Collinearity (correlation of F^-1: which channels trade off) and the Fisher
     eigenspectrum (smallest eigen-directions = what the trace does not see).
  6. Feature x channel matrix  S[f,p] = RMS |d soft_f / d theta_p| / scale_f  (soft-eFEL on
     raw mV, scaled exactly as HybridLoss._efel_feat_loss) + per-channel best handle.
  7. Spike counts of the unperturbed operating points per stim (what the battery evokes).

z-space: --zmode fixed (default) = the pack constants VOLT_NORM_MEAN/STD, i.e. the space
HybridLoss compares traces in (a DC shift IS visible); --zmode row z-scores every trace
(the CNN-input view, DC-blind).  --tSkipMs drops the prefix the loss skips (sim_t_skip_ms).

Ranking, single-vs-multi ratios, collinearity and eigenvectors are sigma-free; sigma only
sets the absolute CRB scale (0.7 ~ the models' own voltage RMSE_z).

Usage (GPU salloc; see scripts/run_sensitivity_salloc.sh):
  L5TTPC_NCOMP=2 python sensitivity_analysis.py --cell l5ttpc \
      --stims Roy500_icaRec_5k Roy1000_icaRec_5k --h5 <pack>.mlPack1.h5 \
      --stimScale 1.0 --simDt 0.2 --tSkipMs 99.8 -o <outDir>
"""
import os, sys, time, argparse, csv, importlib
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_FP32 = "--fp32" in sys.argv
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ["JAX_ENABLE_X64"] = "false" if _FP32 else "true"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", not _FP32)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                  unit_to_phys_np, load_stim_csv,
                                  VOLT_NORM_MEAN, VOLT_NORM_STD)
from toolbox.soft_efel import (soft_efel_features, FEATURE_SCALES, STRONG_FEATURES,
                               ALL_FEATURES, STRONG_PLUS_SUBTHRESHOLD)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="l5ttpc", help="jaxley cell module name")
    p.add_argument("--stims", nargs="+", required=True, help="stim CSV stem(s)")
    p.add_argument("--h5", default=None,
                   help="pack .mlPack1.h5: operating points (test_unit_par) and, without "
                        "--physRange, the phys_par_range")
    p.add_argument("--physRange", default=None,
                   help="yaml holding phys_par_range (input_meta / train_params.voltage_loss "
                        "/ voltage_loss / top level)")
    p.add_argument("--stimScale", type=float, default=None,
                   help="stim multiplier; None -> the cell spec's own value")
    p.add_argument("--simDt", type=float, default=None,
                   help="solver dt in ms; None -> the cell module default")
    p.add_argument("--tMax", default="auto",
                   help="'auto' = len(stim)*dt_stim per stim, else a number in ms")
    p.add_argument("--tSkipMs", type=float, default=0.0,
                   help="drop this prefix (ms) before analysis = HybridLoss sim_t_skip_ms")
    p.add_argument("--zmode", choices=["fixed", "row"], default="fixed")
    p.add_argument("--fp32", action="store_true", help="fp32 solve (default fp64)")
    p.add_argument("--numOp", type=int, default=32, help="operating points")
    p.add_argument("--eps", type=float, default=0.02, help="central FD step in unit space")
    p.add_argument("--sigma", type=float, default=0.7, help="obs. noise std in z units")
    p.add_argument("--solver", default="bwd_euler")
    p.add_argument("--simBatch", type=int, default=128)
    p.add_argument("--recIdx", type=int, default=0, help="recording row of the soma probe")
    p.add_argument("--features", default="STRONG+SUB",
                   help="STRONG | STRONG+SUB | ALL | comma-list of soft_efel names")
    p.add_argument("--spikeThr", type=float, default=-20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fakeSim", action="store_true",
                   help="dry run: synthetic traces instead of jaxley (code-path check)")
    p.add_argument("-o", "--outDir", required=True)
    return p.parse_args()


def resolve_phys_range(args):
    ppr = None
    if args.physRange:
        md = read_yaml(args.physRange, verb=0)
        for path in (("input_meta", "phys_par_range"),
                     ("train_params", "voltage_loss", "phys_par_range"),
                     ("voltage_loss", "phys_par_range"), ("phys_par_range",)):
            d = md
            for k in path:
                d = d.get(k) if isinstance(d, dict) else None
                if d is None:
                    break
            if d:
                ppr = d
                break
    if ppr is None and args.h5:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        ppr = _read_phys_par_range_from_h5(args.h5)
    if ppr is None:
        raise SystemExit("need --physRange <yaml> or --h5 <pack> for phys_par_range")
    centers, logspans = phys_par_range_to_arrays(ppr)
    lin = phys_par_range_linear_mask(ppr)
    return ppr, np.asarray(centers, np.float64), np.asarray(logspans, np.float64), np.asarray(lin, bool)


def operating_points(args, P):
    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as f:
            for key in ("test_unit_par", "valid_unit_par", "train_unit_par"):
                if key in f:
                    U = f[key][:args.numOp].astype(np.float64)[:, :P]
                    print(f"[sens] operating points: {key}[:{U.shape[0]}] of {args.h5}")
                    return U
    rng = np.random.default_rng(args.seed)
    print(f"[sens] operating points: {args.numOp} uniform draws in [-1,1]^{P}")
    return rng.uniform(-1.0, 1.0, size=(args.numOp, P))


def feature_list(spec):
    if spec == "STRONG":
        return list(STRONG_FEATURES)
    if spec == "STRONG+SUB":
        return list(STRONG_PLUS_SUBTHRESHOLD)
    if spec == "ALL":
        return list(ALL_FEATURES)
    feats = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in feats if f not in ALL_FEATURES]
    if bad:
        raise SystemExit(f"unknown feature(s) {bad}; valid: {ALL_FEATURES}")
    return feats


_LOC = {"apical": "api", "dend": "dend", "axonal": "axo", "somatic": "soma", "all": "all"}


def short_name(n):
    """gNaTs2_tbar_NaTs2_t_apical -> NaTs2_t_api ; g_pas_somatic -> pas_soma ; e_pas_all."""
    tail = n.split("_")[-1]
    if tail in _LOC:
        core, loc = n[:-(len(tail) + 1)], _LOC[tail]
    else:
        core, loc = n, None
    if "bar_" in core:
        core = core.split("bar_", 1)[1]
    core = core.replace("g_pas", "pas")
    return f"{core}_{loc}" if loc else core


def spike_counts(V, thr):
    above = V > thr
    return ((~above[:, :-1]) & above[:, 1:]).sum(axis=1)


def main():
    args = get_parser()
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cell}")
    if args.simDt is not None:
        cell_mod._DT = float(args.simDt)
    spec = jaxley_cells.get(args.cell)            # picks up _DT
    param_names = list(cell_mod.PARAM_KEYS)
    P = len(param_names)
    short = [short_name(n) for n in param_names]
    ppr, centers, logspans, lin_mask = resolve_phys_range(args)
    if len(centers) != P:
        raise SystemExit(f"phys_par_range has {len(centers)} rows, cell has {P} params")
    feats = feature_list(args.features)
    Fn = len(feats)
    fscales = np.array([FEATURE_SCALES[f] for f in feats], dtype=np.float64)
    stim_dir = Path(spec.stim_dir)
    dt_stim = float(spec.dt_stim)
    stim_scale_eff = float(spec.stim_scale) if args.stimScale is None else float(args.stimScale)
    ncomp = getattr(cell_mod, "_NCOMP", None)

    print(f"[sens] cell={args.cell} ncomp={ncomp} P={P} solver={args.solver} "
          f"dt={spec.dt} stim_scale={stim_scale_eff} fp64={not args.fp32} "
          f"zmode={args.zmode} tSkip={args.tSkipMs} ms eps={args.eps} sigma={args.sigma}")
    print(f"[sens] channels: {param_names}")
    print(f"[sens] features: {feats}")

    # ---- operating points + perturbation matrix --------------------------------
    U0 = operating_points(args, P)                              # (K,P)
    K = U0.shape[0]
    prior_std = U0.std(axis=0)
    eps = args.eps
    U = np.repeat(U0[:, None, :], 2 * P, axis=1)                # (K,2P,P)
    for p in range(P):
        U[:, 2 * p,     p] += eps
        U[:, 2 * p + 1, p] -= eps
    U_all = np.concatenate([U0, U.reshape(K * 2 * P, P)], axis=0)   # base rows first
    phys_all = unit_to_phys_np(U_all, centers, logspans, linear=lin_mask)
    M = phys_all.shape[0]
    print(f"[sens] {K} operating points -> {M} sims per stim "
          f"(prior std = {np.array2string(prior_std, precision=2)})")

    def simulate(stim):
        if args.fakeSim:
            rng = np.random.default_rng(hash(stim) % (2**32))
            T = int(round(cell_mod._T_MAX / float(spec.dt)))
            w = rng.normal(size=(P, T)) * 4.0
            base = -65.0 + U_all @ w
            spikes = 30.0 * (np.sin(np.linspace(0, 40, T)[None, :] + U_all[:, :1]) > 0.98)
            return (base + spikes).astype(np.float64), float(spec.dt)
        # Pure forward on the cached handle's jitted vmap.  JaxleyBridge.simulate_batch is
        # the TRAINING bridge: its forward captures a jax.vjp whose residuals hold every
        # solver state, which OOMs a 40 GB GPU for the 19-param L5 cell (same reason
        # evaluate_voltage._simulate_nograd exists).  Last chunk padded -> one XLA shape.
        import jax.numpy as jnp
        h = JaxleyBridge.get_handle(args.cell, stim, None, args.solver, args.stimScale)
        pp = jnp.asarray(phys_all.astype(np.float32 if args.fp32 else np.float64))
        bs = args.simBatch
        outs = []
        for i0 in range(0, M, bs):
            pg = pp[i0:i0 + bs]
            ng = pg.shape[0]
            if ng < bs:
                pg = jnp.concatenate(
                    [pg, jnp.broadcast_to(pg[:1], (bs - ng,) + pg.shape[1:])], axis=0)
            v = h.simulate_batch(pg)                                  # (bs, n_rec, T_out)
            outs.append(np.asarray(v[:ng, args.recIdx, :], dtype=np.float64))
        return np.concatenate(outs, axis=0), float(h.out_dt)

    def to_z(V):
        if args.zmode == "fixed":
            return (V - VOLT_NORM_MEAN) / VOLT_NORM_STD
        return (V - V.mean(axis=-1, keepdims=True)) / (V.std(axis=-1, keepdims=True) + 1e-6)

    # ---- per-stim simulation + analysis -----------------------------------------
    F_per, sens_per, raw_per, S_per, spikes_per, tmax_per, nbad_per = {}, {}, {}, {}, {}, {}, {}
    base_traces = {}
    out_dt = None
    for stim in args.stims:
        stim_arr = load_stim_csv(stim_dir / f"{stim}.csv")
        tmax = float(len(stim_arr)) * dt_stim if str(args.tMax).lower() == "auto" else float(args.tMax)
        cell_mod._T_MAX = tmax
        JaxleyBridge.clear_cache()
        tmax_per[stim] = tmax
        print(f"[sens] '{stim}': {len(stim_arr)} pts, t_max {tmax} ms, "
              f"I in [{stim_arr.min()*stim_scale_eff:.3f}, {stim_arr.max()*stim_scale_eff:.3f}] nA; "
              f"simulating {M} cells ...", flush=True)
        t0 = time.time()
        V, out_dt = simulate(stim)                                # (M,T) raw mV
        skip = int(round(args.tSkipMs / out_dt))
        V = V[:, skip:]
        T = V.shape[1]
        Vb, Vp = V[:K], V[K:].reshape(K, 2 * P, T)
        base_traces[stim] = Vb[:4].copy()
        spikes_per[stim] = spike_counts(Vb, args.spikeThr)

        Zp = to_z(Vp)
        J = np.empty((K, T, P))
        dV = np.empty((K, T, P))
        for p in range(P):
            J[:, :, p]  = (Zp[:, 2 * p, :] - Zp[:, 2 * p + 1, :]) / (2.0 * eps)
            dV[:, :, p] = (Vp[:, 2 * p, :] - Vp[:, 2 * p + 1, :]) / (2.0 * eps)
        ok = np.isfinite(J).all(axis=(1, 2))
        nbad_per[stim] = int((~ok).sum())
        if nbad_per[stim]:
            print(f"[sens]   WARNING {nbad_per[stim]} operating points non-finite -> dropped")
        Fm = np.zeros((P, P))
        for k in np.flatnonzero(ok):
            Fm += J[k].T @ J[k]
        Fm /= (max(int(ok.sum()), 1) * T)
        F_per[stim] = Fm
        sens_per[stim] = np.sqrt(np.diag(Fm))
        raw_per[stim] = np.sqrt(np.nanmean(dV[ok] ** 2, axis=(0, 1)))      # mV per unit

        # soft-eFEL features on RAW mV of the perturbed rows (what HybridLoss feeds them).
        fvals = {f: np.full(K * 2 * P, np.nan) for f in feats}
        Vflat = Vp.reshape(K * 2 * P, T)
        for i in range(0, Vflat.shape[0], args.simBatch):
            with torch.no_grad():
                fd = soft_efel_features(torch.tensor(Vflat[i:i + args.simBatch],
                                                     dtype=torch.float32, device=device),
                                        dt_ms=out_dt, only=feats)
            for f in feats:
                fvals[f][i:i + args.simBatch] = fd[f].detach().cpu().numpy()
        S = np.zeros((Fn, P))
        for fi, f in enumerate(feats):
            a = fvals[f].reshape(K, 2 * P)
            for p in range(P):
                d = (a[:, 2 * p] - a[:, 2 * p + 1]) / (2.0 * eps)
                d = d[np.isfinite(d)]
                S[fi, p] = np.sqrt(np.mean(d ** 2)) / fscales[fi] if d.size else 0.0
        S_per[stim] = S

        ev = np.linalg.eigvalsh(Fm)
        sc = spikes_per[stim]
        print(f"[sens]   done in {time.time()-t0:.1f}s | spikes/trace mean {sc.mean():.1f} "
              f"median {np.median(sc):.0f} min {sc.min()} max {sc.max()} | "
              f"Fisher cond {ev[-1]/max(ev[0],1e-30):.2e} (eig {ev[0]:.2e}..{ev[-1]:.2e})", flush=True)
        print(f"[sens]   sens_z = {np.array2string(sens_per[stim], precision=2, max_line_width=200)}")

    stims = list(args.stims)
    F_multi = np.sum([F_per[s] for s in stims], axis=0)
    comb_sens = np.sqrt(np.diag(F_multi))
    comb_raw = np.sqrt(np.sum([raw_per[s] ** 2 for s in stims], axis=0))
    S_multi = np.sqrt(np.sum([S_per[s] ** 2 for s in stims], axis=0))

    sigma = args.sigma

    def crb(Fm):
        lam = 1e-9 * np.trace(Fm) / P
        Finv = np.linalg.inv(Fm + lam * np.eye(P))
        return sigma * np.sqrt(np.clip(np.diag(Finv), 0, None)), Finv

    crb_per = {s: crb(F_per[s])[0] for s in stims}
    crb_multi, Finv_multi = crb(F_multi)
    d = np.sqrt(np.clip(np.diag(Finv_multi), 1e-30, None))
    corr = Finv_multi / np.outer(d, d)
    evals, evecs = np.linalg.eigh(F_multi)
    cond = float(evals[-1] / max(evals[0], 1e-30))
    ident_idx = crb_multi / np.maximum(prior_std, 1e-9)
    ident_per = {s: crb_per[s] / np.maximum(prior_std, 1e-9) for s in stims}

    # feature handles (responsive AND specific).
    rowsum = S_multi.sum(axis=1, keepdims=True) + 1e-12
    handle_score = S_multi * (S_multi / rowsum)
    best_feat = {param_names[p]: feats[int(np.argmax(handle_score[:, p]))] for p in range(P)}

    # ---- report -------------------------------------------------------------------
    single = stims[0]
    W = 14
    print("\n" + "=" * 100)
    print(f"IDENTIFIABILITY  cell={args.cell} ncomp={ncomp} stim_scale={stim_scale_eff} "
          f"dt={spec.dt} zmode={args.zmode} skip={args.tSkipMs} ms  sigma={sigma}  K={K}")
    print(f"  combined Fisher condition number = {cond:.3e}   (per-stim: " +
          ", ".join(f"{s}={np.linalg.cond(F_per[s]):.1e}" for s in stims) + ")")
    print(f"  spikes/trace at the operating points: " +
          ", ".join(f"{s}: {spikes_per[s].mean():.1f}" for s in stims))
    print("-" * 100)
    hdr = f"{'channel':<{W}}{'sens_z':>8}{'raw mV':>8}" + "".join(f"{'ii_'+s.replace('_icaRec_5k',''):>10}" for s in stims) + f"{'ii_ALL':>9}{'CRB_ALL':>9}"
    print(hdr)
    order = np.argsort(-ident_idx)
    n_unid = 0
    for p in order:
        flag = "  <-- unidentifiable" if ident_idx[p] >= 1.0 else ""
        n_unid += ident_idx[p] >= 1.0
        print(f"{short[p]:<{W}}{comb_sens[p]:>8.3f}{comb_raw[p]:>8.2f}" +
              "".join(f"{ident_per[s][p]:>10.2f}" for s in stims) +
              f"{ident_idx[p]:>9.2f}{crb_multi[p]:>9.3f}{flag}")
    print("-" * 100)
    print(f"  ii = CRB/prior_std (identifiability index; <1 = the trace beats the prior). "
          f"{P-n_unid}/{P} channels identifiable from the combined battery.")
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            if abs(corr[i, j]) > 0.8:
                pairs.append((param_names[i], param_names[j], float(corr[i, j])))
    if pairs:
        print("Strongly collinear pairs (|corr|>0.8 in F^-1 -> trade off in the trace):")
        for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"    {short_name(a):<14} <-> {short_name(b):<14} corr={c:+.2f}")
    else:
        print("No channel pair exceeds |corr|>0.8.")
    v0 = evecs[:, 0]
    print(f"Least-observable direction (smallest Fisher eigval={evals[0]:.2e}):")
    for p in np.argsort(-np.abs(v0)):
        if abs(v0[p]) > 0.2:
            print(f"    {v0[p]:+.2f} * {param_names[p]}")
    print("-" * 100)
    print(f"FEATURE x CHANNEL (scaled RMS |dF/dtheta|, combined over {len(stims)} stims)")
    print("feature".ljust(28) + "".join(s[:8].rjust(9) for s in short))
    for fi, f in enumerate(feats):
        print(f[:28].ljust(28) + "".join(f"{S_multi[fi, p]:9.3f}" for p in range(P)))
    print("per-channel best handle (responsive & specific):")
    for p in range(P):
        print(f"    {short[p]:<14} -> {best_feat[param_names[p]]:<28} (S={S_multi[:, p].max():.3f})")
    print("=" * 100 + "\n")

    # ---- outputs ------------------------------------------------------------------
    outDir = args.outDir
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, "crb_by_stim.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "prior_std", "sens_z_combined", "sens_mV_combined"]
                   + [f"CRB_{s}" for s in stims] + ["CRB_combined"]
                   + [f"ii_{s}" for s in stims] + ["ii_combined"])
        for p in range(P):
            w.writerow([param_names[p], f"{prior_std[p]:.4f}", f"{comb_sens[p]:.4f}", f"{comb_raw[p]:.4f}"]
                       + [f"{crb_per[s][p]:.4f}" for s in stims] + [f"{crb_multi[p]:.4f}"]
                       + [f"{ident_per[s][p]:.4f}" for s in stims] + [f"{ident_idx[p]:.4f}"])
    with open(os.path.join(outDir, "sensitivity_by_stim.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel"] + [f"sens_z_{s}" for s in stims] + ["sens_z_combined"]
                   + [f"sens_mV_{s}" for s in stims] + ["sens_mV_combined"])
        for p in range(P):
            w.writerow([param_names[p]] + [f"{sens_per[s][p]:.4f}" for s in stims] + [f"{comb_sens[p]:.4f}"]
                       + [f"{raw_per[s][p]:.4f}" for s in stims] + [f"{comb_raw[p]:.4f}"])
    with open(os.path.join(outDir, "feature_channel_sensitivity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature"] + param_names)
        for fi, f in enumerate(feats):
            w.writerow([f] + [f"{S_multi[fi, p]:.5f}" for p in range(P)])
    np.savez(os.path.join(outDir, "arrays.npz"),
             stims=np.array(stims), channels=np.array(param_names), features=np.array(feats),
             F_per=np.stack([F_per[s] for s in stims]), F_multi=F_multi,
             S_per=np.stack([S_per[s] for s in stims]), S_multi=S_multi,
             sens_z_per=np.stack([sens_per[s] for s in stims]),
             sens_mV_per=np.stack([raw_per[s] for s in stims]),
             crb_per=np.stack([crb_per[s] for s in stims]), crb_multi=crb_multi,
             corr=corr, evals=evals, evecs=evecs, prior_std=prior_std, U0=U0,
             spikes_per=np.stack([spikes_per[s] for s in stims]),
             base_traces=np.stack([base_traces[s] for s in stims]), out_dt=out_dt)

    summary = {
        "cell": args.cell, "ncomp": ncomp, "stims": stims, "stim_scale": stim_scale_eff,
        "solver": args.solver, "dt_ms": float(spec.dt), "out_dt_ms": out_dt, "fp64": not args.fp32,
        "t_max_ms": {s: tmax_per[s] for s in stims}, "t_skip_ms": args.tSkipMs, "zmode": args.zmode,
        "n_op_points": int(K), "eps": float(eps), "sigma": float(sigma),
        "operating_points_from": args.h5, "phys_par_range_rows": len(ppr),
        "fisher_condition_number": cond,
        "fisher_condition_per_stim": {s: float(np.linalg.cond(F_per[s])) for s in stims},
        "spikes_per_trace_mean": {s: float(spikes_per[s].mean()) for s in stims},
        "nonfinite_op_points": nbad_per,
        "channels": [
            {"name": param_names[p], "short": short[p], "prior_std": float(prior_std[p]),
             "sens_z_combined": float(comb_sens[p]), "sens_mV_combined": float(comb_raw[p]),
             "sens_z_per_stim": {s: float(sens_per[s][p]) for s in stims},
             "crb_per_stim": {s: float(crb_per[s][p]) for s in stims},
             "crb_combined": float(crb_multi[p]),
             "ident_index_per_stim": {s: float(ident_per[s][p]) for s in stims},
             "ident_index_combined": float(ident_idx[p]),
             "identifiable": bool(ident_idx[p] < 1.0),
             "best_feature": best_feat[param_names[p]]}
            for p in range(P)],
        "collinear_pairs": [{"a": a, "b": b, "corr": c} for a, b, c in pairs],
        "least_observable_direction": {param_names[p]: float(v0[p]) for p in range(P)},
        "features": feats,
        "S_feature_x_channel": [[float(x) for x in row] for row in S_multi],
    }
    write_yaml(summary, os.path.join(outDir, "summary.yaml"))

    labels = [s.replace("_icaRec_5k", "") for s in stims]
    x = np.arange(P)
    # 1) marginal sensitivity per stim.
    fig, ax = plt.subplots(figsize=(0.75 * P + 3, 4.6))
    wb = 0.8 / len(stims)
    for si, s in enumerate(stims):
        ax.bar(x + si * wb, sens_per[s], wb, label=labels[si])
    ax.set_xticks(x + 0.4 - wb / 2); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("marginal sensitivity ||dV_z/dtheta|| (RMS)")
    ax.set_title(f"{args.cell} nc{ncomp}: per-channel voltage sensitivity by stimulus "
                 f"(scale {stim_scale_eff}, dt {spec.dt}, {args.zmode}-z, skip {args.tSkipMs} ms)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "sensitivity_bars.png"), dpi=120); plt.close(fig)
    # 2) identifiability index per stim + combined, prior line at 1.
    fig, ax = plt.subplots(figsize=(0.75 * P + 3, 4.6))
    wb = 0.8 / (len(stims) + 1)
    for si, s in enumerate(stims):
        ax.bar(x + si * wb, ident_per[s], wb, label=f"{labels[si]} only")
    ax.bar(x + len(stims) * wb, ident_idx, wb, label="all stims", color="k")
    ax.axhline(1.0, color="r", ls="--", lw=1, label="prior (unidentifiable above)")
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - wb / 2); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"CRB / prior std  (sigma={sigma})")
    ax.set_title("Identifiability index: lower = better recovered; >=1 = voltage cannot beat the prior")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "crb_bars.png"), dpi=120); plt.close(fig)
    # 3) collinearity.
    fig, ax = plt.subplots(figsize=(0.55 * P + 3, 0.55 * P + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(P)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(P)); ax.set_yticklabels(short, fontsize=7)
    for i in range(P):
        for j in range(P):
            if abs(corr[i, j]) > 0.5 and i != j:
                ax.text(j, i, f"{corr[i,j]:+.1f}", ha="center", va="center", fontsize=5)
    ax.set_title("Estimator collinearity (combined F^-1); |corr| -> 1 = channels trade off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "collinearity.png"), dpi=120); plt.close(fig)
    # 4) eigenspectrum.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].semilogy(range(1, P + 1), evals[::-1], "o-")
    axes[0].set_xlabel("eigen-index (large -> small)"); axes[0].set_ylabel("Fisher eigenvalue")
    axes[0].set_title(f"Fisher spectrum (cond={cond:.1e})"); axes[0].grid(alpha=0.3)
    im = axes[1].imshow(evecs[:, ::-1], vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axes[1].set_yticks(range(P)); axes[1].set_yticklabels(short, fontsize=7)
    axes[1].set_xlabel("eigenvector (large -> small eigval)")
    axes[1].set_title("Eigenvectors: rightmost = least-observable combination")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "eigenspectrum.png"), dpi=120); plt.close(fig)
    # 5) feature x channel heatmap.
    fig, ax = plt.subplots(figsize=(0.7 * P + 3, 0.45 * Fn + 2))
    im = ax.imshow(S_multi, aspect="auto", cmap="viridis")
    ax.set_xticks(range(P)); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(Fn)); ax.set_yticklabels(feats, fontsize=8)
    for fi in range(Fn):
        for p in range(P):
            ax.text(p, fi, f"{S_multi[fi, p]:.2f}", ha="center", va="center", fontsize=5, color="w")
    for p in range(P):
        fi = int(np.argmax(handle_score[:, p]))
        ax.add_patch(plt.Rectangle((p - .5, fi - .5), 1, 1, fill=False, edgecolor="red", lw=1.5))
    ax.set_title("Feature x channel sensitivity (scaled |dF/dtheta|); red = best per-channel handle")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "feature_heatmap.png"), dpi=120); plt.close(fig)
    # 6) what the battery evokes: base traces of the first op points.
    fig, axes = plt.subplots(len(stims), 1, figsize=(12, 2.2 * len(stims)), sharex=True, squeeze=False)
    for si, s in enumerate(stims):
        ax = axes[si, 0]
        tb = base_traces[s]
        t = args.tSkipMs + np.arange(tb.shape[1]) * out_dt
        for k in range(tb.shape[0]):
            ax.plot(t, tb[k], lw=0.6)
        ax.set_ylabel("mV"); ax.set_title(f"{labels[si]}: spikes/trace mean {spikes_per[s].mean():.1f} "
                                          f"(K={K})", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("ms")
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "stim_traces.png"), dpi=110); plt.close(fig)

    print(f"[sens] wrote csv/yaml/npz + 6 plots to {outDir}/  (elapsed {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
