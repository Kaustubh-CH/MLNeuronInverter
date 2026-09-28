#!/usr/bin/env python3
"""Feature x Channel sensitivity matrix for the CA3 (or any jaxley-cell) inverse.

Companion to `sensitivity_analysis.py` (which builds the VOLTAGE Jacobian ->
Fisher) but works in soft-eFEL FEATURE space.  It answers the two questions the
voltage-only recovery needs:

  (i)  which differentiable soft-eFEL feature is the best HANDLE for each channel
       (i.e. the feature whose value moves most, and most SPECIFICALLY, when that
       one conductance changes) -> tells you what to put in
       voltage_loss.efel_features;
  (ii) how strongly the trace's FEATURES respond to each channel -> fills
       voltage_loss.grad_precond.sensitivity so a weakly-observed-but-identifiable
       channel gets an un-starved gradient (w_p = sens_p ** -exponent).

WHAT it computes (all in the unit-parameter space [-1,1] the CNN predicts):

  * S[f, p] = RMS over K operating points of  |d soft_f / d theta_p| / scale_f
      central finite differences in unit space; scale_f = FEATURE_SCALES[f] so
      features are comparable exactly as HybridLoss._efel_feat_loss divides them.
  * information ADDS across independent stims:  S_multi[f,p] = sqrt(sum_s S_s^2).
  * per-channel BEST feature = argmax_f ( S[f,p] * specificity[f,p] ), where
      specificity[f,p] = S[f,p] / sum_p' S[f,p']  (fraction of feature f's total
      sensitivity attributable to channel p) — rewards features that are BOTH
      responsive to p AND not driven by every other channel.
  * grad_precond.sensitivity vector s_p : the per-channel aggregate feature
      sensitivity (L2 over the requested feature subset), ready to paste into the
      design YAML.  Also emits the raw VOLTAGE marginal sensitivity for reference.

This is the instrument the FEATURE designer uses to confirm a newly-added
per-channel differentiable handle actually lights up its target channel's column
(e.g. a Kdr-specific ap_downstroke feature should dominate row for CA3_gkdrbar_kdr).

Run under an interactive salloc (see run_feature_sensitivity_salloc.sh):
  python feature_channel_sensitivity.py --cell ca3_pyramidal \
      --stims 4k50kInterChaoticB --physRange <hpar-or-sum_train>.yaml \
      --h5 <pack>.mlPack1.h5 --numOp 32 --features STRONG
"""

import os, sys, time, argparse, csv, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                  unit_to_phys_np, load_stim_csv)
from toolbox.soft_efel import (
    soft_efel_features, FEATURE_SCALES, FEATURES, STRONG_FEATURES,
    ALL_FEATURES, DVDT_FEATURES, KCHAN_FEATURES,
)

# CA3 fallback phys range if neither --physRange nor --h5 is given.
_CA3_PHYS_PAR_RANGE = None  # resolved lazily from a run yaml if available


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="ca3_pyramidal", help="jaxley cell module name")
    p.add_argument("--stims", nargs="+", required=True,
                   help="stim CSV stem(s); information adds across independent stims")
    p.add_argument("--stimDir", default=None,
                   help="override the cell's stim dir (else uses spec.stim_dir)")
    p.add_argument("--physRange", default=None,
                   help="yaml with input_meta.phys_par_range (sum_train.yaml or hpar); "
                        "otherwise read from --h5 pack meta")
    p.add_argument("--h5", default=None,
                   help="pack .mlPack1.h5: source of phys_par_range (if no --physRange) "
                        "AND of realistic operating points (test_unit_par)")
    p.add_argument("--numOp", type=int, default=32,
                   help="operating points to average |dFeature/dtheta| over")
    p.add_argument("--eps", type=float, default=0.03,
                   help="central finite-difference step in unit space")
    p.add_argument("--features", default="STRONG",
                   help="'STRONG' | 'ALL' | comma-list of soft_efel feature names")
    p.add_argument("--solver", default="bwd_euler")
    p.add_argument("--simBatch", type=int, default=128)
    p.add_argument("--nativeTmax", action="store_true", default=True,
                   help="set _T_MAX = len(stim)*dt_stim per stim (default on)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outDir", default="feature_sensitivity")
    return p.parse_args()


def resolve_phys_range(args, cell_mod, P):
    ppr = None
    if args.physRange and os.path.exists(args.physRange):
        md = read_yaml(args.physRange, verb=0)
        ppr = (md.get("input_meta", {}).get("phys_par_range")
               or md.get("phys_par_range"))
    if ppr is None and args.h5:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        ppr = _read_phys_par_range_from_h5(args.h5)
    if ppr is None:
        raise SystemExit("need --physRange <yaml> or --h5 <pack> to get phys_par_range")
    centers, logspans = phys_par_range_to_arrays(ppr)
    lin = phys_par_range_linear_mask(ppr)
    return centers[:P].astype(np.float64), logspans[:P].astype(np.float64), lin[:P]


def operating_points(args, P):
    """(K, P) unit vectors: realistic draws from the pack test split if --h5,
    else uniform in [-1,1]."""
    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as f:
            key = "test_unit_par" if "test_unit_par" in f else \
                  ("valid_unit_par" if "valid_unit_par" in f else None)
            if key is not None:
                U = f[key][:args.numOp].astype(np.float64)[:, :P]
                return U
    rng = np.random.default_rng(args.seed)
    return rng.uniform(-1.0, 1.0, size=(args.numOp, P))


def feature_list(spec):
    if spec == "STRONG":
        return list(STRONG_FEATURES)
    if spec == "ALL":
        return list(ALL_FEATURES)
    if spec == "KCHAN":
        # the L4 per-channel handles + their dV/dt & eFEL neighbours, for validating
        # that each new handle isolates its target channel (Kdr / Km).
        return list(dict.fromkeys(
            DVDT_FEATURES + KCHAN_FEATURES +
            ["AHP_depth_abs_slow", "time_to_first_spike", "inv_first_ISI",
             "adaptation_index", "mean_frequency", "AP_amplitude"]))
    feats = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in feats if f not in ALL_FEATURES]
    if bad:
        raise SystemExit(f"unknown feature(s) {bad}; valid: {ALL_FEATURES}")
    return feats


def main():
    args = get_parser()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = jaxley_cells.get(args.cell)
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cell}")
    param_names = list(cell_mod.PARAM_KEYS)
    P = len(param_names)
    centers, logspans, lin_mask = resolve_phys_range(args, cell_mod, P)
    feats = feature_list(args.features)
    F = len(feats)
    scales = np.array([FEATURE_SCALES[f] for f in feats], dtype=np.float64)

    if args.stimDir:
        cell_mod._STIM_DIR = __import__("pathlib").Path(args.stimDir)
    stim_dir = getattr(cell_mod, "_STIM_DIR", spec.stim_dir)
    dt_ms = float(spec.dt_stim)

    print(f"[fs] cell={args.cell} P={P} feats={feats}")
    print(f"[fs] channels={param_names}")

    # --- operating points + central-difference perturbation matrix -----------
    U0 = operating_points(args, P)                       # (K, P)
    K = U0.shape[0]
    eps = args.eps
    # (K, 2P, P): row 2p = +eps on channel p, row 2p+1 = -eps.
    U = np.repeat(U0[:, None, :], 2 * P, axis=1)
    for p in range(P):
        U[:, 2 * p,     p] += eps
        U[:, 2 * p + 1, p] -= eps
    Uf = U.reshape(K * 2 * P, P)
    phys = torch.tensor(unit_to_phys_np(Uf, centers, logspans, linear=lin_mask),
                        dtype=torch.float64, device=device)      # (K*2P, P)

    from pathlib import Path
    S_per_stim = {}      # stim -> (F, P) scaled RMS |dFeature/dtheta|
    Vsens_per_stim = {}  # stim -> (P,) raw voltage marginal sensitivity (reference)
    for stim in args.stims:
        stim_arr = load_stim_csv(Path(stim_dir) / f"{stim}.csv")
        if args.nativeTmax:
            cell_mod._T_MAX = float(len(stim_arr)) * dt_ms
            JaxleyBridge.clear_cache()
        print(f"[fs] '{stim}': simulating {K*2*P} perturbed cells "
              f"(T={len(stim_arr)}) ...", flush=True)
        ts = time.time()
        outs = []
        with torch.no_grad():
            for i in range(0, phys.shape[0], args.simBatch):
                v = JaxleyBridge.simulate_batch(
                    phys[i:i + args.simBatch], args.cell, stim, solver=args.solver)
                outs.append(v[:, 0, :].to(torch.float64).cpu())   # soma row 0, RAW mV
        V = torch.cat(outs, dim=0)                                # (K*2P, T) raw mV

        # soft-eFEL features on RAW mV (what HybridLoss feeds them).
        fvals = {f: np.full(V.shape[0], np.nan) for f in feats}
        for i in range(0, V.shape[0], args.simBatch):
            with torch.no_grad():
                fd = soft_efel_features(V[i:i + args.simBatch].to(device),
                                        dt_ms=dt_ms, only=feats)
            for f in feats:
                fvals[f][i:i + args.simBatch] = fd[f].detach().cpu().numpy()

        # central diff per feature per channel, RMS over operating points.
        Vnp = V.numpy().reshape(K, 2 * P, -1)
        S = np.zeros((F, P))
        for fi, f in enumerate(feats):
            a = fvals[f].reshape(K, 2 * P)
            for p in range(P):
                d = (a[:, 2 * p] - a[:, 2 * p + 1]) / (2.0 * eps)   # (K,)
                d = d[np.isfinite(d)]
                S[fi, p] = np.sqrt(np.mean(d ** 2)) / scales[fi] if d.size else 0.0
        S_per_stim[stim] = S

        # raw voltage marginal sensitivity ||dV_z/dtheta|| (reference for grad_precond).
        Vz = (Vnp - Vnp.mean(-1, keepdims=True)) / (Vnp.std(-1, keepdims=True) + 1e-6)
        vs = np.zeros(P)
        for p in range(P):
            dv = (Vz[:, 2 * p, :] - Vz[:, 2 * p + 1, :]) / (2.0 * eps)  # (K,T)
            vs[p] = np.sqrt(np.nanmean(dv ** 2))
        Vsens_per_stim[stim] = vs
        print(f"[fs]   done in {time.time()-ts:.1f}s", flush=True)

    # information adds across independent stims.
    S_multi = np.sqrt(np.sum([S ** 2 for S in S_per_stim.values()], axis=0))   # (F,P)
    Vsens_multi = np.sqrt(np.sum([v ** 2 for v in Vsens_per_stim.values()], axis=0))

    # specificity: fraction of feature f's total sensitivity due to channel p.
    row = S_multi.sum(axis=1, keepdims=True) + 1e-12
    spec_mat = S_multi / row
    handle_score = S_multi * spec_mat                        # (F,P) responsive AND specific
    best_feat = {param_names[p]: feats[int(np.argmax(handle_score[:, p]))]
                 for p in range(P)}

    # grad_precond.sensitivity: per-channel aggregate FEATURE sensitivity over the
    # subset (L2), plus the best-feature-only variant.  Larger sens -> smaller
    # weight (that channel already gets a strong gradient).
    s_feat_l2  = np.sqrt((S_multi ** 2).sum(axis=0))         # (P,)
    s_feat_max = S_multi.max(axis=0)                         # (P,)

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"FEATURE x CHANNEL SENSITIVITY  (scaled RMS |dF/dtheta|, {len(args.stims)} stim)")
    print("-" * 74)
    hdr = "feature".ljust(22) + "".join(n.split("_")[-1].rjust(9) for n in param_names)
    print(hdr)
    for fi, f in enumerate(feats):
        print(f[:22].ljust(22) + "".join(f"{S_multi[fi, p]:9.3f}" for p in range(P)))
    print("-" * 74)
    print("per-channel BEST handle (responsive & specific):")
    for p in range(P):
        print(f"    {param_names[p]:<20} -> {best_feat[param_names[p]]:<22}"
              f" (S={s_feat_max[p]:.3f})")
    print("-" * 74)
    print("grad_precond.sensitivity candidates (paste into voltage_loss.grad_precond):")
    print(f"    # feature-L2  : {[round(float(x),4) for x in s_feat_l2]}")
    print(f"    # feature-max : {[round(float(x),4) for x in s_feat_max]}")
    print(f"    # voltage-MSE : {[round(float(x),4) for x in Vsens_multi]}")
    print("=" * 74 + "\n")

    # ---- outputs ------------------------------------------------------------
    os.makedirs(args.outDir, exist_ok=True)
    with open(os.path.join(args.outDir, "feature_channel_sensitivity.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature"] + param_names)
        for fi, f in enumerate(feats):
            w.writerow([f] + [f"{S_multi[fi, p]:.5f}" for p in range(P)])

    write_yaml({
        "cell": args.cell, "stims": list(args.stims), "features": feats,
        "n_op_points": int(K), "eps": float(eps),
        "channels": param_names,
        "S_feature_x_channel": [[float(x) for x in row] for row in S_multi],
        "best_feature_per_channel": best_feat,
        "grad_precond_sensitivity": {
            "feature_l2":  [float(x) for x in s_feat_l2],
            "feature_max": [float(x) for x in s_feat_max],
            "voltage_mse": [float(x) for x in Vsens_multi],
        },
    }, os.path.join(args.outDir, "feature_sensitivity_summary.yaml"))

    # heatmap: rows=features, cols=channels (short names), annotate.
    fig, ax = plt.subplots(figsize=(1.1 * P + 3, 0.5 * F + 2))
    im = ax.imshow(S_multi, aspect="auto", cmap="viridis")
    ax.set_xticks(range(P))
    ax.set_xticklabels([n.split("_")[-1] for n in param_names], rotation=30, ha="right")
    ax.set_yticks(range(F)); ax.set_yticklabels(feats, fontsize=8)
    for fi in range(F):
        for p in range(P):
            ax.text(p, fi, f"{S_multi[fi, p]:.2f}", ha="center", va="center",
                    fontsize=6, color="w")
    # star the best handle per channel.
    for p in range(P):
        fi = int(np.argmax(handle_score[:, p]))
        ax.add_patch(plt.Rectangle((p - .5, fi - .5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
    ax.set_title("Feature x Channel sensitivity (scaled |dF/dtheta|)\n"
                 "red box = best per-channel handle")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outDir, "feature_channel_sensitivity.png"), dpi=120)
    plt.close(fig)

    print(f"[fs] wrote CSV + summary.yaml + heatmap to {args.outDir}/  "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
