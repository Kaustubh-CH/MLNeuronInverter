#!/usr/bin/env python3
"""Identifiability / sensitivity analysis for the CA3 (or any jaxley-cell) inverse problem.

WHY: voltage-only training can only recover the conductances the soma trace is
actually *sensitive* to.  This tool quantifies, per stimulus and for the combined
multi-stim battery, HOW identifiable each channel is — independent of the CNN.

WHAT it computes (all in the unit-parameter space the CNN predicts, i.e. the
[-1,1] log-scaled space the data was generated in):

  1. Jacobian  J = d(z-scored V)/d(unit_theta)   via central finite differences,
     at K operating points sampled from the pack's true unit_par distribution.
     (Finite-diff, not autodiff: only needs the proven forward sim.)
  2. Fisher information  F_s = mean_k (J_k^T J_k) / T   per stim s.
     Information from independent stims ADDS:  F_multi = sum_s F_s.
  3. Marginal sensitivity  sqrt(diag(F))  — how much each channel moves the trace.
  4. Cramér-Rao bound  CRB_p = sigma * sqrt((F^-1)_pp)  — the best-possible
     posterior std of channel p.  CRB >= prior std  =>  unidentifiable from voltage.
  5. Collinearity: correlation matrix from F^-1 — which channels TRADE OFF
     (|corr|~1  =>  a degenerate pair the trace cannot separate).
  6. Eigenspectrum of F_multi — smallest eigen-directions are the null-space
     (parameter combinations the voltage does not see).

Most conclusions (ranking, single-vs-multi CRB ratio, collinearity, eigenvectors)
are INDEPENDENT of the noise sigma; sigma only sets the absolute CRB scale.

Outputs under <outDir> (default <modelPath>/sensitivity/):
  sensitivity_bars.png / .csv   per-channel marginal sensitivity, per stim
  crb_bars.png / crb_by_stim.csv  Cramer-Rao bound per channel, each stim + combined
  collinearity.png              P x P channel correlation (combined)
  eigenspectrum.png             F_multi eigenvalues + most-degenerate directions
  summary.yaml                  machine-readable rollup
  (printed) ranked identifiability + interpretation

Usage:
  python sensitivity_analysis.py --modelPath <run_dir>/out [--numOp 24] [--stims a b c]
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox import JaxleyBridge
from toolbox.jaxley_utils import phys_par_range_to_arrays


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", required=True,
                   help="a run's out/ (used only for config: phys_par_range, stims, cell)")
    p.add_argument("--numOp", type=int, default=24,
                   help="number of operating points (cells) to average the Jacobian over")
    p.add_argument("--eps", type=float, default=0.02,
                   help="central finite-difference step in unit space")
    p.add_argument("--sigma", type=float, default=0.7,
                   help="observation noise std in z-scored voltage space (sets CRB scale; "
                        "0.7 ~ the models' own voltage RMSE_z). Ranking/ratios are sigma-free.")
    p.add_argument("--stims", nargs="*", default=None,
                   help="override stim list (default: stim_names_multi or stim_name from yaml)")
    p.add_argument("-o", "--outDir", default=None)
    return p.parse_args()


def main():
    args = get_parser()
    t_start = time.time()

    trainMD = read_yaml(os.path.join(args.modelPath, "sum_train.yaml"), verb=0)
    vl = trainMD["train_params"]["voltage_loss"]
    cell_name  = vl["cell_name_for_sim"]
    clamp_tanh = bool(vl.get("clamp_unit_tanh", False))
    solver     = vl.get("solver", "bwd_euler")
    stim_name  = vl.get("stim_name")
    t_max_override = vl.get("t_max_override")

    # Stim battery.
    if args.stims:
        stims = list(args.stims)
    else:
        stims = vl.get("stim_names_multi") or [stim_name]
    print(f"[sens] cell={cell_name} stims={stims} solver={solver} clamp_tanh={clamp_tanh}")

    # phys_par_range -> centers, logspans.
    phys_par_range = vl.get("phys_par_range")
    if phys_par_range is None:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        phys_par_range = _read_phys_par_range_from_h5(trainMD["train_params"]["full_h5name"])
    centers, logspans = phys_par_range_to_arrays(phys_par_range)
    P = len(centers)

    import importlib
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
    param_names = list(cell_mod.PARAM_KEYS)[:P]

    # t_max_override:auto -> derive from a stim length (all 4 CA3 stims are 500 ms).
    if t_max_override is not None:
        mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
        if isinstance(t_max_override, str) and t_max_override.lower() in ("auto", "stim"):
            from toolbox import jaxley_cells, jaxley_utils as _ju
            from pathlib import Path
            spec = jaxley_cells.get(cell_name)
            sn = stims[0]
            stim_arr = _ju.load_stim_csv(Path(spec.stim_dir) / f"{sn}.csv")
            t_max_override = float(len(stim_arr)) * float(spec.dt_stim)
        mod._T_MAX = float(t_max_override)
        JaxleyBridge.clear_cache()
        print(f"[sens] t_max set to {t_max_override} ms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Operating points: sample K true unit_par vectors from the test split. ──
    K = args.numOp
    h5_path = trainMD["train_params"]["full_h5name"]
    with h5py.File(h5_path, "r") as f:
        true_unit = f["test_unit_par"][:K].astype(np.float64)   # (K, P) in [-1, 1]
    true_unit = true_unit[:, :P]
    K = true_unit.shape[0]
    prior_std = true_unit.std(axis=0)   # spread of the true params (for identifiability ref)
    print(f"[sens] {K} operating points; prior std per param = "
          f"{np.array2string(prior_std, precision=3)}")

    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    def unit_to_phys(u):  # u: (M,P) torch f64 -> phys (M,P)
        return centers_t * torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device),
                                     u * logspans_t)

    def zscore_rows(v):   # v: (M,T) -> per-row z-score
        return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)

    eps = args.eps
    # Build the full perturbation matrix once: for each op-point k and param p,
    # a +eps and -eps row.  Shape (K, 2P, P) -> flatten to (K*2P, P).
    U = np.repeat(true_unit[:, None, :], 2 * P, axis=1)         # (K, 2P, P)
    for p in range(P):
        U[:, 2 * p,     p] += eps
        U[:, 2 * p + 1, p] -= eps
    Uf = torch.tensor(U.reshape(K * 2 * P, P), dtype=torch.float64, device=device)
    phys_f = unit_to_phys(Uf)

    # ── Per-stim Jacobian via central differences, then Fisher information. ──
    F_per_stim = {}          # stim -> (P,P) Fisher
    sens_per_stim = {}       # stim -> (P,) marginal sensitivity sqrt(diag F)
    sim_bs = 128
    for s in stims:
        print(f"[sens] simulating {K*2*P} perturbed cells under '{s}' ...")
        t0 = time.time()
        outs = []
        with torch.no_grad():
            for i in range(0, phys_f.shape[0], sim_bs):
                v = JaxleyBridge.simulate_batch(phys_f[i:i+sim_bs], cell_name, s, solver=solver)
                outs.append(v[:, 0, :].cpu())     # (b, T)
        V = torch.cat(outs, dim=0)                # (K*2P, T)
        Vz = zscore_rows(V).numpy()               # z-scored to match training loss
        T = Vz.shape[1]
        Vz = Vz.reshape(K, 2 * P, T)
        # Central diff:  J[k,:,p] = (V(+) - V(-)) / (2 eps)
        J = np.empty((K, T, P))
        for p in range(P):
            J[:, :, p] = (Vz[:, 2 * p, :] - Vz[:, 2 * p + 1, :]) / (2.0 * eps)
        # Fisher per op-point averaged: F = mean_k (J_k^T J_k) / T
        F = np.zeros((P, P))
        for k in range(K):
            F += J[k].T @ J[k]
        F /= (K * T)
        F_per_stim[s] = F
        sens_per_stim[s] = np.sqrt(np.diag(F))
        print(f"[sens]   '{s}' done in {time.time()-t0:.1f}s; "
              f"marginal sensitivity = {np.array2string(sens_per_stim[s], precision=3)}")

    # Per-stim conditioning (how degenerate is each stimulus on its own).
    for s in stims:
        ev = np.linalg.eigvalsh(F_per_stim[s])
        print(f"[sens]   '{s}' Fisher cond = {ev[-1]/max(ev[0],1e-30):.2f}  "
              f"(eigvals {ev[0]:.1f}..{ev[-1]:.1f})")

    # Combined (information adds across independent stims).
    F_multi = np.sum([F_per_stim[s] for s in stims], axis=0)

    # ── Cramer-Rao bound per stim + combined.  CRB_p = sigma * sqrt((F^-1)_pp). ──
    sigma = args.sigma
    def crb(F):
        # ridge for numerical stability (F can be near-singular = the whole point)
        lam = 1e-9 * np.trace(F) / P
        Finv = np.linalg.inv(F + lam * np.eye(P))
        return sigma * np.sqrt(np.clip(np.diag(Finv), 0, None)), Finv
    crb_per_stim = {s: crb(F_per_stim[s])[0] for s in stims}
    crb_multi, Finv_multi = crb(F_multi)

    # Collinearity (correlation of the estimator covariance) from combined F^-1.
    d = np.sqrt(np.clip(np.diag(Finv_multi), 1e-30, None))
    corr = Finv_multi / np.outer(d, d)

    # Eigenspectrum of combined Fisher (small eigval = degenerate direction).
    evals, evecs = np.linalg.eigh(F_multi)         # ascending
    cond = float(evals[-1] / max(evals[0], 1e-30))

    # Identifiability index = CRB / prior_std ; >=1 means voltage can't beat the prior.
    ident_idx = crb_multi / np.maximum(prior_std, 1e-9)

    # ── Report ──────────────────────────────────────────────────────────────
    single = stims[0]
    print("\n" + "=" * 78)
    print(f"IDENTIFIABILITY (sigma={sigma}, combined over {len(stims)} stim(s))")
    print(f"  Fisher condition number = {cond:.3e}  (>>1 => strong degeneracy)")
    print("-" * 78)
    print(f"{'channel':<20}{'sens(comb)':>11}{'CRB_1stim':>11}{'CRB_multi':>11}"
          f"{'CRB drop':>10}{'ident_idx':>11}")
    comb_sens = np.sqrt(np.diag(F_multi))
    order = np.argsort(-ident_idx)   # worst-identified first
    for p in order:
        drop = 1.0 - crb_multi[p] / max(crb_per_stim[single][p], 1e-30)
        flag = "  <-- unidentifiable" if ident_idx[p] >= 1.0 else ""
        print(f"{param_names[p]:<20}{comb_sens[p]:>11.3f}{crb_per_stim[single][p]:>11.3f}"
              f"{crb_multi[p]:>11.3f}{drop*100:>9.0f}%{ident_idx[p]:>11.2f}{flag}")
    print("-" * 78)
    # Degenerate pairs.
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            if abs(corr[i, j]) > 0.8:
                pairs.append((param_names[i], param_names[j], float(corr[i, j])))
    if pairs:
        print("Strongly collinear channel pairs (|corr|>0.8, trade off in voltage):")
        for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"    {a:<18} <-> {b:<18}  corr={c:+.2f}")
    else:
        print("No channel pair exceeds |corr|>0.8.")
    # Most degenerate direction.
    v0 = evecs[:, 0]
    print(f"Least-observable direction (smallest Fisher eigval={evals[0]:.2e}):")
    for p in np.argsort(-np.abs(v0)):
        if abs(v0[p]) > 0.2:
            print(f"    {v0[p]:+.2f} * {param_names[p]}")
    print("=" * 78 + "\n")

    # ── Outputs ─────────────────────────────────────────────────────────────
    outDir = args.outDir or os.path.join(args.modelPath, "sensitivity")
    os.makedirs(outDir, exist_ok=True)

    import csv
    with open(os.path.join(outDir, "crb_by_stim.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "prior_std", "sens_combined"]
                   + [f"CRB_{s}" for s in stims] + ["CRB_combined", "ident_index"])
        for p in range(P):
            w.writerow([param_names[p], f"{prior_std[p]:.4f}", f"{comb_sens[p]:.4f}"]
                       + [f"{crb_per_stim[s][p]:.4f}" for s in stims]
                       + [f"{crb_multi[p]:.4f}", f"{ident_idx[p]:.4f}"])

    with open(os.path.join(outDir, "sensitivity_by_stim.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel"] + [f"sens_{s}" for s in stims] + ["sens_combined"])
        for p in range(P):
            w.writerow([param_names[p]] + [f"{sens_per_stim[s][p]:.4f}" for s in stims]
                       + [f"{comb_sens[p]:.4f}"])

    short = [s if len(s) <= 12 else s[:11] + "…" for s in stims]

    # 1) Marginal sensitivity grouped bars.
    fig, ax = plt.subplots(figsize=(1.4 * P + 3, 4.5))
    x = np.arange(P); wbar = 0.8 / len(stims)
    for si, s in enumerate(stims):
        ax.bar(x + si * wbar, sens_per_stim[s], wbar, label=short[si])
    ax.set_xticks(x + 0.4 - wbar / 2); ax.set_xticklabels(param_names, rotation=35, ha="right")
    ax.set_ylabel("marginal sensitivity  ‖dV_z/dθ‖ (RMS)")
    ax.set_title("Per-channel voltage sensitivity by stimulus")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "sensitivity_bars.png"), dpi=120)
    plt.close(fig)

    # 2) CRB bars: single vs combined, with prior-std reference line per channel.
    fig, ax = plt.subplots(figsize=(1.4 * P + 3, 4.5))
    ax.bar(x - 0.2, crb_per_stim[single], 0.4, label=f"CRB ({short[0]} only)", color="C1")
    ax.bar(x + 0.2, crb_multi, 0.4, label="CRB (all stims)", color="C0")
    ax.plot(x, prior_std, "k_", markersize=18, label="prior std (unidentifiable line)")
    ax.set_xticks(x); ax.set_xticklabels(param_names, rotation=35, ha="right")
    ax.set_ylabel(f"Cramér-Rao std (unit space, σ={sigma})")
    ax.set_title("Identifiability: lower CRB = better recovered; CRB≥prior ⇒ unidentifiable")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "crb_bars.png"), dpi=120)
    plt.close(fig)

    # 3) Collinearity heatmap.
    fig, ax = plt.subplots(figsize=(0.9 * P + 3, 0.9 * P + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(P)); ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(P)); ax.set_yticklabels(param_names, fontsize=8)
    for i in range(P):
        for j in range(P):
            ax.text(j, i, f"{corr[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Estimator collinearity (from combined F⁻¹)\n|corr|→1 ⇒ channels trade off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "collinearity.png"), dpi=120)
    plt.close(fig)

    # 4) Eigenspectrum.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(range(1, P + 1), evals[::-1], "o-")
    axes[0].set_xlabel("eigen-index (large→small)"); axes[0].set_ylabel("Fisher eigenvalue")
    axes[0].set_title(f"Fisher spectrum (cond={cond:.1e})"); axes[0].grid(alpha=0.3)
    im = axes[1].imshow(evecs[:, ::-1], vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axes[1].set_yticks(range(P)); axes[1].set_yticklabels(param_names, fontsize=8)
    axes[1].set_xlabel("eigenvector (large→small eigval)")
    axes[1].set_title("Eigenvectors — rightmost col = least-observable combo")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outDir, "eigenspectrum.png"), dpi=120)
    plt.close(fig)

    summary = {
        "cell_name": cell_name, "stims": list(stims), "n_op_points": int(K),
        "sigma": float(sigma), "eps": float(eps),
        "fisher_condition_number": cond,
        "channels": [
            {"name": param_names[p], "prior_std": float(prior_std[p]),
             "sens_combined": float(comb_sens[p]),
             "crb_single": float(crb_per_stim[single][p]),
             "crb_combined": float(crb_multi[p]),
             "ident_index": float(ident_idx[p]),
             "identifiable": bool(ident_idx[p] < 1.0)}
            for p in range(P)
        ],
        "collinear_pairs": [{"a": a, "b": b, "corr": c} for a, b, c in pairs],
        "least_observable_direction": {param_names[p]: float(v0[p]) for p in range(P)},
    }
    write_yaml(summary, os.path.join(outDir, "summary.yaml"))
    print(f"[sens] wrote plots + summary to {outDir}/  (elapsed {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
