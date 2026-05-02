#!/usr/bin/env python3
"""Phase 3 validation plotter.

Extends `evaluate_voltage.py` with per-parameter recovery metrics.  Run on
a checkpoint trained against the matched ball_and_stick_bbp synth data
(`ballBBP_matched.hpar.yaml`).

Outputs (under <modelPath>/eval_phase3/, or --outDir):

    summary.yaml                aggregate stats + acceptance verdict
    voltage_metrics.csv         per-sample voltage RMSE_z + spike counts
    voltage_loss_hist.png       hist of per-sample voltage MSE_z
    voltage_rmse_cdf.png        CDF over test samples
    trace_overlay_<i>_*.png     N=10 overlay plots (best/median/worst)
    param_recovery.csv          per-sample (true_unit, pred_unit) for all params
    param_recovery_summary.csv  one row per param: R, RMSE, MAE, EV
    param_recovery_<k>_<name>.png  per-parameter scatter (pred vs true)
    param_recovery_grid.png     12-panel grid overview

Acceptance bar (from PHASE3_TASKS.md):
  * per-parameter explained variance  > 0.85  on ALL 12 params
  * voltage RMSE_z (mean over test)   < 0.30

Usage:
    python plotJaxleyValidation.py --modelPath <run_dir>/out [--numSamples 200]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# JAX env must be set BEFORE jax/jaxley import.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda" if os.environ.get("SLURM_JOBID") else "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)

from toolbox import JaxleyBridge
from toolbox.HybridLoss import _log_jax_devices_once
from toolbox.Util_IOfunc import read_yaml, write_yaml
from toolbox.jaxley_utils import phys_par_range_to_arrays


# ─────────────────────────────────────────────────────────────────────────
# acceptance thresholds — from PHASE3_TASKS.md
# ─────────────────────────────────────────────────────────────────────────
ACCEPT_EV_PER_PARAM = 0.85
ACCEPT_VOLTAGE_RMSE_Z = 0.30


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--modelPath", required=True,
                   help="run dir's `out/` containing blank_model.pth + checkpoints/")
    p.add_argument("-n", "--numSamples", type=int, default=2000,
                   help="evaluate on this many test-split samples (default: 2000)")
    p.add_argument("--numOverlay", type=int, default=10,
                   help="how many overlay plots to save")
    p.add_argument("-o", "--outDir", default=None,
                   help="output dir (default: <modelPath>/eval_phase3)")
    p.add_argument("--simBatch", type=int, default=64,
                   help="batch size for jaxley sim (default: 64)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────────────────

def load_trained_model(modelPath, device):
    """Mirrors evaluate_voltage.py:load_trained_model."""
    sumYaml = os.path.join(modelPath, "sum_train.yaml")
    trainMD = read_yaml(sumYaml, verb=0)

    blankF = os.path.join(modelPath, trainMD["train_params"]["blank_model"])
    ckptF  = os.path.join(modelPath, trainMD["train_params"]["checkpoint_name"])

    print(f"[val] loading blank_model: {blankF}")
    model = torch.load(blankF, map_location=device, weights_only=False)
    print(f"[val] loading checkpoint:  {ckptF}")
    ck = torch.load(ckptF, map_location=device, weights_only=False)

    state = ck["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, trainMD


def load_test_data(trainMD, n_samples):
    """Read test split: voltages on the selected probes + ground-truth unit_par."""
    tp = trainMD["train_params"]
    h5_path = tp["full_h5name"]
    probs = tp["data_conf"]["probs_select"]
    stims = tp["data_conf"]["stims_select"]

    print(f"[val] reading test split from {h5_path}, probs={probs}, stims={stims}")
    with h5py.File(h5_path, "r") as f:
        # (N, T, P, S) fp16 -> select stim 0 -> (N, T, P)
        v_norm = f["test_volts_norm"][:n_samples, :, :, stims[0]].astype(np.float32)
        unit_true = f["test_unit_par"][:n_samples].astype(np.float32)
    # CNN input: select probe columns
    cnn_in = v_norm[:, :, probs]
    # Soma trace for voltage comparison: probsSelect[0] -> soma in synth pack.
    v_soma_z = v_norm[:, :, probs[0]]
    print(f"[val] cnn_in shape = {cnn_in.shape}, v_soma_z shape = {v_soma_z.shape}, "
          f"unit_true shape = {unit_true.shape}")
    return cnn_in, v_soma_z, unit_true


# ─────────────────────────────────────────────────────────────────────────
# CNN forward + jaxley sim
# ─────────────────────────────────────────────────────────────────────────

def cnn_forward(model, cnn_in, device, bs=128):
    """Run the CNN on every test sample.  Returns pred_unit (N, P) on CPU."""
    N = cnn_in.shape[0]
    chunks = []
    with torch.no_grad():
        for i in range(0, N, bs):
            x = torch.from_numpy(cnn_in[i:i+bs]).permute(0, 2, 1).contiguous().to(device)
            y = model(x).float().cpu()
            chunks.append(y)
    return torch.cat(chunks, dim=0)


def jaxley_forward(pred_phys, cell_name, stim_name, sim_bs=64):
    """Run jaxley sim on pred_phys (N, P).  Returns v_sim_mv (N, T) on CPU numpy."""
    N = pred_phys.shape[0]
    chunks = []
    print(f"[val] running jaxley on {N} predictions, batch={sim_bs}...")
    t0 = time.time()
    for i in range(0, N, sim_bs):
        chunk = pred_phys[i:i + sim_bs]
        v = JaxleyBridge.simulate_batch(chunk, cell_name, stim_name)
        chunks.append(v[:, 0, :].cpu())     # soma
    v_sim = torch.cat(chunks, dim=0).numpy()
    print(f"[val] jaxley ran in {time.time()-t0:.1f}s, v_sim shape = {v_sim.shape}")
    return v_sim


# ─────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────

def per_param_metrics(pred_unit, true_unit, param_names):
    """Pearson R, RMSE, MAE, explained variance per parameter.

    Explained variance := 1 - Var(true - pred) / Var(true)
    """
    rows = []
    P = pred_unit.shape[1]
    for k in range(P):
        t = true_unit[:, k].astype(np.float64)
        p = pred_unit[:, k].astype(np.float64)
        # Pearson R
        if t.std() > 0 and p.std() > 0:
            r = float(np.corrcoef(t, p)[0, 1])
        else:
            r = float("nan")
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        mae  = float(np.abs(p - t).mean())
        var_t = float(t.var())
        var_resid = float((t - p).var())
        ev = 1.0 - var_resid / var_t if var_t > 0 else float("nan")
        rows.append({
            "k": k,
            "name": param_names[k],
            "pearson_r": r,
            "rmse": rmse,
            "mae": mae,
            "explained_variance": float(ev),
            "true_mean": float(t.mean()),
            "true_std":  float(t.std()),
            "pred_mean": float(p.mean()),
            "pred_std":  float(p.std()),
        })
    return rows


def voltage_metrics(v_sim_z, v_data_z):
    """Per-sample voltage MSE/RMSE in z-scored space."""
    err_z = ((v_sim_z - v_data_z) ** 2).mean(axis=1)
    rmse_z = np.sqrt(err_z)
    return err_z, rmse_z


# ─────────────────────────────────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_param_scatter(pred_unit, true_unit, param_metrics, out_dir):
    """Per-param scatter + a 12-panel grid overview."""
    P = pred_unit.shape[1]
    # Per-param scatter plots
    for row in param_metrics:
        k = row["k"]
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(true_unit[:, k], pred_unit[:, k], s=4, alpha=0.4)
        lim = (-1.2, 1.2)
        plt.plot(lim, lim, "k--", lw=0.8)
        plt.xlim(lim); plt.ylim(lim)
        plt.xlabel("true unit"); plt.ylabel("pred unit")
        plt.title(f"#{k:02d} {row['name']}\n"
                  f"R={row['pearson_r']:.3f}  EV={row['explained_variance']:.3f}  "
                  f"RMSE={row['rmse']:.3f}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(out_dir, f"param_recovery_{k:02d}_{row['name']}.png")
        plt.savefig(fp, dpi=120)
        plt.close(fig)

    # Grid overview
    ncol = 4
    nrow = (P + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for k in range(P):
        ax = axes[k]
        ax.scatter(true_unit[:, k], pred_unit[:, k], s=2, alpha=0.4)
        lim = (-1.2, 1.2)
        ax.plot(lim, lim, "k--", lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ev = param_metrics[k]['explained_variance']
        passed = "OK" if ev > ACCEPT_EV_PER_PARAM else "FAIL"
        ax.set_title(f"#{k:02d} {param_metrics[k]['name'][:18]}\n"
                     f"EV={ev:.3f} [{passed}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for k in range(P, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "param_recovery_grid.png"), dpi=120)
    plt.close(fig)


def plot_voltage_overlay(v_sim_pre, v_sim_z, v_data_z, rmse_z, spikes_sim, spikes_data,
                          out_dir, n_overlay=10, dt=0.1):
    """N=10 trace overlays — best, median, worst by rmse_z."""
    N, T = v_data_z.shape
    n = min(n_overlay, N)
    order = np.argsort(rmse_z)
    pick = np.concatenate([
        order[: n // 3],
        order[len(order)//2 - n//6 : len(order)//2 + n//6],
        order[-n // 3:],
    ])[:n]
    t_axis = np.arange(T) * dt

    for k, idx in enumerate(pick):
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].plot(t_axis, v_data_z[idx], "k", lw=1.0, label="data (z-scored)")
        axes[0].plot(t_axis, v_sim_z[idx], "C3", lw=1.0, alpha=0.8, label="predicted (z-scored)")
        axes[0].set_ylabel("z-scored V_soma")
        axes[0].set_title(f"Sample #{idx}  rmse_z={rmse_z[idx]:.3f}  "
                          f"spikes sim/data={int(spikes_sim[idx])}/{int(spikes_data[idx])}")
        axes[0].legend(loc="upper right")
        axes[1].plot(t_axis, v_sim_pre[idx], "C3", lw=1.0, label="predicted (mV)")
        axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("V_soma (mV)")
        axes[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trace_overlay_{k:02d}_sample{idx}.png"), dpi=120)
        plt.close(fig)


def plot_voltage_summary(err_z, rmse_z, out_dir):
    """Voltage-loss histogram + RMSE CDF."""
    N = len(err_z)
    fig = plt.figure(figsize=(7, 4))
    plt.hist(err_z, bins=40, color="C0", edgecolor="black", alpha=0.8)
    plt.axvline(err_z.mean(), color="red", linestyle="--", label=f"mean={err_z.mean():.3f}")
    plt.axvline(np.median(err_z), color="orange", linestyle="--", label=f"median={np.median(err_z):.3f}")
    plt.xlabel("per-sample voltage MSE (z-scored)")
    plt.ylabel("count")
    plt.title(f"Voltage loss distribution on test ({N} samples)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "voltage_loss_hist.png"), dpi=120); plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(np.sort(rmse_z), np.linspace(0, 1, N), color="C0", lw=2)
    plt.axvline(ACCEPT_VOLTAGE_RMSE_Z, color="red", linestyle="--",
                label=f"accept threshold = {ACCEPT_VOLTAGE_RMSE_Z}")
    plt.xlabel("voltage RMSE (z-scored)"); plt.ylabel("CDF over test samples")
    plt.title("Test-set voltage-RMSE CDF"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "voltage_rmse_cdf.png"), dpi=120); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainMD = load_trained_model(args.modelPath, device)
    tp = trainMD["train_params"]
    vl = tp["voltage_loss"]
    cell_name      = vl["cell_name_for_sim"]
    clamp_tanh     = bool(vl.get("clamp_unit_tanh", False))
    stim_name      = vl.get("stim_name")
    t_max_override = vl.get("t_max_override")

    # phys_par_range from the H5 meta (matched-data pack stores it under
    # input_meta.phys_par_range; fall back to inline if present).
    phys_par_range = vl.get("phys_par_range")
    if phys_par_range is None:
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        phys_par_range = _read_phys_par_range_from_h5(tp["full_h5name"])

    centers, logspans = phys_par_range_to_arrays(phys_par_range)
    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # t_max_override (matches HybridLoss build path)
    if t_max_override is not None:
        import importlib
        mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
        if isinstance(t_max_override, str) and t_max_override.lower() in ("auto", "stim"):
            from toolbox import jaxley_cells, jaxley_utils as _jutils
            spec = jaxley_cells.get(cell_name)
            sn = stim_name or spec.default_stim_name
            stim_arr = _jutils.load_stim_csv(Path(spec.stim_dir) / f"{sn}.csv")
            t_max_override = float(len(stim_arr)) * float(spec.dt_stim)
        mod._T_MAX = float(t_max_override)
        JaxleyBridge.clear_cache()
        print(f"[val] t_max set to {t_max_override} ms")

    _log_jax_devices_once()

    # Param names — from H5 meta (canonical), fall back to inline phys_par_range len.
    param_names = None
    try:
        with h5py.File(tp["full_h5name"], "r") as fh:
            mblob = json.loads(fh["meta.JSON"][0])
        param_names = (mblob.get("input_meta", {}).get("parName")
                       or mblob.get("parName"))
    except Exception as e:
        print(f"[val] could not read parName from H5: {e}")
    if not param_names:
        param_names = [f"param_{k}" for k in range(len(centers))]
    print(f"[val] {len(param_names)} params: {param_names}")

    # ── data ───────────────────────────────────────────────────────────────
    cnn_in, v_data_z, unit_true = load_test_data(trainMD, args.numSamples)
    N = cnn_in.shape[0]

    # ── CNN forward ────────────────────────────────────────────────────────
    pred_unit = cnn_forward(model, cnn_in, device, bs=128)
    print(f"[val] pred_unit shape={pred_unit.shape}, "
          f"range=[{pred_unit.min():.3f}, {pred_unit.max():.3f}]")

    # The CNN's raw output is the saved unit_par.  When clamp_unit_tanh was
    # used in training, the *effective* unit (the one fed to the simulator)
    # is tanh(pred); record both, but use the squashed one for jaxley sim.
    pred_unit_np = pred_unit.numpy()
    if clamp_tanh:
        pred_unit_eff_np = np.tanh(pred_unit_np)
    else:
        pred_unit_eff_np = pred_unit_np

    # ── unit -> phys, run jaxley ───────────────────────────────────────────
    pred_unit_t = torch.from_numpy(pred_unit_eff_np).double().to(device)
    pred_phys = centers_t * torch.pow(
        torch.tensor(10.0, dtype=torch.float64, device=device),
        pred_unit_t * logspans_t,
    )
    v_sim_pre = jaxley_forward(pred_phys, cell_name, stim_name, sim_bs=args.simBatch)

    # ── voltage metrics ────────────────────────────────────────────────────
    T = min(v_sim_pre.shape[1], v_data_z.shape[1])
    v_sim_pre = v_sim_pre[:, :T]
    v_data_z = v_data_z[:, :T]
    v_sim_z = (v_sim_pre - v_sim_pre.mean(axis=1, keepdims=True)) / (
        v_sim_pre.std(axis=1, keepdims=True) + 1e-6)
    err_z, rmse_z = voltage_metrics(v_sim_z, v_data_z)
    spikes_sim  = ((v_sim_pre[:, 1:] > 0) & (v_sim_pre[:, :-1] <= 0)).sum(axis=1)
    spikes_data = ((v_data_z[:, 1:] > 2.0) & (v_data_z[:, :-1] <= 2.0)).sum(axis=1)
    spike_diff  = spikes_sim - spikes_data

    # ── per-param metrics ──────────────────────────────────────────────────
    # Compare in the EFFECTIVE unit space (what the simulator saw) — this is
    # what the matched-data pipeline is supposed to recover.  When
    # clamp_unit_tanh=True, this is tanh(pred); the true_unit was sampled
    # in [-1, 1] and applied directly (no tanh), so for the simulator both
    # sides correspond to the same physical params iff |pred| ≤ ~1.
    pmet = per_param_metrics(pred_unit_eff_np, unit_true, param_names)

    # ── outputs ────────────────────────────────────────────────────────────
    outDir = args.outDir or os.path.join(args.modelPath, "eval_phase3")
    os.makedirs(outDir, exist_ok=True)

    # voltage_metrics.csv
    with open(os.path.join(outDir, "voltage_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "rmse_z", "mse_z", "spikes_sim", "spikes_data", "spike_diff"])
        for i in range(N):
            w.writerow([i, f"{rmse_z[i]:.4f}", f"{err_z[i]:.4f}",
                        int(spikes_sim[i]), int(spikes_data[i]), int(spike_diff[i])])

    # param_recovery.csv (per-sample)
    with open(os.path.join(outDir, "param_recovery.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["sample"]
        for nm in param_names:
            cols += [f"true_{nm}", f"pred_{nm}"]
        w.writerow(cols)
        for i in range(N):
            row = [i]
            for k in range(len(param_names)):
                row += [f"{unit_true[i, k]:.5f}", f"{pred_unit_eff_np[i, k]:.5f}"]
            w.writerow(row)

    # param_recovery_summary.csv
    with open(os.path.join(outDir, "param_recovery_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "name", "pearson_r", "explained_variance", "rmse", "mae",
                    "true_mean", "true_std", "pred_mean", "pred_std", "pass_EV>0.85"])
        for r in pmet:
            w.writerow([r["k"], r["name"],
                        f"{r['pearson_r']:.4f}", f"{r['explained_variance']:.4f}",
                        f"{r['rmse']:.4f}", f"{r['mae']:.4f}",
                        f"{r['true_mean']:.4f}", f"{r['true_std']:.4f}",
                        f"{r['pred_mean']:.4f}", f"{r['pred_std']:.4f}",
                        bool(r["explained_variance"] > ACCEPT_EV_PER_PARAM)])

    # acceptance verdict
    n_ev_pass = sum(1 for r in pmet if r["explained_variance"] > ACCEPT_EV_PER_PARAM)
    voltage_pass = bool(rmse_z.mean() < ACCEPT_VOLTAGE_RMSE_Z)
    overall_pass = (n_ev_pass == len(pmet)) and voltage_pass

    summary = {
        "n_samples": int(N),
        "voltage_mse_z_mean":     float(err_z.mean()),
        "voltage_mse_z_median":   float(np.median(err_z)),
        "voltage_rmse_z_mean":    float(rmse_z.mean()),
        "voltage_rmse_z_median":  float(np.median(rmse_z)),
        "voltage_rmse_z_threshold": ACCEPT_VOLTAGE_RMSE_Z,
        "voltage_pass": voltage_pass,
        "spike_count_diff_mean_abs": float(np.abs(spike_diff).mean()),
        "spikes_sim_mean":  float(spikes_sim.mean()),
        "spikes_data_mean": float(spikes_data.mean()),
        "params": [{
            "k": r["k"], "name": r["name"],
            "pearson_r": r["pearson_r"],
            "explained_variance": r["explained_variance"],
            "rmse": r["rmse"], "mae": r["mae"],
            "pass": bool(r["explained_variance"] > ACCEPT_EV_PER_PARAM),
        } for r in pmet],
        "param_ev_threshold": ACCEPT_EV_PER_PARAM,
        "params_passed_ev":   int(n_ev_pass),
        "params_total":       int(len(pmet)),
        "overall_pass":       bool(overall_pass),
        "model_path": args.modelPath,
        "cell_name":  cell_name,
    }
    write_yaml(summary, os.path.join(outDir, "summary.yaml"))

    # Plots
    plot_param_scatter(pred_unit_eff_np, unit_true, pmet, outDir)
    plot_voltage_overlay(v_sim_pre, v_sim_z, v_data_z, rmse_z, spikes_sim, spikes_data,
                          outDir, n_overlay=args.numOverlay, dt=0.1)
    plot_voltage_summary(err_z, rmse_z, outDir)

    # ── stdout summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE 3 VALIDATION — {os.path.abspath(outDir)}")
    print("=" * 70)
    print(f"Test samples: {N}")
    print(f"\nVoltage:")
    print(f"  rmse_z mean  = {rmse_z.mean():.4f}  (threshold < {ACCEPT_VOLTAGE_RMSE_Z}) "
          f"[{'PASS' if voltage_pass else 'FAIL'}]")
    print(f"  rmse_z median = {np.median(rmse_z):.4f}")
    print(f"  spike count: sim={spikes_sim.mean():.2f}  data={spikes_data.mean():.2f}  "
          f"|diff|={np.abs(spike_diff).mean():.2f}")
    print(f"\nPer-parameter explained variance (threshold > {ACCEPT_EV_PER_PARAM}):")
    print(f"  {'k':>2}  {'name':<32}  {'R':>7}  {'EV':>7}  {'RMSE':>7}  {'verdict':>7}")
    for r in pmet:
        verdict = "PASS" if r["explained_variance"] > ACCEPT_EV_PER_PARAM else "FAIL"
        print(f"  {r['k']:>2}  {r['name']:<32}  {r['pearson_r']:>+7.3f}  "
              f"{r['explained_variance']:>+7.3f}  {r['rmse']:>7.3f}  {verdict:>7}")
    print(f"\nParameters passing EV threshold: {n_ev_pass}/{len(pmet)}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'} (voltage AND all 12 params)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
