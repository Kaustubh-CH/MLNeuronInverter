#!/usr/bin/env python3
"""Evaluate a HybridLoss/voltage-trained model.

Loads the trained CNN, runs it on the test split, then for each prediction
runs the matching jaxley cell (ball_and_stick_bbp here) to get a simulated
soma trace.  Compares simulated vs ground-truth traces in z-scored space
(matches training loss) AND in raw mV space (re-derived from the per-sample
mean/std stored alongside the data).

Outputs (under <modelPath>/eval/):
  voltage_metrics.csv         per-sample RMSE, peak diff, spike-count diff
  voltage_loss_hist.png       histogram of z-scored voltage MSE per sample
  trace_overlay_<i>.png       N=10 overlay plots (true vs predicted soma)
  summary.yaml                aggregate stats

Usage:
    python evaluate_voltage.py --modelPath <run_dir>/out [--numSamples 200]
"""

import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAX env must be set BEFORE jax/jaxley import.
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
from toolbox.HybridLoss import _log_jax_devices_once  # for device print
from toolbox.jaxley_utils import phys_par_range_to_arrays


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", required=True,
                   help="run dir's `out/` containing blank_model.pth + checkpoints/")
    p.add_argument("-n", "--numSamples", type=int, default=200,
                   help="evaluate on this many test-split samples")
    p.add_argument("--numOverlay", type=int, default=10,
                   help="how many overlay plots to save")
    p.add_argument("-o", "--outDir", default=None,
                   help="output dir (default: <modelPath>/eval)")
    return p.parse_args()


def load_trained_model(modelPath: str, device: torch.device):
    """Mirrors predict.py:load_model — load whole nn.Module, then ckpt's state_dict."""
    sumYaml = os.path.join(modelPath, "sum_train.yaml")
    trainMD = read_yaml(sumYaml, verb=0)

    blankF = os.path.join(modelPath, trainMD["train_params"]["blank_model"])
    ckptF  = os.path.join(modelPath, trainMD["train_params"]["checkpoint_name"])

    print(f"[eval] loading blank_model: {blankF}")
    model = torch.load(blankF, map_location=device, weights_only=False)
    print(f"[eval] loading checkpoint:  {ckptF}")
    ck = torch.load(ckptF, map_location=device, weights_only=False)

    state = ck["model_state"]
    # Strip "module." prefix if present (saved from DDP).
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, trainMD


def load_test_data(trainMD, n_samples):
    """Read first `n_samples` test-split voltages + per-sample mean/std so we
    can de-normalize the z-scored data back to mV."""
    h5_path = trainMD["train_params"]["full_h5name"]
    probs = trainMD["train_params"]["data_conf"]["probs_select"]
    stims = trainMD["train_params"]["data_conf"]["stims_select"]
    soma_idx_in_select = 0  # design YAML pins soma_probe_index=0
    print(f"[eval] reading test split from {h5_path}, probs={probs}, stims={stims}")

    with h5py.File(h5_path, "r") as f:
        # (N, T, P, S) fp16
        v_norm = f["test_volts_norm"][:n_samples, :, :, stims[0]].astype(np.float32)
        # raw phys/unit params (for reference; CNN won't use them with mask_channels=True)
        unit_par = f["test_unit_par"][:n_samples].astype(np.float32)
        # Reconstruct per-sample mean/std from the original raw voltages would
        # require the simRaw.h5 file; for now we only show z-scored comparison.
        # If `<dom>_volts_mean` / `<dom>_volts_std` exist, use them.
        try:
            v_mean = f["test_volts_mean"][:n_samples, :, stims[0]].astype(np.float32)
            v_std  = f["test_volts_std"][:n_samples, :, stims[0]].astype(np.float32)
            have_mvstats = True
        except KeyError:
            v_mean = v_std = None
            have_mvstats = False

    # Pick the soma probe column (probsSelect's 0th -> soma)
    soma_idx_h5 = probs[soma_idx_in_select]
    v_soma_norm = v_norm[:, :, soma_idx_h5]      # (N, T)
    print(f"[eval] v_soma_norm shape = {v_soma_norm.shape}, "
          f"mV-stats available = {have_mvstats}")
    return v_soma_norm, unit_par, v_mean, v_std


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainMD = load_trained_model(args.modelPath, device)

    # Voltage-loss config from sum_train.yaml — has cell_name, phys_par_range, etc.
    vl = trainMD["train_params"]["voltage_loss"]
    cell_name      = vl["cell_name_for_sim"]
    phys_par_range = vl.get("phys_par_range")  # may be None if read from H5 meta
    clamp_tanh     = bool(vl.get("clamp_unit_tanh", False))
    fp64           = bool(vl.get("fp64", False))
    stim_name      = vl.get("stim_name")
    t_max_override = vl.get("t_max_override")
    soma_probe_idx = int(vl.get("soma_probe_index", 0))

    if phys_par_range is None:
        # Read from H5 meta — same fallback as build_hybrid_loss
        from toolbox.HybridLoss import _read_phys_par_range_from_h5
        phys_par_range = _read_phys_par_range_from_h5(trainMD["train_params"]["full_h5name"])

    centers, logspans = phys_par_range_to_arrays(phys_par_range)
    centers_t  = torch.tensor(centers,  dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # Apply t_max_override (matches HybridLoss build path)
    if t_max_override is not None:
        import importlib
        mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
        if isinstance(t_max_override, str) and t_max_override.lower() in ("auto", "stim"):
            from toolbox import jaxley_cells, jaxley_utils as _jutils
            from pathlib import Path
            spec = jaxley_cells.get(cell_name)
            sn = stim_name or spec.default_stim_name
            stim_arr = _jutils.load_stim_csv(Path(spec.stim_dir) / f"{sn}.csv")
            t_max_override = float(len(stim_arr)) * float(spec.dt_stim)
        mod._T_MAX = float(t_max_override)
        JaxleyBridge.clear_cache()
        print(f"[eval] t_max set to {t_max_override} ms")

    _log_jax_devices_once()

    # Load test data
    v_soma_norm, unit_par_true, v_mean_arr, v_std_arr = load_test_data(trainMD, args.numSamples)
    N, T_data = v_soma_norm.shape
    P_cnn = trainMD["train_params"]["model"]["outputSize"]
    print(f"[eval] N={N} test samples, T_data={T_data}, P_cnn={P_cnn}")

    # Build CNN-input shape = (N, T, C) where C = num_probes_after_select.
    # The CNN's input is the raw z-scored voltages on all selected probes.
    # We re-read those for CNN forward.
    h5_path = trainMD["train_params"]["full_h5name"]
    probs = trainMD["train_params"]["data_conf"]["probs_select"]
    with h5py.File(h5_path, "r") as f:
        cnn_in = f["test_volts_norm"][:N, :, :, trainMD["train_params"]["data_conf"]["stims_select"][0]]
    cnn_in = cnn_in[:, :, probs].astype(np.float32)   # (N, T, num_probes)
    print(f"[eval] CNN input shape: {cnn_in.shape}")

    # CNN forward — chunked to fit GPU
    bs = 128
    pred_unit_chunks = []
    with torch.no_grad():
        for i in range(0, N, bs):
            x = torch.from_numpy(cnn_in[i:i+bs]).permute(0, 2, 1).contiguous().to(device)
            y = model(x).float().cpu()
            pred_unit_chunks.append(y)
    pred_unit = torch.cat(pred_unit_chunks, dim=0)
    print(f"[eval] pred_unit shape: {pred_unit.shape}, range [{pred_unit.min():.3f}, {pred_unit.max():.3f}]")

    # unit -> phys (apply tanh if used in training)
    pred_unit_d = pred_unit.double().to(device)
    if clamp_tanh:
        pred_unit_d = torch.tanh(pred_unit_d)
    pred_phys = centers_t * torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device),
                                       pred_unit_d * logspans_t)
    print(f"[eval] pred_phys[0] = {pred_phys[0].cpu().numpy()}")

    # Run jaxley on the predicted phys params (chunked)
    # Outputs (B, n_recorded=1, T_sim).
    sim_chunks = []
    sim_bs = 64
    print(f"[eval] running jaxley on {N} predictions, batch={sim_bs}...")
    t0 = time.time()
    for i in range(0, N, sim_bs):
        chunk = pred_phys[i:i+sim_bs]
        v = JaxleyBridge.simulate_batch(chunk, cell_name, stim_name)
        sim_chunks.append(v[:, 0, :].cpu())   # (B, T_sim)
    v_sim = torch.cat(sim_chunks, dim=0).numpy()  # (N, T_sim)
    t1 = time.time()
    print(f"[eval] jaxley ran in {t1-t0:.1f}s, v_sim shape = {v_sim.shape}")

    # Truncate to overlap
    T = min(v_sim.shape[1], T_data)
    v_sim_pre = v_sim[:, :T]                          # (N, T) in mV (pre z-score)
    v_data_z  = v_soma_norm[:, :T]                     # (N, T) z-scored

    # z-score the predicted trace per-sample to compare vs data (which is z-scored)
    v_sim_z = (v_sim_pre - v_sim_pre.mean(axis=1, keepdims=True)) / (
        v_sim_pre.std(axis=1, keepdims=True) + 1e-6)

    # Per-sample voltage MSE in z-scored space (matches training loss)
    err_z = ((v_sim_z - v_data_z) ** 2).mean(axis=1)
    rmse_z = np.sqrt(err_z)
    print(f"[eval] voltage MSE_z: mean={err_z.mean():.4f} median={np.median(err_z):.4f} "
          f"min={err_z.min():.4f} max={err_z.max():.4f}")
    print(f"[eval] voltage RMSE_z: mean={rmse_z.mean():.3f} median={np.median(rmse_z):.3f}")

    # Spike-count diff (rough: count crossings of 0 mV in v_sim_pre, and z=2 in data)
    spikes_sim  = ((v_sim_pre[:, 1:] > 0) & (v_sim_pre[:, :-1] <= 0)).sum(axis=1)
    spikes_data = ((v_data_z[:, 1:] > 2.0) & (v_data_z[:, :-1] <= 2.0)).sum(axis=1)
    spike_diff  = spikes_sim - spikes_data
    print(f"[eval] spike count: sim mean={spikes_sim.mean():.1f}, "
          f"data mean={spikes_data.mean():.1f}, |diff| mean={np.abs(spike_diff).mean():.2f}")

    # ─── outputs ────────────────────────────────────────────────────────────
    outDir = args.outDir or os.path.join(args.modelPath, "eval")
    os.makedirs(outDir, exist_ok=True)

    # CSV per-sample
    import csv
    with open(os.path.join(outDir, "voltage_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "rmse_z", "mse_z", "spikes_sim", "spikes_data", "spike_diff"])
        for i in range(N):
            w.writerow([i, f"{rmse_z[i]:.4f}", f"{err_z[i]:.4f}",
                        int(spikes_sim[i]), int(spikes_data[i]), int(spike_diff[i])])

    # Summary YAML
    write_yaml({
        "n_samples": int(N),
        "voltage_mse_z_mean": float(err_z.mean()),
        "voltage_mse_z_median": float(np.median(err_z)),
        "voltage_rmse_z_mean": float(rmse_z.mean()),
        "voltage_rmse_z_median": float(np.median(rmse_z)),
        "spike_count_diff_mean_abs": float(np.abs(spike_diff).mean()),
        "spikes_sim_mean":  float(spikes_sim.mean()),
        "spikes_data_mean": float(spikes_data.mean()),
        "model_path": args.modelPath,
        "cell_name": cell_name,
    }, os.path.join(outDir, "summary.yaml"))
    print(f"[eval] wrote summary -> {outDir}/summary.yaml")

    # Histogram
    fig = plt.figure(figsize=(7, 4))
    plt.hist(err_z, bins=40, color="C0", edgecolor="black", alpha=0.8)
    plt.axvline(err_z.mean(), color="red", linestyle="--", label=f"mean={err_z.mean():.3f}")
    plt.axvline(np.median(err_z), color="orange", linestyle="--", label=f"median={np.median(err_z):.3f}")
    plt.xlabel("per-sample voltage MSE (z-scored)")
    plt.ylabel("count")
    plt.title(f"Voltage-loss distribution on test ({N} samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, "voltage_loss_hist.png"), dpi=120)
    plt.close(fig)

    # Overlay plots — pick a mix of best, median, worst samples
    n_overlay = min(args.numOverlay, N)
    order = np.argsort(err_z)
    pick = np.concatenate([
        order[:n_overlay // 3],                              # best
        order[len(order) // 2 - n_overlay // 6 : len(order) // 2 + n_overlay // 6],  # median
        order[-n_overlay // 3:],                             # worst
    ])[:n_overlay]
    dt = 0.1
    t_axis = np.arange(T) * dt   # ms

    for k, idx in enumerate(pick):
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].plot(t_axis, v_data_z[idx], "k", lw=1.0, label="data (z-scored)")
        axes[0].plot(t_axis, v_sim_z[idx],  "C3", lw=1.0, alpha=0.8, label="predicted (z-scored)")
        axes[0].set_ylabel("z-scored V_soma")
        axes[0].set_title(f"Sample #{idx}  rmse_z={rmse_z[idx]:.3f}  "
                          f"spikes sim/data = {int(spikes_sim[idx])}/{int(spikes_data[idx])}")
        axes[0].legend(loc="upper right")

        axes[1].plot(t_axis, v_sim_pre[idx], "C3", lw=1.0, label="predicted (mV, jaxley)")
        axes[1].set_xlabel("time (ms)")
        axes[1].set_ylabel("V_soma (mV) — predicted only")
        axes[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, f"trace_overlay_{k:02d}_sample{idx}.png"), dpi=120)
        plt.close(fig)
    print(f"[eval] wrote {n_overlay} overlay plots to {outDir}/")

    # Aggregate accuracy plot: rmse_z sorted
    fig = plt.figure(figsize=(7, 4))
    plt.plot(np.sort(rmse_z), np.linspace(0, 1, N), color="C0", lw=2)
    plt.xlabel("voltage RMSE (z-scored)")
    plt.ylabel("CDF over test samples")
    plt.title("Test-set voltage-RMSE CDF")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, "voltage_rmse_cdf.png"), dpi=120)
    plt.close(fig)
    print(f"[eval] DONE — see {outDir}/")


if __name__ == "__main__":
    main()
