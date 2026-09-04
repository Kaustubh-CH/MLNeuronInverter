#!/usr/bin/env python3
"""Evaluate a HybridLoss/voltage-trained model.

Loads the trained CNN, runs it on the test split, then for each prediction
runs the matching jaxley cell (ball_and_stick_bbp here) to get a simulated
soma trace.  Compares simulated vs ground-truth traces in z-scored space
(matches training loss) AND in raw mV space (re-derived from the per-sample
mean/std stored alongside the data).

Outputs (under <modelPath>/eval/):
  trace_overlays.pdf          THE figure artifact, self-contained:
                                page 1 = ion-channel recovery grid
                                page 2 = z-scored voltage-MSE histogram
                                page 3 = voltage-RMSE CDF
                                page 4+ = per-sample overlays (--numOverlay, default 50)
  voltage_metrics.csv         per-sample RMSE, peak diff, spike-count diff
  channel_recovery.csv        per-param R²/RMSE/bias
  summary.yaml                aggregate stats

Figures are PDF-only by default.  Pass --savePng to additionally emit the old
per-figure .png files (channel_recovery_grid, voltage_loss_hist,
voltage_rmse_cdf, trace_overlay_<i>_sample<j>) — 50 overlay PNGs is ~12 MB per
run, which is why they are now opt-in.

Usage:
    python evaluate_voltage.py --modelPath <run_dir>/out [--numSamples 200]

    # POOLED runs: score the model separately on each stimulus in the battery.
    # A pooled model sees one untagged trace, so per-stim recovery differs and
    # the default (stims_select[0]) reports only the first protocol.
    for k in 0 1 2 3; do
        python evaluate_voltage.py --modelPath <run_dir>/out --stimIndex $k
    done
"""

import os, sys, time, argparse, json, re
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
from matplotlib.backends.backend_pdf import PdfPages

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
    p.add_argument("--numOverlay", type=int, default=50,
                   help="how many overlay plots to save")
    p.add_argument("-o", "--outDir", default=None,
                   help="output dir (default: <modelPath>/eval)")
    p.add_argument("--stimIndex", type=int, default=None,
                   help="POOLED runs only: which entry of data_conf.stims_select to "
                        "evaluate on (0-based into that list).  A pooled model sees one "
                        "trace at a time and does not know which protocol produced it, so "
                        "each stim is a separate test set.  Default (None) = stims_select[0], "
                        "the historical behaviour.  Changes the default outDir to "
                        "<modelPath>/eval_stim<K>_<stimName> so runs do not clobber each other.")
    p.add_argument("--savePng", action="store_true",
                   help="also write the individual .png figures.  Default is PDF-only: "
                        "everything lands in trace_overlays.pdf (channel grid, per-sample "
                        "overlays, loss histogram, RMSE CDF).")
    p.add_argument("--noGrad", action="store_true",
                   help="Forward-only jaxley (handle.simulate_batch, no jax.vjp graph). "
                        "REQUIRED for the 19-param L5TTPC cell: the training bridge's VJP "
                        "graph OOMs a 40 GB A100 at batch 64.  Results are identical.")
    p.add_argument("--simBatch", type=int, default=64,
                   help="jaxley batch size for the re-simulation (default 64).")
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


def load_test_data(trainMD, n_samples, stim_col):
    """Read first `n_samples` test-split voltages + per-sample mean/std so we
    can de-normalize the z-scored data back to mV.

    `stim_col` is the H5 stim-axis index to slice.  For a joint/single-stim pack
    that is always stims_select[0]; for a POOLED pack the stim axis carries the
    battery, so the caller picks which protocol to evaluate on."""
    h5_path = trainMD["train_params"]["full_h5name"]
    probs = trainMD["train_params"]["data_conf"]["probs_select"]
    stims = trainMD["train_params"]["data_conf"]["stims_select"]
    soma_idx_in_select = 0  # design YAML pins soma_probe_index=0
    print(f"[eval] reading test split from {h5_path}, probs={probs}, stims={stims}, "
          f"stim_col={stim_col}")

    with h5py.File(h5_path, "r") as f:
        # (N, T, P, S) fp16
        v_norm = f["test_volts_norm"][:n_samples, :, :, stim_col].astype(np.float32)
        # raw phys/unit params (for reference; CNN won't use them with mask_channels=True)
        unit_par = f["test_unit_par"][:n_samples].astype(np.float32)
        # Reconstruct per-sample mean/std from the original raw voltages would
        # require the simRaw.h5 file; for now we only show z-scored comparison.
        # If `<dom>_volts_mean` / `<dom>_volts_std` exist, use them.
        try:
            v_mean = f["test_volts_mean"][:n_samples, :, stim_col].astype(np.float32)
            v_std  = f["test_volts_std"][:n_samples, :, stim_col].astype(np.float32)
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


def _simulate_nograd(pred_phys, cell_name, stim_name, bs, solver="bwd_euler"):
    """Forward-only jaxley on the cached handle's jitted vmap -> (N, n_rec, T_ds)
    float64 numpy.  Skips the jax.vjp graph that JaxleyBridge.simulate_batch
    always builds (it is the TRAINING bridge); the 19-param L5TTPC cell OOMs
    there.  Pads the last chunk to `bs` so XLA compiles one shape."""
    import jax.numpy as jnp
    handle = JaxleyBridge.get_handle(cell_name, stim_name, solver=solver)
    pp = jnp.asarray(pred_phys.detach().cpu().numpy())
    outs = []
    for i0 in range(0, pp.shape[0], bs):
        pg = pp[i0:i0 + bs]
        ng = pg.shape[0]
        if ng < bs:
            pg = jnp.concatenate(
                [pg, jnp.broadcast_to(pg[:1], (bs - ng,) + pg.shape[1:])], axis=0)
        v = handle.simulate_batch(pg)                    # (bs, n_rec, T_ds)
        outs.append(np.asarray(v[:ng]))
    return np.concatenate(outs, axis=0)


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
    sim_t_skip_ms  = float(vl.get("sim_t_skip_ms", 0.0))
    sim_dt_ms      = float(vl.get("sim_dt_ms", 0.1))
    sim_skip_bins  = max(0, int(round(sim_t_skip_ms / sim_dt_ms)))
    print(f"[eval] sim_t_skip_ms={sim_t_skip_ms} -> skipping first {sim_skip_bins} bins of jaxley output")

    # ── Which stimulus are we evaluating on? ─────────────────────────────
    # Joint/single packs: the battery lives on the PROBE axis, stims_select has
    # one entry, and the per-probe loop below handles the multi-stim case.
    # Pooled packs: the battery lives on the STIM axis (probs_select == [0]) and
    # the model sees ONE trace with no protocol tag, so each stim is its own test
    # set and must be selected explicitly.
    dcf_eval        = trainMD["train_params"]["data_conf"]
    stims_sel       = list(dcf_eval["stims_select"])
    is_pooled       = bool(dcf_eval.get("pooled_stim_index", False))
    # `pooled_stim_names` is written into the voltage_loss block (HybridLoss owns
    # it), NOT data_conf — check both, and stim_names_multi for joint packs.
    pooled_names    = (vl.get("pooled_stim_names") or dcf_eval.get("pooled_stim_names")
                       or vl.get("stim_names_multi") or dcf_eval.get("stim_names_multi"))
    if args.stimIndex is None:
        stim_col = stims_sel[0]
    else:
        if not 0 <= args.stimIndex < len(stims_sel):
            sys.exit(f"[eval] --stimIndex {args.stimIndex} out of range for "
                     f"stims_select={stims_sel} (len {len(stims_sel)})")
        if not is_pooled and len(stims_sel) == 1:
            sys.exit("[eval] --stimIndex is meaningless for this run: stims_select has one "
                     "entry.  For a JOINT pack the battery is on the probe axis and every "
                     "stim is already evaluated (one row per probe).")
        stim_col = stims_sel[args.stimIndex]
        # The simulated stimulus must match the data column we just selected,
        # otherwise we would compare a trace under stim A to a sim under stim B.
        if pooled_names:
            if stim_col >= len(pooled_names):
                sys.exit(f"[eval] stim column {stim_col} has no name in "
                         f"pooled_stim_names={pooled_names}")
            stim_name = pooled_names[stim_col]
        else:
            # Refusing rather than warning: falling back to vl['stim_name'] would
            # compare data recorded under stim K against a simulation run under
            # stim 0 and still emit a complete, plausible-looking scorecard.
            sys.exit(
                "[eval] FATAL: --stimIndex given but no stim-name list found in metadata "
                "(looked for pooled_stim_names / stim_names_multi under both voltage_loss "
                "and data_conf).  Without it the simulated stimulus cannot be matched to the "
                f"selected data column {stim_col}, and the scores would silently compare "
                f"column {stim_col}'s data against a {stim_name!r} simulation.  "
                "Pass the correct name explicitly or re-run with metadata that carries it.")
        print(f"[eval] --stimIndex {args.stimIndex} -> H5 stim column {stim_col}, "
              f"simulating stim_name={stim_name!r}")

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
    v_soma_norm, unit_par_true, v_mean_arr, v_std_arr = load_test_data(
        trainMD, args.numSamples, stim_col)
    N, T_data = v_soma_norm.shape
    P_cnn = trainMD["train_params"]["model"]["outputSize"]
    print(f"[eval] N={N} test samples, T_data={T_data}, P_cnn={P_cnn}")

    # Build CNN-input shape = (N, T, C) where C = num_probes_after_select.
    # The CNN's input is the raw z-scored voltages on all selected probes.
    # We re-read those for CNN forward.
    h5_path = trainMD["train_params"]["full_h5name"]
    probs = trainMD["train_params"]["data_conf"]["probs_select"]
    with h5py.File(h5_path, "r") as f:
        cnn_in = f["test_volts_norm"][:N, :, :, stim_col]
    cnn_in = cnn_in[:, :, probs].astype(np.float32)   # (N, T, num_probes)
    print(f"[eval] CNN input shape: {cnn_in.shape}")

    # CNN forward — chunked to fit GPU
    bs = 128
    pred_unit_chunks = []
    with torch.no_grad():
        for i in range(0, N, bs):
            # Feed (B, T, C) exactly as Dataloader_H5 does.  Model.forwardCnnOnly
            # does `x.view(-1, C, T)` — a RESHAPE, not a transpose — so the byte
            # layout must match training.  Pre-permuting to (B, C, T) feeds a
            # different byte order and silently corrupts multi-probe (C>1)
            # inputs; it is a no-op for C=1, which is why soma-only looked fine.
            x = torch.from_numpy(cnn_in[i:i+bs]).contiguous().to(device)
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

    # ── Channel-recovery metrics ──────────────────────────────────────────
    # Compare CNN prediction (post-clamp) against ground-truth unit_par.
    # Per-param MSE, R² (explained variance), bias.  Plus an aggregate.
    import importlib
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
    param_names = list(cell_mod.PARAM_KEYS)
    P = len(param_names)
    pred_u = pred_unit_d.cpu().numpy()        # (N, P) post-tanh
    true_u = unit_par_true.astype(np.float64) # (N, P) in [-1, 1]
    # Sanity guard — model output P should match cell PARAM_KEYS length.
    if pred_u.shape[1] != P or true_u.shape[1] != P:
        print(f"[eval] WARN: param count mismatch — pred {pred_u.shape[1]} / "
              f"true {true_u.shape[1]} / cell PARAM_KEYS {P}; channel-recovery"
              f" metrics may be misaligned.")
        P = min(pred_u.shape[1], true_u.shape[1], P)
        pred_u = pred_u[:, :P]; true_u = true_u[:, :P]
        param_names = param_names[:P]

    chan_mse_per_param = ((pred_u - true_u) ** 2).mean(axis=0)
    chan_bias_per_param = (pred_u - true_u).mean(axis=0)
    chan_var_per_param  = true_u.var(axis=0)
    # R² = 1 - MSE/Var, clipped to [-1, 1] for readability
    chan_r2_per_param = 1.0 - chan_mse_per_param / np.maximum(chan_var_per_param, 1e-12)
    chan_mse_overall = float(chan_mse_per_param.mean())
    chan_rmse_overall = float(np.sqrt(chan_mse_overall))
    chan_r2_overall = float(chan_r2_per_param.mean())

    print(f"[eval] channel-recovery (N={pred_u.shape[0]}):")
    print(f"[eval]   overall MSE  = {chan_mse_overall:.4f}  RMSE = {chan_rmse_overall:.4f}  mean R² = {chan_r2_overall:.3f}")
    for i, name in enumerate(param_names):
        print(f"[eval]   {name:<32}  MSE={chan_mse_per_param[i]:.4f}  "
              f"R²={chan_r2_per_param[i]:+.3f}  bias={chan_bias_per_param[i]:+.3f}  "
              f"pred range=[{pred_u[:,i].min():+.2f}, {pred_u[:,i].max():+.2f}]")

    # ── Per-probe simulation ─────────────────────────────────────────────
    # For the multi-stim pack each selected "probe" is the SAME soma recorded
    # under a DIFFERENT stimulus (stim_names_multi, order == probsSelect ==
    # the data's probe axis, see ca3_multistim.hpar.yaml).  So we sim the SAME
    # pred_phys under each probe's stim and compare to that probe's data channel.
    # For a single-stim pack there is just one probe -> one stim (unchanged).
    stim_names_multi = vl.get("stim_names_multi")
    sel_probes = list(probs)                     # h5 probe indices, in CNN-channel order
    if stim_names_multi and len(stim_names_multi) > 1:
        # probe axis order == stim_names_multi order
        sel_stims = [stim_names_multi[p] for p in sel_probes]
    else:
        sel_stims = [stim_name for _ in sel_probes]
    n_probe = len(sel_probes)
    probe_labels = [f"probe{p} ({s})" for p, s in zip(sel_probes, sel_stims)]
    print(f"[eval] {n_probe} probe(s): {probe_labels}")

    sim_bs = int(args.simBatch)
    # Which RECORDING of the simulated cell feeds data channel j?  Multi-probe
    # runs (Exp 2) train with `probe_loss_indices` = the cell's .record() order
    # per data channel; everything else compares recording 0 (soma).
    _pli = vl.get("probe_loss_indices")
    rec_idx = [int(_pli[j]) if (_pli and j < len(_pli)) else 0 for j in range(n_probe)]
    if _pli:
        print(f"[eval] probe_loss_indices={_pli} -> recording per data channel: {rec_idx}")
    sim_solver = str(vl.get("solver", "bwd_euler"))
    sim_cache = {}       # stim_name -> (N, n_rec, T_sim): one sim per distinct stim
    v_sim_pre_all = []   # per-probe (N, T) mV pre-z
    v_sim_z_all   = []   # per-probe (N, T) z-scored
    v_data_z_all  = []   # per-probe (N, T) z-scored data (from cnn_in)
    err_z_all     = []   # per-probe (N,)  voltage MSE_z
    spikes_sim_all = []; spikes_data_all = []
    T = None
    for j, (p_idx, s_name) in enumerate(zip(sel_probes, sel_stims)):
        rec_k = rec_idx[j]
        t0 = time.time()
        if s_name in sim_cache:
            print(f"[eval] {probe_labels[j]}: reusing the {s_name} simulation, recording {rec_k}")
            v_all = sim_cache[s_name]
        else:
            print(f"[eval] running jaxley for {probe_labels[j]} on {N} preds, batch={sim_bs}"
                  f"{' (no-grad)' if args.noGrad else ''}...")
            if args.noGrad:
                v_all = _simulate_nograd(pred_phys, cell_name, s_name, sim_bs, sim_solver)
            else:
                sim_chunks = []
                for i in range(0, N, sim_bs):
                    v = JaxleyBridge.simulate_batch(pred_phys[i:i+sim_bs], cell_name, s_name)
                    sim_chunks.append(v.cpu())              # (B, n_rec, T_sim)
                v_all = torch.cat(sim_chunks, dim=0).numpy()
            sim_cache[s_name] = v_all
        v_sim = v_all[:, rec_k, :]                           # (N, T_sim)
        t1 = time.time()
        # Drop the pre-stim window so the sim time axis aligns with the (already
        # pre-trimmed) data H5.
        if sim_skip_bins > 0:
            v_sim = v_sim[:, sim_skip_bins:]
        if T is None:
            T = min(v_sim.shape[1], T_data)
        v_sim_pre = v_sim[:, :T]                              # (N, T) mV
        v_data_z_p = cnn_in[:, :T, j]                          # (N, T) z-scored data
        v_sim_z = (v_sim_pre - v_sim_pre.mean(axis=1, keepdims=True)) / (
            v_sim_pre.std(axis=1, keepdims=True) + 1e-6)
        err_z = ((v_sim_z - v_data_z_p) ** 2).mean(axis=1)
        sp_sim  = ((v_sim_pre[:, 1:] > 0) & (v_sim_pre[:, :-1] <= 0)).sum(axis=1)
        sp_data = ((v_data_z_p[:, 1:] > 2.0) & (v_data_z_p[:, :-1] <= 2.0)).sum(axis=1)
        v_sim_pre_all.append(v_sim_pre); v_sim_z_all.append(v_sim_z)
        v_data_z_all.append(v_data_z_p); err_z_all.append(err_z)
        spikes_sim_all.append(sp_sim); spikes_data_all.append(sp_data)
        print(f"[eval]   {probe_labels[j]}: jaxley {t1-t0:.1f}s  "
              f"MSE_z mean={err_z.mean():.4f} median={np.median(err_z):.4f}  "
              f"spikes sim/data={sp_sim.mean():.1f}/{sp_data.mean():.1f}")

    # Primary probe (probe 0 = soma under the canonical stim) drives the headline
    # scalar metrics / histogram / CDF so summaries stay comparable across runs.
    v_sim_pre = v_sim_pre_all[0]; v_sim_z = v_sim_z_all[0]
    v_data_z  = v_data_z_all[0];  err_z = err_z_all[0]
    rmse_z = np.sqrt(err_z)
    spikes_sim = spikes_sim_all[0]; spikes_data = spikes_data_all[0]
    spike_diff = spikes_sim - spikes_data
    print(f"[eval] [PRIMARY {probe_labels[0]}] voltage MSE_z: mean={err_z.mean():.4f} "
          f"median={np.median(err_z):.4f} min={err_z.min():.4f} max={err_z.max():.4f}")
    print(f"[eval] [PRIMARY {probe_labels[0]}] voltage RMSE_z: mean={rmse_z.mean():.3f} "
          f"median={np.median(rmse_z):.3f}")
    print(f"[eval] [PRIMARY {probe_labels[0]}] spike count: sim mean={spikes_sim.mean():.1f}, "
          f"data mean={spikes_data.mean():.1f}, |diff| mean={np.abs(spike_diff).mean():.2f}")

    # ─── outputs ────────────────────────────────────────────────────────────
    if args.outDir:
        outDir = args.outDir
    elif args.stimIndex is None:
        outDir = os.path.join(args.modelPath, "eval")
    else:
        # Keep each stim's scores separate — a pooled model has a DIFFERENT
        # recovery per protocol, and overwriting `eval/` would hide that.
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(stim_name))
        outDir = os.path.join(args.modelPath, f"eval_stim{args.stimIndex}_{safe}")
    os.makedirs(outDir, exist_ok=True)

    # CSV per-sample
    import csv
    with open(os.path.join(outDir, "voltage_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "rmse_z", "mse_z", "spikes_sim", "spikes_data", "spike_diff"])
        for i in range(N):
            w.writerow([i, f"{rmse_z[i]:.4f}", f"{err_z[i]:.4f}",
                        int(spikes_sim[i]), int(spikes_data[i]), int(spike_diff[i])])

    # Summary YAML — voltage + channel-recovery metrics together.
    summary = {
        "n_samples": int(N),
        # Which protocol this scorecard is for.  Essential for pooled runs, where
        # the same model has a different recovery per stimulus and several
        # summary.yaml files coexist under one run dir.
        "eval_stim_index": (None if args.stimIndex is None else int(args.stimIndex)),
        "eval_stim_col": int(stim_col),
        "eval_stim_name": (str(stim_name) if stim_name else None),
        "eval_pooled": bool(is_pooled),
        "voltage_mse_z_mean": float(err_z.mean()),
        "voltage_mse_z_median": float(np.median(err_z)),
        "voltage_rmse_z_mean": float(rmse_z.mean()),
        "voltage_rmse_z_median": float(np.median(rmse_z)),
        "spike_count_diff_mean_abs": float(np.abs(spike_diff).mean()),
        "spikes_sim_mean":  float(spikes_sim.mean()),
        "spikes_data_mean": float(spikes_data.mean()),
        # Per-probe voltage fidelity (probe 0 == the *_mean scalars above).
        "voltage_per_probe": [
            {"probe": int(sel_probes[j]), "stim": sel_stims[j],
             "mse_z_mean":   float(err_z_all[j].mean()),
             "mse_z_median": float(np.median(err_z_all[j])),
             "spikes_sim_mean":  float(spikes_sim_all[j].mean()),
             "spikes_data_mean": float(spikes_data_all[j].mean())}
            for j in range(n_probe)
        ],
        "channel_mse_overall":  chan_mse_overall,
        "channel_rmse_overall": chan_rmse_overall,
        "channel_r2_overall":   chan_r2_overall,
        "channel_per_param": [
            {"name": param_names[i],
             "mse":  float(chan_mse_per_param[i]),
             "r2":   float(chan_r2_per_param[i]),
             "bias": float(chan_bias_per_param[i])}
            for i in range(P)
        ],
        "model_path": args.modelPath,
        "cell_name": cell_name,
    }
    write_yaml(summary, os.path.join(outDir, "summary.yaml"))

    # Per-param channel-recovery CSV.
    with open(os.path.join(outDir, "channel_recovery.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "mse", "rmse", "r2", "bias",
                    "pred_min", "pred_max", "true_min", "true_max"])
        for i, nm in enumerate(param_names):
            w.writerow([nm,
                        f"{chan_mse_per_param[i]:.6f}",
                        f"{np.sqrt(chan_mse_per_param[i]):.6f}",
                        f"{chan_r2_per_param[i]:+.4f}",
                        f"{chan_bias_per_param[i]:+.6f}",
                        f"{pred_u[:,i].min():+.4f}", f"{pred_u[:,i].max():+.4f}",
                        f"{true_u[:,i].min():+.4f}", f"{true_u[:,i].max():+.4f}"])

    # Scatter plot grid: pred vs true unit_par per parameter.
    cols = 3
    rows = int(np.ceil(P / cols))
    fig_grid, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows),
                                   squeeze=False)
    for i in range(P):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.scatter(true_u[:, i], pred_u[:, i], s=8, alpha=0.4)
        lim = max(1.05, abs(true_u[:, i]).max(), abs(pred_u[:, i]).max())
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("true unit"); ax.set_ylabel("pred unit")
        ax.set_title(f"{param_names[i]}\nR²={chan_r2_per_param[i]:+.2f}  "
                     f"MSE={chan_mse_per_param[i]:.3f}", fontsize=10)
        ax.grid(alpha=0.3)
    # blank trailing axes
    for j in range(P, rows * cols):
        r, c = divmod(j, cols); axes[r][c].axis("off")
    fig_grid.tight_layout()
    if args.savePng:
        fig_grid.savefig(os.path.join(outDir, "channel_recovery_grid.png"), dpi=120)
    # NOTE: keep fig_grid open — it is re-used as page 1 of trace_overlays.pdf
    # so the PDF is self-contained (ion channels + all voltage overlays).
    print(f"[eval] wrote summary -> {outDir}/summary.yaml")

    # Histogram — kept open, becomes page 2 of the PDF
    fig_hist = plt.figure(figsize=(7, 4))
    plt.hist(err_z, bins=40, color="C0", edgecolor="black", alpha=0.8)
    plt.axvline(err_z.mean(), color="red", linestyle="--", label=f"mean={err_z.mean():.3f}")
    plt.axvline(np.median(err_z), color="orange", linestyle="--", label=f"median={np.median(err_z):.3f}")
    plt.xlabel("per-sample voltage MSE (z-scored)")
    plt.ylabel("count")
    plt.title(f"Voltage-loss distribution on test ({N} samples)")
    plt.legend()
    plt.tight_layout()
    if args.savePng:
        plt.savefig(os.path.join(outDir, "voltage_loss_hist.png"), dpi=120)

    # Aggregate accuracy plot: rmse_z sorted — kept open, becomes page 3 of the PDF
    fig_cdf = plt.figure(figsize=(7, 4))
    plt.plot(np.sort(rmse_z), np.linspace(0, 1, N), color="C0", lw=2)
    plt.xlabel("voltage RMSE (z-scored)")
    plt.ylabel("CDF over test samples")
    plt.title("Test-set voltage-RMSE CDF")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if args.savePng:
        plt.savefig(os.path.join(outDir, "voltage_rmse_cdf.png"), dpi=120)

    # Overlay plots — plot `numOverlay` (default 50) samples spanning the full
    # error range (best -> worst), evenly sampled so no duplicates and the PDF
    # shows the whole quality spectrum rather than only the extremes.
    n_overlay = min(args.numOverlay, N)
    order = np.argsort(err_z)                                 # best -> worst
    sel = np.unique(np.linspace(0, N - 1, n_overlay).astype(int))
    pick = order[sel]
    dt = 0.1
    # x-axis starts at sim_t_skip_ms so plots show absolute simulation time
    # rather than relative-to-trim.  Easier to read against the stim CSV.
    t_axis = sim_t_skip_ms + np.arange(T) * dt   # ms

    pdf_path = os.path.join(outDir, "trace_overlays.pdf")
    pdf = PdfPages(pdf_path)
    # Pages 1-3: ion-channel recovery grid, loss histogram, RMSE CDF — so the
    # PDF is the single self-contained artifact and no PNGs are needed.
    pdf.savefig(fig_grid);  plt.close(fig_grid)
    pdf.savefig(fig_hist);  plt.close(fig_hist)
    pdf.savefig(fig_cdf);   plt.close(fig_cdf)
    # One page per picked sample; one row per probe (data vs predicted-sim in
    # z-scored space).  For the multi-stim pack the rows are the SAME neuron
    # under the 4 different stimuli, so you can see whether the predicted params
    # reproduce every stimulus condition, not just the canonical one.
    for k, idx in enumerate(pick):
        fig, axes = plt.subplots(n_probe, 1, figsize=(11, 2.6 * n_probe + 0.6),
                                 sharex=True, squeeze=False)
        for j in range(n_probe):
            ax = axes[j][0]
            rmse_zj = float(np.sqrt(err_z_all[j][idx]))
            ax.plot(t_axis, v_data_z_all[j][idx], "k", lw=1.0,
                    label="data (z-scored)")
            ax.plot(t_axis, v_sim_z_all[j][idx], "C3", lw=1.0, alpha=0.8,
                    label="predicted (z-scored)")
            ax.set_ylabel("z-scored V")
            ax.set_title(f"{probe_labels[j]}  rmse_z={rmse_zj:.3f}  "
                         f"spikes sim/data = {int(spikes_sim_all[j][idx])}/"
                         f"{int(spikes_data_all[j][idx])}", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        axes[0][0].annotate(f"Sample #{idx}", xy=(0.01, 0.99),
                            xycoords="axes fraction", va="top", fontsize=10,
                            fontweight="bold")
        axes[-1][0].set_xlabel("time (ms)")
        plt.tight_layout()
        if args.savePng:
            plt.savefig(os.path.join(outDir, f"trace_overlay_{k:02d}_sample{idx}.png"), dpi=120)
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    # Compact trace dump so cross-run composite figures can be built without
    # re-simulating (float32, z-scored, all probes, all N samples).
    np.savez_compressed(
        os.path.join(outDir, "traces.npz"),
        t_axis=t_axis.astype(np.float32),
        v_sim_z=np.stack(v_sim_z_all).astype(np.float32),     # (n_probe, N, T)
        v_data_z=np.stack(v_data_z_all).astype(np.float32),
        err_z=np.stack(err_z_all).astype(np.float32),         # (n_probe, N)
        spikes_sim=np.stack(spikes_sim_all), spikes_data=np.stack(spikes_data_all),
        pick=pick, probe_labels=np.array(probe_labels), rec_idx=np.array(rec_idx))
    print(f"[eval] wrote {os.path.join(outDir, 'traces.npz')}")
    print(f"[eval] wrote {pdf_path}: page1=ion-channel grid, page2=loss hist, "
          f"page3=rmse CDF, then {len(pick)} voltage overlays "
          f"({n_probe} probe row(s) each)"
          + ("  [+ .png copies, --savePng]" if args.savePng else "  [PDF-only; --savePng for .png]"))
    print(f"[eval] DONE — see {outDir}/")


if __name__ == "__main__":
    main()
