#!/usr/bin/env python
"""Voltage-error evaluation of a CA3 model on Paula's Roy v2 experimental pack.

Successor of .claude/worktrees/ca3-vo-dtwblur/plot_exp_overlay_roy.py, adapted
to the ca3ft pack layout (RoyExpPack_ca3ft: one H5, <dom>_* keys, 4001 bins
fixed-norm, per-sample stim family) and the v2 rig stimuli (Roy<amp>_icav2_5k =
holding -0.0496 nA + fitted slope x 5k50kInterChaoticB).

Per stimulus family:
  1. CNN on the pack's fixed-norm 4001-bin soma trace -> unit params (tanh),
  2. unit -> physical conductances via the pack's input_meta.phys_par_range,
  3. jaxley re-sim under the family's rig-accurate stimulus (500 ms),
  4. align last 4000 bins vs the recording; score voltage error two ways:
       mse_fixed : MSE in the FIXED z-space (same space as the training loss)
       mse_z     : MSE after per-trace z-scoring (comparable to the Apr-2026
                   plot_exp_overlay_roy.py numbers)
     plus spike counts (upward 0 mV crossings) sim vs data.

Outputs in --outDir: roy_summary.csv (per family), roy_per_sample.csv
(neuron-resolved), Roy<amp>_voltage_overlays.pdf, roy_unit_params.png.

Run zero-shot (k128 out/) and post-fine-tune (fine-tuned out/) with the same
pack and diff the summaries.  Requires JAX_ENABLE_X64=true (set below).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")   # before any jax import

import argparse, importlib, json, time
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox import jaxley_cells
from toolbox.jaxley_utils import (phys_par_range_to_arrays,
                                  VOLT_NORM_MEAN, VOLT_NORM_STD)

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
AMPS = [100, 500, 1000, 1500, 2000]
SIM_BATCH = 32     # fixed sim batch (padded) -> one XLA shape per stim


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default="out")
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_royv2")
    p.add_argument("--cellName", default="ca3_pyramidal")
    p.add_argument("--numOverlay", type=int, default=4)
    p.add_argument("--noClampTanh", action="store_true",
                   help="skip tanh on the CNN output before the unit->phys map")
    p.add_argument("--paramSubset", type=int, nargs="*", default=None,
                   help="cell-param indices the CNN predicts (subset models); "
                        "the rest are filled at unit 0 = cell default")
    p.add_argument("--stimChannel", action="store_true",
                   help="feed the pack's probe 1 (stim, fixed nA scale) as a "
                        "second CNN input channel (ms2ch models)")
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def sim_family(pred_phys, cell_name, stim_name):
    """Chunked, pad-to-SIM_BATCH jaxley forward -> (N, T) soma mV."""
    outs = []
    for i0 in range(0, pred_phys.shape[0], SIM_BATCH):
        pg = pred_phys[i0:i0 + SIM_BATCH]
        ng = pg.shape[0]
        if ng < SIM_BATCH:
            pg = torch.cat([pg, pg[:1].expand(SIM_BATCH - ng, -1)], dim=0)
        v = JaxleyBridge.simulate_batch(pg, cell_name, stim_name)
        outs.append(v[:ng, 0, :].cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_royv2")
    os.makedirs(outDir, exist_ok=True)

    with h5py.File(args.packFile, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        if args.stimChannel:
            X = f[args.dom + "_volts_norm"][:, :, :, 0].astype(np.float32)  # (N,4001,2)
        else:
            X = f[args.dom + "_volts_norm"][:, :, 0, 0].astype(np.float32)  # (N,4001) fixed-z
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)       # (N,4000) mV
        famS = f[args.dom + "_stim_family"][:].astype(str)
        nids = f[args.dom + "_neuron_id"][:].astype(str)

    par_names = meta["input_meta"]["parName"]
    stim_stems = meta["stim_from_label"]["stim_names"]
    fam_order = meta["stim_from_label"]["stim_family_order"]
    centers, logspans = phys_par_range_to_arrays(meta["input_meta"]["phys_par_range"])
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    # icav2 stims are 5000 pts x 0.1 ms = 500 ms
    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()
    P = len(par_names)
    print(f"[royv2] model={args.modelPath} dom={args.dom} N={len(X)} "
          f"clampTanh={not args.noClampTanh} cell={args.cellName}")

    unit_by_amp, summary, per_sample = {}, [], []
    for amp in AMPS:
        fam = f"Roy{amp}"
        m = famS == fam
        if not m.any():
            continue
        data_raw = raw[m]                                     # (N,4000) mV
        T_data = data_raw.shape[1]
        with torch.no_grad():
            xin = X[m] if args.stimChannel else X[m][:, :, None]   # (N,T,C)
            xt = torch.from_numpy(xin).contiguous().to(device)
            pred_unit = model(xt).float().cpu()
        pu = pred_unit.double().to(device)
        if not args.noClampTanh:
            pu = torch.tanh(pu)
        if args.paramSubset:
            full = torch.zeros(pu.shape[0], len(centers), dtype=pu.dtype, device=device)
            full[:, args.paramSubset] = pu
            pu = full
        pred_phys = centers_t * torch.pow(
            torch.tensor(10.0, dtype=torch.float64, device=device), pu * logspans_t)
        unit_by_amp[amp] = pu.cpu().numpy()

        stim_name = stim_stems[fam_order.index(fam)]
        t0 = time.time()
        v_sim = sim_family(pred_phys, args.cellName, stim_name)  # (N, T_sim) mV
        v_al = v_sim[:, -T_data:]
        bad = ~np.isfinite(v_al).all(axis=1)
        if bad.any():
            print(f"[royv2] WARN {fam}: {bad.sum()} non-finite sims dropped from stats")
        ok = ~bad
        mse_fixed = ((zfix(v_al) - zfix(data_raw)) ** 2).mean(axis=1)
        mse_z = ((zscore(v_al) - zscore(data_raw)) ** 2).mean(axis=1)
        sp_sim, sp_data = spikes_pos(v_al), spikes_pos(data_raw)
        summary.append((amp, int(m.sum()), int(bad.sum()),
                        float(mse_fixed[ok].mean()), float(np.median(mse_fixed[ok])),
                        float(mse_z[ok].mean()),
                        float(sp_sim[ok].mean()), float(sp_data.mean())))
        for i, nidx in enumerate(np.flatnonzero(m)):
            per_sample.append((nids[nidx], fam, float(mse_fixed[i]), float(mse_z[i]),
                               int(sp_sim[i]), int(sp_data[i]), bool(bad[i])))
        print(f"[royv2] {fam}: N={int(m.sum())} jaxley {time.time()-t0:.1f}s "
              f"MSE_fixed={mse_fixed[ok].mean():.3f} MSE_z={mse_z[ok].mean():.3f} "
              f"spikes sim/data={sp_sim[ok].mean():.1f}/{sp_data.mean():.1f}")

        dt = 0.1; t_axis = np.arange(T_data) * dt
        with PdfPages(os.path.join(outDir, f"Roy{amp}_voltage_overlays.pdf")) as pdf:
            for idx in range(min(args.numOverlay, int(m.sum()))):
                fig, axes = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True)
                axes[0].plot(t_axis, data_raw[idx], "k", lw=0.8, label="experimental")
                axes[0].plot(t_axis, v_al[idx], "C3", lw=0.8, alpha=0.8, label="sim from pred params")
                axes[0].set_ylabel("V (mV)"); axes[0].legend(loc="upper right", fontsize=8)
                axes[0].set_title(f"{fam} {nids[np.flatnonzero(m)[idx]]}  "
                                  f"MSE_fixed={mse_fixed[idx]:.3f} MSE_z={mse_z[idx]:.3f}  "
                                  f"spikes sim/data={int(sp_sim[idx])}/{int(sp_data[idx])}", fontsize=9)
                axes[1].plot(t_axis, zfix(data_raw[idx:idx+1])[0], "k", lw=0.8)
                axes[1].plot(t_axis, zfix(v_al[idx:idx+1])[0], "C3", lw=0.8, alpha=0.8)
                axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("fixed-z V")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with open(os.path.join(outDir, "roy_summary.csv"), "w") as fo:
        fo.write("amp,n,n_bad,mse_fixed_mean,mse_fixed_median,mse_z_mean,spikes_sim,spikes_data\n")
        for row in summary:
            fo.write(",".join(str(r) for r in row) + "\n")
    with open(os.path.join(outDir, "roy_per_sample.csv"), "w") as fo:
        fo.write("neuron_id,family,mse_fixed,mse_z,spikes_sim,spikes_data,sim_nonfinite\n")
        for row in per_sample:
            fo.write(",".join(str(r) for r in row) + "\n")

    # unit-parameter consistency across amplitudes
    tags = [f"Roy{a}" for a in AMPS if a in unit_by_amp]
    cols = 3; rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for pi in range(P):
        ax = axes[pi // cols][pi % cols]
        vals = [unit_by_amp[a][:, pi] for a in AMPS if a in unit_by_amp]
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        ax.axhspan(-1, 1, color="green", alpha=0.05); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[pi], fontsize=10); ax.grid(alpha=0.3, axis="y")
    for j in range(P, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Predicted CA3 unit params on Roy v2 recordings ({args.dom})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outDir, "roy_unit_params.png"), dpi=120)
    print(f"[royv2] wrote {outDir}/roy_summary.csv, roy_per_sample.csv, overlays, unit params")


if __name__ == "__main__":
    main()
