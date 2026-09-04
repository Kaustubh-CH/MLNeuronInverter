#!/usr/bin/env python
"""Evaluate a stim-as-channel single-sweep model on held-out Roy neurons.

Same flow as royexp_ft/plot_exp_overlay_royv2.py, but reads the 2-channel
RoyExpStimCh pack and feeds the model the channels it was trained on
(--channels "0 1" for the stim-channel arm, "0" for the A/B control).
Unit->phys ranges come from the MODEL's sum_train (wide box).  Sims run under
the family's exact recorded Roy<amp>_icaRec_5k stimulus.

Key diagnostic on top of the usual metrics: PARAMS vs AMPLITUDE — with the
drive visible as an input channel the predicted params for the same neuron
should stop drifting with stimulus amplitude.  Writes roy_summary.csv,
roy_per_sample.csv, overlays, roy_unit_params.png in --outDir.
Requires JAX_ENABLE_X64=true (set below).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse, importlib, json, time
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox.jaxley_utils import (phys_par_range_to_arrays,
                                  VOLT_NORM_MEAN, VOLT_NORM_STD)
from toolbox.soft_dtw import soft_dtw_loss

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_stimch/RoyExpStimCh.mlPack1.h5"
AMPS = [100, 500, 1000, 1500, 2000]
SIM_BATCH = 16


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default="out")
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_stimch")
    p.add_argument("--cellName", default="ca3_pyramidal")
    p.add_argument("--channels", default="0 1",
                   help="input channels, space-separated ('0 1' stim-ch arm, '0' control)")
    p.add_argument("--numOverlay", type=int, default=4)
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def main():
    args = get_parser()
    chans = [int(c) for c in args.channels.split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_stimch")
    os.makedirs(outDir, exist_ok=True)

    vl = trainMD["train_params"]["voltage_loss"]
    centers, logspans = phys_par_range_to_arrays(vl["phys_par_range"])
    par_names = ["g_leak", "gbar_na3", "gkdrbar_kdr", "gkabar_kap", "gbar_km", "gkdbar_kd"]
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)
    P = len(par_names)

    with h5py.File(args.packFile, "r") as f:
        X = f[args.dom + "_volts_norm"][:, :, :, 0].astype(np.float32)  # (N,4001,2)
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)       # (N,4000)
        famS = f[args.dom + "_stim_family"][:].astype(str)
        nids = f[args.dom + "_neuron_id"][:].astype(str)
    print(f"[stimch] model={args.modelPath} channels={chans} N={len(X)} "
          f"spans={logspans.tolist()}")

    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()
    import jax.numpy as jnp

    unit_by_amp, summary, per_sample = {}, [], []
    for amp in AMPS:
        fam = f"Roy{amp}"
        m = famS == fam
        if not m.any():
            continue
        data_raw = raw[m]
        T_data = data_raw.shape[1]
        with torch.no_grad():
            xt = torch.from_numpy(X[m][:, :, chans]).contiguous().to(device)
            pu = torch.tanh(model(xt).double()).cpu()
        pred_phys = (centers_t.cpu() * torch.pow(
            torch.tensor(10.0, dtype=torch.float64), pu * logspans_t.cpu())).numpy()
        unit_by_amp[amp] = pu.numpy()

        handle = JaxleyBridge.get_handle(args.cellName, f"Roy{amp}_icaRec_5k")
        pp = jnp.asarray(pred_phys)
        outs = []
        t0 = time.time()
        for i0 in range(0, pp.shape[0], SIM_BATCH):
            pg = pp[i0:i0 + SIM_BATCH]
            ng = pg.shape[0]
            if ng < SIM_BATCH:
                pg = jnp.concatenate(
                    [pg, jnp.broadcast_to(pg[:1], (SIM_BATCH - ng,) + pg.shape[1:])], 0)
            outs.append(np.asarray(handle.simulate_batch(pg))[:ng, 0, :])
        v_al = np.concatenate(outs, 0)[:, -T_data:]
        ok = np.isfinite(v_al).all(axis=1)
        v_al = np.nan_to_num(v_al, nan=0.0, posinf=0.0, neginf=0.0)
        mse_fixed = ((zfix(v_al) - zfix(data_raw)) ** 2).mean(axis=1)
        zs, zd = zscore(v_al), zscore(data_raw)
        mse_z = ((zs - zd) ** 2).mean(axis=1)
        dtw_z = np.array([float(soft_dtw_loss(
            torch.tensor(zs[i:i + 1], dtype=torch.float64),
            torch.tensor(zd[i:i + 1], dtype=torch.float64),
            gamma=0.1, n_points=256, band_ms=8.0)) for i in range(len(zs))])
        sp_s, sp_d = spikes_pos(v_al), spikes_pos(data_raw)
        summary.append((amp, int(m.sum()), int((~ok).sum()),
                        float(mse_fixed[ok].mean()), float(np.median(mse_fixed[ok])),
                        float(mse_z[ok].mean()), float(dtw_z[ok].mean()),
                        float(sp_s[ok].mean()), float(sp_d.mean())))
        for i, nidx in enumerate(np.flatnonzero(m)):
            per_sample.append((nids[nidx], fam, float(mse_fixed[i]), float(mse_z[i]),
                               float(dtw_z[i]), int(sp_s[i]), int(sp_d[i]),
                               bool(~ok[i])))
        print(f"[stimch] {fam}: N={int(m.sum())} jaxley {time.time()-t0:.1f}s "
              f"MSE_fixed={mse_fixed[ok].mean():.3f} MSE_z={mse_z[ok].mean():.3f} "
              f"DTW_z={dtw_z[ok].mean():.3f} "
              f"spikes sim/data={sp_s[ok].mean():.1f}/{sp_d.mean():.1f}")

        dt = 0.1; t_axis = np.arange(T_data) * dt
        with PdfPages(os.path.join(outDir, f"Roy{amp}_voltage_overlays.pdf")) as pdf:
            for idx in range(min(args.numOverlay, int(m.sum()))):
                fig, ax = plt.subplots(figsize=(11, 3.0))
                ax.plot(t_axis, data_raw[idx], "k", lw=0.7, label="experimental")
                ax.plot(t_axis, v_al[idx], "C3", lw=0.7, alpha=0.85, label="sim")
                ax.set_title(f"{fam} {nids[np.flatnonzero(m)[idx]]}  "
                             f"MSE_z={mse_z[idx]:.3f} DTW_z={dtw_z[idx]:.3f}  "
                             f"spikes {int(sp_s[idx])}/{int(sp_d[idx])}", fontsize=9)
                ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with open(os.path.join(outDir, "roy_summary.csv"), "w") as fo:
        fo.write("amp,n,n_bad,mse_fixed_mean,mse_fixed_median,mse_z_mean,dtw_z_mean,"
                 "spikes_sim,spikes_data\n")
        for row in summary:
            fo.write(",".join(str(r) for r in row) + "\n")
    with open(os.path.join(outDir, "roy_per_sample.csv"), "w") as fo:
        fo.write("neuron_id,family,mse_fixed,mse_z,dtw_z,spikes_sim,spikes_data,sim_nonfinite\n")
        for row in per_sample:
            fo.write(",".join(str(r) for r in row) + "\n")

    tags = [f"Roy{a}" for a in AMPS if a in unit_by_amp]
    cols = 3; rows_ = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4.6 * cols, 3.4 * rows_), squeeze=False)
    for pi in range(P):
        ax = axes[pi // cols][pi % cols]
        vals = [unit_by_amp[a][:, pi] for a in AMPS if a in unit_by_amp]
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        ax.axhspan(-1, 1, color="green", alpha=0.05); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[pi], fontsize=10); ax.grid(alpha=0.3, axis="y")
    for j in range(P, rows_ * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Unit params vs amplitude (channels={chans}); flat across families "
                 "= amplitude-invariant (the stim channel's job)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outDir, "roy_unit_params.png"), dpi=120)
    print(f"[stimch] wrote {outDir}/roy_summary.csv, per-sample, overlays, unit params")


if __name__ == "__main__":
    main()
