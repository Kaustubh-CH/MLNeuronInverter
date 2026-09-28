#!/usr/bin/env python
"""Evaluate the neuron-level joint 5-sweep model on held-out neurons.

Per test sample (= neuron sweep-combo): CNN on the (4001,5) battery -> tanh ->
ONE theta -> jaxley sim under all five Roy<amp>_icaRec_5k stims -> score sim i
vs the combo's channel-i recording (mse_fixed / mse_z / dtw_z / spikes).
Unit->phys mapping comes from the MODEL's sum_train voltage_loss (wide box).

Also reports the metric this architecture exists for: PER-NEURON PARAM
CONSISTENCY -- the std of theta across a neuron's sweep combos (a per-cell
model should give the same params regardless of which sweeps it saw).

Outputs in --outDir: roy_summary.csv (per family), per_neuron.csv (spikes and
mse per neuron x family + param std), Roy<amp>_voltage_overlays.pdf,
neuron_param_consistency.png.  Requires JAX_ENABLE_X64=true (set below).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse, importlib, json, collections
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                  unit_to_phys_np, VOLT_NORM_MEAN, VOLT_NORM_STD)
from toolbox.soft_dtw import soft_dtw_loss

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_neuron5/RoyExpNeuron5.mlPack1.h5"
FAMS = ["Roy100", "Roy500", "Roy1000", "Roy1500", "Roy2000"]
AMPS = [100, 500, 1000, 1500, 2000]
SIM_BATCH = 16


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default="out")
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_neuron5")
    p.add_argument("--cellName", default="ca3_pyramidal")
    p.add_argument("--numOverlay", type=int, default=6)
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_neuron5")
    os.makedirs(outDir, exist_ok=True)

    vl = trainMD["train_params"]["voltage_loss"]
    centers, logspans = phys_par_range_to_arrays(vl["phys_par_range"])
    lin_mask = phys_par_range_linear_mask(vl["phys_par_range"])
    par_names = ["g_leak", "gbar_na3", "gkdrbar_kdr", "gkabar_kap", "gbar_km", "gkdbar_kd"]
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)
    print(f"[n5] model={args.modelPath} spans={logspans.tolist()}")

    with h5py.File(args.packFile, "r") as f:
        X = f[args.dom + "_volts_norm"][:, :, :, 0].astype(np.float32)   # (M,4001,5)
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)        # (M,4000,5)
        nids = f[args.dom + "_neuron_id"][:].astype(str)
    M, T_data = raw.shape[0], raw.shape[1]

    with torch.no_grad():
        xt = torch.from_numpy(X).contiguous().to(device)
        pu = torch.tanh(model(xt).double()).cpu().numpy()               # (M,6) unit
    phys = unit_to_phys_np(pu, centers, logspans, linear=lin_mask)

    # per-neuron consistency: std of unit params across a neuron's combos
    by_n = collections.defaultdict(list)
    for i, n in enumerate(nids):
        by_n[n].append(i)
    print("[n5] per-neuron unit-param mean +- std across combos:")
    cons_rows = []
    for n, idxs in sorted(by_n.items()):
        mu, sd = pu[idxs].mean(0), pu[idxs].std(0)
        cons_rows.append((n, mu, sd))
        print(f"  {n}: " + " ".join(f"{par_names[j][:9]}={mu[j]:+.2f}±{sd[j]:.2f}"
                                    for j in range(6)))

    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()
    import jax.numpy as jnp

    pp_all = jnp.asarray(phys)
    summary, per_neuron = [], collections.defaultdict(dict)
    for ci, (fam, amp) in enumerate(zip(FAMS, AMPS)):
        handle = JaxleyBridge.get_handle(args.cellName, f"Roy{amp}_icaRec_5k")
        outs = []
        for i0 in range(0, M, SIM_BATCH):
            pg = pp_all[i0:i0 + SIM_BATCH]
            ng = pg.shape[0]
            if ng < SIM_BATCH:
                pg = jnp.concatenate(
                    [pg, jnp.broadcast_to(pg[:1], (SIM_BATCH - ng,) + pg.shape[1:])], 0)
            outs.append(np.asarray(handle.simulate_batch(pg))[:ng, 0, :])
        v_al = np.concatenate(outs, 0)[:, -T_data:]
        data = raw[:, :, ci]
        ok = np.isfinite(v_al).all(axis=1)
        v_al = np.nan_to_num(v_al, nan=0.0, posinf=0.0, neginf=0.0)
        mse_fixed = ((zfix(v_al) - zfix(data)) ** 2).mean(axis=1)
        zs, zd = zscore(v_al), zscore(data)
        mse_z = ((zs - zd) ** 2).mean(axis=1)
        dtw_z = np.array([float(soft_dtw_loss(
            torch.tensor(zs[i:i + 1], dtype=torch.float64),
            torch.tensor(zd[i:i + 1], dtype=torch.float64),
            gamma=0.1, n_points=256, band_ms=8.0)) for i in range(M)])
        sp_s, sp_d = spikes_pos(v_al), spikes_pos(data)
        summary.append((amp, M, int((~ok).sum()),
                        float(mse_fixed[ok].mean()), float(np.median(mse_fixed[ok])),
                        float(mse_z[ok].mean()), float(dtw_z[ok].mean()),
                        float(sp_s[ok].mean()), float(sp_d.mean())))
        for n, idxs in by_n.items():
            per_neuron[n][fam] = (float(np.mean(sp_s[idxs])), float(np.mean(sp_d[idxs])),
                                  float(np.mean(mse_z[idxs])))
        print(f"[n5] {fam}: MSE_fixed={mse_fixed[ok].mean():.3f} "
              f"MSE_z={mse_z[ok].mean():.3f} DTW_z={dtw_z[ok].mean():.3f} "
              f"spikes sim/data={sp_s[ok].mean():.1f}/{sp_d.mean():.1f}")

        dt = 0.1; t_axis = np.arange(T_data) * dt
        with PdfPages(os.path.join(outDir, f"{fam}_voltage_overlays.pdf")) as pdf:
            done = set()
            for i in range(M):                      # first combo of each neuron
                if nids[i] in done or len(done) >= args.numOverlay:
                    continue
                done.add(nids[i])
                fig, ax = plt.subplots(figsize=(11, 3.0))
                ax.plot(t_axis, data[i], "k", lw=0.7, label="experimental")
                ax.plot(t_axis, v_al[i], "C3", lw=0.7, alpha=0.85, label="sim (per-neuron θ)")
                ax.set_title(f"{fam} {nids[i]}  MSE_z={mse_z[i]:.3f} DTW_z={dtw_z[i]:.3f} "
                             f"spikes {int(sp_s[i])}/{int(sp_d[i])}", fontsize=9)
                ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with open(os.path.join(outDir, "roy_summary.csv"), "w") as fo:
        fo.write("amp,n,n_bad,mse_fixed_mean,mse_fixed_median,mse_z_mean,dtw_z_mean,"
                 "spikes_sim,spikes_data\n")
        for row in summary:
            fo.write(",".join(str(r) for r in row) + "\n")
    with open(os.path.join(outDir, "per_neuron.csv"), "w") as fo:
        fo.write("neuron_id," + ",".join(f"{f}_sim,{f}_data,{f}_msez" for f in FAMS)
                 + "," + ",".join(f"std_{p}" for p in par_names) + "\n")
        for n, mu, sd in cons_rows:
            cells = []
            for f in FAMS:
                s, d, mz = per_neuron[n][f]
                cells += [f"{s:.2f}", f"{d:.2f}", f"{mz:.3f}"]
            fo.write(n + "," + ",".join(cells) + ","
                     + ",".join(f"{x:.3f}" for x in sd) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), squeeze=False)
    for j in range(6):
        ax = axes[j // 3][j % 3]
        names = [n for n, _, _ in cons_rows]
        ax.errorbar(range(len(names)), [m[j] for _, m, _ in cons_rows],
                    yerr=[s[j] for _, _, s in cons_rows], fmt="o", ms=4, capsize=3)
        ax.axhspan(-1, 1, color="green", alpha=0.05); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[j], fontsize=9); ax.grid(alpha=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[-6:] for n in names], fontsize=6, rotation=45)
    fig.suptitle("Per-neuron unit params (mean ± std over sweep combos) — "
                 "small bars = consistent per-cell θ", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outDir, "neuron_param_consistency.png"), dpi=120)
    print(f"[n5] wrote {outDir}/roy_summary.csv, per_neuron.csv, overlays, consistency png")


if __name__ == "__main__":
    main()
