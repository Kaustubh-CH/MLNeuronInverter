#!/usr/bin/env python
"""Cross-cell-model probe: score the L5TTPC jaxley model on Roy v2 recordings.

User request 2026-08-26: "run it on the L5TTPC model we have to see if it will
improve the predictions".  Same flow as royexp_ft_57631428/plot_exp_overlay_royv2.py
but the CNN + unit->phys mapping + re-simulation all come from a 19-parameter
L5TTPC_jaxley_nc2 run (cell l5ttpc) instead of the 6-par CA3 model:

  1. CNN on the ca3ft pack's fixed-norm soma trace, leading-edge padded from
     4001 to the model's num_time_bins (5001; the pad region is rest),
  2. unit -> physical via the MODEL's own input_meta.phys_par_range (19 par;
     the CA3 pack meta is ignored for the mapping),
  3. jaxley re-sim with cell l5ttpc under the family's Roy<amp>_icav2_5k
     stimulus (same stim dir as CA3, 5000 pts x 0.1 ms = 500 ms),
  4. align last 4000 bins vs the recording; score mse_fixed / mse_z / dtw_z /
     spike counts exactly as the CA3 royv2 scorer does.

The L5 run trained with clamp_unit_tanh False; default is faithful (no tanh),
--clampTanh bounds the unit params to the trained box instead.
Requires JAX_ENABLE_X64=true (set below).
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
from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                  unit_to_phys_torch, VOLT_NORM_MEAN, VOLT_NORM_STD)
from toolbox.soft_dtw import soft_dtw_loss

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
MODEL_DEF = ("/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/l5ttpc_supervised/"
             "L5TTPC_jaxley_nc2/super/out")
AMPS = [500, 1000, 1500, 2000]    # Roy100 dropped from the study (user, 2026-08-26)
SIM_BATCH = 8                     # L5 morphology is heavier than the CA3 soma


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default=MODEL_DEF)
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default=None, help="default <modelPath>/exp_royv2_l5")
    p.add_argument("--cellName", default="l5ttpc")
    p.add_argument("--numOverlay", type=int, default=4)
    p.add_argument("--clampTanh", action="store_true",
                   help="tanh the CNN output (model trained WITHOUT the clamp)")
    return p.parse_args()


def zscore(v):
    return (v - v.mean(axis=1, keepdims=True)) / (v.std(axis=1, keepdims=True) + 1e-6)


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def sim_family(pred_phys, cell_name, stim_name):
    """Chunked NO-GRAD jaxley forward -> (N, T) soma mV.

    Calls the handle's jitted vmap forward directly instead of
    JaxleyBridge.simulate_batch: the latter always builds the jax.vjp graph
    (it is the training bridge) and OOMs on the L5 cell (fp64 x 5000 steps).
    """
    import jax.numpy as jnp
    handle = JaxleyBridge.get_handle(cell_name, stim_name)
    pp = jnp.asarray(pred_phys.detach().cpu().numpy())
    outs = []
    for i0 in range(0, pp.shape[0], SIM_BATCH):
        pg = pp[i0:i0 + SIM_BATCH]
        ng = pg.shape[0]
        if ng < SIM_BATCH:
            pg = jnp.concatenate(
                [pg, jnp.broadcast_to(pg[:1], (SIM_BATCH - ng,) + pg.shape[1:])], axis=0)
        v = handle.simulate_batch(pg)                 # (B, n_rec, T_ds)
        outs.append(np.asarray(v[:ng, 0, :]))
    return np.concatenate(outs, axis=0)


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(args.modelPath, device)
    outDir = args.outDir or os.path.join(args.modelPath, "exp_royv2_l5")
    os.makedirs(outDir, exist_ok=True)

    # 19-par mapping from the MODEL, not the CA3 pack
    im = trainMD["input_meta"]
    par_names = im["parName"]
    T_model = int(im["num_time_bins"])
    centers, logspans = phys_par_range_to_arrays(im["phys_par_range"])
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)
    linear_t   = torch.tensor(phys_par_range_linear_mask(im["phys_par_range"]), device=device)
    P = len(par_names)

    with h5py.File(args.packFile, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        X = f[args.dom + "_volts_norm"][:, :, 0, 0].astype(np.float32)  # (N,4001) fixed-z
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)       # (N,4000) mV
        famS = f[args.dom + "_stim_family"][:].astype(str)
        nids = f[args.dom + "_neuron_id"][:].astype(str)

    stim_stems = meta["stim_from_label"]["stim_names"]
    fam_order = meta["stim_from_label"]["stim_family_order"]

    # icav2 stims are 5000 pts x 0.1 ms = 500 ms
    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()
    print(f"[royv2-l5] model={args.modelPath} P={P} T_model={T_model} dom={args.dom} "
          f"N={len(X)} clampTanh={args.clampTanh} cell={args.cellName}")

    unit_by_amp, summary, per_sample = {}, [], []
    for amp in AMPS:
        fam = f"Roy{amp}"
        m = famS == fam
        if not m.any():
            continue
        data_raw = raw[m]                                     # (N,4000) mV
        T_data = data_raw.shape[1]
        x = X[m]
        if x.shape[1] < T_model:      # leading pad: 0-100 ms of the 5k stim is rest
            x = np.pad(x, ((0, 0), (T_model - x.shape[1], 0)), mode="edge")
        elif x.shape[1] > T_model:
            x = x[:, :T_model]
        with torch.no_grad():
            xt = torch.from_numpy(x[:, :, None]).contiguous().to(device)
            pred_unit = model(xt).float().cpu()
        pu = pred_unit.double().to(device)
        if args.clampTanh:
            pu = torch.tanh(pu)
        pred_phys = unit_to_phys_torch(pu, centers_t, logspans_t, linear_t)
        unit_by_amp[amp] = pu.cpu().numpy()

        stim_name = stim_stems[fam_order.index(fam)]
        t0 = time.time()
        v_sim = sim_family(pred_phys, args.cellName, stim_name)  # (N, T_sim) mV
        v_al = v_sim[:, -T_data:]
        bad = ~np.isfinite(v_al).all(axis=1)
        if bad.any():
            print(f"[royv2-l5] WARN {fam}: {bad.sum()} non-finite sims dropped from stats")
        ok = ~bad
        v_al_f = np.nan_to_num(v_al, nan=0.0, posinf=0.0, neginf=0.0)
        mse_fixed = ((zfix(v_al_f) - zfix(data_raw)) ** 2).mean(axis=1)
        zs, zd = zscore(v_al_f), zscore(data_raw)
        mse_z = ((zs - zd) ** 2).mean(axis=1)
        dtw_z = np.array([float(soft_dtw_loss(
            torch.tensor(zs[i:i + 1], dtype=torch.float64),
            torch.tensor(zd[i:i + 1], dtype=torch.float64),
            gamma=0.1, n_points=256, band_ms=8.0)) for i in range(len(zs))])
        sp_sim, sp_data = spikes_pos(v_al_f), spikes_pos(data_raw)
        if not ok.any():
            print(f"[royv2-l5] {fam}: ALL {int(m.sum())} sims non-finite -- skipped")
            summary.append((amp, int(m.sum()), int(bad.sum()),
                            np.nan, np.nan, np.nan, np.nan, np.nan, float(sp_data.mean())))
            continue
        summary.append((amp, int(m.sum()), int(bad.sum()),
                        float(mse_fixed[ok].mean()), float(np.median(mse_fixed[ok])),
                        float(mse_z[ok].mean()), float(dtw_z[ok].mean()),
                        float(sp_sim[ok].mean()), float(sp_data.mean())))
        for i, nidx in enumerate(np.flatnonzero(m)):
            per_sample.append((nids[nidx], fam, float(mse_fixed[i]), float(mse_z[i]),
                               float(dtw_z[i]),
                               int(sp_sim[i]), int(sp_data[i]), bool(bad[i])))
        print(f"[royv2-l5] {fam}: N={int(m.sum())} jaxley {time.time()-t0:.1f}s "
              f"MSE_fixed={mse_fixed[ok].mean():.3f} MSE_z={mse_z[ok].mean():.3f} "
              f"DTW_z={dtw_z[ok].mean():.3f} "
              f"spikes sim/data={sp_sim[ok].mean():.1f}/{sp_data.mean():.1f}")

        dt = 0.1; t_axis = np.arange(T_data) * dt
        with PdfPages(os.path.join(outDir, f"Roy{amp}_voltage_overlays.pdf")) as pdf:
            for idx in range(min(args.numOverlay, int(m.sum()))):
                fig, axes = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True)
                axes[0].plot(t_axis, data_raw[idx], "k", lw=0.8, label="experimental")
                axes[0].plot(t_axis, v_al_f[idx], "C3", lw=0.8, alpha=0.8,
                             label="L5TTPC sim from pred params")
                axes[0].set_ylabel("V (mV)"); axes[0].legend(loc="upper right", fontsize=8)
                axes[0].set_title(f"{fam} {nids[np.flatnonzero(m)[idx]]}  "
                                  f"MSE_fixed={mse_fixed[idx]:.3f} MSE_z={mse_z[idx]:.3f} "
                                  f"DTW_z={dtw_z[idx]:.3f}  "
                                  f"spikes sim/data={int(sp_sim[idx])}/{int(sp_data[idx])}",
                                  fontsize=9)
                axes[1].plot(t_axis, zfix(data_raw[idx:idx+1])[0], "k", lw=0.8)
                axes[1].plot(t_axis, zfix(v_al_f[idx:idx+1])[0], "C3", lw=0.8, alpha=0.8)
                axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("fixed-z V")
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
    cols = 4; rows = int(np.ceil(P / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), squeeze=False)
    for pi in range(P):
        ax = axes[pi // cols][pi % cols]
        vals = [unit_by_amp[a][:, pi] for a in AMPS if a in unit_by_amp]
        ax.boxplot(vals, tick_labels=tags, showfliers=False)
        ax.axhspan(-1, 1, color="green", alpha=0.05); ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(par_names[pi], fontsize=8); ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=7)
    for j in range(P, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"L5TTPC unit params on Roy v2 recordings ({args.dom}); "
                 "green = trained [-1,1] box (model trained UNCLAMPED)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outDir, "roy_unit_params.png"), dpi=120)
    print(f"[royv2-l5] wrote {outDir}/roy_summary.csv, roy_per_sample.csv, overlays, unit params")


if __name__ == "__main__":
    main()
