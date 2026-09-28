#!/usr/bin/env python
"""Overlay experimental voltage traces with the sims from EVERY model stage.

For each stimulus family and each of --numOverlay sweeps of the chosen domain,
one panel: recorded trace (black) + one colored line per stage (CNN -> tanh ->
param_subset expand -> jaxley re-sim under the family's rig stimulus).

  ./plot_stage_overlays.py --dom test  --models zero-shot=./out_vo ft2000=./out_ft
  (models given as label=modelPath, evaluated left to right; missing dirs skipped)

Outputs in --outDir (default ./stage_overlays):
  stage_overlays_<dom>.pdf   one page per family
  stage_mse_<dom>.csv        family x stage: mse_fixed mean + spike counts
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse, importlib, json
import numpy as np, h5py, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evaluate_voltage import load_trained_model
from toolbox import JaxleyBridge
from toolbox.jaxley_utils import (phys_par_range_to_arrays,
                                  VOLT_NORM_MEAN, VOLT_NORM_STD)

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
AMPS = [100, 500, 1000, 1500, 2000]
SIM_BATCH = 32


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="label=modelPath ...")
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--dom", default="test", choices=["train", "valid", "test"])
    p.add_argument("--outDir", default="stage_overlays")
    p.add_argument("--cellName", default="ca3_pyramidal")
    p.add_argument("--numOverlay", type=int, default=4)
    p.add_argument("--paramSubset", type=int, nargs="*", default=[0, 5])
    return p.parse_args()


def zfix(v):
    return (v - VOLT_NORM_MEAN) / VOLT_NORM_STD


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def sim_family(pred_phys, cell_name, stim_name):
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
    os.makedirs(args.outDir, exist_ok=True)

    models = []
    for spec in args.models:
        label, path = spec.split("=", 1)
        if os.path.isfile(os.path.join(path, "sum_train.yaml")):
            m, _ = load_trained_model(path, device)
            models.append((label, m))
            print(f"[stages] loaded {label} <- {path}")
        else:
            print(f"[stages] SKIP {label}: no sum_train.yaml in {path}")
    if not models:
        raise SystemExit("no models loaded")

    with h5py.File(args.packFile, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        X = f[args.dom + "_volts_norm"][:, :, 0, 0].astype(np.float32)
        raw = f[args.dom + "_raw_volts_mV"][:].astype(np.float32)
        famS = f[args.dom + "_stim_family"][:].astype(str)
        nids = f[args.dom + "_neuron_id"][:].astype(str)

    stim_stems = meta["stim_from_label"]["stim_names"]
    fam_order = meta["stim_from_label"]["stim_family_order"]
    centers, logspans = phys_par_range_to_arrays(meta["input_meta"]["phys_par_range"])
    centers_t = torch.tensor(centers, dtype=torch.float64, device=device)
    logspans_t = torch.tensor(logspans, dtype=torch.float64, device=device)

    mod = importlib.import_module(f"toolbox.jaxley_cells.{args.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()

    colors = ["C3", "C0", "C2", "C4", "C5"]
    rows = []
    with PdfPages(os.path.join(args.outDir, f"stage_overlays_{args.dom}.pdf")) as pdf:
        for amp in AMPS:
            fam = f"Roy{amp}"
            m = famS == fam
            if not m.any():
                continue
            data_raw = raw[m]
            T = data_raw.shape[1]
            stim_name = stim_stems[fam_order.index(fam)]
            sims, stats = {}, {}
            xt = torch.from_numpy(X[m][:, :, None]).contiguous().to(device)
            for label, model in models:
                with torch.no_grad():
                    pu = torch.tanh(model(xt).double())
                if args.paramSubset:
                    full = torch.zeros(pu.shape[0], len(centers), dtype=pu.dtype, device=device)
                    full[:, args.paramSubset] = pu
                    pu = full
                phys = centers_t * torch.pow(
                    torch.tensor(10.0, dtype=torch.float64, device=device), pu * logspans_t)
                v = sim_family(phys, args.cellName, stim_name)[:, -T:]
                sims[label] = v
                mse = float(((zfix(v) - zfix(data_raw)) ** 2).mean())
                stats[label] = (mse, float(spikes_pos(v).mean()))
                rows.append((args.dom, fam, label, mse,
                             stats[label][1], float(spikes_pos(data_raw).mean())))
                print(f"[stages] {args.dom} {fam} {label}: MSE_fixed={mse:.3f} "
                      f"spikes sim/data={stats[label][1]:.1f}/{spikes_pos(data_raw).mean():.1f}")

            nshow = min(args.numOverlay, int(m.sum()))
            t_axis = np.arange(T) * 0.1
            fig, axes = plt.subplots(nshow, 1, figsize=(12, 2.6 * nshow),
                                     sharex=True, squeeze=False)
            for i in range(nshow):
                ax = axes[i][0]
                ax.plot(t_axis, data_raw[i], "k", lw=0.9, label="experimental")
                for (label, _), c in zip(models, colors):
                    ax.plot(t_axis, sims[label][i], c, lw=0.8, alpha=0.75, label=label)
                ax.set_ylabel("V (mV)")
                ax.set_title(f"{fam}  {nids[np.flatnonzero(m)[i]]}", fontsize=9)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=7, ncol=len(models) + 1)
            axes[-1][0].set_xlabel("time (ms)")
            hdr = "   ".join(f"{l}: MSE {stats[l][0]:.2f}, spk {stats[l][1]:.1f}"
                             for l, _ in models)
            fig.suptitle(f"{fam} ({args.dom})   data spk "
                         f"{spikes_pos(data_raw).mean():.1f}   |   {hdr}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)

    with open(os.path.join(args.outDir, f"stage_mse_{args.dom}.csv"), "w") as fo:
        fo.write("dom,family,stage,mse_fixed_mean,spikes_sim,spikes_data\n")
        for r in rows:
            fo.write(",".join(str(x) for x in r) + "\n")
    print(f"[stages] wrote {args.outDir}/stage_overlays_{args.dom}.pdf and stage_mse_{args.dom}.csv")


if __name__ == "__main__":
    main()
