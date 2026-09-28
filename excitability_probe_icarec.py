#!/usr/bin/env python
"""Can the CA3 model fire at experimental rates under the TRUE recorded drive?

Finding 12 follow-up: every model (any loss, any training stim, exact recorded
stimulus) tops out at ~2 spikes vs 8-15 in the data, with predictions saturating
the "maximally excitable" corner of the unit box.  This probe asks whether ANY
parameter setting -- inside or far OUTSIDE the trained box -- reproduces the
data's firing rate under Roy1000/2000_icaRec_5k:

  variants per family:
    pred        mean icarec-ft prediction (tanh'd) on that family's sweeps
    pred x f    the prediction scaled f=1.5/2/3 past the corner (unbounded)
    corner x s  pure excitable template [-1,+1,-1,-1,-1,-1] * s, s=1/1.5/2/3
    na3 = v     prediction with only unit-na3 forced to 1.5/2/3

  unit -> phys: center * 10^(u * logspan); logspan 0.5 => u=2 is 10x center.

If nothing in this (generous) family fires ~10 spikes, the wall is the CELL
MODEL (missing channels / soma-only geometry), not the parameter box, and no
training-side lever can close it.  Outputs: printed table + overlays PDF +
probe_summary.csv in --outDir.  Requires JAX_ENABLE_X64=true (set below).
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
from toolbox.jaxley_utils import phys_par_range_to_arrays, phys_par_range_linear_mask, unit_to_phys_np

PACK_DEF = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
MODEL_DEF = ("/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_royexp_ft_icarec/"
             "RoyExpChaotic/ft_icarec/out")
AMPS = [1000, 2000]
CORNER = np.array([-1.0, +1.0, -1.0, -1.0, -1.0, -1.0])   # leak-,na3+,K- x4
SIM_BATCH = 16


def spikes_pos(v, thr=0.0):
    return ((v[:, 1:] > thr) & (v[:, :-1] <= thr)).sum(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--modelPath", default=MODEL_DEF)
    p.add_argument("--packFile", default=PACK_DEF)
    p.add_argument("--outDir", default="excitability_probe")
    p.add_argument("--cellName", default="ca3_pyramidal")
    a = p.parse_args()
    os.makedirs(a.outDir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainMD = load_trained_model(a.modelPath, device)

    with h5py.File(a.packFile, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
        X = f["test_volts_norm"][:, :, 0, 0].astype(np.float32)
        raw = f["test_raw_volts_mV"][:].astype(np.float32)
        famS = f["test_stim_family"][:].astype(str)
    par_names = meta["input_meta"]["parName"]
    centers, logspans = phys_par_range_to_arrays(meta["input_meta"]["phys_par_range"])
    lin_mask = phys_par_range_linear_mask(meta["input_meta"]["phys_par_range"])

    mod = importlib.import_module(f"toolbox.jaxley_cells.{a.cellName}")
    mod._T_MAX = 500.0
    JaxleyBridge.clear_cache()

    import jax.numpy as jnp
    rows = []
    for amp in AMPS:
        fam = f"Roy{amp}"
        m = famS == fam
        data_raw = raw[m]
        sp_data = spikes_pos(data_raw).mean()
        with torch.no_grad():
            xt = torch.from_numpy(X[m][:, :, None]).contiguous().to(device)
            u_pred = torch.tanh(model(xt).double()).cpu().numpy().mean(axis=0)

        variants = [("pred", u_pred)]
        variants += [(f"pred x{f}", u_pred * f) for f in (1.5, 2.0, 3.0)]
        variants += [(f"corner x{s}", CORNER * s) for s in (1.0, 1.5, 2.0, 3.0)]
        for v in (1.5, 2.0, 3.0):
            u = u_pred.copy(); u[1] = v
            variants.append((f"na3={v}", u))

        U = np.stack([u for _, u in variants])                     # (V, 6)
        phys = unit_to_phys_np(U, centers, logspans, linear=lin_mask)
        pp = jnp.asarray(phys)
        if pp.shape[0] < SIM_BATCH:
            pp = jnp.concatenate(
                [pp, jnp.broadcast_to(pp[:1], (SIM_BATCH - pp.shape[0], pp.shape[1]))], 0)
        handle = JaxleyBridge.get_handle(a.cellName, f"Roy{amp}_icaRec_5k")
        v_sim = np.asarray(handle.simulate_batch(pp))[:len(variants), 0, :]
        v_al = v_sim[:, -data_raw.shape[1]:]

        print(f"\n=== Roy{amp}_icaRec (data spikes mean {sp_data:.1f}) ===")
        print(f"{'variant':>12} {'spikes':>7} {'Vmin':>8} {'Vmax':>8}  unit params")
        sp = spikes_pos(np.nan_to_num(v_al, nan=0.0))
        for i, (name, u) in enumerate(variants):
            finite = np.isfinite(v_al[i]).all()
            print(f"{name:>12} {sp[i]:>7d} {np.nanmin(v_al[i]):>8.1f} "
                  f"{np.nanmax(v_al[i]):>8.1f}  "
                  f"[{', '.join(f'{x:+.2f}' for x in u)}]"
                  + ("" if finite else "  NON-FINITE"))
            rows.append((amp, name, int(sp[i]), float(sp_data),
                         float(np.nanmin(v_al[i])), float(np.nanmax(v_al[i])),
                         bool(finite)))

        with PdfPages(os.path.join(a.outDir, f"Roy{amp}_probe_overlays.pdf")) as pdf:
            t = np.arange(data_raw.shape[1]) * 0.1
            for i, (name, _) in enumerate(variants):
                fig, ax = plt.subplots(figsize=(11, 3.0))
                ax.plot(t, data_raw[0], "k", lw=0.7, label="experimental (sweep 0)")
                ax.plot(t, v_al[i], "C3", lw=0.7, alpha=0.85, label=f"sim {name}")
                ax.set_title(f"Roy{amp}_icaRec  {name}: {sp[i]} spikes "
                             f"(data mean {sp_data:.1f})", fontsize=9)
                ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
                ax.legend(loc="upper right", fontsize=7)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with open(os.path.join(a.outDir, "probe_summary.csv"), "w") as fo:
        fo.write("amp,variant,spikes_sim,spikes_data,vmin,vmax,finite\n")
        for r in rows:
            fo.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[probe] wrote {a.outDir}/probe_summary.csv + overlays; "
          f"par order = {par_names}")


if __name__ == "__main__":
    main()
