#!/usr/bin/env python3
"""Why is the L5 nc2 model silent under the Roy currents, and what makes it fire like the cells?

For each VARIANT of the fixed (non-trainable) dendritic passive properties and/or the stimulus
scale c, simulate the BBP-default cell + nBox random draws from the training box under
Roy500/1000/1500/2000 `_icaRec_5k` (the rig-recorded currents) and a -0.1 nA step, and measure
exactly what the data give (scripts/rin_fit.py, same metric as the recordings):
  * apparent input resistance R (MOhm) + tau (ms) from the sub-threshold V~I fit,
  * steady-state R from the -0.1 nA step,
  * spikes per trace (upward -20 mV crossings, t >= 100 ms = the recorded window),
  * AP peak / min V / depolarisation-block flag.

Variant syntax (repeatable --variant): name:key=val,key=val with keys
  gpas   dendritic (basal+apical) g_pas, S/cm^2   (BBP 3e-5)
  cm     dendritic cm, uF/cm^2                    (BBP 2.0)
  scale  multiplier on the Roy / step current     (1.0 = real current)
The dendritic knobs map onto l5ttpc._DEND_GPAS / _DEND_CM (read at cell build).

  srun -n1 --gpus=1 python scripts/l5_excitability_probe.py --variant base: \
       --variant g1e5:gpas=1e-5 --variant c3:scale=3 -o <outdir>
"""
import os, sys, json, time, argparse
os.environ.setdefault("L5TTPC_NCOMP", "2")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np
import h5py

ROY_DIR = Path("/pscratch/sd/k/ktub1999/main/DL4neurons2/stims")
PILOT_H5 = "/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/l5ttpc_nc2_bbp_synth.mlPack1.h5"
AMPS = [500, 1000, 1500, 2000]


def parse_variant(s):
    name, _, kv = s.partition(":")
    v = {"name": name, "gpas": 3e-5, "cm": 2.0, "scale": 1.0}
    for tok in [t for t in kv.split(",") if t]:
        k, _, x = tok.partition("=")
        v[k.strip()] = float(x)
    return v


def make_step_dir(root):
    """-0.1 nA step 100..400 ms in a 500 ms / 0.1 ms CSV, next to symlinks of the Roy CSVs."""
    root.mkdir(parents=True, exist_ok=True)
    w = np.zeros(5000, np.float32); w[1000:4000] = -0.1
    np.savetxt(root / "step_m0p1_5k.csv", w, fmt="%.6f")   # one value per line (load_stim_csv)
    for a in AMPS:
        dst = root / f"Roy{a}_icaRec_5k.csv"
        if not dst.exists():
            dst.symlink_to(ROY_DIR / f"Roy{a}_icaRec_5k.csv")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--nBox", type=int, default=32)
    ap.add_argument("--h5", default=PILOT_H5, help="pack whose meta phys_par_range = the box")
    ap.add_argument("--simDt", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import importlib, jax.numpy as jnp
    from toolbox import JaxleyBridge
    from toolbox.jaxley_utils import (phys_par_range_to_arrays, phys_par_range_linear_mask,
                                      unit_to_phys_np, load_stim_csv)
    from rin_fit import fit_rin, spikes
    mod = importlib.import_module("toolbox.jaxley_cells.l5ttpc")
    mod._DT = args.simDt; mod._T_MAX = 500.0
    mod._STIM_DIR = make_step_dir(out / "stims")

    with h5py.File(args.h5, "r") as f:
        meta = json.loads(f["meta.JSON"][0])
    ppr = (meta.get("input_meta") or {}).get("phys_par_range") or meta["phys_par_range"]
    c, ls = phys_par_range_to_arrays(ppr); lin = phys_par_range_linear_mask(ppr)
    P = len(mod.PARAM_KEYS)
    U = np.random.default_rng(args.seed).uniform(-1, 1, (args.nBox, P))
    phys = np.concatenate([np.array([[mod._DEFAULTS[k] for k in mod.PARAM_KEYS]]),
                           unit_to_phys_np(U, c, ls, lin)], 0)          # row 0 = BBP default
    B = phys.shape[0]
    print(f"[probe] {B} cells (row 0 = BBP default), box from {args.h5}", flush=True)

    rows, traces = [], {}
    for vs in args.variant:
        v = parse_variant(vs)
        mod._DEND_GPAS, mod._DEND_CM = v["gpas"], v["cm"]
        JaxleyBridge.clear_cache()
        for stim in ["step_m0p1_5k"] + [f"Roy{a}_icaRec_5k" for a in AMPS]:
            t0 = time.time()
            h = JaxleyBridge.get_handle("l5ttpc", stim, None, "bwd_euler", v["scale"])
            V = np.asarray(h.simulate_batch(jnp.asarray(phys)))[:, 0, :].astype(np.float64)
            dt = float(h.out_dt)
            I = load_stim_csv(mod._STIM_DIR / f"{stim}.csv").astype(np.float64) * v["scale"]
            t = np.arange(V.shape[1]) * dt
            I_t = np.interp(t, np.arange(len(I)) * 0.1, I)
            w = t >= 99.8                                   # the recorded window
            traces[f"{v['name']}|{stim}"] = V[:, ::1].astype(np.float32)
            for b in range(B):
                vb = V[b]
                rec = dict(variant=v["name"], gpas=v["gpas"], cm=v["cm"], scale=v["scale"],
                           stim=stim, cell=b, default=int(b == 0),
                           vrest=float(vb[(t > 80) & (t < 99)].mean()),
                           vmax=float(vb[w].max()), vmin=float(vb[w].min()),
                           n_spk=int(len(spikes(vb[w]))),
                           block=int((vb[w] > -40).mean() > 0.2))
                if stim.startswith("step"):
                    rec["R_step"] = float((vb[(t > 350) & (t < 399)].mean() - rec["vrest"]) / -0.1)
                else:
                    fr = fit_rin(vb[w], I_t[w], dt)
                    rec.update(R_fit=fr["R"], tau_fit=fr["tau"], r2_fit=fr["r2"])
                rows.append(rec)
            d = [r for r in rows if r["variant"] == v["name"] and r["stim"] == stim]
            spk = np.array([r["n_spk"] for r in d])
            extra = (f"R_step def {d[0]['R_step']:.1f} MOhm box med {np.median([r['R_step'] for r in d[1:]]):.1f}"
                     if stim.startswith("step") else
                     f"R_fit def {d[0]['R_fit']:.1f} box med {np.nanmedian([r['R_fit'] for r in d[1:]]):.1f}")
            print(f"[probe] {v['name']:>10} {stim:<18} spikes def {spk[0]:2d} box mean {spk[1:].mean():5.2f} "
                  f"fire {np.mean(spk[1:] > 0):.2f} block {np.mean([r['block'] for r in d[1:]]):.2f} "
                  f"vmax def {d[0]['vmax']:+6.1f} | {extra}  ({time.time()-t0:.0f}s)", flush=True)

    import csv
    keys = sorted({k for r in rows for k in r})
    with open(out / "probe_rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    np.savez_compressed(out / "probe_traces.npz", **{k.replace("|", "__"): x for k, x in traces.items()},
                        phys=phys, unit=U, param_keys=np.array(mod.PARAM_KEYS))
    print(f"[probe] wrote {out}/probe_rows.csv, probe_traces.npz", flush=True)


if __name__ == "__main__":
    main()
