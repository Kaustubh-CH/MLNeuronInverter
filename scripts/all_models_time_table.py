#!/usr/bin/env python
"""Every trained model of record: samples/epoch, GPUs, measured s/epoch (steady
state from tb_logs), epochs run, measured train hours, and the projected cost
of 100 epochs at that rate.  Also GPU-seconds per sample so any other sample
count can be estimated.  Scans the jaxley run trees; skips dirs without
sum_train.yaml + tb_logs.
"""
import csv, glob, os, re, sys
import numpy as np, yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOTS = [
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/*/*/*/out",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/*/out",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/roy2k_*/out_*",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/*/*/out",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_voltage_only/*/*/*/out",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/*/out",
]

def tb_epoch_times(out):
    for sub in glob.glob(f"{out}/tb_logs/*"):
        b = os.path.basename(sub).strip()
        if b.startswith("epoch time") and b.endswith("_tot"):
            vals = {}
            for ev in sorted(glob.glob(f"{sub}/events.*")):
                try:
                    ea = EventAccumulator(ev); ea.Reload()
                    for t in ea.Tags().get("scalars", []):
                        for e in ea.Scalars(t): vals[e.step] = e.value
                except Exception: pass
            if vals:
                v = np.array([vals[k] for k in sorted(vals)])
                return len(v), float(np.median(v[-10:])), float(v.sum())
    return None, None, None

def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d

seen, recs = set(), []
for pat in ROOTS:
    for out in sorted(glob.glob(pat)):
        out = os.path.realpath(out)
        if out in seen or not os.path.isfile(f"{out}/sum_train.yaml") or not os.path.isdir(f"{out}/tb_logs"):
            continue
        seen.add(out)
        try:
            st = yaml.safe_load(open(f"{out}/sum_train.yaml"))
        except Exception:
            continue
        tp = st.get("train_params", {}); vl = tp.get("voltage_loss") or {}
        n_ep, sec, tot = tb_epoch_times(out)
        if not n_ep or n_ep < 2: continue
        samples = tp.get("train_glob_sampl") or g(tp, "data_conf", "max_glob_samples_per_epoch") or 0
        # numGlobSamp is a CAP; the loader uses min(cap, rows in the pack) -> read the pack
        try:
            import h5py
            with h5py.File(tp["full_h5name"], "r") as f:
                n_h5 = int(f["train_unit_par"].shape[0])
            samples = min(samples, n_h5) if samples else n_h5
        except Exception:
            pass
        gpus = int(tp.get("world_size") or 1)
        r2 = None
        cellnm = str(tp.get("cell_name") or "")
        if os.path.isfile(f"{out}/eval/summary.yaml") and not cellnm.startswith("RoyExp"):
            try: r2 = yaml.safe_load(open(f"{out}/eval/summary.yaml")).get("channel_r2_overall")
            except Exception: pass
        loss = "param-MSE" if not tp.get("use_voltage_loss") else (
            ("DTW" if vl.get("dtw_weight") else "MSE") + ("+eFEL" if vl.get("efel_weight") else "")
            + ("+ch" if (vl.get("channel_weight") or 0) > 0 else ""))
        rel = out.replace("/pscratch/sd/k/ktub1999/tmp_neuInv/", "")
        recs.append(dict(
            model=rel, cell=tp.get("cell_name") or tp.get("myId"), stim=(vl.get("stim_names_multi") or vl.get("pooled_stim_names")
                  or vl.get("stim_name") or g(vl, "stim_from_label", "stim_names") or ""),
            loss=loss, samples=int(samples), gpus=gpus, n_ep=n_ep, sec=sec, tot_h=tot / 3600,
            est100=sec * 100 / 3600, gpus_per_sample=(sec * gpus / samples if samples else float("nan")),
            r2=r2, design=tp.get("design") or tp.get("myId"),
        ))

MIN_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
recs = [r for r in recs if r["n_ep"] >= MIN_EP]
recs.sort(key=lambda r: (r["model"]))
print("| model (under tmp_neuInv/) | cell | loss | samples/epoch | GPUs | s/epoch | epochs run | measured h | est. h / 100 ep | GPU-s per sample | mean R² |")
print("|" + "---|" * 11)
for r in recs:
    r2s = "" if r["r2"] is None else f"{r['r2']:.3f}"
    print(f"| {r['model']} | {r['cell']} | {r['loss']} | {r['samples']:,} | {r['gpus']} | {r['sec']:.0f} | {r['n_ep']} | "
          f"{r['tot_h']:.1f} | {r['est100']:.1f} | {r['gpus_per_sample']:.3f} | {r2s} |")
print(f"\n{len(recs)} runs")
