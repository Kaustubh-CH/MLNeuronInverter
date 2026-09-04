#!/usr/bin/env python
"""Table of every chaoticRamp CA3 run of record: recipe knobs, samples, epochs,
seconds/epoch (steady state, from tb_logs), total train time, nodes/GPUs (sacct),
mean R2 + per-channel R2 (from the run's eval/summary.yaml).

Sources: vo_ledger/results.csv (summary_path column) + the three pre-DTW
chaoticRamp runs (supervised / voltage-MSE / voltage-eFEL).
"""
import csv, glob, os, re, subprocess, sys
import numpy as np, yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

J = "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3"
rows = []
with open(f"{J}/vo_ledger/results.csv") as fh:
    for r in csv.DictReader(fh):
        rows.append((r["design"], os.path.dirname(os.path.dirname(r["summary_path"]))))
for d in ["ca3_supervised_chaoticramp", "ca3_voltonly_mse_chaoticramp", "ca3_voltonly_efel_chaoticramp"]:
    for o in sorted(glob.glob(f"{J}/{d}/*/*/out")):
        if os.path.isfile(f"{o}/eval/summary.yaml"):
            rows.append((d.replace("ca3_", ""), o))

def tb_epoch_times(out):
    for sub in glob.glob(f"{out}/tb_logs/*"):
        if os.path.basename(sub).strip().startswith("epoch time") and sub.rstrip("/").endswith("_tot"):
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

recs = []
for name, out in rows:
    st = yaml.safe_load(open(f"{out}/sum_train.yaml"))
    tp = st["train_params"]; vl = tp.get("voltage_loss") or {}
    dc = tp.get("data_conf", {}); tc = tp.get("train_conf", {})
    jid = str(tp.get("job_id", "")); m = re.search(r"(\d{7,9})", jid); jnum = m.group(1) if m else ""
    n_ep, sec_ep, tot = tb_epoch_times(out)
    S = yaml.safe_load(open(f"{out}/eval/summary.yaml"))
    r2 = {}
    with open(f"{out}/eval/channel_recovery.csv") as fh:
        for rec in csv.DictReader(fh): r2[rec["param"]] = float(rec["r2"])
    ef = vl.get("efel_weight", 0); ef = ef if isinstance(ef, (int, float)) else f"{ef.get('start')}->{ef.get('end')}/{ef.get('epochs')}ep"
    recs.append(dict(
        name=name, out=out, jnum=jnum, design=tp.get("design") or tp.get("myId"),
        samples=g(tp, "train_glob_sampl") or g(tp, "data_conf", "max_glob_samples_per_epoch") or "?",
        epochs=st.get("epoch_stop"), n_ep=n_ep, sec_ep=sec_ep, tot_h=(tot / 3600 if tot else None),
        gbs=g(tc, "batch_size"), world=tp.get("world_size"), lr=g(tc, "optimizer", "lr") or g(tp, "iniLR"),
        use_v=tp.get("use_voltage_loss"),
        dtw=vl.get("dtw_weight", 0), gamma=vl.get("dtw_gamma"), band=vl.get("dtw_band_ms"), npts=vl.get("dtw_n_points"),
        mse=vl.get("mse_weight"), efel=ef, feats=vl.get("efel_features"),
        precond=g(vl, "grad_precond") is not None and g(vl, "grad_precond"),
        blur=vl.get("blur_weight"), tanh=vl.get("clamp_unit_tanh"), fp64=vl.get("fp64"),
        r2m=S["channel_r2_overall"], msez=S["voltage_mse_z_mean"], sp=f"{S['spikes_sim_mean']:.1f}/{S['spikes_data_mean']:.1f}",
        r2=r2, stims=vl.get("stim_names_multi") or vl.get("pooled_stim_names") or vl.get("stim_name"),
    ))

# sacct for wall/nodes
jids = ",".join(sorted({r["jnum"] for r in recs if r["jnum"]}))
sac = {}
if jids:
    txt = subprocess.run(["sacct", "-j", jids, "-X", "-P", "-n", "-o", "JobID,Elapsed,NNodes,Start,State,JobName"],
                         capture_output=True, text=True).stdout
    for line in txt.splitlines():
        p = line.split("|")
        if len(p) >= 6: sac[p[0]] = p
for r in recs:
    p = sac.get(r["jnum"]); r["wall"] = p[1] if p else "?"; r["nodes"] = p[2] if p else "?"; r["start"] = p[3][:10] if p else "?"
    r["state"] = p[4] if p else "?"

recs.sort(key=lambda r: -r["r2m"])
K = ["CA3_g_leak", "CA3_gbar_na3", "CA3_gkdrbar_kdr", "CA3_gkabar_kap", "CA3_gbar_km", "CA3_gkdbar_kd"]
print("| run | date | job | nodes | samples | epochs | s/epoch | train h | wall | loss knobs | mean R² | leak | na3 | kdr | kap | km | kd | mse_z | spikes |")
print("|" + "---|" * 19)
for r in recs:
    knobs = []
    if not r["use_v"]: knobs.append("param-MSE (supervised)")
    else:
        if r["dtw"]: knobs.append(f"DTW {r['dtw']} g{r['gamma']} band{r['band']} n{r['npts']}")
        if r["mse"] not in (None, 0) and not r["dtw"]: knobs.append(f"MSE {r['mse']}")
        if r["efel"] not in (0, None, 0.0): knobs.append(f"eFEL {r['efel']} {r['feats'] or ''}")
        if r["precond"]: knobs.append(f"precond {r['precond']}")
        if r["blur"]: knobs.append(f"blur {r['blur']}")
        if r["tanh"]: knobs.append("tanh")
    print(f"| {r['name']} | {r['start']} | {r['jnum']} | {r['nodes']}n×{r['world']}gpu | {r['samples']} | {r['n_ep'] or r['epochs']} | "
          f"{r['sec_ep']:.0f} | {r['tot_h']:.1f} | {r['wall']} | {'; '.join(knobs)} | {r['r2m']:.3f} | "
          + " | ".join(f"{r['r2'].get(k, float('nan')):+.2f}" for k in K)
          + f" | {r['msez']:.2f} | {r['sp']} |")
print()
for r in recs: print(f"{r['name']:24s} lr={r['lr']} gbs={r['gbs']} stims={r['stims']} state={r['state']}  {r['out']}")
