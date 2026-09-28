#!/usr/bin/env python
"""Collect model-ladder training results (out/eval/summary.yaml of every run) into one
markdown table: rung x stim x mode -> mean channel R2, per-parameter R2, voltage MSE_z,
spike counts.  Usage: python scripts/collect_ladder_results.py [run_root] [out.md]"""
import glob, os, sys, yaml
import re
# run roots: the HOME tree (ladder runs made while pscratch was over quota) and the SCRATCH tree
# (the fp32/dt0.2 pilot + its supervised twin + the 200k run); comma-separated override in argv[1].
roots = (sys.argv[1].split(",") if len(sys.argv) > 1 else
         ["/global/homes/k/ktub1999/tmp_neuInv/model_ladder", "/pscratch/sd/k/ktub1999/tmp_neuInv/model_ladder"])
out = sys.argv[2] if len(sys.argv) > 2 else "docs/model_ladder/results.md"
STIM = {"icb": "InterChaoticB", "cr": "chaoticRamp", "icb4k": "InterChaoticB 400ms dt0.2 (run.py box)"}
DESIGN = re.compile(r"^ladder_(?P<rung>.+?)_(?P<stag>icb4k|icb|cr)_(?P<mode>sup|vo)(?P<extra>.*)$")
rows = []
for f in sorted(sum((glob.glob(f"{r}/ladder_*/*/*/out/eval/summary.yaml") for r in roots), [])):
    d = yaml.safe_load(open(f)); design = f.split("/")[-6]; suffix = f.split("/")[-4]
    m = DESIGN.match(design)
    if not m:
        print(f"skip (unparsed design name): {design}"); continue
    rung, stag, mode = m.group("rung"), m.group("stag"), m.group("mode")
    if "smoke" in suffix:                       # 2-epoch pipeline smokes are not results
        continue
    stim = STIM[stag]
    per = d.get("channel_per_param", [])
    per_s = ", ".join(f"{p['name'].replace('gbar_','').replace('_somatic','_som').replace('_axonal','_ax').replace('_apical','_api')[:22]} {p['r2']:.2f}" for p in per)
    lg = os.path.join(os.path.dirname(os.path.dirname(f)), "..", "log.train")
    ep = ""
    try:
        import re
        txt = open(os.path.join(os.path.dirname(f), "..", "..", "log.train")).read()
        m = re.findall(r"Epoch (\d+) took", txt); ep = m[-1] if m else ""
    except Exception: pass
    rows.append((rung, stim, mode, d.get("channel_r2_overall", float("nan")), d.get("voltage_mse_z_mean", float("nan")),
                 d.get("voltage_mse_z_median", float("nan")), d.get("spikes_sim_mean", float("nan")), d.get("spikes_data_mean", float("nan")),
                 d.get("n_samples", ""), ep, per_s, suffix))
lines = ["| rung | stim | mode | mean R2 | MSE_z mean / median | spikes sim / data | n | epochs | per-parameter R2 | run |",
         "|" + "---|" * 10]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | **{r[3]:.3f}** | {r[4]:.2f} / {r[5]:.2f} | {r[6]:.1f} / {r[7]:.1f} | {r[8]} | {r[9]} | {r[10]} | {r[11]} |")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print(f"\nwrote {out} ({len(rows)} runs)")
