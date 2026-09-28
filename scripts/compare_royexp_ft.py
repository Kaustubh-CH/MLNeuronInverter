#!/usr/bin/env python3
"""Compare Roy-exp fine-tunes of the L5 nc2 model (2026-09-24: c = 3 vs x1) from the outputs of
plot_exp_overlay_royv2_l5dt02.py (<out>/exp_royv2_l5dt02/{roy_summary.csv, roy_traces.npz}).

Per model x Roy amplitude: spikes sim/data (mean), per-trace spike RMSE, envelope error
(mse_fixed, mse_z, dtw_z), and apparent R_in from scripts/rin_fit.py fitted against the UNSCALED
rig current (MOhm per rig-nA, the recordings' metric: 146 MOhm, IQR 88-218).  R_in medians keep
fits with r2 >= --minR2 only.

  python scripts/compare_royexp_ft.py c3=<run>/out x1_200k=<run>/out x1_pilot=<run>/out \\
         --outCsv docs/model_ladder/sensitivity/royexp_ft_c3/compare.csv
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rin_fit import fit_rin

AMPS = [500, 1000, 1500, 2000]


def rin_stats(V, I, dt, min_r2, ok=None):
    R = []
    for i in range(len(V)):
        if ok is not None and not ok[i]:
            continue
        fr = fit_rin(V[i], I, dt)
        if fr["r2"] >= min_r2:
            R.append(fr["R"])
    R = np.array(R)
    if not len(R):
        return np.nan, np.nan, np.nan, 0
    return float(np.median(R)), float(np.percentile(R, 25)), float(np.percentile(R, 75)), len(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=<run>/out  or  label=<overlay dir>")
    ap.add_argument("--sub", default="exp_royv2_l5dt02")
    ap.add_argument("--minR2", type=float, default=0.5)
    ap.add_argument("--outCsv", default=None)
    a = ap.parse_args()

    rows, data_done = [], set()
    for spec in a.runs:
        label, path = spec.split("=", 1)
        d = path if os.path.exists(os.path.join(path, "roy_summary.csv")) else os.path.join(path, a.sub)
        tr = np.load(os.path.join(d, "roy_traces.npz"), allow_pickle=True)
        dt = float(tr["out_dt"]); c = float(tr["stim_scale"])
        summ = {int(r["amp"]): r for r in csv.DictReader(open(os.path.join(d, "roy_summary.csv")))}
        for amp in AMPS:
            fam = f"Roy{amp}"
            if f"{fam}_sim" not in tr:
                continue
            Vd, Vs, ok, I = tr[f"{fam}_data"], tr[f"{fam}_sim"], tr[f"{fam}_sim_ok"], tr[f"{fam}_I_rig"]
            sp = lambda v: ((v[:, 1:] > 0) & (v[:, :-1] <= 0)).sum(axis=1)
            ss, sd = sp(Vs), sp(Vd)
            if fam not in data_done:                 # recordings are identical across runs
                r = rin_stats(Vd, I, dt, a.minR2)
                rows.append(dict(model="recordings", scale="", amp=amp, n=len(Vd), spikes_sim="",
                                 spikes_data=f"{sd.mean():.2f}", spike_rmse="", mse_fixed="",
                                 mse_z="", dtw_z="", R_med=f"{r[0]:.1f}", R_q25=f"{r[1]:.1f}",
                                 R_q75=f"{r[2]:.1f}", R_n=r[3]))
                data_done.add(fam)
            r = rin_stats(Vs, I, dt, a.minR2, ok)
            s = summ[amp]
            rows.append(dict(model=label, scale=c, amp=amp, n=int(ok.sum()),
                             spikes_sim=f"{ss[ok].mean():.2f}", spikes_data=f"{sd.mean():.2f}",
                             spike_rmse=f"{np.sqrt(((ss - sd)[ok] ** 2).mean()):.2f}",
                             mse_fixed=f"{float(s['mse_fixed_mean']):.3f}",
                             mse_z=f"{float(s['mse_z_mean']):.3f}", dtw_z=f"{float(s['dtw_z_mean']):.3f}",
                             R_med=f"{r[0]:.1f}", R_q25=f"{r[1]:.1f}", R_q75=f"{r[2]:.1f}", R_n=r[3]))

    keys = list(rows[0].keys())
    rows.sort(key=lambda r: (r["amp"], r["model"] != "recordings"))
    print("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    if a.outCsv:
        os.makedirs(os.path.dirname(a.outCsv) or ".", exist_ok=True)
        with open(a.outCsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {a.outCsv}")


if __name__ == "__main__":
    main()
