#!/usr/bin/env python3
"""Compare ALL_CELLS_Inhibitory error (same-cell / intrapolation / extrapolation)
between the half-data run (55038357) and the full-data run (39334328).

Error = unit-space prediction MSE (matches sum_pred testLossMSE), computed directly
from each MLoutput.h5 (ground_truth_upar vs predict_upar)."""
import h5py, yaml, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS_Inhibitory"
RUNS = [("39334328", "full data", "#1f77b4"),
        ("55038357", "half data", "#ff7f0e")]
DOMAINS = [  # label, subdir, sum_pred yaml
    ("same-cell",     "predictionResutlsH5",       "sum_pred_nif.yaml"),
    ("intrapolation", "predictionlOntraNewClone",  "sum_pred_nifALL_CELLS_Inhibitory_Intrapolated.yaml"),
    ("extrapolation", "predictionlOntraNewCell",   "sum_pred_nifALL_CELLS_Inhibitory_Extrapolation.yaml"),
]

def load(job, sub):
    with h5py.File(f"{BASE}/{job}/{sub}/MLoutput.h5", "r") as h:
        return h["ground_truth_upar"][:], h["predict_upar"][:]

# param names (17) from one sum_pred yaml
y0 = yaml.safe_load(open(f"{BASE}/39334328/out/sum_pred_nif.yaml"))
pnames = [r[0] for r in y0["residual_mean_std"]]

# collect metrics
mse   = {}     # (job, dom) -> overall MSE
permae = {}    # (job, dom) -> per-param MAE (17,)
for job, _, _ in RUNS:
    for dlabel, sub, _ in DOMAINS:
        gt, pr = load(job, sub)
        res = pr - gt
        mse[(job, dlabel)] = float(np.mean(res**2))
        permae[(job, dlabel)] = np.mean(np.abs(res), axis=0)

# ---------- figure ----------
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.32)

# Panel A: grouped bars of overall MSE
axA = fig.add_subplot(gs[0])
dlabels = [d[0] for d in DOMAINS]
x = np.arange(len(dlabels)); w = 0.36
for i, (job, jlabel, col) in enumerate(RUNS):
    vals = [mse[(job, d)] for d in dlabels]
    bars = axA.bar(x + (i - 0.5) * w, vals, w, label=f"{jlabel} ({job})", color=col)
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)
# annotate % change half vs full
for j, d in enumerate(dlabels):
    full, half = mse[("39334328", d)], mse[("55038357", d)]
    pct = 100 * (half - full) / full
    axA.text(x[j], max(full, half) + 0.014, f"{pct:+.1f}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             color="green" if pct < 0 else "firebrick")
axA.set_xticks(x); axA.set_xticklabels(dlabels, fontsize=12)
axA.set_ylabel("unit-space prediction MSE", fontsize=12)
axA.set_title("ALL_CELLS_Inhibitory: prediction error vs training-data size\n"
              "(half = each cell's contribution halved)", fontsize=13)
axA.legend(fontsize=11); axA.grid(axis="y", alpha=0.3)
axA.set_ylim(0, max(mse.values()) * 1.22)

# Panel B: per-parameter MAE, one cluster per domain (full vs half)
axB = fig.add_subplot(gs[1])
xp = np.arange(len(pnames)); ww = 0.13
offs = 0
styles = {"same-cell": ("o", "-"), "intrapolation": ("s", "--"), "extrapolation": ("^", ":")}
colors = {"39334328": "#1f77b4", "55038357": "#ff7f0e"}
for dlabel, sub, _ in DOMAINS:
    for job, jlabel, _ in RUNS:
        m = permae[(job, dlabel)]
        axB.plot(xp, m, marker=styles[dlabel][0], linestyle=styles[dlabel][1],
                 color=colors[job], alpha=0.85, markersize=5,
                 label=f"{dlabel} – {jlabel}")
axB.set_xticks(xp)
axB.set_xticklabels(pnames, rotation=90, fontsize=8)
axB.set_ylabel("per-parameter MAE (unit space)", fontsize=12)
axB.set_title("Per-parameter mean absolute error", fontsize=12)
axB.legend(fontsize=8, ncol=3, loc="upper left")
axB.grid(axis="y", alpha=0.3)

out = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/inh_halved_vs_full_error.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

# also dump the table
print("\n%-15s %12s %12s %8s" % ("domain", "full(39334328)", "half(55038357)", "Δ%"))
for d in dlabels:
    full, half = mse[("39334328", d)], mse[("55038357", d)]
    print("%-15s %12.4f %12.4f %+8.1f" % (d, full, half, 100*(half-full)/full))
