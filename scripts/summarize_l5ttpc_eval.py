#!/usr/bin/env python
"""Summarise the L5TTPC ncomp=2 test-split evaluations in l5ttpc_eval/<run>/.

Prints a markdown table (mean R2, per-channel R2 for the channels that matter,
voltage MSE_z, spikes) and draws ONE composite figure: for every run, the
predicted-vs-true soma trace of the median-error test sample (z-scored), so
all nine runs can be compared on one page.
"""
import csv, glob, os, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "l5ttpc_eval"
ORDER = ["supervised_nc2", "voltage_only_nc2", "hybrid_nc2", "multiprobe_nc2",
         "multistim_nc2", "multistim_efel", "hybrid_efel", "paramonly_3stim",
         "paramonly_ft80k"]
KEY = {  # short label -> param name in channel_recovery.csv
    "Na_soma": "gNaTs2_tbar_NaTs2_t_somatic", "SKv3_soma": "gSKv3_1bar_SKv3_1_somatic",
    "e_pas": "e_pas_all", "Na_axon": "gNaTa_tbar_NaTa_t_axonal", "Ih_dend": "gIhbar_Ih_dend",
    "cm_soma": "cm_somatic", "Na_apic": "gNaTs2_tbar_NaTs2_t_apical",
    "SKv3_apic": "gSKv3_1bar_SKv3_1_apical", "Im_apic": "gImbar_Im_apical",
    "CaLVA_soma": "gCa_LVAstbar_Ca_LVAst_somatic", "K_Tst_axon": "gK_Tstbar_K_Tst_axonal",
}

rows, traces = [], {}
for name in ORDER:
    d = os.path.join(ROOT, name)
    sy = os.path.join(d, "summary.yaml")
    if not os.path.isfile(sy):
        rows.append((name, None)); continue
    S = yaml.safe_load(open(sy))
    r2 = {}
    with open(os.path.join(d, "channel_recovery.csv")) as fh:
        for rec in csv.DictReader(fh):
            r2[rec["param"]] = float(rec["r2"])
    rows.append((name, S, r2))
    tz = os.path.join(d, "traces.npz")
    if os.path.isfile(tz):
        traces[name] = np.load(tz, allow_pickle=True)

# ── table ──────────────────────────────────────────────────────────────
hdr = ["run", "mean R²"] + list(KEY) + ["mse_z", "spikes sim/data"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for row in rows:
    name = row[0]
    if row[1] is None:
        print(f"| {name} | (not scored) |" + " |" * (len(hdr) - 2)); continue
    S, r2 = row[1], row[2]
    cells = [name, f"{S['channel_r2_overall']:.3f}"]
    cells += [f"{r2.get(KEY[k], float('nan')):+.2f}" for k in KEY]
    cells += [f"{S['voltage_mse_z_mean']:.2f}",
              f"{S['spikes_sim_mean']:.1f}/{S['spikes_data_mean']:.1f}"]
    print("| " + " | ".join(cells) + " |")

# ── composite overlay: median-error soma trace per run ─────────────────
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
names = [n for n in ORDER if n in traces]
if names:
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.1 * len(names) + 1.2),
                             sharex=True, facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        Z = traces[name]
        err = Z["err_z"][0]; i = int(np.argsort(err)[len(err) // 2])   # median sample
        t = Z["t_axis"]
        ax.set_facecolor("#fcfcfb")
        ax.plot(t, Z["v_data_z"][0, i], color=INK, lw=1.2, label="true (test data)")
        ax.plot(t, Z["v_sim_z"][0, i], color=ORANGE, lw=1.2, alpha=0.9, label="predicted params, re-simulated")
        ax.text(0.995, 0.95, f"{name}   median sample #{i}   mse_z {err[i]:.2f}   "
                f"spikes pred/true {int(Z['spikes_sim'][0, i])}/{int(Z['spikes_data'][0, i])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec=GRID, lw=0.8))
        ax.set_ylabel("z-scored V", fontsize=8.5, color=INK2)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[-1].set_xlabel("time (ms)", fontsize=9, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=9.5, ncol=2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("L5TTPC ncomp=2 runs: predicted-vs-true soma voltage on the held-out test split\n"
                 "(median-error sample of 200 per run, z-scored; CNN prediction re-simulated in jaxley at ncomp=2)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(ROOT, "l5ttpc_test_overlays_composite")
    fig.savefig(out + ".png", dpi=160); fig.savefig(out + ".pdf")
    print("wrote", out + ".png/.pdf")
