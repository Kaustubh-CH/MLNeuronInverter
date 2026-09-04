#!/usr/bin/env python
"""Train / validation loss per epoch for every L5TTPC (ncomp=2) run, read from
each run's TensorBoard event files.  Small multiples, one panel per run.
Each run has its own objective, so panels are NOT comparable on y -- the panel
title names the objective."""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

T = "/pscratch/sd/k/ktub1999/tmp_neuInv"
V = T + "/jaxley_voltage_only"
RUNS = [
    # (short name, objective line, samples/16 GPU, s per epoch, tb_logs dir)
    ("supervised nc2", "param-MSE, no simulator, 200 ep", "15.9k, 0.75 s/ep",
     T + "/jaxley_ca3/l5ttpc_supervised/L5TTPC_jaxley_nc2/super/out/tb_logs"),
    ("voltage-only nc2", "voltage MSE, 1 probe", "15.4k, 566 s/ep",
     V + "/l5ttpc_jaxley_ncomp2/L5TTPC_jaxley_nc2/55135593/out/tb_logs"),
    ("hybrid nc2", "channel + voltage MSE", "15.4k, 566 s/ep",
     V + "/l5ttpc_jaxley_hybrid/L5TTPC_jaxley_nc2/salloc_55144257/out/tb_logs"),
    ("multiprobe nc2", "voltage MSE, 4 probes", "7.2k, 268 s/ep",
     V + "/l5ttpc_multiprobe/L5TTPC_multiprobe/salloc_55163753/out/tb_logs"),
    ("multistim nc2", "voltage MSE, 3 stims joint", "7.7k, 1124 s/ep",
     V + "/l5ttpc_multistim/L5TTPC_multistim/salloc_55166043/out/tb_logs"),
    ("multistim + eFEL", "eFEL 1.0 + MSE 0.1, 3 stims (3 legs)", "15.4k, 2210 s/ep",
     V + "/l5ttpc_multistim_efel/L5TTPC_multistim/salloc_55370141/out/tb_logs"),
    ("hybrid + eFEL", "channel + voltage + eFEL, 1 stim", "15.4k, 752 s/ep",
     V + "/l5ttpc_jaxley_hybrid_efel/L5TTPC_multistim/reg15/out/tb_logs"),
    ("param-only, 3 stims", "param-MSE, 4 GPU", "4.0k, 0.55 s/ep",
     V + "/l5ttpc_multistim_paramonly/L5TTPC_multistim/debug_55374316/out/tb_logs"),
    ("param-only fine-tune 80k", "param-MSE, 1 GPU", "80k, 5.7 s/ep",
     V + "/l5ttpc_multistim_paramonly/L5TTPC_multistim/finetune_55375352/out/tb_logs"),
]

def read_curve(tbdir, tag_sub):
    """Merge every event file under tb_logs/<dir containing tag_sub>/ by step."""
    pts = {}
    for sub in sorted(glob.glob(os.path.join(tbdir, "*"))):
        b = os.path.basename(sub).strip()
        # subdirs are ' loss _train', ' loss _val', 'epoch time (sec) _train',
        # 'glob_speed (k samp:sec) _train', ... -> keep ONLY the loss ones
        if not (b.startswith("loss") and b.endswith(tag_sub)):
            continue
        for ev in sorted(glob.glob(os.path.join(sub, "events.*"))):
            try:
                ea = EventAccumulator(ev); ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    for e in ea.Scalars(tag):
                        pts[e.step] = e.value          # later legs override
            except Exception as ex:
                print("WARN", ev, ex, file=sys.stderr)
    if not pts:
        return np.array([]), np.array([])
    s = np.array(sorted(pts)); return s, np.array([pts[k] for k in s])

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), facecolor="#fcfcfb")
for ax, (name, obj, cost, tbdir) in zip(axes.ravel(), RUNS):
    st, tr = read_curve(tbdir, "_train")
    sv, va = read_curve(tbdir, "_val")
    ax.set_facecolor("#fcfcfb")
    ax.plot(st, tr, color=BLUE, lw=2, label="train")
    ax.plot(sv, va, color=ORANGE, lw=2, label="validation")
    if len(sv):
        ib = int(np.argmin(va))
        ax.plot(sv[ib], va[ib], "o", ms=8, mfc=ORANGE, mec="#fcfcfb", mew=2)
        ax.text(0.98, 0.97, f"best val {va[ib]:.3f} @ ep {sv[ib]}   |   last val {va[-1]:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec=GRID, lw=0.8))
        ax.text(sv[-1], va[-1], "  val", color=INK2, fontsize=8.5, va="center")
    if len(st):
        ax.text(st[-1], tr[-1], "  train", color=INK2, fontsize=8.5, va="center")
        ax.set_xlim(0, st[-1] * 1.18)
    ax.set_title(f"{name}\n{obj}   |   {cost}", fontsize=9.5, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel("epoch", fontsize=8.5, color=INK2)
    ax.set_ylabel("loss (per-run objective)", fontsize=8.5, color=INK2)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", frameon=False, fontsize=9.5, ncol=2, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("L5TTPC ncomp=2 runs (2026-06-27 to 07-08): train vs validation loss per epoch\n"
             "Each panel is its own objective; y-scales are not comparable across panels. "
             "Samples per epoch and seconds per epoch (16 GPUs unless noted) in each title.",
             fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = sys.argv[1] if len(sys.argv) > 1 else "l5ttpc_train_val_curves"
fig.savefig(out + ".png", dpi=170); fig.savefig(out + ".pdf")
print("wrote", out + ".png", out + ".pdf")
