"""Grouped bar chart: sims/sec vs batch size, across every GPU config + CPU.

Lays out a 2x2 grid:

    L5TTPC x fwd        | L5TTPC x fwd_bwd
    --------------------|----------------------
    ball_and_stick x fwd| ball_and_stick x fwd_bwd

Within each panel: x = batch size, one bar per GPU config (+ CPU
single-thread reference for fwd panels). OOM bars are hatched and
labelled.

Output: $PSCRATCH/tmp_neuInv/jaxley_gpu_bench/bar_grouped.pdf

Usage:
    conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
    python -m toolbox.tests.plot_bench_bar_b128
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Crisp axes everywhere.
plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.linewidth":   1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "pdf.fonttype":     42,   # embed TrueType so text stays sharp.
    "ps.fonttype":      42,
})


GPU_BASE = Path("/pscratch/sd/k/ktub1999/tmp_neuInv")
CPU_CSV  = GPU_BASE / "jaxley_solver_bench" / "summary_timing.csv"

# (path, label, color)
GPU_CONFIGS = [
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp1" / "bench_gpu.csv",
        "GPU NCOMP=1",                "tab:green"),
    (GPU_BASE / "jaxley_gpu_bench_ncomp2" / "bench_gpu.csv",
        "GPU NCOMP=2",                "tab:orange"),
    (GPU_BASE / "jaxley_gpu_bench"        / "bench_gpu.csv",
        "GPU NCOMP=4 (baseline)",     "tab:blue"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp2_ckpt_10x100" / "bench_gpu.csv",
        "GPU NCOMP=2 + ckpt[10,100]", "tab:red"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp4_ckpt_10x100" / "bench_gpu.csv",
        "GPU NCOMP=4 + ckpt[10,100]", "tab:purple"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp4_ckpt_100x10" / "bench_gpu.csv",
        "GPU NCOMP=4 + ckpt[100,10]", "tab:brown"),
]
CPU_LABEL = "CPU bwd_euler (1 thread)"
CPU_COLOR = "0.35"

SOLVER  = "bwd_euler"
BATCHES = [1, 4, 16, 64, 128]


# ----------------------------- data loaders -----------------------------

def _gpu_value(csv_path: Path, cell: str, B: int, mode: str):
    if not csv_path.exists():
        return None, "missing"
    df = pd.read_csv(csv_path)
    sub = df[(df["cell"] == cell) & (df["solver"] == SOLVER)
             & (df["B"] == B) & (df["mode"] == mode)]
    if sub.empty:
        return None, "n/a"
    row  = sub.iloc[0]
    note = str(row.get("note", "") or "")
    sps  = pd.to_numeric(row["sims_per_sec"], errors="coerce")
    if pd.isna(sps):
        if "RESOURCE_EXHAUSTED" in note or "OOM" in note.upper():
            return None, "OOM"
        return None, (note[:12] or "n/a")
    return float(sps), ""


def _cpu_value(cell: str, B: int):
    if not CPU_CSV.exists():
        return None
    df = pd.read_csv(CPU_CSV)
    sub = df[(df["cell"] == cell) & (df["solver"] == SOLVER) & (df["B"] == B)]
    if sub.empty:
        return None
    sps = pd.to_numeric(sub.iloc[0]["sims_per_sec"], errors="coerce")
    return float(sps) if not pd.isna(sps) else None


# ----------------------------- plotting -----------------------------

def _plot_panel(ax, cell: str, mode: str, title: str, gpu_configs):
    n_series = len(gpu_configs) + 1   # + CPU
    bar_w    = 0.85 / n_series
    group_x  = np.arange(len(BATCHES), dtype=float)
    offsets  = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

    # CPU reference (only fwd; no fwd_bwd CPU bench was run).
    if mode == "fwd":
        cpu_vals = [_cpu_value(cell, B) for B in BATCHES]
    else:
        cpu_vals = [None] * len(BATCHES)
    cpu_y = [v if v is not None else 0.0 for v in cpu_vals]
    bars  = ax.bar(group_x + offsets[0], cpu_y, bar_w,
                   color=CPU_COLOR, edgecolor="black", lw=0.6)
    for i, v in enumerate(cpu_vals):
        if v is None:
            bars[i].set_alpha(0.15)

    # GPU configs.
    all_max = max(cpu_y) if cpu_y else 1.0
    for k, (path, _, color) in enumerate(gpu_configs, start=1):
        ys, notes = [], []
        for B in BATCHES:
            sps, note = _gpu_value(path, cell, B, mode)
            ys.append(sps if sps is not None else 0.0)
            notes.append(note)
        all_max = max(all_max, max(ys) if ys else 0.0)
        bars = ax.bar(group_x + offsets[k], ys, bar_w,
                      color=color, edgecolor="black", lw=0.6)
        for i, (v, n) in enumerate(zip(ys, notes)):
            if v == 0.0:
                bars[i].set_hatch("///")
                bars[i].set_alpha(0.4)
                if n:
                    ax.text(group_x[i] + offsets[k], 0.5, n,
                            ha="center", va="bottom",
                            fontsize=8, color="firebrick",
                            rotation=90, fontweight="bold")

    ax.set_xticks(group_x)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("batch size  B")
    ax.set_ylabel("sims / sec")
    ax.set_yscale("log")
    if all_max > 0:
        ax.set_ylim(bottom=0.1, top=all_max * 3.0)
    ax.set_title(title)
    ax.grid(True, axis="y", which="major", alpha=0.4, lw=0.6)
    ax.grid(True, axis="y", which="minor", alpha=0.18, lw=0.4)
    ax.tick_params(axis="both", which="both", direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)


def main():
    bas_configs = [c for c in GPU_CONFIGS
                   if c[0] == GPU_BASE / "jaxley_gpu_bench" / "bench_gpu.csv"]

    print("L5TTPC bwd_euler (sims/sec):")
    for path, label, _ in GPU_CONFIGS:
        for mode in ("fwd", "fwd_bwd"):
            row = []
            for B in BATCHES:
                sps, note = _gpu_value(path, "L5TTPC", B, mode)
                row.append(f"B{B}={('%.1f'%sps) if sps else note}")
            print(f"  {label:30s} {mode:8s}: " + "  ".join(row))

    print("\nball_and_stick bwd_euler (sims/sec, baseline only):")
    for mode in ("fwd", "fwd_bwd"):
        row = []
        for B in BATCHES:
            sps, note = _gpu_value(bas_configs[0][0], "ball_and_stick", B, mode)
            row.append(f"B{B}={('%.1f'%sps) if sps else note}")
        print(f"  GPU NCOMP=4 (baseline)         {mode:8s}: " + "  ".join(row))

    print("\nCPU bwd_euler (sims/sec, fwd only):")
    for cell in ("L5TTPC", "ball_and_stick"):
        row = []
        for B in BATCHES:
            v = _cpu_value(cell, B)
            row.append(f"B{B}={('%.1f'%v) if v else '----'}")
        print(f"  {cell:16s}: " + "  ".join(row))

    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(2, 2, left=0.07, right=0.985,
                           top=0.92, bottom=0.18,
                           wspace=0.20, hspace=0.42)
    ax_l_fwd = fig.add_subplot(gs[0, 0])
    ax_l_bwd = fig.add_subplot(gs[0, 1])
    ax_b_fwd = fig.add_subplot(gs[1, 0])
    ax_b_bwd = fig.add_subplot(gs[1, 1])

    _plot_panel(ax_l_fwd, "L5TTPC", "fwd",
                "L5TTPC — forward only (bwd_euler)", GPU_CONFIGS)
    _plot_panel(ax_l_bwd, "L5TTPC", "fwd_bwd",
                "L5TTPC — forward + backward (bwd_euler)", GPU_CONFIGS)
    _plot_panel(ax_b_fwd, "ball_and_stick", "fwd",
                "ball_and_stick — forward only (bwd_euler)", bas_configs)
    _plot_panel(ax_b_bwd, "ball_and_stick", "fwd_bwd",
                "ball_and_stick — forward + backward (bwd_euler)", bas_configs)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CPU_COLOR,
                             ec="black", lw=0.6, label=CPU_LABEL)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=color,
                              ec="black", lw=0.6, label=label)
                for _, label, color in GPU_CONFIGS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.02), frameon=False)

    fig.suptitle(
        "jaxley voltage-loss throughput — grouped bars by batch size  "
        "(dt=0.1 ms, t_max=100 ms, A100; CPU = single-thread reference)",
        fontsize=13, y=0.97,
    )

    out = GPU_BASE / "jaxley_gpu_bench" / "bar_grouped.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)            # vector PDF; no bbox_inches='tight'
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
