"""Pull every completed bench CSV into a single comparison figure.

Reads CPU bench (`bench_solvers.py` output) + GPU bench variants (`bench_gpu_l5ttpc.py`
output: NCOMP=4 baseline, NCOMP=2 ablation, NCOMP=1, NCOMP=2 + checkpoint).
Plots sims/sec vs batch size, log–log, with one line per config.

Output: `$NEUINV_BENCH_OUT_ROOT/comparison_plot.png` (default
$PSCRATCH/tmp_neuInv/jaxley_gpu_bench/comparison_plot.png).

Usage:
    conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
    python -m toolbox.tests.plot_bench_comparison
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ───────────────────────── sources ─────────────────────────

# (path, label, color, marker, linestyle, has_mode_column)
GPU_BASE = Path("/pscratch/sd/k/ktub1999/tmp_neuInv")
CPU_CSV  = GPU_BASE / "jaxley_solver_bench" / "summary_timing.csv"

GPU_SOURCES = [
    (GPU_BASE / "jaxley_gpu_bench"        / "bench_gpu.csv",
        "GPU NCOMP=4 (baseline)",     "tab:blue",   "o", "-"),
    (GPU_BASE / "jaxley_gpu_bench_ncomp2" / "bench_gpu.csv",
        "GPU NCOMP=2",                "tab:orange", "s", "-"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp1" / "bench_gpu.csv",
        "GPU NCOMP=1",                "tab:green",  "^", "-"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp2_ckpt_10x100" / "bench_gpu.csv",
        "GPU NCOMP=2 + ckpt[10,100]", "tab:red",    "D", "-"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp4_ckpt_10x100" / "bench_gpu.csv",
        "GPU NCOMP=4 + ckpt[10,100]", "tab:purple", "v", "-"),
    (GPU_BASE / "jaxley_gpu_bench_4way" / "ncomp4_ckpt_100x10" / "bench_gpu.csv",
        "GPU NCOMP=4 + ckpt[100,10]", "tab:brown",  "P", "-"),
]


def _load_gpu(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Drop rows that errored or didn't run.
    df = df[df["sims_per_sec"].notna() & (df["note"].fillna("") == "")]
    df["sims_per_sec"] = pd.to_numeric(df["sims_per_sec"], errors="coerce")
    df["B"] = pd.to_numeric(df["B"], errors="coerce")
    return df


def _load_cpu(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["note"].fillna("") == ""]
    df["sims_per_sec"] = pd.to_numeric(df["sims_per_sec"], errors="coerce")
    df["B"] = pd.to_numeric(df["B"], errors="coerce")
    df["mode"] = "fwd"  # CPU bench is forward-only
    return df


def _plot_panel(ax, cell: str, mode: str, gpu_dfs, cpu_df, title: str):
    plotted = False

    # CPU line — only available for fwd; same for ball_and_stick & L5TTPC.
    if mode == "fwd":
        cpu = cpu_df[(cpu_df["cell"] == cell) & (cpu_df["solver"] == "bwd_euler")]
        if not cpu.empty:
            ax.plot(cpu["B"], cpu["sims_per_sec"],
                    color="gray", marker="x", linestyle="--",
                    label="CPU bwd_euler (single thread)", alpha=0.85, lw=1.6)
            plotted = True

    for path, label, color, marker, ls in GPU_SOURCES:
        df = gpu_dfs.get(str(path))
        if df is None or df.empty:
            continue
        sub = df[(df["cell"] == cell) & (df["solver"] == "bwd_euler")
                 & (df["mode"] == mode)]
        if sub.empty:
            continue
        sub = sub.sort_values("B")
        ax.plot(sub["B"], sub["sims_per_sec"],
                color=color, marker=marker, linestyle=ls, label=label,
                alpha=0.92, lw=1.8, ms=6)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=11)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B")
    ax.set_ylabel("sims/sec")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best")


def main():
    cpu_df  = _load_cpu(CPU_CSV)
    gpu_dfs = {str(p): _load_gpu(p) for p, *_ in GPU_SOURCES}

    print(f"CPU rows: {len(cpu_df)}")
    for p, *_ in GPU_SOURCES:
        n = len(gpu_dfs.get(str(p), pd.DataFrame()))
        print(f"  GPU {p.name}: {n} rows  ({'present' if p.exists() else 'MISSING'})")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_panel(axes[0][0], "L5TTPC",         "fwd",     gpu_dfs, cpu_df,
                "L5TTPC — forward only (bwd_euler)")
    _plot_panel(axes[0][1], "L5TTPC",         "fwd_bwd", gpu_dfs, cpu_df,
                "L5TTPC — forward + backward (bwd_euler)")
    _plot_panel(axes[1][0], "ball_and_stick", "fwd",     gpu_dfs, cpu_df,
                "ball_and_stick — forward only (bwd_euler)")
    _plot_panel(axes[1][1], "ball_and_stick", "fwd_bwd", gpu_dfs, cpu_df,
                "ball_and_stick — forward + backward (bwd_euler)")

    fig.suptitle(
        "jaxley voltage-loss throughput — CPU vs GPU variants  "
        "(dt=0.1 ms, t_max=100 ms, A100)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = GPU_BASE / "jaxley_gpu_bench" / "comparison_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
