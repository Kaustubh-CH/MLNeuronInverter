#!/usr/bin/env python
"""Side-by-side comparison of sensitivity_variation.py runs (e.g. L5TTPC nc1 / nc2 / nc4).

  python scripts/compare_sensvar_ncomp.py nc1=<dir> nc2=<dir> nc4=<dir> -o <outDir>

Reads each <dir>/variation_matrix.csv (stims x channels, mean temporal std in mV) and writes
  compare_heatmaps.png   one annotated panel per run, shared log colour scale (raw mV)
  compare_bars.png       per stim, grouped bars channel x run (raw mV)
  compare_table.md       stim x channel table with one column per run
  compare_matrix.csv     long-form (run, stim, channel, mV)
"""
import sys, os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_matrix(path):
    rows = list(csv.DictReader(open(path)))
    ch = [k for k in rows[0] if k != "stim"]
    stims = [r["stim"] for r in rows]
    M = np.array([[float(r[c]) for c in ch] for r in rows])
    return stims, ch, M


_LOC = {"apical": "api", "dend": "dend", "axonal": "axo", "somatic": "soma", "all": "all"}


def short(n):
    tail = n.split("_")[-1]
    core, loc = (n[:-(len(tail) + 1)], _LOC[tail]) if tail in _LOC else (n, None)
    if "bar_" in core:
        core = core.split("bar_", 1)[1]
    core = core.replace("g_pas", "pas")
    return f"{core}_{loc}" if loc else core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=<combined dir>")
    ap.add_argument("-o", "--outDir", required=True)
    ap.add_argument("--vmin", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.outDir, exist_ok=True)

    runs = []
    for spec in args.runs:
        label, d = spec.split("=", 1)
        p = os.path.join(d, "variation_matrix.csv")
        if not os.path.exists(p):
            print(f"[cmp] WARN missing {p}, skipping {label}")
            continue
        runs.append((label, *read_matrix(p)))
    if not runs:
        raise SystemExit("no runs")
    stims, ch = runs[0][1], runs[0][2]
    for label, s, c, _ in runs[1:]:
        if s != stims or c != ch:
            raise SystemExit(f"{label}: stim/channel layout differs from {runs[0][0]}")
    labels = [r[0] for r in runs]
    Ms = [r[3] for r in runs]
    S, P = Ms[0].shape
    sshort = [s.replace("_icaRec_5k", "").replace("5k50kInterChaoticB", "ICB x1.5") for s in stims]
    cshort = [short(c) for c in ch]
    vmax = max(np.nanmax(M) for M in Ms)

    # 1) heatmaps, raw mV, shared log scale, annotated.
    fig, axes = plt.subplots(len(runs), 1, figsize=(0.75 * P + 3, (0.42 * S + 1.4) * len(runs)),
                             squeeze=False)
    for ax, label, M in zip(axes[:, 0], labels, Ms):
        im = ax.imshow(np.clip(M, args.vmin, None), aspect="auto", cmap="viridis",
                       norm=LogNorm(vmin=args.vmin, vmax=vmax))
        ax.set_yticks(range(S)); ax.set_yticklabels(sshort, fontsize=8)
        ax.set_xticks(range(P)); ax.set_xticklabels(cshort, rotation=40, ha="right", fontsize=7)
        for i in range(S):
            for j in range(P):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}" if v >= 0.01 else "<.01", ha="center", va="center",
                        fontsize=5.5, color="w" if v < 0.3 * vmax else "k")
        ax.set_title(f"{label}: mean temporal std of soma V (mV) when sweeping ONE channel over its box",
                     fontsize=9, loc="left")
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, label="mV (log scale)")
    fig.savefig(os.path.join(args.outDir, "compare_heatmaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2) grouped bars per stim.
    fig, axes = plt.subplots(S, 1, figsize=(0.75 * P + 3, 2.4 * S), sharex=True, squeeze=False)
    x = np.arange(P); wb = 0.8 / len(runs)
    for i, ax in enumerate(axes[:, 0]):
        for k, (label, M) in enumerate(zip(labels, Ms)):
            ax.bar(x + k * wb, np.clip(M[i], args.vmin, None), wb, label=label)
        ax.set_yscale("log"); ax.set_ylim(args.vmin, vmax * 1.5)
        ax.set_ylabel("mV"); ax.grid(alpha=0.3, axis="y", which="both")
        ax.set_title(sshort[i], fontsize=9, loc="left")
        if i == 0:
            ax.legend(fontsize=8, ncol=len(runs))
    axes[-1, 0].set_xticks(x + 0.4 - wb / 2)
    axes[-1, 0].set_xticklabels(cshort, rotation=40, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.outDir, "compare_bars.png"), dpi=130)
    plt.close(fig)

    # 3) tables.
    with open(os.path.join(args.outDir, "compare_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run", "stim", "channel", "variation_mV"])
        for label, M in zip(labels, Ms):
            for i in range(S):
                for j in range(P):
                    w.writerow([label, stims[i], ch[j], f"{M[i, j]:.5f}"])
    with open(os.path.join(args.outDir, "compare_table.md"), "w") as fh:
        for i in range(S):
            fh.write(f"\n### {sshort[i]}\n\n| channel | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n")
            for j in range(P):
                fh.write(f"| {cshort[j]} | " + " | ".join(f"{M[i, j]:.2f}" for M in Ms) + " |\n")
    # console summary: per run, channels with >= 0.5 mV under Roy2000-or-less and under ICB.
    for label, M in zip(labels, Ms):
        roy = [s for s in range(S) if stims[s].startswith("Roy")]
        best_roy = M[roy].max(axis=0) if roy else np.zeros(P)
        n_roy = int((best_roy >= 0.5).sum())
        print(f"[cmp] {label}: channels >= 0.5 mV under the best Roy stim: {n_roy}/{P} "
              f"({', '.join(cshort[j] for j in np.argsort(-best_roy)[:n_roy])})")
    print(f"[cmp] wrote {args.outDir}/compare_{{heatmaps,bars}}.png, compare_table.md, compare_matrix.csv")


if __name__ == "__main__":
    main()
