#!/usr/bin/env python3
"""Compare two sensitivity_variation runs (e.g. native vs interpolated stims),
and the voltage-metric vs the eFEL-metric within each run.

For every channel it reports, on the stims common to both runs:
  * Spearman rank correlation of the per-stim variation between run A and run B
    (how much does interpolating to a fixed length reorder the stim ranking?)
  * the #1 stim in each run and whether it changed
  * top-3 overlap
It also reports, within each run, the Spearman correlation between the voltage
metric and the eFEL metric (do they rank stims the same way?).

Usage:
  python sensitivity_variation_compare.py \
     --a native=<runA/combined> --b interp4000=<runB/combined> \
     --outDir <dir>
"""

import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox.Util_IOfunc import write_yaml


def read_matrix(path):
    if not os.path.exists(path):
        return None, None, None
    with open(path) as fh:
        r = list(csv.reader(fh))
    header = r[0][1:]
    stims = [ln[0] for ln in r[1:]]
    vals = np.asarray([[float(x) for x in ln[1:]] for ln in r[1:]], dtype=np.float64)
    return stims, header, vals


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def align(stimsA, valsA, stimsB, valsB):
    """Restrict both to common stims, same order."""
    common = [s for s in stimsA if s in set(stimsB)]
    ia = {s: i for i, s in enumerate(stimsA)}
    ib = {s: i for i, s in enumerate(stimsB)}
    A = valsA[[ia[s] for s in common]]
    B = valsB[[ib[s] for s in common]]
    return common, A, B


def parse_kv(s):
    label, _, path = s.partition("=")
    return label, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="labelA=<runA/combined>")
    ap.add_argument("--b", required=True, help="labelB=<runB/combined>")
    ap.add_argument("--outDir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outDir, exist_ok=True)

    (la, pa), (lb, pb) = parse_kv(args.a), parse_kv(args.b)
    sA, chA, vA = read_matrix(os.path.join(pa, "variation_matrix.csv"))
    sB, chB, vB = read_matrix(os.path.join(pb, "variation_matrix.csv"))
    assert chA == chB, "channel columns differ between runs"
    channels = chA
    common, VA, VB = align(sA, vA, sB, vB)
    P = len(channels)
    print(f"[cmp] {la} vs {lb}: {len(common)} common stims, {P} channels")

    # eFEL aggregates (optional).
    eA = read_matrix(os.path.join(pa, "efel_variation_aggregate.csv"))
    eB = read_matrix(os.path.join(pb, "efel_variation_aggregate.csv"))
    have_efel = eA[0] is not None and eB[0] is not None
    if have_efel:
        _, EA, EB = align(eA[0], eA[2], eB[0], eB[2])

    report = {"runA": {"label": la, "path": pa}, "runB": {"label": lb, "path": pb},
              "n_common_stims": len(common), "per_channel": {}}

    print("\n" + "=" * 92)
    print(f"VOLTAGE metric: {la}  vs  {lb}   (Spearman ρ of per-stim ranking; best stim each)")
    print("-" * 92)
    print(f"{'channel':<18}{'ρ(volt)':>9}{'best '+la:>22}{'best '+lb:>22}{'top3∩':>7}")
    for p in range(P):
        rho = spearman(VA[:, p], VB[:, p])
        bestA = common[int(np.nanargmax(VA[:, p]))]
        bestB = common[int(np.nanargmax(VB[:, p]))]
        t3A = {common[i] for i in np.argsort(-VA[:, p])[:3]}
        t3B = {common[i] for i in np.argsort(-VB[:, p])[:3]}
        ov = len(t3A & t3B)
        star = "" if bestA == bestB else "  *changed"
        print(f"{channels[p]:<18}{rho:>9.3f}{bestA:>22}{bestB:>22}{ov:>7}{star}")
        entry = {"spearman_voltage": None if np.isnan(rho) else round(rho, 4),
                 f"best_{la}": bestA, f"best_{lb}": bestB,
                 "best_changed": bestA != bestB, "top3_overlap": ov}
        report["per_channel"][channels[p]] = entry

    if have_efel:
        print("-" * 92)
        print(f"eFEL metric: {la} vs {lb}, and VOLTAGE-vs-eFEL agreement within each run")
        print(f"{'channel':<18}{'ρ(efel)':>9}{'ρ V–eFEL '+la:>16}{'ρ V–eFEL '+lb:>16}")
        for p in range(P):
            rho_e = spearman(EA[:, p], EB[:, p])
            rho_ve_a = spearman(VA[:, p], EA[:, p])
            rho_ve_b = spearman(VB[:, p], EB[:, p])
            print(f"{channels[p]:<18}{rho_e:>9.3f}{rho_ve_a:>16.3f}{rho_ve_b:>16.3f}")
            report["per_channel"][channels[p]].update({
                "spearman_efel": None if np.isnan(rho_e) else round(rho_e, 4),
                f"spearman_volt_vs_efel_{la}": None if np.isnan(rho_ve_a) else round(rho_ve_a, 4),
                f"spearman_volt_vs_efel_{lb}": None if np.isnan(rho_ve_b) else round(rho_ve_b, 4)})
    print("=" * 92 + "\n")

    write_yaml(report, os.path.join(args.outDir, "compare_summary.yaml"))

    # Scatter: per-channel voltage variation A vs B (log-log), y=x reference.
    ncol = 3; nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for p in range(P):
        ax = axes[p]
        x, y = VA[:, p], VB[:, p]
        ax.scatter(x, y, s=14, alpha=0.6)
        lim = [min(x.min(), y.min()) * 0.9 + 1e-6, max(x.max(), y.max()) * 1.1 + 1e-6]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{la} var (mV)", fontsize=8)
        ax.set_ylabel(f"{lb} var (mV)", fontsize=8)
        rho = spearman(x, y)
        ax.set_title(f"{channels[p]}  (ρ={rho:.2f})", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for p in range(P, len(axes)):
        axes[p].axis("off")
    fig.suptitle(f"Per-stim voltage variation: {la} vs {lb}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(args.outDir, "compare_scatter.png"), dpi=130)
    plt.close(fig)
    print(f"[cmp] wrote compare_summary.yaml + compare_scatter.png -> {args.outDir}/")


if __name__ == "__main__":
    main()
