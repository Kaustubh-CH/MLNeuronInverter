#!/usr/bin/env python3
"""Merge chunked sensitivity_variation.py outputs into one combined result.

Each chunk (one per GPU) wrote a `part_<k>/` dir with variation_matrix.csv,
variation_matrix_peak.csv and per_stim_variation.pdf covering its slice of
stims.  This concatenates the CSV rows, rebuilds the cross-stim summary plots
(summary_heatmap.png, ranking_per_channel.png) + summary.yaml + rankings from
the full matrix, and gs-concatenates the per-part PDFs.

Usage:
  python sensitivity_variation_merge.py --partsDir <dir with part_*> \
         --outDir <combined out> [--cell ca3_pyramidal] [--topK 15]
"""

import os, sys, csv, glob, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from toolbox.Util_IOfunc import write_yaml
import sensitivity_variation as sv


def read_matrix(path):
    """Return (stims list, param_names list, values ndarray)."""
    with open(path) as fh:
        r = list(csv.reader(fh))
    header = r[0][1:]
    stims, rows = [], []
    for line in r[1:]:
        stims.append(line[0])
        rows.append([float(x) for x in line[1:]])
    return stims, header, np.asarray(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partsDir", required=True)
    ap.add_argument("--outDir", required=True)
    ap.add_argument("--cell", default="ca3_pyramidal")
    ap.add_argument("--nSamples", type=int, default=500)
    ap.add_argument("--stimDir", default="")
    ap.add_argument("--topK", type=int, default=15)
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.partsDir, "part_*")))
    if not parts:
        raise SystemExit(f"no part_* dirs under {args.partsDir}")
    os.makedirs(args.outDir, exist_ok=True)

    stims, param_names = [], None
    mean_rows, peak_rows = [], []
    efel_blocks, efel_feats = [], None
    blur_blocks, dtw_blocks = [], []
    sysvar_blocks, smoothvar_blocks = [], []
    part_pdfs = []
    for pd in parts:
        mpath = os.path.join(pd, "variation_matrix.csv")
        ppath = os.path.join(pd, "variation_matrix_peak.csv")
        if not os.path.exists(mpath):
            print(f"[merge] WARN missing {mpath}, skipping part {pd}")
            continue
        s, hdr, M = read_matrix(mpath)
        _, _, Pk = read_matrix(ppath)
        param_names = param_names or hdr
        stims += s; mean_rows.append(M); peak_rows.append(Pk)
        pdf = os.path.join(pd, "per_stim_variation.pdf")
        if os.path.exists(pdf):
            part_pdfs.append(pdf)
        efp = os.path.join(pd, "efel_variation_per_feature.npz")
        if os.path.exists(efp):
            z = np.load(efp, allow_pickle=True)
            efel_blocks.append(z["efel_var"])
            efel_feats = [str(x) for x in z["features"]]
        for nm, store in (("blur", blur_blocks), ("dtw", dtw_blocks),
                          ("sysvar", sysvar_blocks), ("smoothvar", smoothvar_blocks)):
            fp = os.path.join(pd, f"{nm}_variation_aggregate.csv")
            if os.path.exists(fp):
                _, _, Mn = read_matrix(fp)
                store.append(Mn)

    var_mean = np.vstack(mean_rows)
    var_peak = np.vstack(peak_rows)
    S, P = var_mean.shape
    print(f"[merge] combined {S} stims x {P} channels from {len(parts)} parts")

    # Combined CSVs.
    def dump(fname, M):
        with open(os.path.join(args.outDir, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stim"] + list(param_names))
            for i, s in enumerate(stims):
                w.writerow([s] + [f"{M[i, p]:.5f}" for p in range(P)])
    dump("variation_matrix.csv", var_mean)
    dump("variation_matrix_peak.csv", var_peak)

    col_max = np.nanmax(var_mean, axis=0)
    var_norm = var_mean / np.where(col_max > 0, col_max, 1.0)
    dump("variation_matrix_normalized.csv", var_norm)

    with open(os.path.join(args.outDir, "ranking_per_channel.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "rank", "stim", "variation_mV", "variation_norm"])
        for p in range(P):
            order = np.argsort(-var_mean[:, p])
            for rank, i in enumerate(order):
                w.writerow([param_names[p], rank + 1, stims[i],
                            f"{var_mean[i, p]:.5f}", f"{var_norm[i, p]:.4f}"])

    summary = {"cell": args.cell, "n_stims": int(S), "n_channels": int(P),
               "n_samples_per_set": int(args.nSamples), "stim_dir": args.stimDir,
               "best_stim_per_channel": {}}
    for p in range(P):
        order = np.argsort(-var_mean[:, p])
        summary["best_stim_per_channel"][param_names[p]] = {
            "best": stims[order[0]], "variation_mV": float(var_mean[order[0], p]),
            "top3": [{"stim": stims[i], "variation_mV": float(var_mean[i, p])}
                     for i in order[:3]]}
    # eFEL: concatenate per-feature blocks (S,P,F) and rebuild efel outputs.
    if efel_blocks and efel_feats is not None:
        efel_var = np.concatenate(efel_blocks, axis=0)
        if efel_var.shape[0] == len(stims):
            sv.write_efel_outputs(args.outDir, stims, param_names, efel_var,
                                  efel_feats, args.topK, summary)
            print(f"[merge] combined eFEL variation {efel_var.shape}")
        else:
            print(f"[merge] WARN eFEL block rows {efel_var.shape[0]} != stims {len(stims)}; skipping eFEL merge")

    # blur / dtw: concatenate per-part aggregate rows and rebuild outputs.
    for nm, blocks, unit, title in [
        ("blur", blur_blocks, "mean blurred-MSE to default (mV^2)",
         "Multi-scale blurred-MSE variation (dist to default; per-channel norm)\n"
         "brighter = sweeping this channel moves the blurred trace more"),
        ("dtw", dtw_blocks, "mean soft-DTW divergence to default",
         "Soft-DTW divergence variation (dist to default; per-channel norm)\n"
         "brighter = sweeping this channel moves the time-aligned trace more"),
        ("sysvar", sysvar_blocks, "parameter-explained (systematic) std (mV)",
         "Systematic (parameter-explained) variance (per-channel norm)\n"
         "chaos-robust: brighter = this channel SYSTEMATICALLY moves the trace"),
        ("smoothvar", smoothvar_blocks, "smoothed-ensemble std (mV)",
         "Smoothed-ensemble variance (per-channel norm)\n"
         "brighter = this channel moves the low-pass (jitter-free) envelope more")]:
        if not blocks:
            continue
        mat = np.vstack(blocks)
        if mat.shape[0] == len(stims):
            sv.write_scalar_metric_outputs(args.outDir, stims, param_names, mat,
                                           nm, args.topK, summary, unit, title)
            print(f"[merge] combined {nm} variation {mat.shape}")
        else:
            print(f"[merge] WARN {nm} rows {mat.shape[0]} != stims {len(stims)}; skipping")

    write_yaml(summary, os.path.join(args.outDir, "summary.yaml"))

    # Cross-stim summary plots (reuse the main script's plotters).
    sv._plot_summary_heatmap(args.outDir, stims, param_names, var_mean, var_norm)
    sv._plot_ranking_per_channel(args.outDir, stims, param_names, var_mean, args.topK)

    # Concatenate the per-part per-stim PDFs with ghostscript.
    if part_pdfs:
        out_pdf = os.path.join(args.outDir, "per_stim_variation.pdf")
        cmd = ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
               f"-sOutputFile={out_pdf}"] + part_pdfs
        try:
            subprocess.run(cmd, check=True)
            print(f"[merge] wrote combined PDF {out_pdf} ({len(part_pdfs)} parts)")
        except Exception as e:
            print(f"[merge] gs concat failed ({e}); per-part PDFs left in place")

    print(f"[merge] DONE -> {args.outDir}/")


if __name__ == "__main__":
    main()
