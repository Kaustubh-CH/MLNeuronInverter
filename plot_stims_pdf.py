#!/usr/bin/env python3
"""Plot every stimulus CSV in a directory into one multipage PDF (4 per page).

Usage:
  python plot_stims_pdf.py --stimDir <dir> --out <file.pdf> [--dtStim 0.1]
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_current(path):
    try:
        a = np.loadtxt(path)
        return a if a.ndim == 1 else a[:, -1]
    except ValueError:
        a = np.genfromtxt(path, delimiter=",", names=True)
        c = a.dtype.names
        pick = next((x for x in c if "data" in x.lower() or "scaled" in x.lower()), c[-1])
        return np.asarray(a[pick], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimDir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtStim", type=float, default=0.1)
    ap.add_argument("--perPage", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.stimDir, "*.csv")))
    print(f"[plot] {len(files)} stims from {args.stimDir}")
    per = args.perPage
    with PdfPages(args.out) as pdf:
        for start in range(0, len(files), per):
            batch = files[start:start + per]
            fig, axes = plt.subplots(len(batch), 1, figsize=(11, 2.2 * len(batch)))
            axes = np.atleast_1d(axes)
            for ax, f in zip(axes, batch):
                name = os.path.splitext(os.path.basename(f))[0]
                cur = load_current(f)
                t = np.arange(len(cur)) * args.dtStim
                ax.plot(t, cur, lw=0.7, color="C0")
                ax.set_title(f"{name}   (n={len(cur)}, {t[-1]:.0f} ms)", fontsize=9)
                ax.set_xlabel("t (ms)", fontsize=8); ax.set_ylabel("I (nA)", fontsize=8)
                ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
