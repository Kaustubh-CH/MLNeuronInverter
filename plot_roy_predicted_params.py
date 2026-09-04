#!/usr/bin/env python3
"""Plot the ion-channel / passive parameters the probescan_exc model predicted
from Paula's Roy* traces.

  page 1  unit space -- one subplot per stimulus amplitude, all 15 parameters
          against the +-1 band the model was trained inside
  page 2  physical space -- small multiples, one panel per parameter in its own
          units, vs stimulus amplitude
  page 3  the three out-of-range treatments (exact / minmax / default), shown
          only for the parameters where they actually differ

Run: shifter --image=balewski/ubu20-neuron8:v5 python3 plot_roy_predicted_params.py [modelDir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ALL_CELLS = "/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS"
MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(ALL_CELLS, "probescan_exc_p0_56351363"))
TAG = {"probescan_exc_p0_56351363": "",
       "probescan_exc_p0_k128_56588087": "_k128"
       }.get(os.path.basename(MODEL_DIR),
             "_" + os.path.basename(MODEL_DIR))
DSET = sys.argv[2] if len(sys.argv) > 2 else ""   # e.g. "Blank350"
DTAG = ("_" + DSET.lower()) if DSET else ""
PRED_ROOT = os.path.join(MODEL_DIR, "predict_royPaula" + DSET)
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "exp_data_paula", f"roy_predicted_channel_params{TAG}{DTAG}.pdf")
AMPS = [100, 500, 1000, 1500, 2000]
VARIANTS = ["exact", "minmax", "default"]

SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#d9d8d3"
# amplitude is ordered magnitude -> ordinal ramp, monotone in lightness so the
# order reads even in greyscale / colour-blindness.  viridis[0.05,0.75]:
# min step dL* 13.0, min pairwise dE00 17.5, min contrast on surface 2.04:1.
RAMP = ["#5ec962", "#1fa287", "#2a788e", "#3e4a89", "#471365"]
BAND = "#e8e7e2"      # the +-1 trained region
WARN = "#e34948"      # out-of-range markers
# variants are categorical, not ordered
VAR_C = {"exact": "#0d366b", "minmax": "#eb6834", "default": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9,
    "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "font.size": 9,
})


def load():
    md = yaml.safe_load(open(os.path.join(MODEL_DIR, "out", "sum_train.yaml")))
    im = md["input_meta"]
    include = im["include"]
    # some sum_train.yaml carry base values as strings ('8e-05'), coerce first
    im["base_values"] = [float(x) for x in im["base_values"]]
    names = [im["parName"][i] for i in include]
    units = [im["phys_par_range"][i][2] for i in include]

    unitP, physP = {}, {v: {} for v in VARIANTS}
    for amp in AMPS:
        d = os.path.join(PRED_ROOT, f"Roy{amp}")
        unitP[amp] = np.array([pd.read_csv(f)["unit_params_predict"].values
                               for f in sorted(glob.glob(os.path.join(d, "unitParam*.csv")))])
        for v in VARIANTS:
            P = np.atleast_2d(np.loadtxt(os.path.join(d, f"{v}Converted1.csv")))
            physP[v][amp] = P[:, include]
    return names, units, unitP, physP, im


def style(ax):
    ax.grid(True, alpha=0.55, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    names, units, unitP, physP, im = load()
    npar = len(names)
    color = {a: RAMP[k] for k, a in enumerate(AMPS)}
    handles = [Line2D([0], [0], marker="o", ls="none", color=color[a],
                      markersize=7, label=f"Roy{a}") for a in AMPS]
    with PdfPages(OUT_PDF) as pdf:
        # ---- page 1: unit space, one subplot per stimulus --------------------
        fig, axes = plt.subplots(1, len(AMPS), figsize=(15, 8.5), sharey=True)
        y = np.arange(npar)
        lo = min(unitP[a].min() for a in AMPS) - 0.15
        hi = max(unitP[a].max() for a in AMPS) + 0.15
        for k, a in enumerate(AMPS):
            ax = axes[k]
            U = unitP[a]
            ax.axvspan(-1, 1, color=BAND, zorder=0)
            ax.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
            for j in range(npar):
                ax.scatter(U[:, j], np.full(U.shape[0], y[j]), s=30,
                           color=color[a], edgecolor=SURFACE, linewidth=0.7, zorder=3)
                bad = np.abs(U[:, j]) > 1
                if bad.any():
                    ax.scatter(U[bad, j], np.full(bad.sum(), y[j]), s=95,
                               facecolor="none", edgecolor=WARN, linewidth=1.3, zorder=4)
            n_oor = int(np.sum(np.abs(U) > 1))
            ax.set_title(f"Roy{a}\n{n_oor}/{U.size} outside ±1", fontsize=9.5,
                         color=INK, pad=8)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("unit parameter")
            style(ax)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names, fontsize=8.5)
        axes[0].set_ylim(npar - 0.5, -0.5)
        fig.legend(handles=handles + [Line2D([0], [0], marker="o", ls="none",
                                             markerfacecolor="none", markeredgecolor=WARN,
                                             markersize=9, label="outside trained ±1")],
                   loc="lower center", frameon=False, ncol=6, fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))
        n_oor = sum(int(np.sum(np.abs(unitP[a]) > 1)) for a in AMPS)
        n_tot = sum(unitP[a].size for a in AMPS)
        fig.suptitle("Predicted channel parameters in unit space — one panel per stimulus",
                     fontsize=13, y=0.985, color=INK)
        fig.text(0.5, 0.945, "shaded band = the ±1 range the model was trained inside · "
                             f"5 sweeps per stimulus · {n_oor}/{n_tot} predictions "
                             f"({100*n_oor/n_tot:.0f}%) fall outside it · "
                             f"{os.path.basename(MODEL_DIR)}",
                 fontsize=8.5, color=INK_2, ha="center")
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: physical space, small multiples --------------------------
        ncol = 4
        nrow = int(np.ceil(npar / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 9.5))
        axes = np.atleast_2d(axes)
        x = np.arange(len(AMPS))
        for j in range(nrow * ncol):
            ax = axes[j // ncol, j % ncol]
            if j >= npar:
                ax.axis("off")
                continue
            med = []
            for k, a in enumerate(AMPS):
                v = physP["exact"][a][:, j]
                ax.scatter(np.full(v.shape, x[k]), v, s=30, color=color[a],
                           edgecolor=SURFACE, linewidth=0.7, zorder=3)
                med.append(np.median(v))
            ax.plot(x, med, color=INK_MUTED, lw=1.2, zorder=2)
            if units[j] == "S/cm2":
                ax.set_yscale("log")
            ax.axhline(im["base_values"][im["include"][j]], color=WARN, lw=1.0,
                       ls="--", zorder=1, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in AMPS], fontsize=7.5, rotation=45)
            ax.set_title(names[j], fontsize=8, color=INK)
            ax.set_ylabel(units[j], fontsize=8)
            style(ax)
            if j // ncol == nrow - 1:
                ax.set_xlabel("Roy stim amplitude", fontsize=8)
        fig.suptitle("Predicted channel parameters in physical units vs stimulus amplitude",
                     fontsize=13, y=0.985, color=INK)
        fig.text(0.5, 0.958, "exact conversion · one dot per sweep · gray line = median · "
                             "dashed red = base value · conductances on log axes",
                 fontsize=8.5, color=INK_2, ha="center")
        fig.legend(handles=handles, loc="lower center", ncol=len(AMPS),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.035, 1, 0.948])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: where the three treatments differ ------------------------
        diff_j = [j for j in range(npar)
                  if any(not np.allclose(physP[v][a][:, j], physP["exact"][a][:, j],
                                         rtol=1e-9)
                         for v in ("minmax", "default") for a in AMPS)]
        if diff_j:
            ncol = min(3, len(diff_j))
            nrow = int(np.ceil(len(diff_j) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow),
                                     squeeze=False)
            for cell, j in enumerate(diff_j):
                ax = axes[cell // ncol, cell % ncol]
                for vi, v in enumerate(VARIANTS):
                    off = (vi - 1) * 0.22
                    for k, a in enumerate(AMPS):
                        vals = physP[v][a][:, j]
                        ax.scatter(np.full(vals.shape, k + off), vals, s=34,
                                   color=VAR_C[v], edgecolor=SURFACE,
                                   linewidth=0.7, zorder=3)
                if units[j] == "S/cm2":
                    ax.set_yscale("log")
                ax.axhline(im["base_values"][im["include"][j]], color=WARN, lw=1.0,
                           ls="--", zorder=1, alpha=0.8)
                ax.set_xticks(np.arange(len(AMPS)))
                ax.set_xticklabels([f"Roy{a}" for a in AMPS], fontsize=8, rotation=45)
                ax.set_title(names[j], fontsize=9, color=INK)
                ax.set_ylabel(units[j], fontsize=8)
                style(ax)
            for cell in range(len(diff_j), nrow * ncol):
                axes[cell // ncol, cell % ncol].axis("off")
            fig.legend(handles=[Line2D([0], [0], marker="o", ls="none",
                                       color=VAR_C[v], markersize=7, label=v)
                                for v in VARIANTS],
                       loc="lower center", ncol=3, frameon=False, fontsize=9,
                       bbox_to_anchor=(0.5, 0.0))
            fig.suptitle("The three out-of-range treatments, where they differ",
                         fontsize=13, y=0.985, color=INK)
            fig.text(0.5, 0.95, "only parameters with at least one prediction outside ±1 "
                                "are shown; the other "
                                f"{npar - len(diff_j)} are identical in all three · "
                                "dashed red = base value",
                     fontsize=8.5, color=INK_2, ha="center")
            fig.tight_layout(rect=[0, 0.07, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"wrote {OUT_PDF}")
    print(f"{npar} parameters x {len(AMPS)} amplitudes x "
          f"{unitP[AMPS[0]].shape[0]} sweeps; {len(diff_j)} params differ across variants")
    print("\nmedian physical value per amplitude (exact):")
    print("  %-32s %-9s" % ("param", "unit") + "".join("%12s" % f"Roy{a}" for a in AMPS))
    for j, nm in enumerate(names):
        row = "  %-32s %-9s" % (nm[:32], units[j])
        for a in AMPS:
            row += "%12.4g" % np.median(physP["exact"][a][:, j])
        print(row)


if __name__ == "__main__":
    main()
