#!/usr/bin/env python3
"""Visualize the "Roy*" voltage traces in exp_data_paula as a multi-page PDF.

Selection is driven by the protocol name stored in each ABF header, not by the
file name: every sweep whose protocol starts with "Roy" is included.  That picks
up Roy100/500/1000/1500/2000 (5 repeats each) and leaves out the IV curve and the
gap-free recording.

Run inside the shifter container (it has pyabf + matplotlib):
    shifter --image=balewski/ubu20-neuron8:v5 python3 plot_roy_traces.py
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import pyabf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_data_paula")
OUT_PDF = os.path.join(DATA_DIR, "roy_voltage_traces.pdf")
PREFIX = "Roy"
SPIKE_THRESH_MV = -10.0

# --- palette -----------------------------------------------------------------
# Stimulus amplitude is ordered magnitude, so it gets a one-hue ordinal ramp
# (blue, steps 250-650) rather than categorical hues.  Validated light-mode:
# monotone L, min dL 0.093, light end 2.06:1 on the surface, hue spread 3 deg.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d9d8d3"
STIM = "#52514e"          # injected current: context, not an identity series
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "font.size": 9,
})


def load_roy_traces(data_dir):
    """Return list of dicts, one per Roy*.abf sweep, sorted by amplitude then file."""
    out = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".abf"):
            continue
        abf = pyabf.ABF(os.path.join(data_dir, fname))
        if not abf.protocol.startswith(PREFIX):
            continue
        m = re.match(r"Roy(\d+)", abf.protocol)
        abf.setSweep(0, channel=0)          # channel 0 = membrane potential (mV)
        t, v = abf.sweepX.copy(), abf.sweepY.copy()
        abf.setSweep(0, channel=1)          # channel 1 = injected current (pA)
        i = abf.sweepY.copy()
        out.append({
            "file": fname,
            "protocol": abf.protocol,
            "amp": int(m.group(1)) if m else -1,
            "t": t, "v": v, "i": i,
            "rate": abf.dataRate,
        })
    out.sort(key=lambda d: (d["amp"], d["file"]))
    return out


def n_spikes(v, thresh=SPIKE_THRESH_MV):
    """Count upward threshold crossings."""
    return int(np.count_nonzero((v[:-1] < thresh) & (v[1:] >= thresh)))


def style(ax, xlab=None, ylab=None):
    ax.grid(True, alpha=0.55, linewidth=0.5)
    ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)


def main():
    traces = load_roy_traces(DATA_DIR)
    if not traces:
        raise SystemExit(f"no {PREFIX}* protocols found in {DATA_DIR}")

    amps = sorted({d["amp"] for d in traces})
    color = {a: RAMP[k % len(RAMP)] for k, a in enumerate(amps)}
    by_amp = {a: [d for d in traces if d["amp"] == a] for a in amps}

    # Common axes limits so every panel in the document is directly comparable.
    vmin = min(d["v"].min() for d in traces)
    vmax = max(d["v"].max() for d in traces)
    pad = 0.06 * (vmax - vmin)
    VLIM = (vmin - pad, vmax + pad)
    imin = min(d["i"].min() for d in traces)
    imax = max(d["i"].max() for d in traces)
    ipad = 0.08 * (imax - imin)
    ILIM = (imin - ipad, imax + ipad)
    TMAX = max(d["t"][-1] for d in traces)

    handles = [Line2D([0], [0], color=color[a], lw=2, label=f"Roy{a}") for a in amps]

    with PdfPages(OUT_PDF) as pdf:
        # ---- page 1: every trace, rows = amplitude, cols = repeat ------------
        ncol = max(len(v) for v in by_amp.values())
        fig, axes = plt.subplots(len(amps), ncol, figsize=(13, 9.5),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for r, a in enumerate(amps):
            for c in range(ncol):
                ax = axes[r, c]
                if c >= len(by_amp[a]):
                    ax.axis("off")
                    continue
                d = by_amp[a][c]
                ax.plot(d["t"], d["v"], color=color[a], lw=0.6)
                ax.text(0.03, 0.94, f"{d['file'][:-4]}  ·  {n_spikes(d['v'])} sp",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, color=INK_MUTED)
                ax.set_ylim(*VLIM)
                ax.set_xlim(0, TMAX)
                style(ax)
                if c == 0:
                    ax.set_ylabel(f"Roy{a}\nV (mV)", fontsize=8.5, color=INK)
                if r == len(amps) - 1:
                    ax.set_xlabel("time (s)")
        fig.suptitle("Roy* protocols — membrane potential, all 25 sweeps",
                     fontsize=13, y=0.985, color=INK, x=0.5, ha="center")
        fig.text(0.5, 0.955,
                 "rows = chaotic-stimulus amplitude (pA scale factor) · columns = the 5 repeats · "
                 f"shared axes · 'sp' = upward crossings of {SPIKE_THRESH_MV:g} mV",
                 fontsize=8.5, color=INK_2, ha="center")
        fig.legend(handles=handles, loc="lower center", ncol=len(amps), frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.035, 1, 0.945])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: repeats overlaid, one panel per amplitude ---------------
        fig, axes = plt.subplots(len(amps), 1, figsize=(13, 9.5), sharex=True, sharey=True)
        for ax, a in zip(np.atleast_1d(axes), amps):
            for d in by_amp[a]:
                ax.plot(d["t"], d["v"], color=color[a], lw=0.55, alpha=0.85)
            spk = [n_spikes(d["v"]) for d in by_amp[a]]
            ax.legend(handles=[Line2D([0], [0], color=color[a], lw=2)],
                      labels=[f"Roy{a}  ·  {len(by_amp[a])} repeats overlaid  ·  "
                              f"spikes {min(spk)}–{max(spk)}"],
                      loc="upper right", frameon=False, fontsize=9,
                      handlelength=1.6, borderpad=0.1)
            ax.set_ylim(*VLIM)
            ax.set_xlim(0, TMAX)
            style(ax, ylab="V (mV)")
        np.atleast_1d(axes)[-1].set_xlabel("time (s)")
        fig.suptitle("Repeat-to-repeat reproducibility — each amplitude's 5 sweeps overlaid",
                     fontsize=13, y=0.985, color=INK)
        fig.text(0.5, 0.957, "same y-scale throughout; a thick-looking trace is repeat jitter",
                 fontsize=8.5, color=INK_2, ha="center")
        fig.tight_layout(rect=[0, 0, 1, 0.948])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- pages 3+: one page per amplitude, stimulus + each repeat --------
        # Unlike pages 1-2, these auto-scale to the amplitude on the page: the
        # global scale is set by the Roy2000 spikes and flattens Roy100's
        # subthreshold response into a line.
        for a in amps:
            grp = by_amp[a]
            gv = np.concatenate([d["v"] for d in grp])
            gvpad = 0.08 * (gv.max() - gv.min())
            gvlim = (gv.min() - gvpad, gv.max() + gvpad)
            gi = grp[0]["i"]
            gipad = 0.10 * (gi.max() - gi.min())
            gilim = (gi.min() - gipad, gi.max() + gipad)

            fig, axes = plt.subplots(len(grp) + 1, 1, figsize=(13, 9.5), sharex=True,
                                     gridspec_kw={"height_ratios": [0.85] + [1] * len(grp)})
            ax0 = axes[0]
            ax0.plot(grp[0]["t"], grp[0]["i"], color=STIM, lw=0.55)
            ax0.set_ylim(*gilim)
            style(ax0, ylab="I inj (pA)")
            ax0.text(0.995, 0.92, "injected current (recorded command, repeat 1)",
                     transform=ax0.transAxes, ha="right", va="top",
                     fontsize=8.5, color=INK_2)

            for ax, d in zip(axes[1:], grp):
                ax.plot(d["t"], d["v"], color=color[a], lw=0.6)
                ax.set_ylim(*gvlim)
                style(ax, ylab="V (mV)")
                ax.text(0.995, 0.92, f"{d['file'][:-4]}  ·  {n_spikes(d['v'])} spikes",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5, color=INK_2)
            axes[-1].set_xlabel("time (s)")
            axes[-1].set_xlim(0, TMAX)

            hold = float(np.median(grp[0]["i"]))
            fig.suptitle(f"Roy{a} — chaotic current stimulus and the 5 voltage responses",
                         fontsize=13, y=0.985, color=INK)
            fig.text(0.5, 0.957,
                     f"{grp[0]['rate'] / 1000:.0f} kHz · {grp[0]['t'][-1]:.2f} s sweeps · "
                     f"I spans {grp[0]['i'].min():.0f} to {grp[0]['i'].max():.0f} pA "
                     f"(median hold {hold:.0f} pA) · axes auto-scaled to Roy{a} "
                     f"(pages 1-2 hold the shared cross-amplitude scale)",
                     fontsize=8.5, color=INK_2, ha="center")
            fig.tight_layout(rect=[0, 0, 1, 0.948])
            pdf.savefig(fig)
            plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = "Roy* voltage traces — exp_data_paula"
        meta["Subject"] = f"{len(traces)} sweeps, protocols {', '.join('Roy%d' % a for a in amps)}"

    print(f"wrote {OUT_PDF}")
    print(f"{len(traces)} sweeps across {len(amps)} protocols: "
          + ", ".join(f"Roy{a} (n={len(by_amp[a])})" for a in amps))
    for d in traces:
        print(f"  {d['file']}  {d['protocol']:14s} V [{d['v'].min():7.1f},{d['v'].max():7.1f}] mV  "
              f"{n_spikes(d['v'])} spikes")


if __name__ == "__main__":
    main()
