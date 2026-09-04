#!/usr/bin/env python3
"""Plot every recording in exp_data_paula: one page per ABF file.

Each page has two subplots: the stimulus on top and the recorded response
below.  Covers all 27 files — the IV curve (30 sweeps overlaid), the 25
Roy100..Roy2000 chaotic-stimulus sweeps, and the long gap-free VC recording
(where the roles flip: command voltage on top, recorded current below).

Run inside the shifter container (it has pyabf + matplotlib):
    shifter --image=balewski/ubu20-neuron8:v5 python3 plot_all_stims_paula.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

import pyabf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_data_paula")
OUT_PDF = os.path.join(DATA_DIR, "all_stims_and_voltages.pdf")
SPIKE_THRESH_MV = -10.0

# --- palette (same validated set as plot_roy_traces.py) ----------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d9d8d3"
STIM = "#52514e"          # stimulus is context, not an identity series
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
AMP_COLOR = {100: RAMP[0], 500: RAMP[1], 1000: RAMP[2], 1500: RAMP[3], 2000: RAMP[4]}
SEQ_CMAP = LinearSegmentedColormap.from_list("blues_seq", RAMP)

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


def style(ax, xlab=None, ylab=None):
    ax.grid(True, alpha=0.55, linewidth=0.5)
    ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)


def n_spikes(v, thresh=SPIKE_THRESH_MV):
    return int(np.count_nonzero((v[:-1] < thresh) & (v[1:] >= thresh)))


def minmax_decimate(t, y, nbins=4000):
    """Per-bin min/max envelope: keeps every transient at a fraction of the points."""
    n = len(y)
    if n <= 2 * nbins:
        return t, y
    edge = (n // nbins) * nbins
    yb = y[:edge].reshape(nbins, -1)
    tb = t[:edge].reshape(nbins, -1)
    lo, hi = yb.argmin(axis=1), yb.argmax(axis=1)
    rows = np.arange(nbins)
    idx = np.sort(np.concatenate([rows * yb.shape[1] + lo, rows * yb.shape[1] + hi]))
    return t[:edge][idx], y[:edge][idx]


def new_page(title, subtitle):
    fig, (ax_s, ax_v) = plt.subplots(
        2, 1, figsize=(13, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.2]})
    fig.suptitle(title, fontsize=13, y=0.985, color=INK)
    fig.text(0.5, 0.955, subtitle, fontsize=8.5, color=INK_2, ha="center")
    return fig, ax_s, ax_v


def finish_page(pdf, fig, ax_v, xlab="time (s)"):
    ax_v.set_xlabel(xlab)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    pdf.savefig(fig)
    plt.close(fig)


def page_iv(pdf, abf, fname):
    nsw = abf.sweepCount
    colors = [SEQ_CMAP(k / max(nsw - 1, 1)) for k in range(nsw)]
    steps = []
    fig, ax_s, ax_v = new_page(
        f"{fname} — {abf.protocol}",
        f"{nsw} sweeps overlaid · color = sweep order (light → dark) · "
        f"{abf.dataRate / 1000:.0f} kHz · {abf.sweepLengthSec:.2f} s per sweep")
    for k in range(nsw):
        abf.setSweep(k, channel=1)
        ax_s.plot(abf.sweepX, abf.sweepY, color=colors[k], lw=0.5)
        steps.append(float(np.median(abf.sweepY[len(abf.sweepY) // 2 - 500:
                                                len(abf.sweepY) // 2 + 500])))
        abf.setSweep(k, channel=0)
        ax_v.plot(abf.sweepX, abf.sweepY, color=colors[k], lw=0.5)
    style(ax_s, ylab="I inj (pA)")
    style(ax_v, ylab="V (mV)")
    ax_s.text(0.995, 0.94,
              f"current steps, mid-sweep level {min(steps):.0f} to {max(steps):.0f} pA",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2)
    finish_page(pdf, fig, ax_v)


def page_roy(pdf, abf, fname, amp):
    c = AMP_COLOR.get(amp, RAMP[-1])
    abf.setSweep(0, channel=0)
    t, v = abf.sweepX.copy(), abf.sweepY.copy()
    abf.setSweep(0, channel=1)
    i = abf.sweepY.copy()
    fig, ax_s, ax_v = new_page(
        f"{fname} — {abf.protocol}",
        f"{abf.dataRate / 1000:.0f} kHz · {abf.sweepLengthSec:.2f} s sweep · "
        f"I spans {i.min():.0f} to {i.max():.0f} pA (median hold {np.median(i):.0f} pA)")
    ax_s.plot(t, i, color=STIM, lw=0.55)
    style(ax_s, ylab="I inj (pA)")
    ax_s.text(0.995, 0.94, "injected current (recorded command)",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2)
    ax_v.plot(t, v, color=c, lw=0.6)
    style(ax_v, ylab="V (mV)")
    ax_v.text(0.995, 0.94,
              f"membrane potential · {n_spikes(v)} spikes "
              f"(upward crossings of {SPIKE_THRESH_MV:g} mV)",
              transform=ax_v.transAxes, ha="right", va="top", fontsize=8.5, color=INK_2)
    finish_page(pdf, fig, ax_v)


def page_gapfree(pdf, abf, fname):
    # VC recording: channel 1 = command/membrane voltage, channel 0 = current.
    abf.setSweep(0, channel=1)
    tv, v = minmax_decimate(abf.sweepX, abf.sweepY)
    abf.setSweep(0, channel=0)
    ti, i = minmax_decimate(abf.sweepX, abf.sweepY)
    fig, ax_s, ax_v = new_page(
        f"{fname} — {abf.protocol}",
        f"voltage clamp · {abf.dataRate / 1000:.0f} kHz · {abf.sweepLengthSec:.1f} s "
        "continuous recording · traces min/max-decimated for plotting")
    ax_s.plot(tv, v, color=STIM, lw=0.55)
    style(ax_s, ylab="V command (mV)")
    ax_s.text(0.995, 0.94, "holding potential", transform=ax_s.transAxes,
              ha="right", va="top", fontsize=8.5, color=INK_2)
    ax_v.plot(ti, i, color=RAMP[2], lw=0.5)
    style(ax_v, ylab="I recorded (pA)")
    ax_v.text(0.995, 0.94, "membrane current response", transform=ax_v.transAxes,
              ha="right", va="top", fontsize=8.5, color=INK_2)
    finish_page(pdf, fig, ax_v)


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".abf"))
    with PdfPages(OUT_PDF) as pdf:
        for fname in files:
            abf = pyabf.ABF(os.path.join(DATA_DIR, fname))
            proto = abf.protocol
            if proto.startswith("IV"):
                page_iv(pdf, abf, fname)
            elif proto.startswith("Roy"):
                amp = int("".join(ch for ch in proto.split()[0] if ch.isdigit()) or -1)
                page_roy(pdf, abf, fname, amp)
            else:
                page_gapfree(pdf, abf, fname)
            print(f"page: {fname}  {proto}")
        meta = pdf.infodict()
        meta["Title"] = "All stimuli and responses — exp_data_paula"
        meta["Subject"] = f"{len(files)} ABF files, one page each (stimulus + response)"
    print(f"wrote {OUT_PDF} ({len(files)} pages)")


if __name__ == "__main__":
    main()
