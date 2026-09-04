#!/usr/bin/env python3
"""Overlay the measured Roy traces (model input) against the NEURON traces
simulated from the parameters the model predicted from them (model output),
for each of the three out-of-range treatments.

One page per stimulus.  Rows: stimulus, measured, exact / minmax / default
re-simulations, then up to two controls:

  ALL_CELLS mean base   the u=0 point of the probescan parameterisation.  This is
                        NOT any real cell's biophysics -- it is a 46-cell mean,
                        and it differs from the L5_TTPC1cADpyr0 defaults by up to
                        ~90x on individual conductances.
  BBP cell defaults     the unmodified cell as BBP ships it; the biophysically
                        meaningful "does this cell fire like the recording?".

Both live on the same 4000-bin / 0.1 ms grid:
  * experiment : ABF sweep decimated 50 kHz -> 0.1 ms, 400 ms window
  * simulation : run.py drops data[:1000], i.e. the 100 ms zero pad of
                 5k50kInterChaoticB, leaving exactly the experimental window

Run: shifter --image=balewski/ubu20-neuron8:v5 python3 \
         plot_roy_input_vs_output.py [modelDir] [cellName]
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ALL_CELLS = "/pscratch/sd/k/ktub1999/tmp_neuInv/bbp3/ALL_CELLS"
MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(ALL_CELLS, "probescan_exc_p0_56351363"))
CELL = sys.argv[2] if len(sys.argv) > 2 else "L5_TTPC1cADpyr0"
DSET = sys.argv[3] if len(sys.argv) > 3 else ""   # e.g. "Blank350"
DTAG = ("_" + DSET.lower()) if DSET else ""
DEFAULT_CELL = "L5_TTPC1cADpyr0"

TAG = {"probescan_exc_p0_56351363": "",
       "probescan_exc_p0_k128_56588087": "_k128"
       }.get(os.path.basename(MODEL_DIR), "_" + os.path.basename(MODEL_DIR))
CELL_TAG = "" if CELL == DEFAULT_CELL else "_" + CELL

EXP_DIR = "/global/homes/k/ktub1999/ExperimentalData/PyForEphys/RoyPaula" + DSET
SIM_ROOT = "/pscratch/sd/k/ktub1999/RoyPaulaSims/runs2"
CTL_ROOT = "/pscratch/sd/k/ktub1999/RoyPaulaSims/CONTROLS"
PRED_ROOT = os.path.join(MODEL_DIR, "predict_royPaula" + DSET)
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_data_paula",
                       f"roy_input_vs_neuron_output{TAG}{CELL_TAG}{DTAG}.pdf")
AMPS = [100, 500, 1000, 1500, 2000]
VARIANTS = ["exact", "minmax", "default"]
DT_MS = 0.1
N_TBIN = 4000

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
EXP_C = "#0d366b"     # measured -- the reference series
VAR_C = {"exact": "#eb6834", "minmax": "#c026a1", "default": "#1baf7a"}
MEAN_C = "#8a8880"    # controls stay neutral so they never read as a result
TRUE_C = "#52514e"
STIM_C = "#52514e"
SPIKE_THRESH = -10.0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "font.size": 9,
})


def load_jobs(path):
    jobs = {}
    if path and os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2:
                jobs[int(p[0])] = p[1]
    return jobs


def jobs_for(variant):
    """Per-cell map, falling back to the original cell's un-suffixed files."""
    p = os.path.join(PRED_ROOT, f"roy_sim_jobs_{CELL}_{variant}.txt")
    if os.path.exists(p):
        return load_jobs(p)
    if CELL == DEFAULT_CELL:
        legacy = ("roy_sim_jobs.txt" if variant == "exact"
                  else f"roy_sim_jobs_{variant}.txt")
        return load_jobs(os.path.join(PRED_ROOT, legacy))
    return {}


def n_spikes(v, thresh=SPIKE_THRESH):
    return int(np.count_nonzero((v[:-1] < thresh) & (v[1:] >= thresh)))


def spike_label(arr):
    s = [n_spikes(v) for v in arr]
    return f"{min(s)}" if min(s) == max(s) else f"{min(s)}–{max(s)}"


def load_sim(jid):
    """Return (nsim, 4000) somatic mV from every rank's h5 for this job."""
    out = []
    for f in sorted(glob.glob(os.path.join(SIM_ROOT, f"{jid}_1", "*", "*.h5"))):
        with h5py.File(f, "r") as hf:
            if "volts" not in hf:
                continue
            v = np.array(hf["volts"])          # (nsamp, tbin, probe, stim)
            while v.ndim < 4:
                v = v[..., None]
            out.append(v[:, :, 0, 0].astype(np.float32))   # probe 0 = soma
    return np.concatenate(out, axis=0) if out else None


def style(ax, ylab=None):
    ax.grid(True, alpha=0.55, linewidth=0.5)
    ax.set_axisbelow(True)
    if ylab:
        ax.set_ylabel(ylab)


def main():
    jobs = {v: jobs_for(v) for v in VARIANTS}
    # the ALL_CELLS-mean control was only ever run for the original cell
    meanjobs = (load_jobs(os.path.join(ALL_CELLS, "probescan_exc_p0_56351363",
                                       "predict_royPaula", "CONTROL", "base_jobs.txt"))
                if CELL == DEFAULT_CELL else {})
    truejobs = load_jobs(os.path.join(CTL_ROOT, CELL, "true_jobs.txt"))

    t = np.arange(N_TBIN) * DT_MS
    pages, summary = 0, []

    with PdfPages(OUT_PDF) as pdf:
        for amp in AMPS:
            expF = os.path.join(EXP_DIR, f"Roy{amp}.mlPack1.h5")
            if not os.path.exists(expF):
                continue
            with h5py.File(expF, "r") as hf:
                exp = np.array(hf["raw_volts_mV"])
                stim = np.array(hf["stim_pA"])[0]
                meta = json.loads(hf["meta.JSON"][0])
            stim = stim - np.median(stim)

            sims = {}
            for v in VARIANTS:
                s = load_sim(jobs[v][amp]) if amp in jobs[v] else None
                if s is not None:
                    sims[v] = s[:, :N_TBIN]
            if not sims:
                print(f"Roy{amp}: no NEURON output yet")
                continue

            panels = [(exp, EXP_C, f"measured (model input) · {exp.shape[0]} sweeps · "
                                   f"{spike_label(exp)} spikes")]
            for v in VARIANTS:
                if v in sims:
                    panels.append((sims[v], VAR_C[v],
                                   f"NEURON from predicted params — {v} · "
                                   f"{sims[v].shape[0]} paramsets · "
                                   f"{spike_label(sims[v])} spikes"))
            ctls = {}
            for key, jm, c, lab in [
                    ("mean", meanjobs, MEAN_C,
                     "control: ALL_CELLS mean base params (u=0, not a real cell)"),
                    ("true", truejobs, TRUE_C,
                     f"control: unmodified BBP {CELL} defaults")]:
                a = load_sim(jm[amp]) if amp in jm else None
                if a is not None:
                    a = a[:1, :N_TBIN]
                    ctls[key] = a
                    panels.append((a, c, f"{lab} · {spike_label(a)} spikes"))

            nrow = 1 + len(panels)
            fig, axes = plt.subplots(nrow, 1, figsize=(13, 1.95 * nrow), sharex=True,
                                     gridspec_kw={"height_ratios": [0.6] + [1] * len(panels)})
            axes[0].plot(t, stim, color=STIM_C, lw=0.55)
            style(axes[0], ylab="I inj (pA)")
            axes[0].text(0.995, 0.9, f"stimulus  ·  {meta['Stim']}  ·  peak "
                                     f"{meta['stim_peak_pA']:.0f} pA",
                         transform=axes[0].transAxes, ha="right", va="top",
                         fontsize=8.5, color=INK_2)

            for ax, (arr, c, lab) in zip(axes[1:], panels):
                for w in arr:
                    ax.plot(t, w, color=c, lw=0.55, alpha=0.85)
                ax.legend(handles=[Line2D([0], [0], color=c, lw=2)], labels=[lab],
                          loc="upper right", frameon=False, fontsize=9,
                          handlelength=1.6, borderpad=0.1)
                style(ax, ylab="V (mV)")

            allv = [p[0] for p in panels]
            lo = min(np.nanmin(a) for a in allv) - 5
            hi = max(np.nanmax(a) for a in allv) + 5
            for ax in axes[1:]:
                ax.set_ylim(lo, hi)
            axes[-1].set_xlabel("time (ms)")
            axes[-1].set_xlim(0, t[-1])

            fig.suptitle(f"Roy{amp} — measured trace vs NEURON re-simulation in {CELL}",
                         fontsize=13, y=0.995, color=INK)
            fig.text(0.5, 0.973, f"{CELL} · stim 5k50kInterChaoticB × "
                                 f"{meta['stim_peak_pA']/6822.4:.3f} nA · dt 0.1 ms · "
                                 "shared V scale across trace panels · "
                                 f"spikes = upward crossings of {SPIKE_THRESH:g} mV · "
                                 f"{os.path.basename(MODEL_DIR)}",
                     fontsize=8.5, color=INK_2, ha="center")
            fig.tight_layout(rect=[0, 0, 1, 0.963])
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
            summary.append((amp, spike_label(exp),
                            {v: spike_label(sims[v]) for v in sims},
                            {k: spike_label(a) for k, a in ctls.items()}))

    print(f"wrote {OUT_PDF} ({pages} pages)   cell={CELL}   "
          f"model={os.path.basename(MODEL_DIR)}\n")
    print("%-9s %10s %10s %10s %10s %10s %10s" % ("stim", "measured", "exact",
                                                  "minmax", "default",
                                                  "meanBase", "cellDflt"))
    for amp, e, sv, cv in summary:
        print("%-9s %10s %10s %10s %10s %10s %10s" % (
            f"Roy{amp}", e, sv.get("exact", "-"), sv.get("minmax", "-"),
            sv.get("default", "-"), cv.get("mean", "-"), cv.get("true", "-")))


if __name__ == "__main__":
    main()
