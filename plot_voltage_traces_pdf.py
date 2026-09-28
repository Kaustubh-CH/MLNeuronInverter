#!/usr/bin/env python
"""Multi-page PDF: one voltage trace per page (raw mV) for the ball-and-stick pack.

Each page overlays the dataset mean trace (faint gray) behind the individual
trace so how much it deviates from the population is obvious.  The page title
lists the 4 physical parameters that generated it and the per-trace RMS
deviation from the mean.
"""
import json
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from toolbox.jaxley_utils import VOLT_NORM_MEAN, VOLT_NORM_STD

H5  = "/pscratch/sd/k/ktub1999/synthetic_ball_data/ball_synth_v1/ball_and_stick_synth.mlPack1.h5"
OUT = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/voltage_traces_perpage.pdf"
N   = 50                      # pages / traces

f = h5py.File(H5, "r")
meta = json.loads(f["meta.JSON"][0])
dt = meta["timeAxis"]["step"]
parName = meta["parName"]
Ntot = f["train_volts_norm"].shape[0]
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(Ntot, size=N, replace=False))
vnorm = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32)
phys  = f["train_phys_par"][idx]
f.close()

v = vnorm * VOLT_NORM_STD + VOLT_NORM_MEAN      # raw mV, (N, T)
T = v.shape[1]
t = np.arange(T) * dt

mean_trace = v.mean(axis=0)                      # population mean (of the sample)
rms_dev = np.sqrt(((v - mean_trace) ** 2).mean(axis=1))   # per-trace deviation
ymin, ymax = v.min() - 5, v.max() + 5

with PdfPages(OUT) as pdf:
    for i in range(N):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(t, mean_trace, lw=1.0, color="0.7", label="dataset mean (n=%d)" % N)
        ax.plot(t, v[i], lw=0.6, color="navy", label="this trace")
        ax.axhline(-20, color="crimson", ls="--", lw=0.7)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
        pstr = "  ".join(f"{n}={phys[i,j]:.4g}" for j, n in enumerate(parName))
        ax.set_title(f"sample {idx[i]}   RMS dev from mean = {rms_dev[i]:.2f} mV\n{pstr}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)
        print(f"page {i+1}/{N}: sample {idx[i]}  rms_dev={rms_dev[i]:.2f}", end="\r")

print(f"\nsaved: {OUT}")
print(f"RMS deviation from population mean (mV): "
      f"min {rms_dev.min():.2f}, median {np.median(rms_dev):.2f}, max {rms_dev.max():.2f}")
print(f"For scale, population std averaged over time = "
      f"{v.std(axis=0).mean():.2f} mV; mean trace ranges "
      f"[{mean_trace.min():.1f}, {mean_trace.max():.1f}] mV")
