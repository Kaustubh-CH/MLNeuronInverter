#!/usr/bin/env python
"""Plot 100 voltage traces (raw mV) from the ball-and-stick synthetic pack."""
import json
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox.jaxley_utils import VOLT_NORM_MEAN, VOLT_NORM_STD

H5  = "/pscratch/sd/k/ktub1999/synthetic_ball_data/ball_synth_v1/ball_and_stick_synth.mlPack1.h5"
OUT_OVERLAY = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/voltage_traces_overlay.png"
OUT_GRID    = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/voltage_traces_grid.png"
N = 100

f = h5py.File(H5, "r")
meta = json.loads(f["meta.JSON"][0])
dt = meta["timeAxis"]["step"]
Ntot = f["train_volts_norm"].shape[0]
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(Ntot, size=N, replace=False))
vnorm = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32)   # (N, T)
f.close()

# de-normalize to raw mV (same fixed scale HybridLoss uses)
v = vnorm * VOLT_NORM_STD + VOLT_NORM_MEAN
T = v.shape[1]
t = np.arange(T) * dt

# --- overlay of all 100 ---
fig, ax = plt.subplots(figsize=(15, 6))
for i in range(N):
    ax.plot(t, v[i], lw=0.4, alpha=0.4, color="steelblue")
ax.axhline(-20, color="crimson", ls="--", lw=0.8, label="spike thr (-20 mV)")
ax.set_title(f"{N} ball-and-stick voltage traces (soma, raw mV)")
ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(OUT_OVERLAY, dpi=110)
print("saved:", OUT_OVERLAY)

# --- small-multiples grid of 25 individual traces ---
ng = 25
fig, axes = plt.subplots(5, 5, figsize=(18, 11), sharex=True)
for a, ax in enumerate(axes.reshape(-1)):
    ax.plot(t, v[a], lw=0.5, color="navy")
    ax.axhline(-20, color="crimson", ls="--", lw=0.5)
    ax.set_ylim(v.min() - 5, v.max() + 5)
    ax.tick_params(labelsize=6)
    ax.set_title(f"sample {idx[a]}", fontsize=7)
fig.suptitle(f"25 individual ball-and-stick voltage traces (soma, raw mV)", fontsize=13)
fig.supxlabel("time (ms)"); fig.supylabel("V (mV)")
fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(OUT_GRID, dpi=100)
print("saved:", OUT_GRID)

print(f"raw-mV range across {N} traces: [{v.min():.1f}, {v.max():.1f}] mV")
