#!/usr/bin/env python
"""Plot 100 voltage traces (raw mV) from the ca3_synth_v2 dataset."""
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ca3_synth_v2 pack's own recovered normalization (NOT jaxley_utils; residual 0.002 mV)
GEN_MEAN, GEN_STD = -62.0714, 14.9950

H5  = "/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_synth_v2/ca3_pyramidal_synth.mlPack1.h5"
OUT_OVERLAY = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/synth_plots/ca3_traces_overlay.png"
OUT_GRID    = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/synth_plots/ca3_traces_grid.png"
N = 100

f = h5py.File(H5, "r")
dt = 0.1
Ntot = f["train_volts_norm"].shape[0]
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(Ntot, size=N, replace=False))
v = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32) * GEN_STD + GEN_MEAN
f.close()
T = v.shape[1]; t = np.arange(T) * dt

# --- overlay of all 100 ---
fig, ax = plt.subplots(figsize=(15, 6))
for i in range(N):
    ax.plot(t, v[i], lw=0.4, alpha=0.4, color="teal")
ax.axhline(-20, color="crimson", ls="--", lw=0.8, label="spike thr (-20 mV)")
ax.set_title(f"{N} CA3 pyramidal voltage traces (soma, raw mV) — ca3_synth_v2")
ax.set_xlabel("time (ms)"); ax.set_ylabel("V (mV)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(OUT_OVERLAY, dpi=110)
print("saved:", OUT_OVERLAY)

# --- small-multiples grid of 25 ---
fig, axes = plt.subplots(5, 5, figsize=(18, 11), sharex=True)
for a, ax in enumerate(axes.reshape(-1)):
    ax.plot(t, v[a], lw=0.5, color="darkslategray")
    ax.axhline(-20, color="crimson", ls="--", lw=0.5)
    ax.set_ylim(v.min() - 5, v.max() + 5)
    ax.tick_params(labelsize=6)
    ax.set_title(f"sample {idx[a]}", fontsize=7)
fig.suptitle("25 individual CA3 pyramidal voltage traces (soma, raw mV)", fontsize=13)
fig.supxlabel("time (ms)"); fig.supylabel("V (mV)")
fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(OUT_GRID, dpi=100)
print("saved:", OUT_GRID)

print(f"raw-mV range across {N} traces: [{v.min():.1f}, {v.max():.1f}] mV")
