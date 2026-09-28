#!/usr/bin/env python
"""Characterize voltage-trace variation across the ball-and-stick synthetic dataset."""
import h5py, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H5 = "/pscratch/sd/k/ktub1999/synthetic_ball_data/ball_synth_v1/ball_and_stick_synth.mlPack1.h5"
OUT = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/ball_variation.png"

f = h5py.File(H5, "r")
meta = json.loads(f["meta.JSON"][0])
dt = meta["timeAxis"]["step"]
parName = meta["parName"]

V = f["train_volts_norm"]          # (40000, 5001, 1, 1) fp16
N = V.shape[0]
T = V.shape[1]
t = np.arange(T) * dt

# Load a random subset into RAM to keep it light
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(N, size=4000, replace=False))
volts = V[idx, :, 0, 0].astype(np.float32)   # (4000, 5001)
phys = f["train_phys_par"][idx]               # (4000, 4)

# --- per-timepoint statistics across samples ---
mean_t = volts.mean(axis=0)
std_t  = volts.std(axis=0)
p05 = np.percentile(volts, 5, axis=0)
p95 = np.percentile(volts, 95, axis=0)
p25 = np.percentile(volts, 25, axis=0)
p75 = np.percentile(volts, 75, axis=0)

# --- scalar variation metrics ---
# mean pairwise variability: average std across time; and dispersion of each trace
per_trace_range = volts.max(axis=1) - volts.min(axis=1)
avg_std = std_t.mean()

print(f"Subset: {volts.shape[0]} traces x {T} timepoints ({t[-1]:.0f} ms)")
print(f"Global voltage(norm) range: [{volts.min():.2f}, {volts.max():.2f}]")
print(f"Mean across-sample std (per timepoint), averaged over time: {avg_std:.3f}")
print(f"Median cross-sample std at a timepoint: {np.median(std_t):.3f}")
print(f"Max cross-sample std at a timepoint:    {std_t.max():.3f}")
print(f"Per-trace peak-to-peak (norm): mean {per_trace_range.mean():.2f}, "
      f"min {per_trace_range.min():.2f}, max {per_trace_range.max():.2f}")

fig, ax = plt.subplots(2, 2, figsize=(16, 10))

# (0,0) overlay of 200 random traces
ax0 = ax[0, 0]
for i in range(200):
    ax0.plot(t, volts[i], lw=0.3, alpha=0.25, color="steelblue")
ax0.set_title("200 random normalized voltage traces (soma)")
ax0.set_xlabel("time (ms)"); ax0.set_ylabel("V (norm, mean0/std1)")

# (0,1) mean +/- envelope
ax1 = ax[0, 1]
ax1.fill_between(t, p05, p95, color="orange", alpha=0.3, label="5-95 pct")
ax1.fill_between(t, p25, p75, color="darkorange", alpha=0.4, label="25-75 pct")
ax1.plot(t, mean_t, color="black", lw=0.8, label="mean")
ax1.set_title("Cross-sample envelope per timepoint")
ax1.set_xlabel("time (ms)"); ax1.set_ylabel("V (norm)"); ax1.legend(loc="upper right")

# (1,0) cross-sample std over time
ax2 = ax[1, 0]
ax2.plot(t, std_t, color="crimson", lw=0.6)
ax2.axhline(avg_std, color="k", ls="--", lw=0.8, label=f"mean std={avg_std:.3f}")
ax2.set_title("Across-sample std vs time (higher = more variation)")
ax2.set_xlabel("time (ms)"); ax2.set_ylabel("std across samples"); ax2.legend()

# (1,1) per-trace peak-to-peak histogram
ax3 = ax[1, 1]
ax3.hist(per_trace_range, bins=60, color="seagreen", alpha=0.8)
ax3.set_title("Per-trace peak-to-peak amplitude (norm units)")
ax3.set_xlabel("max - min of each trace"); ax3.set_ylabel("count")

fig.suptitle("Ball-and-stick synthetic dataset — voltage-trace variation (train split, 4000-trace subset)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=110)
print("saved:", OUT)

# --- parameter coverage summary ---
print("\nPhysical parameter ranges in subset:")
for j, nm in enumerate(parName):
    print(f"  {nm:12s}: [{phys[:,j].min():.4g}, {phys[:,j].max():.4g}]  "
          f"mean {phys[:,j].mean():.4g}")
f.close()
