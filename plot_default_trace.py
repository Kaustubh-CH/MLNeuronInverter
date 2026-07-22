#!/usr/bin/env python
"""Ball-and-stick voltage trace at DEFAULT ion-channel values vs the dataset.

Defaults (NEURON hh, == centers of phys_par_range):
    HH_gNa=0.12, HH_gK=0.036, HH_gLeak=0.0003, Leak_gLeak=0.0001  (S/cm^2)

IMPORTANT: ball_synth_v1 was normalized with MEAN=-73.97, STD=42.65 (recovered
by resimulating a stored sample; residual 0.27 mV). The constants currently in
toolbox/jaxley_utils.py (-60.095/18.95) are DIFFERENT and would give a wrong mV
scale here, so we de-normalize the dataset with the pack's own constants.
"""
import json
import h5py, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox import JaxleyBridge
from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS

# --- pack's ACTUAL generation-time normalization (not jaxley_utils' current) ---
GEN_MEAN, GEN_STD = -73.9724, 42.6517

CELL = "ball_and_stick"
STIM = "5k50kInterChaoticB"
H5   = "/pscratch/sd/k/ktub1999/synthetic_ball_data/ball_synth_v1/ball_and_stick_synth.mlPack1.h5"
OUT  = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/default_trace.png"

# --- simulate at defaults (true raw mV) ---
p = torch.tensor([[_DEFAULTS[k] for k in PARAM_KEYS]], dtype=torch.float32)
v = JaxleyBridge.simulate_batch(p, CELL, STIM).detach().cpu().numpy()[0]   # (n_probes, T)
soma = v[0]; T = soma.shape[0]; dt = 0.1; t = np.arange(T) * dt

# --- dataset population, de-normalized with the pack's own constants ---
f = h5py.File(H5, "r")
Ntot = f["train_volts_norm"].shape[0]
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(Ntot, size=200, replace=False))
pop = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32) * GEN_STD + GEN_MEAN
f.close()
Tp = min(T, pop.shape[1]); t = t[:Tp]; soma = soma[:Tp]; pop = pop[:, :Tp]
pop_mean = pop.mean(0)

fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
ax[0].fill_between(t, np.percentile(pop, 5, 0), np.percentile(pop, 95, 0),
                   color="0.82", label="dataset 5-95 pct")
ax[0].fill_between(t, np.percentile(pop, 25, 0), np.percentile(pop, 75, 0),
                   color="0.65", label="dataset 25-75 pct")
ax[0].plot(t, pop_mean, color="0.35", lw=0.9, label="dataset mean")
ax[0].plot(t, soma, color="navy", lw=0.7, label="DEFAULT params (soma)")
ax[0].axhline(-20, color="crimson", ls="--", lw=0.7)
ax[0].set_ylabel("V (mV)")
ax[0].set_title("Ball-and-stick at DEFAULT ion-channel values vs dataset population "
                "(pack norm MEAN=-73.97/STD=42.65)\n"
                + "  ".join(f"{k}={_DEFAULTS[k]:.4g}" for k in PARAM_KEYS), fontsize=11)
ax[0].legend(loc="upper right", fontsize=8)

ax[1].plot(t, soma, color="navy", lw=0.7, label="soma")
if v.shape[0] > 1:
    ax[1].plot(t, v[1][:Tp], color="darkorange", lw=0.7, alpha=0.8, label="dend")
ax[1].axhline(-20, color="crimson", ls="--", lw=0.7, label="spike thr (-20 mV)")
ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("V (mV)")
ax[1].set_title("Default-parameter trace (recorded probes, true raw mV)", fontsize=11)
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print("saved:", OUT)
print(f"default soma range [{soma.min():.1f}, {soma.max():.1f}] mV")
print(f"dataset(pop) range [{pop.min():.1f}, {pop.max():.1f}] mV, mean-trace "
      f"[{pop_mean.min():.1f}, {pop_mean.max():.1f}]")
print(f"RMS(default - dataset mean) = {np.sqrt(((soma-pop_mean)**2).mean()):.2f} mV "
      f"(pop std avg over time = {pop.std(0).mean():.2f} mV)")
