#!/usr/bin/env python
"""CA3 pyramidal default-parameter voltage trace vs the ca3_synth_v2 dataset.

Defaults (== centers of phys_par_range):
    CA3_g_leak=3.9417e-5, CA3_gbar_na3=0.04, CA3_gkdrbar_kdr=0.01,
    CA3_gkabar_kap=0.04, CA3_gbar_km=5.2e-4, CA3_gkdbar_kd=2.5e-4  (S/cm^2)

Pack norm constants are RECOVERED from the data (resimulate a stored sample and
linear-fit), not taken from jaxley_utils — ball_synth_v1 proved those can be
stale for a given pack. See project_ball_synth_v1_norm_mismatch.
"""
import json
import h5py, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox import JaxleyBridge
from toolbox.jaxley_cells.ca3_pyramidal import _DEFAULTS, PARAM_KEYS

CELL = "ca3_pyramidal"
STIM = "5k50kInterChaoticB"
H5   = "/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_synth_v2/ca3_pyramidal_synth.mlPack1.h5"
OUT  = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/ca3_default_trace.png"

# --- simulate default (true raw mV) ---
p = torch.tensor([[_DEFAULTS[k] for k in PARAM_KEYS]], dtype=torch.float32)
print("default params:", {k: float(_DEFAULTS[k]) for k in PARAM_KEYS})
soma = JaxleyBridge.simulate_batch(p, CELL, STIM).detach().cpu().numpy()[0, 0]
T = soma.shape[0]; dt = 0.1; t = np.arange(T) * dt
print(f"default CA3 soma range [{soma.min():.1f}, {soma.max():.1f}] mV")

# --- recover this pack's normalization from a near-default stored sample ---
f = h5py.File(H5, "r")
u = f["train_unit_par"][:]
j = int(np.linalg.norm(u, axis=1).argmin())
phys_j = np.array(f["train_phys_par"][j], dtype=np.float32)
vnorm_j = f["train_volts_norm"][j, :, 0, 0].astype(np.float64)
resim_j = JaxleyBridge.simulate_batch(torch.tensor(phys_j[None]), CELL, STIM
                                      ).detach().cpu().numpy()[0, 0].astype(np.float64)
Tm = min(len(resim_j), len(vnorm_j))
STD, MEAN = np.polyfit(vnorm_j[:Tm], resim_j[:Tm], 1)   # resim = STD*vnorm + MEAN
resid = np.sqrt((resim_j[:Tm] - (STD*vnorm_j[:Tm] + MEAN))**2).mean()
print(f"recovered pack norm: MEAN={MEAN:.4f} STD={STD:.4f} (fit residual {resid:.3f} mV)")

# --- population de-normalized with the recovered constants ---
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(f["train_volts_norm"].shape[0], size=200, replace=False))
pop = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32) * STD + MEAN
f.close()
Tp = min(T, pop.shape[1]); t = t[:Tp]; soma = soma[:Tp]; pop = pop[:, :Tp]
pop_mean = pop.mean(0)

fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
ax[0].fill_between(t, np.percentile(pop, 5, 0), np.percentile(pop, 95, 0),
                   color="0.82", label="dataset 5-95 pct")
ax[0].fill_between(t, np.percentile(pop, 25, 0), np.percentile(pop, 75, 0),
                   color="0.65", label="dataset 25-75 pct")
ax[0].plot(t, pop_mean, color="0.35", lw=0.9, label="dataset mean")
ax[0].plot(t, soma, color="navy", lw=0.7, label="DEFAULT params")
ax[0].axhline(-20, color="crimson", ls="--", lw=0.7)
ax[0].set_ylabel("V (mV)")
ax[0].set_title(f"CA3 pyramidal at DEFAULT ion-channel values vs ca3_synth_v2 population "
                f"(recovered pack norm MEAN={MEAN:.2f}/STD={STD:.2f})", fontsize=11)
ax[0].legend(loc="upper right", fontsize=8)

ax[1].plot(t, soma, color="navy", lw=0.7)
ax[1].axhline(-20, color="crimson", ls="--", lw=0.7, label="spike thr (-20 mV)")
ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("V (mV)")
ax[1].set_title("CA3 default-parameter trace (soma, true raw mV)\n"
                + "  ".join(f"{k}={_DEFAULTS[k]:.4g}" for k in PARAM_KEYS), fontsize=10)
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print("saved:", OUT)
print(f"dataset(pop) range [{pop.min():.1f}, {pop.max():.1f}] mV, mean-trace "
      f"[{pop_mean.min():.1f}, {pop_mean.max():.1f}]")
print(f"RMS(default - dataset mean) = {np.sqrt(((soma-pop_mean)**2).mean()):.2f} mV "
      f"(pop std avg over time = {pop.std(0).mean():.2f} mV)")
