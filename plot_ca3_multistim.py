#!/usr/bin/env python
"""CA3 pyramidal multi-stim per-page PDF.

For N samples drawn from ca3_synth_v2, simulate the CA3 soma response under the
canonical 3-stim set (chaotic / step500 / chirp) and write one PDF page per
sample, with a subplot per stim (true raw mV, so no normalization constants).

CA3 is single-compartment (soma only), so "multi-probe" is not possible; this
shows the same cell/params under multiple stimuli instead.
"""
import sys, json
import h5py, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from toolbox import JaxleyBridge
from toolbox.jaxley_cells.ca3_pyramidal import PARAM_KEYS

CELL  = "ca3_pyramidal"
STIMS = ["5k50kInterChaoticB", "5k0step_500", "5k0chirp"]
LABELS = ["chaotic", "step500", "chirp"]
H5    = "/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_synth_v2/ca3_pyramidal_synth.mlPack1.h5"
OUT   = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/synth_plots/ca3_multistim_perpage.pdf"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
dt = 0.1

# --- draw N sample parameter sets from the pack ---
f = h5py.File(H5, "r")
Ntot = f["train_phys_par"].shape[0]
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(Ntot, size=N, replace=False))
phys = np.array(f["train_phys_par"][idx], dtype=np.float32)   # (N, 6)
f.close()
p = torch.from_numpy(phys)

# --- simulate all N under each stim (one vmapped batch per stim) ---
sims = {}
for s in STIMS:
    v = JaxleyBridge.simulate_batch(p, CELL, s).detach().cpu().numpy()[:, 0, :]  # (N, T)
    sims[s] = v
    print(f"[sim] {s}: shape {v.shape}  range [{v.min():.1f}, {v.max():.1f}] mV", flush=True)

# global y-limits per stim (shared across pages so pages are comparable)
ylim = {s: (sims[s].min() - 5, sims[s].max() + 5) for s in STIMS}

with PdfPages(OUT) as pdf:
    for i in range(N):
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=False)
        for a, (s, lab) in enumerate(zip(STIMS, LABELS)):
            v = sims[s][i]
            t = np.arange(len(v)) * dt
            axes[a].plot(t, v, lw=0.6, color="navy")
            axes[a].axhline(-20, color="crimson", ls="--", lw=0.6)
            axes[a].set_ylim(*ylim[s])
            axes[a].set_ylabel("V (mV)")
            axes[a].set_title(f"stim: {lab} ({s})", fontsize=10, loc="left")
        axes[-1].set_xlabel("time (ms)")
        pstr = "  ".join(f"{k}={phys[i,j]:.3g}" for j, k in enumerate(PARAM_KEYS))
        fig.suptitle(f"CA3 sample {idx[i]}   (soma, raw mV)\n{pstr}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  page {i+1}/{N}", flush=True)

print("saved:", OUT)
