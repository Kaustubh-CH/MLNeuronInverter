#!/usr/bin/env python
"""Distribution of soft-eFEL features vs ion-channel value for the ball-and-stick pack.

Replicates the HybridLoss voltage path:
  * data pack stores voltages in FIXED z-space (VOLT_NORM_MEAN/STD)
  * de-normalize back to raw mV:  v_raw = v_norm * STD + MEAN
  * run toolbox.soft_efel.soft_efel_features on the raw-mV (B, T) tensor

One PDF page per ion channel; each page has a subplot per eFEL feature,
y = feature value, x = that channel's physical (S/cm^2) value.
"""
import os, json
import h5py, numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from toolbox.jaxley_utils import VOLT_NORM_MEAN, VOLT_NORM_STD
from toolbox.soft_efel import soft_efel_features, FEATURES, STRONG_FEATURES

H5  = "/pscratch/sd/k/ktub1999/synthetic_ball_data/ball_synth_v1/ball_and_stick_synth.mlPack1.h5"
OUT = "/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/efel_vs_channel.pdf"
N_SAMPLES = 3000
CHUNK     = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"

f = h5py.File(H5, "r")
meta = json.loads(f["meta.JSON"][0])
parName = meta["parName"]                      # [HH_gNa, HH_gK, HH_gLeak, Leak_gLeak]
units   = [r[2] for r in meta["phys_par_range"]]
dt_ms   = meta["timeAxis"]["step"]

Ntot = f["train_volts_norm"].shape[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(Ntot, size=min(N_SAMPLES, Ntot), replace=False))

phys = f["train_phys_par"][idx]                # (N, 4) physical S/cm^2
volts_norm = f["train_volts_norm"][idx, :, 0, 0].astype(np.float32)   # (N, T) z-space
f.close()

# --- de-normalize to raw mV exactly as HybridLoss does for the eFEL side ---
volts_raw = volts_norm * VOLT_NORM_STD + VOLT_NORM_MEAN
print(f"[{DEV}] N={volts_raw.shape[0]} T={volts_raw.shape[1]}  "
      f"raw-mV range [{volts_raw.min():.1f}, {volts_raw.max():.1f}]")

# --- compute all 11 soft-eFEL features in chunks (memory: (B,nmax,T) tensors) ---
feat = {ff: [] for ff in FEATURES}
with torch.no_grad():
    for c in range(0, volts_raw.shape[0], CHUNK):
        vb = torch.tensor(volts_raw[c:c+CHUNK], dtype=torch.float32, device=DEV)
        d = soft_efel_features(vb, dt_ms=dt_ms, only=None)   # thr=-20 mV default
        for ff in FEATURES:
            feat[ff].append(d[ff].detach().cpu().numpy())
        print(f"  chunk {c//CHUNK+1}/{-(-volts_raw.shape[0]//CHUNK)}", end="\r")
feat = {ff: np.concatenate(v) for ff, v in feat.items()}
print("\nfeature extraction done")

# --- plot: one page per ion channel, subplot per feature ---
nfeat = len(FEATURES)
ncol = 3
nrow = int(np.ceil(nfeat / ncol))

with PdfPages(OUT) as pdf:
    for ch, (cname, cunit) in enumerate(zip(parName, units)):
        x = phys[:, ch]
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4 * nrow))
        axes = np.array(axes).reshape(-1)
        for a, ff in enumerate(FEATURES):
            ax = axes[a]
            y = feat[ff]
            strong = ff in STRONG_FEATURES
            ax.scatter(x, y, s=6, alpha=0.25,
                       color="tab:blue" if strong else "tab:gray")
            # running median trend (20 bins over the channel range)
            bins = np.linspace(x.min(), x.max(), 21)
            bc = 0.5 * (bins[:-1] + bins[1:])
            which = np.digitize(x, bins) - 1
            med = [np.nanmedian(y[which == b]) if np.any(which == b) else np.nan
                   for b in range(20)]
            ax.plot(bc, med, color="crimson", lw=1.6)
            ax.set_title(f"{ff}{' *' if strong else ''}", fontsize=10)
            ax.set_xlabel(f"{cname} ({cunit})", fontsize=8)
            ax.set_ylabel(ff, fontsize=8)
            ax.tick_params(labelsize=7)
        for a in range(nfeat, len(axes)):
            axes[a].axis("off")
        fig.suptitle(f"Soft-eFEL features vs {cname}  —  ball_and_stick_synth "
                     f"(N={len(x)})   [* = STRONG_FEATURES, red = binned median]",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        fig.savefig(OUT.replace(".pdf", f"_{cname}.png"), dpi=90)
        plt.close(fig)
        print(f"page {ch+1}/{len(parName)}: {cname}")

print("saved:", OUT)

# --- correlation summary: which feature responds most to which channel ---
print("\nSpearman-ish |corr| of each feature vs each channel:")
print(f"{'feature':22s} " + " ".join(f"{p:>12s}" for p in parName))
for ff in FEATURES:
    y = feat[ff]
    row = []
    for ch in range(len(parName)):
        x = phys[:, ch]
        m = np.isfinite(y) & np.isfinite(x)
        r = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 3 else np.nan
        row.append(r)
    tag = "*" if ff in STRONG_FEATURES else " "
    print(f"{ff:20s}{tag} " + " ".join(f"{v:>12.3f}" for v in row))
