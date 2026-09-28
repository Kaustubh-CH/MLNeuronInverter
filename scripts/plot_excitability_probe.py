#!/usr/bin/env python3
"""Figure for the L5 nc2 excitability probe (scripts/l5_excitability_probe.py) vs the Roy recordings.

  A  input resistance: recordings (per-trace V~I fit, Roy500-2000) vs each model variant's
     BBP-default cell (-0.1 nA step, x the variant's stim scale = MOhm per RIG nA)
  B  spikes/trace vs Roy amplitude: recordings (mean, IQR) vs model box mean per variant,
     plus the best-matching box draw at c=3 and c=4
  C  Roy1000 / Roy2000 traces: one recorded neuron vs the default cell at x1 / x3 / x4
  python scripts/plot_excitability_probe.py <probe_dir> <out.png>
"""
import sys, glob, collections
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rin_fit import fit_rin

D, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
EXP = "/pscratch/sd/k/ktub1999/RoyExpPack_l5dt02/RoyExpChaotic.mlPack1.h5"
AMPS = [500, 1000, 1500, 2000]
EX_NEURON = "20260330_ch2_c1"

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/g*/probe_rows.csv"))])
tr = {}
for f in sorted(glob.glob(f"{D}/g*/probe_traces.npz")):
    z = np.load(f)
    tr.update({k: z[k] for k in z.files if "__" in k})

# ── recordings ──
dR, dS, ex = [], collections.defaultdict(list), {}
with h5py.File(EXP, "r") as f:
    for dom in ["train", "valid", "test"]:
        V = f[dom + "_raw_volts_mV"][:].astype(float); I = f[dom + "_stim_pA"][:].astype(float) / 1e3
        fam = f[dom + "_stim_family"][:].astype(str); nid = f[dom + "_neuron_id"][:].astype(str)
        for k in range(len(V)):
            if fam[k] == "Roy100":
                continue
            r = fit_rin(V[k], I[k], 0.1)
            dR.append(r["R"]); dS[int(fam[k][3:])].append(r["n_spk"])
            if nid[k] == EX_NEURON and fam[k] not in ex:
                ex[fam[k]] = V[k]

variants = ["base", "g1e5", "g3e6", "cm1", "g1e5cm1", "g3e6cm1", "g1e6cm1", "c2", "c3", "c4"]
col = {"base": "k", "g1e6cm1": "tab:purple", "c2": "tab:orange", "c3": "tab:red", "c4": "tab:brown"}
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

ax = fig.add_subplot(gs[0, 0])
st = df[(df.stim == "step_m0p1_5k") & (df.default == 1)].set_index("variant")
ax.axhspan(np.percentile(dR, 25), np.percentile(dR, 75), color="tab:green", alpha=0.18,
           label=f"recordings IQR (median {np.median(dR):.0f})")
ax.axhline(np.median(dR), color="tab:green")
x = np.arange(len(variants))
ax.bar(x, [st.loc[v, "R_step"] for v in variants],
       color=[col.get(v, "tab:gray") for v in variants])
ax.set_xticks(x); ax.set_xticklabels(variants, rotation=45, ha="right")
ax.set_ylabel("input resistance (MOhm per rig nA)")
ax.set_title("A  input resistance: default cell vs recordings")
ax.legend(fontsize=8, loc="upper left")

ax = fig.add_subplot(gs[0, 1])
m = np.array([np.mean(dS[a]) for a in AMPS])
lo = np.array([np.percentile(dS[a], 25) for a in AMPS]); hi = np.array([np.percentile(dS[a], 75) for a in AMPS])
ax.fill_between(AMPS, lo, hi, color="tab:green", alpha=0.18)
ax.plot(AMPS, m, "o-", color="tab:green", lw=2.5, label="recordings (mean, IQR)")
roy = df[df.stim.str.startswith("Roy")].copy(); roy["amp"] = roy.stim.str.extract(r"Roy(\d+)")[0].astype(int)
piv = roy.pivot_table(index=["variant", "cell"], columns="amp", values="n_spk")
for v in ["base", "g1e6cm1", "c2", "c3", "c4"]:
    ax.plot(AMPS, piv.loc[v].iloc[1:].mean().values, "s--", color=col[v], label=f"{v} box mean")
for v in ["c3", "c4"]:
    box = piv.loc[v].iloc[1:]
    b = np.sqrt(((box - pd.Series(dict(zip(AMPS, m)))) ** 2).mean(1)).idxmin()
    ax.plot(AMPS, piv.loc[v].loc[b].values, ":", color=col[v], lw=2, label=f"{v} best draw #{b}")
ax.set_xlabel("Roy amplitude"); ax.set_ylabel("spikes / trace"); ax.legend(fontsize=7)
ax.set_title("B  f-I: recordings vs model box (32 draws)")

ax = fig.add_subplot(gs[0, 2])
fr = roy[roy.default == 0].groupby(["variant", "amp"]).n_spk.apply(lambda s: (s > 0).mean()).unstack()
for v in ["base", "g1e6cm1", "c2", "c3", "c4"]:
    ax.plot(AMPS, fr.loc[v].values, "o-", color=col[v], label=v)
ax.set_ylim(0, 1.05); ax.set_xlabel("Roy amplitude"); ax.set_ylabel("fraction of box draws that fire")
ax.set_title("C  how much of the training box is supra-threshold"); ax.legend(fontsize=8)

for j, a in enumerate([1000, 2000]):
    ax = fig.add_subplot(gs[1, j])
    t_rec = 100 + np.arange(4000) * 0.1
    if f"Roy{a}" in ex:
        ax.plot(t_rec, ex[f"Roy{a}"], color="tab:green", lw=0.9, label=f"recording {EX_NEURON}")
    for v in ["base", "c3", "c4"]:
        k = f"{v}__Roy{a}_icaRec_5k"
        if k in tr:
            V = tr[k][0]; t = np.arange(len(V)) * 0.2
            ax.plot(t, V, color=col[v], lw=0.8, alpha=0.85, label=f"{v} default cell")
    ax.set_xlim(100, 500); ax.set_xlabel("ms"); ax.set_ylabel("mV")
    ax.set_title(f"{'D' if j == 0 else 'E'}  Roy{a}: recording vs default L5 nc2"); ax.legend(fontsize=7, loc="upper right")

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
txt = ["Roy_N == 5k50kInterChaoticB x N/4022 (corr 0.9993)",
       "   -> ICB x1.5 (training) == 'Roy6000'", "",
       f"recordings: R_in median {np.median(dR):.0f} MOhm, "
       f"spikes {', '.join(f'{x:.1f}' for x in m)}",
       f"model (BBP default, x1): R_in {st.loc['base','R_step']:.0f} MOhm",
       f"dendritic leak/10 + cm 1: R_in {st.loc['g1e6cm1','R_step']:.0f} MOhm (Ih + soma/axon cap it)",
       f"stim x3 (== 1/3 membrane area): {st.loc['c3','R_step']:.0f} MOhm per rig nA",
       f"stim x4: {st.loc['c4','R_step']:.0f} MOhm per rig nA"]
ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9)
fig.suptitle("L5 nc2 excitability vs Paula's Roy recordings (dt 0.2, fp64, 32 box draws + BBP default)")
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print("wrote", OUT)
