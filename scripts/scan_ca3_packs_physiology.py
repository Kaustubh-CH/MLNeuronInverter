#!/usr/bin/env python
"""Scan every synthetic CA3 data pack for physiological plausibility and compare
with Paula's real recordings.

Per pack: de-normalise stored voltages to mV, then per trace measure resting V,
min/max V, spike count (0 mV upward crossings), spike peak, spike width at half
height, AHP depth, depolarisation-block plateaus, non-finite and out-of-range
values.  Prints a markdown table, writes <out>.md, a per-pack example-trace PDF
and a summary PNG.
"""
import glob, json, os, sys
import numpy as np, h5py, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = "/pscratch/sd/k/ktub1999/synthetic_ca3_data"
EXP = "/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5"
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/model_ladder/physiology/ca3_packs_physiology"
NS = 400                      # traces sampled per pack (strided over the train split)
XM, XS = -60.0951997, 18.95055671   # fixed sim-side norm (toolbox/jaxley_utils.py)

def spike_stats(v, dt):
    """v: (T,) mV.  Returns dict of per-trace physiological measures."""
    T = len(v)
    up = np.where((v[1:] > 0) & (v[:-1] <= 0))[0] + 1
    n = len(up)
    peaks, widths, ahps = [], [], []
    for i, s in enumerate(up):
        e = up[i + 1] if i + 1 < n else T
        seg = v[s:e]
        if len(seg) < 3: continue
        pk = int(np.argmax(seg)); peak = float(seg[pk])
        # width at half height between threshold (-20 mV) and peak
        half = (peak + (-20.0)) / 2.0
        above = np.where(seg > half)[0]
        widths.append(float((above[-1] - above[0] + 1) * dt) if len(above) else np.nan)
        peaks.append(peak)
        win = seg[pk:pk + int(20 / dt)]
        ahps.append(float(win.min()) if len(win) else np.nan)
    rest = float(np.median(v[: int(50 / dt)]))          # first 50 ms (pre-stim / early)
    # depolarisation block: any stretch >= 50 ms continuously above -20 mV
    hi = v > -20.0
    block = False
    run = 0
    for h in hi:
        run = run + 1 if h else 0
        if run * dt >= 50.0: block = True; break
    return dict(rest=rest, vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)), n=n,
                peak=float(np.mean(peaks)) if peaks else np.nan,
                width=float(np.nanmean(widths)) if widths else np.nan,
                ahp=float(np.nanmean(ahps)) if ahps else np.nan, block=block,
                nonfinite=int((~np.isfinite(v)).sum()), oor=int(((v > 60) | (v < -120)).sum()))

def scan_traces(V, dt):
    rows = [spike_stats(v, dt) for v in V]
    A = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    ntr = len(rows)
    return dict(
        n_tr=ntr, rest=np.nanmedian(A["rest"]), rest_iqr=(np.nanpercentile(A["rest"], 25), np.nanpercentile(A["rest"], 75)),
        vmin=np.nanmin(A["vmin"]), vmax=np.nanmax(A["vmax"]),
        silent=100 * np.mean(A["n"] == 0), spk_med=np.nanmedian(A["n"]), spk_p95=np.nanpercentile(A["n"], 95),
        spk_max=np.nanmax(A["n"]), peak=np.nanmedian(A["peak"]), width=np.nanmedian(A["width"]),
        ahp=np.nanmedian(A["ahp"]), block=100 * np.mean(A["block"]),
        nonfinite=100 * np.mean(A["nonfinite"] > 0), oor=100 * np.mean(A["oor"] > 0), per=A)

# ── which models were trained from which pack ────────────────────────────
trained = {}
for out in glob.glob("/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/*/*/*/out") + \
           glob.glob("/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/*/out*") + \
           glob.glob("/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/*/out"):
    sy = os.path.join(out, "sum_train.yaml")
    if not os.path.isfile(sy): continue
    try:
        h5 = yaml.safe_load(open(sy))["train_params"].get("full_h5name", "")
    except Exception:
        continue
    trained.setdefault(os.path.realpath(h5) if h5 else "", []).append(out.split("/tmp_neuInv/")[-1])

# ── packs ─────────────────────────────────────────────────────────────────
packs = sorted(glob.glob(f"{D}/*/*.mlPack1.h5"))
results = []
pdf = PdfPages(OUT + "_examples.pdf"); os.makedirs(OUT + "_pages", exist_ok=True)
for p in packs:
    name = p.split("/")[-2]
    with h5py.File(p, "r") as f:
        m = json.loads(f["meta.JSON"][0])
        X = f["train_volts_norm"]
        N, T, P, S = X.shape
        idx = np.linspace(0, N - 1, min(NS, N)).astype(int)
        v = X[idx].astype(np.float32)                      # (n, T, P, S)
        has_mean = "train_volts_mean" in f
        if has_mean:
            mu = f["train_volts_mean"][idx].astype(np.float32); sd = f["train_volts_std"][idx].astype(np.float32)
    si = m.get("simu_info", {}); cs = si.get("cell_spec", {})
    dt = float(cs.get("_DT", 0.1)); tmax = float(cs.get("_T_MAX", T * dt))
    stims = m.get("stim_names_multi") or si.get("stim_names_multi") or m.get("stim_names") or [""]
    pr = m.get("phys_par_range") or []
    span = si.get("log_halfspan", pr[0][1] if pr else None)
    vary = m.get("num_varied_phys_par", len(pr))
    norm_mode = "per-sample" if has_mean else "fixed"
    if has_mean:
        mv = v * sd[:, None, :, :] + mu[:, None, :, :]
    else:
        mv = v * XS + XM
    # heuristic check that the fixed constants apply: rest should sit near -65 mV
    per_chan = []
    for pi in range(P):
        for sj in range(S):
            st = scan_traces(mv[:, :, pi, sj], dt)
            st["chan"] = f"{m.get('probe_names', ['soma'])[pi] if pi < len(m.get('probe_names', [])) else pi}/{stims[sj] if sj < len(stims) else sj}" \
                if (P > 1 or S > 1) else stims[0]
            per_chan.append(st)
    results.append(dict(name=name, path=p, N=N, T=T, P=P, S=S, dt=dt, tmax=tmax, stims=stims, span=span, vary=vary,
                        norm=norm_mode, vinit=cs.get("_V_INIT"), date=m.get("pack_info", {}).get("date", ""),
                        chans=per_chan, models=trained.get(os.path.realpath(p), [])))
    # example page: 6 traces spanning the spike-count range, first channel; plus other channels if any
    t = np.arange(T) * dt
    nch = P * S
    fig, axes = plt.subplots(nch, 1, figsize=(12, 2.4 * nch + 0.8), squeeze=False)
    for ci, st in enumerate(per_chan):
        pi, sj = divmod(ci, S)
        n_sp = st["per"]["n"]; order = np.argsort(n_sp)
        pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]
        ax = axes[ci][0]
        for k in pick:
            ax.plot(t, mv[k, :, pi, sj], lw=0.7, alpha=0.85, label=f"{int(n_sp[k])} sp")
        ax.axhline(0, color="#999", lw=0.5, ls="--"); ax.axhline(-65, color="#999", lw=0.5, ls=":")
        ax.set_ylabel("mV"); ax.legend(fontsize=7, ncol=6, loc="upper right", frameon=False)
        ax.set_title(f"{name}  [{st['chan']}]  rest {st['rest']:.1f} mV, silent {st['silent']:.0f}%, spikes med/p95 "
                     f"{st['spk_med']:.0f}/{st['spk_p95']:.0f}, peak {st['peak']:.0f} mV, width {st['width']:.2f} ms, "
                     f"block {st['block']:.0f}%, Vmax {st['vmax']:.0f}", fontsize=8.5, loc="left")
    axes[-1][0].set_xlabel("time (ms)")
    fig.suptitle(f"{name}: N={N:,} T={T} P={P} S={S} dt={dt} ms  span=±{span} log10  varied={vary}  norm={norm_mode}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); pdf.savefig(fig); fig.savefig(f"{OUT}_pages/{name}.png", dpi=70); plt.close(fig)

# ── experimental reference ────────────────────────────────────────────────
with h5py.File(EXP, "r") as f:
    raw = f["train_raw_volts_mV"][:].astype(np.float32); fam = f["train_stim_family"][:].astype(str)
exp_all = scan_traces(raw, 0.1)
exp_fam = {fa: scan_traces(raw[fam == fa], 0.1) for fa in sorted(set(fam))}
fig, axes = plt.subplots(len(exp_fam), 1, figsize=(12, 2.4 * len(exp_fam) + 0.8), squeeze=False)
for ax, (fa, st) in zip(axes[:, 0], exp_fam.items()):
    sel = np.where(fam == fa)[0]; n_sp = st["per"]["n"]; order = np.argsort(n_sp)
    for k in order[np.linspace(0, len(order) - 1, 6).astype(int)]:
        ax.plot(np.arange(raw.shape[1]) * 0.1, raw[sel[k]], lw=0.7, alpha=0.85, label=f"{int(n_sp[k])} sp")
    ax.axhline(0, color="#999", lw=0.5, ls="--"); ax.axhline(-65, color="#999", lw=0.5, ls=":")
    ax.set_ylabel("mV"); ax.legend(fontsize=7, ncol=6, loc="upper right", frameon=False)
    ax.set_title(f"EXPERIMENT {fa}: rest {st['rest']:.1f}, silent {st['silent']:.0f}%, spikes med/p95 {st['spk_med']:.0f}/"
                 f"{st['spk_p95']:.0f}, peak {st['peak']:.0f} mV, width {st['width']:.2f} ms, AHP {st['ahp']:.0f}, Vmax {st['vmax']:.0f}",
                 fontsize=8.5, loc="left")
axes[-1][0].set_xlabel("time (ms)"); fig.suptitle("Paula's Roy recordings (train split, raw mV)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96)); pdf.savefig(fig); fig.savefig(f"{OUT}_pages/EXPERIMENT.png", dpi=70); plt.close(fig); pdf.close()

# ── table ─────────────────────────────────────────────────────────────────
lines = []
hdr = "| pack | date | N | stims (channels) | box ±log10 | varied | norm | rest mV | silent % | spikes med / p95 / max | peak mV | width ms | AHP mV | block % | non-finite % | >+60 or <−120 % | models trained |"
lines.append(hdr); lines.append("|" + "---|" * 17)
def row(name, date, N, ch, span, vary, norm, st, models):
    return (f"| {name} | {date} | {N:,} | {ch} | {span} | {vary} | {norm} | {st['rest']:.1f} | {st['silent']:.0f} | "
            f"{st['spk_med']:.0f} / {st['spk_p95']:.0f} / {st['spk_max']:.0f} | {st['peak']:.0f} | {st['width']:.2f} | {st['ahp']:.0f} | "
            f"{st['block']:.0f} | {st['nonfinite']:.1f} | {st['oor']:.1f} | {models} |")
lines.append(row("**EXPERIMENT (Paula, all families)**", "2026-04", raw.shape[0], "Roy100..2000", "—", "—", "raw mV", exp_all, "reference"))
for fa, st in exp_fam.items():
    lines.append(row(f"  exp {fa}", "", int((fam == fa).sum()), fa, "—", "—", "raw mV", st, ""))
for r in results:
    for ci, st in enumerate(r["chans"]):
        lines.append(row(r["name"] if ci == 0 else "", r["date"] if ci == 0 else "", r["N"] if ci == 0 else 0,
                         st["chan"], r["span"], r["vary"], r["norm"], st,
                         (", ".join(sorted(set(x.split("/")[1] if "/" in x else x for x in r["models"])))[:80] if ci == 0 else "")))
open(OUT + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT}.md and {OUT}_examples.pdf")
