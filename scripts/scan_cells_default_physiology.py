#!/usr/bin/env python
"""Cross-CELL comparison: every registered jaxley cell simulated at its DEFAULT
parameters under one stimulus battery (steps, ramp, two chaotic stims), plus a
physiology scan of the existing datasets of the non-CA3 cells.

Outputs <out>.md (table), <out>_examples.pdf (page 1: cells x stims default
traces; then one page per dataset), <out>_pages/*.png.
Run on CPU: JAX_PLATFORMS=cpu JAX_ENABLE_X64=true.
"""
import glob, json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "true")
import numpy as np, h5py, torch
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.getcwd())
from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import load_stim_csv
from toolbox.physio_stats import spike_stats, scan_traces, verdict_default, verdict_box

def default_flat(cell_name):
    """The cell's OWN built defaults (cell.get_parameters() after build_fn), one
    value per CNN index; multi-compartment entries are averaged.  This is what
    jaxley simulates when no override is given."""
    spec = jaxley_cells.get(cell_name)
    cell, idx = spec.build_fn()
    P = len(spec.param_keys); row = [None] * P
    for e, i in zip(cell.get_parameters(), idx):
        (k,) = e.keys()
        if row[i] is None: row[i] = float(np.asarray(e[k]).mean())
    return torch.tensor([row], dtype=torch.float64)

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/model_ladder/physiology/cells_default_physiology"
CELLS = sys.argv[2].split(",") if len(sys.argv) > 2 else \
    ["single_comp", "ball_and_stick", "ball_and_stick_bbp", "ca3_pyramidal", "l5ttpc", "L5PC_jaxley"]
STIMS = ["5k0step_200", "5k0step_500", "BBP_Exp_Step1000", "5k0ramp", "5k50kInterChaoticB", "5kChaoticRamp"]
XM, XS = -60.0951997, 18.95055671
DATASETS = {   # cell -> packs
    "ca3_pyramidal": ["/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_5kinterchaoticB_v1/ca3_pyramidal_synth.mlPack1.h5",
                      "/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v3/ca3_pyramidal_synth.mlPack1.h5"],
    "ball_and_stick": sorted(glob.glob("/pscratch/sd/k/ktub1999/synthetic_ball_data/*/*.mlPack1.h5")),
    "ball_and_stick_bbp": sorted(glob.glob("/pscratch/sd/k/ktub1999/synthetic_bbp_data/*/*.mlPack1.h5")),
    "l5ttpc (ncomp=2 packs)": ["/pscratch/sd/k/ktub1999/l5ttpc_jaxley_nc2_data/L5TTPC_jaxley_nc2.mlPack1.h5",
                               "/pscratch/sd/k/ktub1999/l5ttpc_multiprobe_data/L5TTPC_multiprobe.mlPack1.h5",
                               "/pscratch/sd/k/ktub1999/l5ttpc_multistim_data/L5TTPC_multistim.mlPack1.h5"],
    "model ladder (2026-09)": sorted(glob.glob(os.environ.get("LADDER_DATA_ROOT", "/global/homes/k/ktub1999/model_ladder_data") + "/*/*.mlPack1.h5")),
}
if os.environ.get("LADDER_ONLY"):        # just the new ladder packs
    DATASETS = {"model ladder (2026-09)": DATASETS["model ladder (2026-09)"]}

pdf = PdfPages(OUT + "_examples.pdf"); os.makedirs(OUT + "_pages", exist_ok=True)
lines = ["| cell | params | probes | ncomp | stim | stim scale | max I (nA) | rest mV | spikes | peak mV | width ms | AHP mV | block | Vmin/Vmax | verdict | sim s |",
         "|" + "---|" * 16]
# ── 1. default cells x stimulus battery ────────────────────────────────────
traces = {}
stim_dir = jaxley_cells.get("ca3_pyramidal").stim_dir
imax = {s: float(np.abs(load_stim_csv(stim_dir / f"{s}.csv")).max()) for s in STIMS}
for cell in CELLS:
    spec = jaxley_cells.get(cell)
    p = default_flat(cell)
    if os.environ.get("L5_BBP_DEFAULTS") and cell in ("l5ttpc", "L5TTPC", "L5PC_jaxley"):
        # BBP biophysics.hoc values (bench_jaxley_cells reference row) instead of the
        # placeholder 1e-5 defaults the cell file builds with.
        from toolbox.tests.bench_jaxley_cells import _default_params_tensor
        p = _default_params_tensor("L5TTPC", 1, torch.float64)
        print(f"[{cell}] using BBP biophysics.hoc defaults", flush=True)
    print(f"[{cell}] defaults: " + ", ".join(f"{k}={v:.4g}" for k, v in zip(spec.param_keys, p[0].tolist())), flush=True)
    ncomp = os.environ.get("L5TTPC_NCOMP", "4") if "l5" in cell.lower() else ("2 (soma+stick)" if "ball" in cell else 1)
    scale = float(getattr(spec, "stim_scale", 1.0))
    for s in STIMS:
        t0 = time.time()
        try:
            h = JaxleyBridge.get_handle(cell, s)
            v = np.asarray(h.simulate_batch(jnp.asarray(p.numpy())))[0]        # (n_rec, T)
            dtms = spec.dt * h.downsample_step
            soma = v[0]; el = time.time() - t0
            st = spike_stats(soma, dtms)
            traces[(cell, s)] = (np.arange(len(soma)) * dtms, soma, st)
            vd = verdict_default(st)
            lines.append(f"| {cell} | {len(spec.param_keys)} | {v.shape[0]} | {ncomp} | {s} | {scale:g} | {imax[s]*scale:.2f} | {st['rest']:.1f} | {st['n']} | "
                         f"{st['peak']:.0f} | {st['width']:.2f} | {st['ahp']:.0f} | {'YES' if st['block'] else 'no'} | "
                         f"{st['vmin']:.0f}/{st['vmax']:.0f} | {'PASS' if not vd else 'FAIL: ' + ', '.join(vd)} | {el:.0f} |")
            print(lines[-1], flush=True)
        except Exception as ex:
            lines.append(f"| {cell} | | | {ncomp} | {s} | | {imax[s]:.2f} | ERROR {str(ex)[:60]} |" + " |" * 8); print(lines[-1], flush=True)
# page 1: grid
nc, ns = len(CELLS), len(STIMS)
fig, axes = plt.subplots(nc, ns, figsize=(3.6 * ns, 2.1 * nc + 0.8), squeeze=False, sharex=True)
for i, cell in enumerate(CELLS):
    for j, s in enumerate(STIMS):
        ax = axes[i][j]
        if (cell, s) in traces:
            t, v, st = traces[(cell, s)]
            ax.plot(t, v, lw=0.7, color="#2a78d6"); ax.axhline(0, color="#999", lw=0.4, ls="--")
            ax.text(0.02, 0.95, f"{st['n']} sp, rest {st['rest']:.0f}, peak {st['peak']:.0f}{', BLOCK' if st['block'] else ''}",
                    transform=ax.transAxes, va="top", fontsize=7)
        else:
            ax.text(0.5, 0.5, "failed", ha="center", transform=ax.transAxes)
        if i == 0: ax.set_title(s, fontsize=8.5)
        sc = float(getattr(jaxley_cells.get(cell), "stim_scale", 1.0))
        ax.text(0.98, 0.95, f"x{sc:g}: max |I| {imax[s]*sc:.2f} nA", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#555")
        if j == 0: ax.set_ylabel(f"{cell}\nmV", fontsize=8.5)
        ax.tick_params(labelsize=7)
for j in range(ns): axes[-1][j].set_xlabel("time (ms)", fontsize=8)
fig.suptitle("Every registered jaxley cell at its DEFAULT parameters, soma voltage under one stimulus battery (fp64, CPU)",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.965)); pdf.savefig(fig); fig.savefig(f"{OUT}_pages/00_default_cells.png", dpi=80); plt.close(fig)

# ── 2. datasets of the other cells ─────────────────────────────────────────
lines += ["", "| dataset | cell | N | T | probes×stims | box ±log10 | stim scale | ncomp | norm used | rest mV | silent % | spikes med / p95 / max | peak mV | width ms | AHP mV | block % | non-finite % | out-of-range % | verdict |",
          "|" + "---|" * 19]
for cell, packs in ({} if os.environ.get("SKIP_DATASETS") else DATASETS).items():
    for pth in packs:
        if not os.path.isfile(pth): continue
        name = "/".join(pth.split("/")[-2:])
        with h5py.File(pth, "r") as f:
            m = json.loads(f["meta.JSON"][0]); X = f["train_volts_norm"]; N, T, P, S = X.shape
            idx = np.linspace(0, N - 1, min(300, N)).astype(int); v = X[idx].astype(np.float32)
            has_mean = "train_volts_mean" in f
            if has_mean: mu = f["train_volts_mean"][idx].astype(np.float32); sd = f["train_volts_std"][idx].astype(np.float32)
        si = m.get("simu_info", {}); cs = si.get("cell_spec", {})
        # trace grid = meta.timeAxis.step (= max(_DT, _DT_STIM)); _DT alone is wrong when the
        # solver ran finer than the 10 kHz stim grid (CA3) and the bridge decimated the output.
        dt = float((m.get("timeAxis") or {}).get("step") or max(float(cs.get("_DT", 0.1)), float(cs.get("_DT_STIM", 0.1))))
        nm = m.get("norm") or {}
        if has_mean:
            mv = v * sd[:, None] + mu[:, None]; used = "per-sample"
        elif abs(float(v.mean(axis=1).mean())) < 0.02 and abs(float(v.std(axis=1).mean()) - 1.0) < 0.02:
            # every trace has mean 0 / std 1 -> an old per-sample z-scored pack whose
            # mean/std were never stored; mV cannot be recovered.  Report in z-units.
            mv = v; used = "per-sample z (mV NOT recoverable)"
        else:
            xm, xs = float(nm.get("volts_xm", XM)), float(nm.get("volts_xs", XS)); mv = v * xs + xm; used = f"fixed {xm:.1f}/{xs:.1f}"
            r = float(np.median(mv[:, : int(50 / dt), 0, 0]))
            if not -95 < r < -45:      # wrong constants -> try the May-2026 ball constants
                mv2 = v * 42.65 - 73.97; r2 = float(np.median(mv2[:, : int(50 / dt), 0, 0]))
                if -95 < r2 < -45: mv, used = mv2, "fixed -73.97/42.65 (ball v1)"
        span = si.get("log_halfspan", (m.get("phys_par_range") or [[0, "?"]])[0][1])
        sscale = cs.get("_STIM_SCALE", 1.0); ncomp_m = cs.get("_NCOMP")
        probes = m.get("probe_names", ["soma"]); stims = m.get("stim_names_multi") or si.get("stim_names_multi") or m.get("stim_names") or ["?"]
        fig, axes = plt.subplots(P * S, 1, figsize=(12, 2.3 * P * S + 0.8), squeeze=False)
        for ci in range(P * S):
            pi, sj = divmod(ci, S); st = scan_traces(mv[:, :, pi, sj], dt)
            ch = f"{probes[pi] if pi < len(probes) else pi}/{stims[sj] if sj < len(stims) else sj}"
            vb = verdict_box(st) if "NOT recoverable" not in used else ["z-units only"]
            lines.append(f"| {name if ci == 0 else ''} | {cell if ci == 0 else ''} | {N:,} | {T} | {ch} | {span} | {sscale:g} | {ncomp_m or ''} | {used} | {st['rest']:.1f} | "
                         f"{st['silent']:.0f} | {st['spk_med']:.0f} / {st['spk_p95']:.0f} / {st['spk_max']:.0f} | {st['peak']:.0f} | "
                         f"{st['width']:.2f} | {st['ahp']:.0f} | {st['block']:.0f} | {st['nonfinite']:.1f} | {st['oor']:.1f} | {'PASS' if not vb else 'FAIL: ' + ', '.join(vb)} |")
            n_sp = st["per"]["n"]; order = np.argsort(n_sp); ax = axes[ci][0]
            for k in order[np.linspace(0, len(order) - 1, 6).astype(int)]:
                ax.plot(np.arange(T) * dt, mv[k, :, pi, sj], lw=0.7, alpha=0.85, label=f"{int(n_sp[k])} sp")
            ax.axhline(0, color="#999", lw=0.5, ls="--"); ax.set_ylabel("mV"); ax.legend(fontsize=7, ncol=6, loc="upper right", frameon=False)
            ax.set_title(f"{name} [{ch}] rest {st['rest']:.1f}, silent {st['silent']:.0f}%, spikes med/p95 {st['spk_med']:.0f}/{st['spk_p95']:.0f}, "
                         f"peak {st['peak']:.0f}, width {st['width']:.2f}, block {st['block']:.0f}%, Vmin/max {st['vmin']:.0f}/{st['vmax']:.0f}", fontsize=8.5, loc="left")
        axes[-1][0].set_xlabel("time (ms)")
        fig.suptitle(f"{name} ({cell}): N={N:,} T={T} P={P} S={S} box ±{span} norm={used}", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96)); pdf.savefig(fig); fig.savefig(f"{OUT}_pages/{name.replace('/', '__')}.png", dpi=70); plt.close(fig)
        print(f"scanned {name}", flush=True)
pdf.close()
open(OUT + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nwrote {OUT}.md, {OUT}_examples.pdf, {OUT}_pages/")
