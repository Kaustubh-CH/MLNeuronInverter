#!/usr/bin/env python
"""Stimulus-amplitude sweep for one jaxley cell: which `stim_scale` makes the
cell's response PHYSIOLOGICAL under each stimulus, both at the default
parameters and across the +-log_halfspan parameter box the packs sample?

For every (stim, scale):
  * simulate the default cell and NBOX uniform draws from the unit box,
  * measure rest / spikes / peak / width / AHP / block / Vmin / Vmax
    (toolbox.physio_stats, the same definitions the pack scans use),
  * PASS/FAIL against docs/model_ladder/README.md criteria.

Writes <out>.md (table), <out>.json (all numbers), <out>.png (rows = scales,
cols = stims; default trace bold + box draws spanning the spike-count range).

    python scripts/stim_scale_scan.py --cell single_comp --scales 0.02,0.05,0.1 \
        --stims 5k50kInterChaoticB,5kChaoticRamp --nbox 128 --out docs/model_ladder/scan_single_comp

Env: JAX_PLATFORMS (cuda|cpu), JAX_ENABLE_X64=true, L5TTPC_NCOMP for l5ttpc.
"""
import argparse, json, os, sys, time
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox import JaxleyBridge, jaxley_cells
from toolbox.jaxley_utils import load_stim_csv, build_phys_par_range, phys_par_range_to_arrays, unit_to_phys_np, phys_par_range_linear_mask
from toolbox.physio_stats import spike_stats, scan_traces, verdict_default, verdict_box


def cell_module(name):
    import importlib
    mod_name = {"single_comp": "soma_only", "L5TTPC": "l5ttpc", "L5_TTPC1cADpyr0": "l5ttpc"}.get(name, name)
    return importlib.import_module(f"toolbox.jaxley_cells.{mod_name}")


def default_row(mod):
    return np.array([float(mod._DEFAULTS[k]) for k in mod.PARAM_KEYS], dtype=np.float64)


def simulate(handle, phys, batch):
    """Forward-only (no VJP graph) on the jitted vmap, chunked + padded to `batch`."""
    import jax.numpy as jnp
    outs = []
    for i0 in range(0, len(phys), batch):
        pg = phys[i0:i0 + batch]; ng = len(pg)
        if ng < batch:
            pg = np.concatenate([pg, np.repeat(pg[:1], batch - ng, axis=0)], axis=0)
        v = handle.simulate_batch(jnp.asarray(pg))
        outs.append(np.asarray(v[:ng]))
    return np.concatenate(outs, axis=0)          # (N, n_rec, T)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--stims", default="5k50kInterChaoticB,5kChaoticRamp")
    ap.add_argument("--scales", required=True, help="comma list of stim multipliers")
    ap.add_argument("--nbox", type=int, default=128, help="uniform draws from the unit box (0 = defaults only)")
    ap.add_argument("--halfspan", type=float, default=0.5, help="log10 half-span of the box")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    stims = [s for s in a.stims.split(",") if s]
    scales = [float(x) for x in a.scales.split(",")]
    mod = cell_module(a.cell); spec = jaxley_cells.get(a.cell)
    P = len(spec.param_keys); ncomp = getattr(mod, "_NCOMP", None)
    centers = default_row(mod)
    ppr = build_phys_par_range(mod, a.halfspan)          # per-key overrides (l5 cm / e_pas)
    box_c, box_s = phys_par_range_to_arrays(ppr)
    rng = np.random.default_rng(a.seed)
    unit = rng.uniform(-1, 1, size=(a.nbox, P)) if a.nbox > 0 else np.zeros((0, P))
    phys = np.concatenate([centers[None], unit_to_phys_np(unit, box_c.astype(np.float64), box_s.astype(np.float64), phys_par_range_linear_mask(ppr))], axis=0)
    print("[scan] box:", "; ".join(f"{k}: {c:.4g} x10^+-{s:.3g}" for k, c, s in zip(spec.param_keys, box_c, box_s)), flush=True)
    imax = {s: float(np.abs(load_stim_csv(spec.stim_dir / f"{s}.csv")).max()) for s in stims}
    print(f"[scan] cell={a.cell} P={P} ncomp={ncomp} stims={stims} scales={scales} nbox={a.nbox} "
          f"halfspan={a.halfspan} platform={os.environ.get('JAX_PLATFORMS')}", flush=True)

    res = {}; traces = {}
    lines = ["| cell | ncomp | stim | scale | max I (nA) | DEFAULT rest | spikes | peak | width ms | AHP | block | Vmin/Vmax | default verdict | "
             "BOX silent % | spikes med/p95/max | peak | width | AHP | block % | oor % | non-finite % | Vmin/Vmax | box verdict | sim s |",
             "|" + "---|" * 25]
    for s in stims:
        for sc in scales:
            t0 = time.time()
            h = JaxleyBridge.get_handle(a.cell, s, stim_scale=sc)
            dtms = spec.dt * h.downsample_step
            v = simulate(h, phys, a.batch)[:, 0, :]          # soma
            el = time.time() - t0
            d = spike_stats(v[0], dtms); dv = verdict_default(d)
            row = dict(stim=s, scale=sc, imax=imax[s] * sc, default=d, default_fail=dv, sim_s=el)
            if a.nbox > 0:
                b = scan_traces(v[1:], dtms); bv = verdict_box(b)
                per = b.pop("per"); row.update(box=b, box_fail=bv, box_nspk=per["n"].tolist())
            else:
                b = None; bv = []
            res[(s, sc)] = row; traces[(s, sc)] = v
            bx = (f"{b['silent']:.0f} | {b['spk_med']:.0f}/{b['spk_p95']:.0f}/{b['spk_max']:.0f} | {b['peak']:.0f} | {b['width']:.2f} | "
                  f"{b['ahp']:.0f} | {b['block']:.0f} | {b['oor']:.0f} | {b['nonfinite']:.1f} | {b['vmin']:.0f}/{b['vmax']:.0f} | "
                  f"{'PASS' if not bv else 'FAIL: ' + ', '.join(bv)}") if b else "| | | | | | | | | -"
            lines.append(f"| {a.cell} | {ncomp} | {s} | {sc:g} | {imax[s]*sc:.2f} | {d['rest']:.1f} | {d['n']} | {d['peak']:.0f} | {d['width']:.2f} | "
                         f"{d['ahp']:.0f} | {'YES' if d['block'] else 'no'} | {d['vmin']:.0f}/{d['vmax']:.0f} | "
                         f"{'PASS' if not dv else 'FAIL: ' + ', '.join(dv)} | {bx} | {el:.0f} |")
            print(lines[-1], flush=True)

    # ── figure ────────────────────────────────────────────────────────────
    ns, nk = len(stims), len(scales)
    fig, axes = plt.subplots(nk, ns, figsize=(6.2 * ns, 2.3 * nk + 0.9), squeeze=False, sharex=True)
    for j, s in enumerate(stims):
        for i, sc in enumerate(scales):
            ax = axes[i][j]; v = traces[(s, sc)]; r = res[(s, sc)]
            t = np.arange(v.shape[1]) * spec.dt * JaxleyBridge.get_handle(a.cell, s, stim_scale=sc).downsample_step
            if a.nbox > 0:
                nspk = np.asarray(r["box_nspk"]); order = np.argsort(nspk)
                for k in order[np.linspace(0, len(order) - 1, 5).astype(int)]:
                    ax.plot(t, v[1 + k], lw=0.5, alpha=0.55, color="#e07b39")
            ax.plot(t, v[0], lw=0.9, color="#1f5fbf")
            ax.axhline(0, color="#999", lw=0.4, ls="--")
            d = r["default"]; b = r.get("box")
            txt = f"x{sc:g} (max {r['imax']:.2f} nA): default {d['n']} sp, peak {d['peak']:.0f}, AHP {d['ahp']:.0f}, Vmin {d['vmin']:.0f}" + \
                  (" BLOCK" if d["block"] else "") + (" | " + ("PASS" if not r["default_fail"] else "FAIL"))
            if b:
                txt += f"\nbox: silent {b['silent']:.0f}%, spikes {b['spk_med']:.0f}/{b['spk_p95']:.0f}, block {b['block']:.0f}%, oor {b['oor']:.0f}%" + \
                       " | " + ("PASS" if not r["box_fail"] else "FAIL " + ", ".join(r["box_fail"]))
            ax.text(0.01, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
                    color=("#1a7f37" if not (r["default_fail"] or r.get("box_fail")) else "#b3261e"))
            ax.set_ylim(min(-130, float(np.nanmin(v[0])) - 5), max(60, float(np.nanmax(v[0])) + 5))
            if i == 0: ax.set_title(s, fontsize=9)
            if j == 0: ax.set_ylabel(f"scale {sc:g}\nmV", fontsize=8)
            ax.tick_params(labelsize=7)
    for j in range(ns): axes[-1][j].set_xlabel("time (ms)", fontsize=8)
    fig.suptitle(a.title or f"{a.cell} (ncomp={ncomp}): stimulus-scale sweep, default (blue) + {a.nbox} box draws (orange, +-{a.halfspan} log10)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(a.out + ".png", dpi=85); plt.close(fig)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    js = {f"{s}|{sc:g}": {k: v for k, v in r.items() if k != "box_nspk"} for (s, sc), r in res.items()}
    json.dump(dict(cell=a.cell, ncomp=ncomp, halfspan=a.halfspan, nbox=a.nbox, rows=js), open(a.out + ".json", "w"),
              indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[scan] wrote {a.out}.md/.png/.json", flush=True)


if __name__ == "__main__":
    main()
