#!/usr/bin/env python
"""Stimulus + voltage figure for the model ladder AFTER the fixes.

--simulate: for each cell, both stims: default trace + NBOX box draws -> npz
--plot    : combine npz files into one figure: rows = rungs, cols = stims; each cell of
            the grid has the SCALED injected current (nA) above the soma voltage (mV).
Run --simulate twice for l5ttpc (L5TTPC_NCOMP=4 and =2) since ncomp is fixed per process.
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
sys.path.insert(0, os.getcwd())

STIMS = ["5k50kInterChaoticB", "5kChaoticRamp"]
LABEL = {"single_comp": "point process\n(1 comp, 3 HH params)", "ball_and_stick": "ball and stick\n(HH, 4 params)",
         "ball_and_stick_bbp": "ball and stick\n(BBP channels, 13 params)", "ca3_pyramidal": "CA3 (reference,\nunchanged, 6 params)",
         "l5ttpc_nc2": "L5 2 comp/branch\n(19 params)", "l5ttpc_nc4": "L5 4 comp/branch\n(19 params)"}


def simulate(cells, nbox, seed, out):
    import jax.numpy as jnp
    from toolbox import JaxleyBridge, jaxley_cells
    from toolbox.jaxley_utils import load_stim_csv, build_phys_par_range, phys_par_range_to_arrays, unit_to_phys_np, phys_par_range_linear_mask
    from toolbox.physio_stats import spike_stats
    import importlib
    res = {}
    for cell in cells:
        mod = importlib.import_module("toolbox.jaxley_cells." + {"single_comp": "soma_only"}.get(cell, cell))
        spec = jaxley_cells.get(cell); key = cell + (f"_nc{mod._NCOMP}" if cell == "l5ttpc" else "")
        ppr = build_phys_par_range(mod, 0.5); c, s = phys_par_range_to_arrays(ppr)
        rng = np.random.default_rng(seed); P = len(spec.param_keys)
        unit = rng.uniform(-1, 1, size=(nbox, P))
        phys = np.concatenate([np.array([[mod._DEFAULTS[k] for k in spec.param_keys]]), unit_to_phys_np(unit, c.astype(np.float64), s.astype(np.float64), phys_par_range_linear_mask(ppr))], 0)
        for st in STIMS:
            h = JaxleyBridge.get_handle(cell, st)
            stim = load_stim_csv(spec.stim_dir / f"{st}.csv") * h.stim_scale
            v = np.asarray(h.simulate_batch(jnp.asarray(phys)))[:, 0, :]
            dt = spec.dt * h.downsample_step
            n = [spike_stats(x, dt)["n"] for x in v]
            res[f"{key}|{st}|stim"] = stim; res[f"{key}|{st}|v"] = v; res[f"{key}|{st}|dt"] = dt
            res[f"{key}|{st}|scale"] = h.stim_scale; res[f"{key}|{st}|nspk"] = np.array(n)
            print(f"[sim] {key} {st} x{h.stim_scale:g}: spikes {n}", flush=True)
    np.savez(out, **res); print("wrote", out)


def plot(npzs, out, order):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = {}
    for f in npzs:
        z = np.load(f); d.update({k: z[k] for k in z.files})
    keys = [k for k in order if f"{k}|{STIMS[0]}|v" in d]
    fig = plt.figure(figsize=(13, 2.9 * len(keys) + 0.8))
    gs = fig.add_gridspec(len(keys) * 2, 2, height_ratios=[1, 2.6] * len(keys), hspace=0.08, wspace=0.12)
    for i, k in enumerate(keys):
        for j, st in enumerate(STIMS):
            stim = d[f"{k}|{st}|stim"]; v = d[f"{k}|{st}|v"]; dt = float(d[f"{k}|{st}|dt"]); sc = float(d[f"{k}|{st}|scale"])
            nspk = d[f"{k}|{st}|nspk"]; t = np.arange(len(stim)) * 0.1; tv = np.arange(v.shape[1]) * dt
            a0 = fig.add_subplot(gs[2 * i, j]); a1 = fig.add_subplot(gs[2 * i + 1, j], sharex=a0)
            a0.plot(t, stim, color="#444", lw=0.7); a0.axhline(0, color="#aaa", lw=0.4)
            a0.set_ylabel("I (nA)", fontsize=8); a0.tick_params(labelsize=7, labelbottom=False)
            a0.text(0.01, 0.92, f"{st}  x{sc:g}  (peak {stim.max():.2f} nA, min {stim.min():.2f} nA)", transform=a0.transAxes, va="top", fontsize=7.5)
            for b in range(1, v.shape[0]):
                a1.plot(tv, v[b], lw=0.5, alpha=0.5, color="#e07b39")
            a1.plot(tv, v[0], lw=0.9, color="#1f5fbf")
            a1.axhline(0, color="#aaa", lw=0.4, ls="--"); a1.set_ylim(min(-110, v[0].min() - 5), max(55, v[0].max() + 5))
            a1.text(0.01, 0.95, f"default: {int(nspk[0])} spikes, peak {v[0].max():.0f} mV, min {v[0].min():.0f} mV; box draws: {', '.join(str(int(x)) for x in nspk[1:])} spikes",
                    transform=a1.transAxes, va="top", fontsize=7.5, color="#1f5fbf")
            a1.tick_params(labelsize=7)
            if j == 0: a1.set_ylabel(LABEL.get(k, k) + "\nmV", fontsize=8)
            if i == len(keys) - 1: a1.set_xlabel("time (ms)", fontsize=8)
            else: a1.tick_params(labelbottom=False)
    fig.suptitle("Model ladder after the fixes: scaled injected current (top) and soma voltage (bottom); blue = default parameters, orange = draws from the +-0.5-decade box",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out, dpi=90, bbox_inches="tight"); fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", default=None, help="comma list of cells")
    ap.add_argument("--nbox", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", default=None, help="comma list of npz files")
    a = ap.parse_args()
    if a.simulate: simulate(a.simulate.split(","), a.nbox, a.seed, a.out)
    if a.plot: plot(a.plot.split(","), a.out, ["single_comp", "ball_and_stick", "ball_and_stick_bbp", "l5ttpc_nc2", "l5ttpc_nc4", "ca3_pyramidal"])
