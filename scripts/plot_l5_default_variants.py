#!/usr/bin/env python
"""Default-parameter L5 soma traces under the cheaper solver settings.
--simulate: at the current L5TTPC_NCOMP and precision (JAX_ENABLE_X64), run the default
cell for --dts on both stims -> npz.   --plot: overlay per ncomp (rows) x stim (cols):
fp64 dt 0.1 (reference), fp64 dt 0.2, fp64 dt 0.25, fp32 dt 0.1."""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
sys.path.insert(0, os.getcwd())
STIMS = ["5k50kInterChaoticB", "5kChaoticRamp"]

def simulate(dts, out, nsamples=0, halfspan=1.0, seed=7):
    """Row 0 = BBP default; rows 1..nsamples = random draws in the DL4neurons2 run.py
    convention (u ~ U(-1,1); conductances base*10^(u*halfspan), e_pas/cm linear).
    The same seed gives the SAME draws for every ncomp / precision / dt."""
    import jax, jax.numpy as jnp
    from toolbox import JaxleyBridge
    from toolbox.jaxley_cells import l5ttpc
    from toolbox.jaxley_utils import build_phys_par_range, phys_par_range_to_arrays, phys_par_range_linear_mask, unit_to_phys_np
    prec = "fp64" if jax.config.jax_enable_x64 else "fp32"
    ppr = build_phys_par_range(l5ttpc, halfspan); c, sp = phys_par_range_to_arrays(ppr); lin = phys_par_range_linear_mask(ppr)
    P = len(l5ttpc.PARAM_KEYS)
    unit = np.random.default_rng(seed).uniform(-1, 1, size=(nsamples, P)) if nsamples else np.zeros((0, P))
    p = np.concatenate([np.array([[l5ttpc._DEFAULTS[k] for k in l5ttpc.PARAM_KEYS]]),
                        unit_to_phys_np(unit, c.astype(np.float64), sp.astype(np.float64), lin)], 0)
    p = p.astype(np.float64 if prec == "fp64" else np.float32)
    res = {"unit": unit}
    for dt in dts:
        l5ttpc._DT = dt; JaxleyBridge.clear_cache()
        for st in STIMS:
            h = JaxleyBridge.get_handle("l5ttpc", st)
            v = np.asarray(h.simulate_batch(jnp.asarray(p)))[:, 0, :]          # (1+nsamples, T)
            k = f"nc{l5ttpc._NCOMP}|{prec}|dt{dt:g}|{st}"
            res[k] = v; res[k + "|out_dt"] = h.out_dt
            print(f"[sim] {k}: {v.shape}, out_dt {h.out_dt}, spikes {[(np.diff((x>0).astype(int))==1).sum() for x in v]}", flush=True)
    np.savez(out, **res); print("wrote", out)

def spikes(v, dt):
    up = np.where((v[1:] > 0) & (v[:-1] <= 0))[0] + 1
    return up * dt

def plot(npzs, out, ncomps):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    d = {}
    for f in npzs:
        z = np.load(f); d.update({k: z[k] for k in z.files})
    variants = [("fp64", 0.1, "#222222", 1.3, "fp64 dt 0.1 (reference)"), ("fp64", 0.2, "#1f77b4", 0.8, "fp64 dt 0.2"),
                ("fp64", 0.25, "#2ca02c", 0.8, "fp64 dt 0.25"), ("fp32", 0.1, "#ff7f0e", 0.8, "fp32 dt 0.1")]
    tg = np.arange(0, 500, 0.1)
    any_key = next(k for k in d if k.endswith(f"|{STIMS[0]}"))
    nrows = d[any_key].shape[0] if d[any_key].ndim == 2 else 1
    unit = d.get("unit", np.zeros((0, 19)))
    try:
        from toolbox.jaxley_cells import l5ttpc; names = l5ttpc.PARAM_KEYS
    except Exception: names = [f"p{i}" for i in range(19)]
    pdf = PdfPages(os.path.splitext(out)[0] + ".pdf")
    for r in range(nrows):
        fig, axes = plt.subplots(len(ncomps), 2, figsize=(15, 2.7 * len(ncomps) + 0.8), squeeze=False, sharex=True)
        _plot_page(fig, axes, d, ncomps, variants, tg, r)
        if r == 0:
            title = "L5 default (BBP) parameters"
        else:
            u = unit[r - 1]; big = np.argsort(-np.abs(u))[:6]
            title = f"L5 random draw #{r} (run.py convention: conductances base*10^u, e_pas/cm linear); largest |u|: " + \
                    ", ".join(f"{names[i].replace('bar_', '_').replace('_somatic', '_som').replace('_axonal', '_ax').replace('_apical', '_api')[:22]}={u[i]:+.2f}" for i in big)
        fig.suptitle(title + " -- solver precision / time-step variants overlaid, ncomp 1-4 per branch", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); pdf.savefig(fig)
        if r == 0: fig.savefig(out, dpi=95)
        if r == 1: fig.savefig(os.path.splitext(out)[0] + '_draw1.png', dpi=95)
        plt.close(fig)
    pdf.close(); print("wrote", out, "and", os.path.splitext(out)[0] + ".pdf", f"({nrows} pages)")


def _plot_page(fig, axes, d, ncomps, variants, tg, r):
    for i, nc in enumerate(ncomps):
        for j, st in enumerate(STIMS):
            ax = axes[i][j]; ref = None; notes = []
            for prec, dt, col, lw, lab in variants:
                k = f"nc{nc}|{prec}|dt{dt:g}|{st}"
                if k not in d: continue
                vv = d[k]; v = vv[r] if vv.ndim == 2 else vv
                odt = float(d[k + "|out_dt"]); t = np.arange(len(v)) * odt
                vg = np.interp(tg, t, v); n = len(spikes(vg, 0.1))
                if ref is None:
                    ref = vg; notes.append(f"{lab}: {n} sp")
                else:
                    rmse = np.sqrt(np.mean((vg - ref) ** 2)); notes.append(f"{lab}: {n} sp, RMSE {rmse:.1f} mV")
                ax.plot(t, v, color=col, lw=lw, alpha=0.9 if prec == "fp64" and dt == 0.1 else 0.75, label=lab)
            ax.axhline(0, color="#aaa", lw=0.4, ls="--"); ax.set_ylim(min(-100, float(np.nanmin(ref)) - 5) if ref is not None else -100, 55)
            ax.text(0.01, 0.97, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=7.2)
            if i == 0: ax.set_title(st + f"  (x1.5)", fontsize=10)
            if j == 0: ax.set_ylabel(f"L5 ncomp={nc}\nmV", fontsize=9)
            if i == len(ncomps) - 1: ax.set_xlabel("time (ms)", fontsize=9)
            ax.tick_params(labelsize=8)
    axes[0][1].legend(fontsize=7.5, loc="lower right", ncol=2, frameon=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--dts", default="0.1,0.2,0.25")
    ap.add_argument("--out", required=True); ap.add_argument("--plot", default=None); ap.add_argument("--ncomps", default="1,2,3,4")
    ap.add_argument("--nsamples", type=int, default=0); ap.add_argument("--halfspan", type=float, default=1.0); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.simulate: simulate([float(x) for x in a.dts.split(",")], a.out, a.nsamples, a.halfspan, a.seed)
    if a.plot: plot(a.plot.split(","), a.out, [int(x) for x in a.ncomps.split(",")])
