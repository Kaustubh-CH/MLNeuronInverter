#!/usr/bin/env python
"""Accuracy of cheaper L5 solver settings: forward traces at (ncomp = env, dt in --dts) for
the default cell + NBOX box draws, both stims, saved to npz (--simulate); --compare puts
everything on the 0.1 ms grid and reports, vs the reference (ncomp 4, dt 0.1): spike-count
agreement, mean |spike-time shift| of matched spikes (ms), RMSE (mV)."""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cuda"); os.environ.setdefault("JAX_ENABLE_X64", "true"); os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
sys.path.insert(0, os.getcwd())
STIMS = ["5k50kInterChaoticB", "5kChaoticRamp"]

def simulate(dts, nbox, seed, out):
    import jax.numpy as jnp
    from toolbox import JaxleyBridge
    from toolbox.jaxley_cells import l5ttpc
    from toolbox.jaxley_utils import build_phys_par_range, phys_par_range_to_arrays, unit_to_phys_np, phys_par_range_linear_mask
    ppr = build_phys_par_range(l5ttpc, 0.5); c, s = phys_par_range_to_arrays(ppr)
    rng = np.random.default_rng(seed); P = len(l5ttpc.PARAM_KEYS)
    unit = rng.uniform(-1, 1, size=(nbox, P))
    phys = np.concatenate([np.array([[l5ttpc._DEFAULTS[k] for k in l5ttpc.PARAM_KEYS]]), unit_to_phys_np(unit, c.astype(np.float64), s.astype(np.float64), phys_par_range_linear_mask(ppr))], 0)
    res = {}
    for dt in dts:
        l5ttpc._DT = dt; JaxleyBridge.clear_cache()
        for st in STIMS:
            h = JaxleyBridge.get_handle("l5ttpc", st)
            v = []
            for i0 in range(0, len(phys), 32):
                pg = phys[i0:i0 + 32]
                if len(pg) < 32: pg = np.concatenate([pg, np.repeat(pg[:1], 32 - len(pg), 0)])
                v.append(np.asarray(h.simulate_batch(jnp.asarray(pg)))[:, 0, :])
            v = np.concatenate(v)[:len(phys)]
            res[f"nc{l5ttpc._NCOMP}|dt{dt:g}|{st}"] = v; res[f"nc{l5ttpc._NCOMP}|dt{dt:g}|{st}|out_dt"] = h.out_dt
            print(f"[sim] nc{l5ttpc._NCOMP} dt={dt:g} {st}: {v.shape} out_dt={h.out_dt}", flush=True)
    np.savez(out, **res); print("wrote", out)

def spikes(v, dt):
    up = np.where((v[1:] > 0) & (v[:-1] <= 0))[0] + 1
    return up * dt

def compare(npzs, out):
    d = {}
    for f in npzs:
        z = np.load(f); d.update({k: z[k] for k in z.files})
    keys = sorted({"|".join(k.split("|")[:2]) for k in d if k.endswith("|out_dt")})
    t_ref = np.arange(0, 500, 0.1)
    lines = ["| config | stim | traces | spike count: exact / within 1 | mean abs dt of matched spikes (ms) | RMSE vs ref (mV) | RMSE vs same ncomp dt 0.1 |", "|" + "---|" * 7]
    for k in keys:
        nc, dts = k.split("|")[0], k.split("|")[1]
        for st in STIMS:
            key = f"{nc}|{dts}|{st}"
            if key not in d: continue
            v = d[key]; odt = float(d[key + "|out_dt"]); t = np.arange(v.shape[1]) * odt
            V = np.stack([np.interp(t_ref, t, x) for x in v])
            refk = f"nc4|dt0.1|{st}"; samek = f"{nc}|dt0.1|{st}"
            def stat(refkey):
                if refkey not in d: return None
                r = d[refkey]; rdt = float(d[refkey + "|out_dt"]); tr = np.arange(r.shape[1]) * rdt
                R = np.stack([np.interp(t_ref, tr, x) for x in r])
                n = min(len(V), len(R)); exact = 0; within = 0; shifts = []; rmse = []
                for i in range(n):
                    a, b = spikes(V[i], 0.1), spikes(R[i], 0.1)
                    exact += len(a) == len(b); within += abs(len(a) - len(b)) <= 1
                    for tb in b:
                        if len(a): shifts.append(np.min(np.abs(a - tb)))
                    rmse.append(np.sqrt(np.mean((V[i] - R[i]) ** 2)))
                return n, 100 * exact / n, 100 * within / n, (np.mean(shifts) if shifts else np.nan), float(np.mean(rmse))
            sr = stat(refk); ss = stat(samek)
            if sr is None: continue
            lines.append(f"| {nc} dt {dts[2:]} | {st} | {sr[0]} | {sr[1]:.0f} % / {sr[2]:.0f} % | {sr[3]:.2f} | {sr[4]:.2f} | {ss[4]:.2f} |")
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--dts", default="0.1,0.2,0.25")
    ap.add_argument("--nbox", type=int, default=32); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True); ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    if a.simulate: simulate([float(x) for x in a.dts.split(",")], a.nbox, a.seed, a.out)
    if a.compare: compare(a.compare.split(","), a.out)
