"""NEURON-side reference simulation for the CA3 Pyramidal model.

Loads the Adapting-CA3 morphology + mechanisms, runs a list of protocols,
writes one .npz per (test, condition) into RESULTS_DIR for the comparison
harness to consume.

Protocols
---------
test1_rest          : I=0 nA for 500 ms, check steady state
test2_subthresh     : I=+0.05 nA, 100..400 ms
test3_suprathresh   : I=+0.30 nA, 100..400 ms
test4_fI            : I-step sweep [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
test5_paramsweep    : 50 random ḡ-perturbations at I=0.3 nA
test6_apshape       : single AP at I=1.0 nA, short
test7_walltime      : 100 traces × 500 ms, report wall-time

Each .npz carries:  t (ms), v_soma (mV), i_stim (nA), gbar (dict), wallclock (s)

CLI
---
    python toolbox/tests/sim_neuron_ca3.py --tests rest subthresh suprathresh fI ...
    python toolbox/tests/sim_neuron_ca3.py --tests all
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────
HOC_DIR = Path(
    "/global/homes/k/ktub1999/mainDL4/DL4neurons2/Adapting CA3 Pyramidal Neuron"
)
RESULTS_DIR = Path(
    "/pscratch/sd/k/ktub1999/ca3_compare/neuron"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Defaults baked into the hoc — used as fallback if NEURON not available
# (we still want the file importable for argparse/sanity)
_DEFAULTS = {
    "g_leak":     3.9417e-5,
    "gbar_na3":   0.04,
    "gkdrbar_kdr":0.01,
    "gkabar_kap": 0.04,
    "gbar_km":    5.2e-4,
    "gkdbar_kd":  2.5e-4,
}

# ── numerics ────────────────────────────────────────────────────────────────
DT      = 0.025      # ms — internal NEURON dt (40 kHz)
DT_OUT  = 0.1        # ms — output sampling rate (matches Jaxley dt)
TSTOP   = 500.0      # ms

V_INIT  = -65.0
CELSIUS = 34.0


def _load_neuron():
    """Import neuron and load mechanisms.  Compile mod files if needed."""
    import neuron
    from neuron import h
    h.load_file("stdrun.hoc")

    # nrnivmodl-compiled mech library lives at HOC_DIR/x86_64/.libs/libnrnmech.so
    mech_lib = HOC_DIR / "x86_64" / ".libs" / "libnrnmech.so"
    if not mech_lib.exists():
        # Try standard nrnivmodl location: <HOC_DIR>/x86_64/special links the libs.
        # Fallback: nrnivmodl was run differently — search.
        mech_lib_alt = HOC_DIR / "x86_64" / "libnrnmech.so"
        if mech_lib_alt.exists():
            mech_lib = mech_lib_alt
        else:
            raise FileNotFoundError(
                f"libnrnmech.so not found under {HOC_DIR/'x86_64'}.\n"
                f"Compile with:  cd {HOC_DIR}/mechanisms && nrnivmodl"
            )
    h.nrn_load_dll(str(mech_lib))
    h.celsius = CELSIUS
    return h


def build_cell(h, gbar=None):
    """Recreate the soma section + mechanisms from morphology_mechanisms.hoc."""
    g = dict(_DEFAULTS)
    if gbar is not None:
        g.update(gbar)

    soma = h.Section(name="soma")
    soma.nseg = 1
    soma.L    = 50.0
    soma.diam = 50.0
    soma.cm   = 1.41
    soma.Ra   = 150.0

    soma.insert("leak")
    soma.insert("na3")
    soma.insert("kdr")
    soma.insert("kap")
    soma.insert("km")
    soma.insert("kd")
    soma.insert("cacum")

    soma.e_leak = 93.9115         # ⚠ verbatim from hoc — likely typo
    soma.ena = 55.0
    soma.ek  = -90.0
    soma.depth_cacum = 25.0

    soma.g_leak       = g["g_leak"]
    soma.gbar_na3     = g["gbar_na3"]
    soma.gkdrbar_kdr  = g["gkdrbar_kdr"]
    soma.gkabar_kap   = g["gkabar_kap"]
    soma.gbar_km      = g["gbar_km"]
    soma.gkdbar_kd    = g["gkdbar_kd"]

    return soma


def run_protocol(h, soma, i_stim_array, t_array, tstop=TSTOP):
    """i_stim_array, t_array: numpy arrays sampled at DT (NEURON dt)."""
    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 0
    iclamp.dur   = 1e9

    stim_vec = h.Vector(i_stim_array.astype(np.float64))
    t_vec    = h.Vector(t_array.astype(np.float64))
    stim_vec.play(iclamp._ref_amp, t_vec, True)

    v_rec = h.Vector()
    v_rec.record(soma(0.5)._ref_v)
    t_rec = h.Vector()
    t_rec.record(h._ref_t)

    h.dt        = DT
    h.tstop     = tstop
    h.v_init    = V_INIT
    h.secondorder = 0       # backward Euler, matches Jaxley's bwd_euler
    h.finitialize(V_INIT)

    t0 = time.time()
    h.run()
    wall = time.time() - t0

    t = np.asarray(t_rec)
    v = np.asarray(v_rec)
    return t, v, wall


def _resample_to_grid(t, v, grid):
    """Linear-interpolate (t, v) onto an evenly-spaced grid."""
    return np.interp(grid, t, v)


def _make_step(amp_nA, t_on=100.0, t_off=400.0, tstop=TSTOP, dt=DT):
    t_arr = np.arange(0.0, tstop + dt/2, dt)
    i_arr = np.zeros_like(t_arr)
    i_arr[(t_arr >= t_on) & (t_arr < t_off)] = amp_nA
    return t_arr, i_arr


def _save(name, t_grid, v, i_grid, gbar, wall):
    out = RESULTS_DIR / f"{name}.npz"
    np.savez_compressed(
        out,
        t=t_grid.astype(np.float32),
        v_soma=v.astype(np.float32),
        i_stim=i_grid.astype(np.float32),
        gbar=json.dumps(gbar),
        wallclock=np.float32(wall),
    )
    return out


# ── individual tests ────────────────────────────────────────────────────────

def run_test1_rest(h):
    soma = build_cell(h)
    t_arr, i_arr = _make_step(0.0)            # zero current
    t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr)
    grid = np.arange(0.0, TSTOP + DT_OUT/2, DT_OUT)
    v_g = _resample_to_grid(t_n, v_n, grid)
    i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
    _save("test1_rest", grid, v_g, i_g, _DEFAULTS, wall)


def run_test2_subthresh(h):
    soma = build_cell(h)
    t_arr, i_arr = _make_step(0.05)
    t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr)
    grid = np.arange(0.0, TSTOP + DT_OUT/2, DT_OUT)
    v_g = _resample_to_grid(t_n, v_n, grid)
    i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
    _save("test2_subthresh", grid, v_g, i_g, _DEFAULTS, wall)


def run_test3_suprathresh(h):
    soma = build_cell(h)
    t_arr, i_arr = _make_step(0.30)
    t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr)
    grid = np.arange(0.0, TSTOP + DT_OUT/2, DT_OUT)
    v_g = _resample_to_grid(t_n, v_n, grid)
    i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
    _save("test3_suprathresh", grid, v_g, i_g, _DEFAULTS, wall)


def run_test4_fI(h):
    amps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    for a in amps:
        soma = build_cell(h)
        t_arr, i_arr = _make_step(a)
        t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr)
        grid = np.arange(0.0, TSTOP + DT_OUT/2, DT_OUT)
        v_g = _resample_to_grid(t_n, v_n, grid)
        i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
        _save(f"test4_fI_amp{a:.2f}".replace(".", "p"), grid, v_g, i_g, _DEFAULTS, wall)


def run_test5_paramsweep(h, n=50, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(_DEFAULTS.keys())
    for i in range(n):
        # Each ḡ scaled by 10^U(-0.5, 0.5).
        scales = 10.0 ** rng.uniform(-0.5, 0.5, size=len(keys))
        gbar = {k: float(_DEFAULTS[k] * s) for k, s in zip(keys, scales)}
        soma = build_cell(h, gbar=gbar)
        t_arr, i_arr = _make_step(0.30)
        t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr)
        grid = np.arange(0.0, TSTOP + DT_OUT/2, DT_OUT)
        v_g = _resample_to_grid(t_n, v_n, grid)
        i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
        _save(f"test5_sweep_{i:03d}", grid, v_g, i_g, gbar, wall)


def run_test6_apshape(h):
    """Short 200 ms run, I=1.0 nA, capture single AP shape."""
    soma = build_cell(h)
    t_arr, i_arr = _make_step(1.0, t_on=50.0, t_off=200.0, tstop=200.0)
    t_n, v_n, wall = run_protocol(h, soma, i_arr, t_arr, tstop=200.0)
    grid = np.arange(0.0, 200.0 + DT_OUT/2, DT_OUT)
    v_g = _resample_to_grid(t_n, v_n, grid)
    i_g = _resample_to_grid(t_n, np.interp(t_n, t_arr, i_arr), grid)
    _save("test6_apshape", grid, v_g, i_g, _DEFAULTS, wall)


def run_test7_walltime(h, n=100):
    """Run n traces of 500 ms with default params, suprathreshold step."""
    times = []
    for i in range(n):
        soma = build_cell(h)
        t_arr, i_arr = _make_step(0.30)
        _, _, wall = run_protocol(h, soma, i_arr, t_arr)
        times.append(wall)
    times = np.asarray(times, dtype=np.float32)
    np.savez_compressed(RESULTS_DIR / "test7_walltime.npz",
                         per_trace_s=times,
                         total_s=times.sum(),
                         mean_s=times.mean(),
                         std_s=times.std(),
                         n=n)
    print(f"NEURON wall-time: {times.mean()*1000:.1f} ± {times.std()*1000:.1f} ms / trace, "
          f"{times.sum():.1f} s total over {n} traces")


# ── CLI ─────────────────────────────────────────────────────────────────────

_ALL = ["rest", "subthresh", "suprathresh", "fI", "paramsweep", "apshape", "walltime"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", nargs="+", default=["all"],
                   help=f"any of: {_ALL} or 'all'")
    p.add_argument("--n-sweep", type=int, default=50)
    p.add_argument("--n-walltime", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tests = _ALL if "all" in args.tests else args.tests

    h = _load_neuron()
    print(f"NEURON loaded.  Mechanisms from {HOC_DIR}/x86_64/.")
    print(f"Writing results to {RESULTS_DIR}")

    if "rest"        in tests: run_test1_rest(h)
    if "subthresh"   in tests: run_test2_subthresh(h)
    if "suprathresh" in tests: run_test3_suprathresh(h)
    if "fI"          in tests: run_test4_fI(h)
    if "paramsweep"  in tests: run_test5_paramsweep(h, n=args.n_sweep, seed=args.seed)
    if "apshape"     in tests: run_test6_apshape(h)
    if "walltime"    in tests: run_test7_walltime(h, n=args.n_walltime)

    print("Done.")


if __name__ == "__main__":
    main()
