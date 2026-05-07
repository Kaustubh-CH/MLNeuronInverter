"""CA3 NEURON-vs-Jaxley comparison harness.

Reads the NEURON-side .npz outputs written by ``sim_neuron_ca3.py``, runs
the Jaxley CA3 model under identical conditions (same step amplitudes, same
random ḡ-perturbations for the sweep test), computes per-test metrics, and
emits overlay plots + a summary CSV.

Tests
-----
test1_rest         : ‖V_n - V_j‖ over last 100 ms; pass if mean‖ΔV‖ < 0.5 mV
test2_subthresh    : RMSE; pass if RMSE < 1.5 mV and AP count = 0 on both
test3_suprathresh  : spike count match, peak ΔV < 5 mV, mean spike-time Δ < 1 ms
test4_fI           : per-amp spike count diff ≤ 1
test5_paramsweep   : corr(spike_count_n, spike_count_j) > 0.95
test6_apshape      : peak Δ < 3 mV, ½-width Δ < 0.3 ms, AHP Δ < 3 mV
test7_walltime     : numbers reported (informational, no pass criterion)

CLI
---
    python toolbox/tests/test_ca3_neuron_vs_jaxley.py --tests all
    python toolbox/tests/test_ca3_neuron_vs_jaxley.py --tests rest fI walltime

Outputs
-------
    docs/ca3/<test>_overlay.png
    docs/ca3/summary.csv
    docs/ca3/walltime.txt
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

# We stay headless; Jaxley imports happen lazily inside runners so this
# script can be imported on a node without GPU.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ── paths ───────────────────────────────────────────────────────────────────
NEURON_RESULTS = Path("/pscratch/sd/k/ktub1999/ca3_compare/neuron")
JAXLEY_RESULTS = Path("/pscratch/sd/k/ktub1999/ca3_compare/jaxley")
PLOTS_DIR      = Path("/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/docs/ca3")
JAXLEY_RESULTS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DT_OUT = 0.1   # ms — output sampling rate, matches sim_neuron_ca3 grid
TSTOP_DEFAULT = 500.0
TSTOP_AP = 200.0


# ── lazy Jaxley runner ─────────────────────────────────────────────────────

def _jaxley_runner():
    """Return a closure run_jaxley(gbar_dict, i_stim, tstop, dt) -> (t, v, wall)."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU is fine for the comparison
    import jax, jax.numpy as jnp
    import jaxley as jx
    from toolbox.jaxley_cells import get
    from toolbox.jaxley_cells.ca3_pyramidal import _PARAM_MAP

    spec = get("ca3_pyramidal")

    # Map CNN-name -> jaxley channel-param key for direct .set() access.
    cnn_to_jax = {cnn: jax_key for cnn, jax_key in _PARAM_MAP}

    def run(gbar_dict, i_stim_np, tstop, dt):
        # Build a fresh cell so default-param mutation between runs is isolated.
        cell, _ = spec.build_fn()
        # Re-set ḡ to whatever the caller asked for (defaults already set by builder).
        for cnn_name, val in gbar_dict.items():
            jax_key = cnn_to_jax.get("CA3_" + cnn_name) or cnn_to_jax.get(cnn_name)
            if jax_key is None:
                # Caller may have used the .mod-style names ("g_leak", "gbar_na3"...)
                # — translate via _PARAM_MAP by suffix match.
                for cnn, jx_key in _PARAM_MAP:
                    if cnn.endswith(cnn_name):
                        jax_key = jx_key
                        break
            if jax_key is None:
                raise KeyError(f"unknown ḡ key: {cnn_name}")
            cell.branch(0).set(jax_key, val)

        # Re-init states at the original v_init so gates start from steady state.
        cell.init_states(delta_t=dt)

        stim_jnp = jnp.asarray(i_stim_np.astype(np.float32))
        ds = spec.stim_attach_fn(cell, stim_jnp)
        t0 = time.time()
        v = np.asarray(jx.integrate(cell, data_stimuli=ds, t_max=tstop, delta_t=dt))
        wall = time.time() - t0
        # v has shape (1, T+1).  Build t-grid.
        t = np.arange(v.shape[1], dtype=np.float32) * dt
        return t, v[0], wall

    return run


# ── analysis helpers ────────────────────────────────────────────────────────

def detect_spikes(v, dt, height=0.0, distance_ms=2.0):
    pk_idx, _ = find_peaks(v, height=height, distance=int(distance_ms / dt))
    return pk_idx, pk_idx.astype(np.float32) * dt


def coincidence_fraction(t_n, t_j, window_ms=2.0):
    if len(t_n) == 0 and len(t_j) == 0:
        return 1.0
    if len(t_n) == 0 or len(t_j) == 0:
        return 0.0
    hits = sum(np.any(np.abs(t_n - t) <= window_ms) for t in t_j)
    return hits / max(len(t_n), len(t_j))


def half_width(v, peak_idx, dt):
    """Width (ms) at half-peak above baseline (taken as 10 ms pre-peak)."""
    pre_idx = max(0, peak_idx - int(10.0 / dt))
    baseline = v[pre_idx]
    half = baseline + 0.5 * (v[peak_idx] - baseline)
    # left
    li = peak_idx
    while li > 0 and v[li] > half:
        li -= 1
    ri = peak_idx
    while ri < len(v) - 1 and v[ri] > half:
        ri += 1
    return (ri - li) * dt


def ahp_depth(v, peak_idx, dt, window_ms=20.0):
    """Min voltage in window after peak − pre-peak baseline."""
    pre_idx = max(0, peak_idx - int(10.0 / dt))
    baseline = v[pre_idx]
    end = min(len(v), peak_idx + int(window_ms / dt))
    return float(baseline - v[peak_idx:end].min())


# ── summary table accumulator ──────────────────────────────────────────────

class Summary:
    def __init__(self):
        self.rows = []

    def add(self, test, metric, neuron_val, jaxley_val, delta, passfail):
        self.rows.append((test, metric, neuron_val, jaxley_val, delta, passfail))

    def write_csv(self, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test", "metric", "neuron", "jaxley", "delta", "pass"])
            for r in self.rows:
                w.writerow(r)
        print(f"summary -> {path}")


# ── per-test runners ───────────────────────────────────────────────────────

def _load_neuron(name):
    path = NEURON_RESULTS / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing NEURON result {path}. Run sim_neuron_ca3.py first.")
    return np.load(path, allow_pickle=True)


def _plot_overlay(name, t_n, v_n, t_j, v_j, i_stim, title, extra_lines=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t_n, v_n, label="NEURON", color="tab:blue", lw=1.5)
    axes[0].plot(t_j, v_j, label="Jaxley", color="tab:orange", lw=1.2, ls="--")
    axes[0].set_ylabel("V (mV)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(len(i_stim)) * DT_OUT, i_stim, color="gray", lw=1)
    axes[1].set_ylabel("I (nA)")
    axes[1].set_xlabel("t (ms)")
    axes[1].grid(alpha=0.3)
    if extra_lines:
        for line in extra_lines:
            axes[0].text(0.01, 0.01, line, transform=axes[0].transAxes,
                         va="bottom", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    out = PLOTS_DIR / f"{name}_overlay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _gbar_from_npz(npz):
    return json.loads(str(npz["gbar"]))


def run_test1_rest(run_jaxley, summary):
    npz = _load_neuron("test1_rest")
    t_n, v_n = npz["t"], npz["v_soma"]
    i_stim = npz["i_stim"]
    gbar   = _gbar_from_npz(npz)
    t_j, v_j, wall = run_jaxley(gbar, i_stim, TSTOP_DEFAULT, DT_OUT)
    # Compare last 100 ms
    n = min(len(v_n), len(v_j))
    last = int(100.0 / DT_OUT)
    err = np.abs(v_n[n-last:n] - v_j[n-last:n])
    mean_err = float(err.mean())
    passed = mean_err < 0.5
    summary.add("test1_rest", "mean|ΔV| (last 100ms, mV)", float(v_n[n-last:n].mean()),
                float(v_j[n-last:n].mean()), mean_err, passed)
    _plot_overlay("test1_rest", t_n, v_n, t_j, v_j, i_stim,
                   f"Test 1 — Rest (I=0).  mean|ΔV|={mean_err:.3f} mV  pass={passed}")


def run_test2_subthresh(run_jaxley, summary):
    npz = _load_neuron("test2_subthresh")
    t_n, v_n, i_stim = npz["t"], npz["v_soma"], npz["i_stim"]
    gbar = _gbar_from_npz(npz)
    t_j, v_j, _ = run_jaxley(gbar, i_stim, TSTOP_DEFAULT, DT_OUT)
    n = min(len(v_n), len(v_j))
    rmse = float(np.sqrt(np.mean((v_n[:n] - v_j[:n]) ** 2)))
    pk_n, _ = detect_spikes(v_n[:n], DT_OUT)
    pk_j, _ = detect_spikes(v_j[:n], DT_OUT)
    passed = (rmse < 1.5) and (len(pk_n) == 0) and (len(pk_j) == 0)
    summary.add("test2_subthresh", "RMSE (mV)", "—", "—", rmse, passed)
    summary.add("test2_subthresh", "spike count", len(pk_n), len(pk_j),
                 len(pk_n) - len(pk_j), passed)
    _plot_overlay("test2_subthresh", t_n, v_n, t_j, v_j, i_stim,
                   f"Test 2 — Subthreshold I=+0.05 nA.  RMSE={rmse:.2f} mV  pass={passed}")


def run_test3_suprathresh(run_jaxley, summary):
    npz = _load_neuron("test3_suprathresh")
    t_n, v_n, i_stim = npz["t"], npz["v_soma"], npz["i_stim"]
    gbar = _gbar_from_npz(npz)
    t_j, v_j, _ = run_jaxley(gbar, i_stim, TSTOP_DEFAULT, DT_OUT)
    n = min(len(v_n), len(v_j))
    pk_n, ts_n = detect_spikes(v_n[:n], DT_OUT)
    pk_j, ts_j = detect_spikes(v_j[:n], DT_OUT)
    if len(pk_n) == 0 and len(pk_j) == 0:
        # Both sub-threshold — fall back to RMSE comparison.
        peak_d = float(abs(v_n[:n].max() - v_j[:n].max()))
        spike_t_d = 0.0
        passed = peak_d < 5.0
    else:
        peak_d = float(abs(v_n[pk_n].max() - v_j[pk_j].max())) if len(pk_n) and len(pk_j) else float("inf")
        if len(pk_n) and len(pk_j):
            m = min(len(ts_n), len(ts_j))
            spike_t_d = float(np.abs(ts_n[:m] - ts_j[:m]).mean())
        else:
            spike_t_d = float("inf")
        passed = (len(pk_n) == len(pk_j)) and (peak_d < 5.0) and (spike_t_d < 1.0)
    summary.add("test3_suprathresh", "spike count", len(pk_n), len(pk_j),
                 len(pk_n) - len(pk_j), passed)
    summary.add("test3_suprathresh", "max peak |ΔV| (mV)", "—", "—", peak_d, passed)
    summary.add("test3_suprathresh", "mean spike-time |Δ| (ms)", "—", "—", spike_t_d, passed)
    _plot_overlay("test3_suprathresh", t_n, v_n, t_j, v_j, i_stim,
                   f"Test 3 — I=+0.30 nA.  spikes N/J={len(pk_n)}/{len(pk_j)}  "
                   f"peakΔ={peak_d:.2f}  spkΔ={spike_t_d:.2f}  pass={passed}")


def run_test4_fI(run_jaxley, summary):
    amps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    counts_n, counts_j = [], []
    for a in amps:
        tag = f"test4_fI_amp{a:.2f}".replace(".", "p")
        npz = _load_neuron(tag)
        v_n, i_stim = npz["v_soma"], npz["i_stim"]
        gbar = _gbar_from_npz(npz)
        _, v_j, _ = run_jaxley(gbar, i_stim, TSTOP_DEFAULT, DT_OUT)
        pk_n, _ = detect_spikes(v_n, DT_OUT)
        pk_j, _ = detect_spikes(v_j, DT_OUT)
        counts_n.append(len(pk_n))
        counts_j.append(len(pk_j))
        ok = abs(len(pk_n) - len(pk_j)) <= 1
        summary.add("test4_fI", f"amp={a} count", len(pk_n), len(pk_j),
                     len(pk_n) - len(pk_j), ok)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(amps, counts_n, "o-", label="NEURON", color="tab:blue")
    ax.plot(amps, counts_j, "s--", label="Jaxley", color="tab:orange")
    ax.set_xlabel("I (nA)"); ax.set_ylabel("# spikes / 300 ms step")
    ax.set_title("Test 4 — f-I curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test4_fI_overlay.png", dpi=120)
    plt.close(fig)


def run_test5_paramsweep(run_jaxley, summary, n=50):
    counts_n, counts_j = [], []
    for i in range(n):
        try:
            npz = _load_neuron(f"test5_sweep_{i:03d}")
        except FileNotFoundError:
            break
        v_n, i_stim = npz["v_soma"], npz["i_stim"]
        gbar = _gbar_from_npz(npz)
        _, v_j, _ = run_jaxley(gbar, i_stim, TSTOP_DEFAULT, DT_OUT)
        pk_n, _ = detect_spikes(v_n, DT_OUT)
        pk_j, _ = detect_spikes(v_j, DT_OUT)
        counts_n.append(len(pk_n))
        counts_j.append(len(pk_j))

    counts_n = np.array(counts_n); counts_j = np.array(counts_j)
    if len(counts_n) > 1 and counts_n.std() > 0 and counts_j.std() > 0:
        rho = float(np.corrcoef(counts_n, counts_j)[0, 1])
    else:
        rho = float("nan")
    passed = (len(counts_n) > 0) and (rho > 0.95)
    summary.add("test5_paramsweep", f"corr (n={len(counts_n)})", "—", "—", rho, passed)

    fig, ax = plt.subplots(figsize=(6, 6))
    mx = max(counts_n.max() if len(counts_n) else 1, counts_j.max() if len(counts_j) else 1, 1)
    ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.5)
    ax.scatter(counts_n, counts_j, s=20, alpha=0.7)
    ax.set_xlabel("NEURON spike count"); ax.set_ylabel("Jaxley spike count")
    ax.set_title(f"Test 5 — param sweep  corr={rho:.3f}  pass={passed}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test5_paramsweep_overlay.png", dpi=120)
    plt.close(fig)


def run_test6_apshape(run_jaxley, summary):
    npz = _load_neuron("test6_apshape")
    t_n, v_n, i_stim = npz["t"], npz["v_soma"], npz["i_stim"]
    gbar = _gbar_from_npz(npz)
    _, v_j, _ = run_jaxley(gbar, i_stim, TSTOP_AP, DT_OUT)
    n = min(len(v_n), len(v_j))
    pk_n, _ = detect_spikes(v_n[:n], DT_OUT)
    pk_j, _ = detect_spikes(v_j[:n], DT_OUT)
    if len(pk_n) == 0 or len(pk_j) == 0:
        summary.add("test6_apshape", "spike found", len(pk_n), len(pk_j),
                     len(pk_n) - len(pk_j), False)
        _plot_overlay("test6_apshape", t_n, v_n, t_n[:n], v_j[:n], i_stim,
                       "Test 6 — AP shape (no spike on at least one stack)")
        return
    pi_n, pi_j = int(pk_n[0]), int(pk_j[0])
    peak_d = float(abs(v_n[pi_n] - v_j[pi_j]))
    hw_n = half_width(v_n[:n], pi_n, DT_OUT)
    hw_j = half_width(v_j[:n], pi_j, DT_OUT)
    ahp_n = ahp_depth(v_n[:n], pi_n, DT_OUT)
    ahp_j = ahp_depth(v_j[:n], pi_j, DT_OUT)
    ok = (peak_d < 3.0) and (abs(hw_n - hw_j) < 0.3) and (abs(ahp_n - ahp_j) < 3.0)
    summary.add("test6_apshape", "peak ΔV (mV)", float(v_n[pi_n]), float(v_j[pi_j]),
                 peak_d, ok)
    summary.add("test6_apshape", "½-width (ms)", hw_n, hw_j, abs(hw_n - hw_j), ok)
    summary.add("test6_apshape", "AHP depth (mV)", ahp_n, ahp_j, abs(ahp_n - ahp_j), ok)
    _plot_overlay("test6_apshape", t_n[:n], v_n[:n], t_n[:n], v_j[:n], i_stim,
                   f"Test 6 — AP shape.  peakΔ={peak_d:.2f}  ½wΔ={abs(hw_n-hw_j):.2f}  "
                   f"AHPΔ={abs(ahp_n-ahp_j):.2f}  pass={ok}")


def run_test7_walltime(run_jaxley, summary, n=20):
    """Jaxley wall-time + comparison vs NEURON's saved value."""
    # Quick one-time JIT warmup
    npz = _load_neuron("test3_suprathresh")
    gbar = _gbar_from_npz(npz)
    _, _, _ = run_jaxley(gbar, npz["i_stim"], TSTOP_DEFAULT, DT_OUT)  # warmup
    times = []
    for _ in range(n):
        _, _, w = run_jaxley(gbar, npz["i_stim"], TSTOP_DEFAULT, DT_OUT)
        times.append(w)
    times = np.array(times)
    summary.add("test7_walltime", "Jaxley mean (ms/trace)", "—", "—",
                 float(times.mean() * 1000), True)

    nrn_path = NEURON_RESULTS / "test7_walltime.npz"
    if nrn_path.exists():
        nz = np.load(nrn_path)
        with open(PLOTS_DIR / "walltime.txt", "w") as f:
            f.write(
                f"NEURON  mean: {float(nz['mean_s'])*1000:.1f} ms/trace, "
                f"std {float(nz['std_s'])*1000:.1f}, n={int(nz['n'])}\n"
                f"Jaxley  mean: {times.mean()*1000:.1f} ms/trace, "
                f"std {times.std()*1000:.1f}, n={n}  (warm; first JIT call excluded)\n"
            )


# ── CLI ─────────────────────────────────────────────────────────────────────

_ALL = ["rest", "subthresh", "suprathresh", "fI", "paramsweep", "apshape", "walltime"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", nargs="+", default=["all"])
    p.add_argument("--n-walltime", type=int, default=20)
    args = p.parse_args()

    tests = _ALL if "all" in args.tests else args.tests
    print(f"Running CA3 comparison: {tests}")

    run_jaxley = _jaxley_runner()
    summary = Summary()

    if "rest"        in tests: run_test1_rest(run_jaxley, summary)
    if "subthresh"   in tests: run_test2_subthresh(run_jaxley, summary)
    if "suprathresh" in tests: run_test3_suprathresh(run_jaxley, summary)
    if "fI"          in tests: run_test4_fI(run_jaxley, summary)
    if "paramsweep"  in tests: run_test5_paramsweep(run_jaxley, summary)
    if "apshape"     in tests: run_test6_apshape(run_jaxley, summary)
    if "walltime"    in tests: run_test7_walltime(run_jaxley, summary, n=args.n_walltime)

    summary.write_csv(PLOTS_DIR / "summary.csv")

    # Pretty-print
    print()
    print("=" * 90)
    print(f"{'test':<22}{'metric':<32}{'NEURON':>12}{'Jaxley':>12}{'Δ':>10}  pass")
    print("-" * 90)
    for r in summary.rows:
        nv = f"{r[2]:.3f}" if isinstance(r[2], float) else str(r[2])
        jv = f"{r[3]:.3f}" if isinstance(r[3], float) else str(r[3])
        dv = f"{r[4]:.3f}" if isinstance(r[4], float) else str(r[4])
        print(f"{r[0]:<22}{r[1]:<32}{nv:>12}{jv:>12}{dv:>10}  {r[5]}")
    print("=" * 90)


if __name__ == "__main__":
    main()
