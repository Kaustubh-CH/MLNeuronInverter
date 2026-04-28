"""Solver + cell + batch sweep bench at dt=0.1 ms, t_max=100 ms.

Single-thread CPU.  Does NOT go through `JaxleyBridge` — builds its own
jit'd vmap simulate closure per (cell, solver) combo so the bridge's
cache keying doesn't need to grow config knobs that only the bench uses.

Modes
-----
Full matrix (default):

    python -m toolbox.tests.bench_solvers

Scaling-only (called repeatedly from a shell loop with different
OMP_NUM_THREADS values; appends one row to scaling_log.csv each call):

    OMP_NUM_THREADS=4 python -m toolbox.tests.bench_solvers --only-scaling

Outputs (under `$NEUINV_BENCH_OUT`, default
`/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_solver_bench/`):

    traces/<cell>_<solver>.png   voltage trace at B=1 with reference overlay
    summary_timing.csv           full matrix timing
    summary_timing.txt           human-readable sibling
    summary_bars.png             grouped bars at B=16
    scaling_log.csv              one row per scaling-mode invocation
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

# Single-thread CPU — must be set before jax/numpy pick up thread counts.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "inter_op_parallelism_threads=1",
)

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox import jaxley_cells, jaxley_utils


OUT_DIR = Path(
    os.environ.get(
        "NEUINV_BENCH_OUT",
        "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_solver_bench",
    )
)

DT = 0.1
T_MAX = 100.0

REF_SOLVER = "bwd_euler"
REF_DT = 0.025

CELLS_DEFAULT = ["single_comp", "ball_and_stick", "L5TTPC"]
SOLVERS_DEFAULT = ["bwd_euler", "crank_nicolson", "exp_euler", "fwd_euler"]
BATCHES_DEFAULT = [1, 4, 16, 64]


# ─────────────────────────────────────────────────────────────────────────
# defaults
# ─────────────────────────────────────────────────────────────────────────

# L5TTPC defaults ported from toolbox/tests/bench_jaxley_cells.py; matches
# biophysics.hoc values used by the canonical NEURON reference.
_L5TTPC_DEFAULTS = [
    0.026145, 0.004226, 0.000143, 0.000080, 3.137968,
    0.089259, 0.006827, 0.007104, 0.000990, 0.973538,
    0.008752, 3e-5, 1.0, 0.303472, 0.983955,
    0.000333, 3e-5, 1.0, -75.0,
]


def _default_params(cell_name):
    if cell_name == "single_comp":
        from toolbox.jaxley_cells.soma_only import _DEFAULTS, PARAM_KEYS
        return [_DEFAULTS[k] for k in PARAM_KEYS]
    if cell_name == "ball_and_stick":
        from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
        return [_DEFAULTS[k] for k in PARAM_KEYS]
    if cell_name in ("L5TTPC", "L5_TTPC1cADpyr0"):
        return list(_L5TTPC_DEFAULTS)
    raise KeyError(cell_name)


# ─────────────────────────────────────────────────────────────────────────
# simulate-builder (no bridge)
# ─────────────────────────────────────────────────────────────────────────

def build_simulate(cell_name, solver, dt, t_max):
    """Return (jit-vmapped simulate_batch, P) for the given config."""
    import jaxley as jx

    spec = jaxley_cells.get(cell_name)
    cell, entry_to_cnn_idx = spec.build_fn()

    stim_path = Path(spec.stim_dir) / f"{spec.default_stim_name}.csv"
    stim_csv = jaxley_utils.load_stim_csv(stim_path)
    stim_up = jaxley_utils.upsample_stim(stim_csv, spec.dt_stim, dt, t_max)
    stim_jnp = jnp.asarray(stim_up[np.newaxis, :])
    data_stim = spec.stim_attach_fn(cell, stim_jnp)

    default_params = cell.get_parameters()

    def _key(entry):
        (k,) = entry.keys()
        return k

    default_shapes = [e[_key(e)].shape for e in default_params]

    def simulate_one(flat_phys):
        params = []
        for e, idx, shape in zip(default_params, entry_to_cnn_idx, default_shapes):
            k = _key(e)
            val = jnp.broadcast_to(flat_phys[idx:idx + 1], shape)
            params.append({k: val})
        v = jx.integrate(
            cell,
            params=params,
            delta_t=dt,
            t_max=t_max,
            data_stimuli=data_stim,
            solver=solver,
        )
        return v

    sim = jax.jit(jax.vmap(simulate_one, in_axes=0))
    return sim, len(spec.param_keys)


# ─────────────────────────────────────────────────────────────────────────
# timing
# ─────────────────────────────────────────────────────────────────────────

def time_warm(sim, params, n_warm=1, n_timed=3):
    """Return (min_s, mean_s) across `n_timed` timed calls after `n_warm`."""
    for _ in range(n_warm):
        out = sim(params)
        _ = np.asarray(out)
    ts = []
    for _ in range(n_timed):
        t0 = time.time()
        out = sim(params)
        _ = np.asarray(out)
        ts.append(time.time() - t0)
    return float(min(ts)), float(np.mean(ts))


# ─────────────────────────────────────────────────────────────────────────
# matrix runner
# ─────────────────────────────────────────────────────────────────────────

def _extract_soma_trace(v_np):
    """`v` shape can be (B, n_rec, T) or (n_rec, T).  Return trace of first
    sample's first recorded compartment."""
    if v_np.ndim == 3:
        return v_np[0, 0]
    if v_np.ndim == 2:
        return v_np[0]
    return v_np


def run_matrix(cells, solvers, batches, build_reference=True):
    rows = []
    plot_traces = {}     # (cell, solver) -> trace
    reference_traces = {}  # cell -> trace  (bwd_euler, dt=0.025)

    if build_reference:
        for cell in cells:
            if cell in ("L5TTPC", "L5_TTPC1cADpyr0"):
                print(f"[ref] skipping {cell} reference (too expensive).")
                continue
            print(f"[ref] {cell}  solver={REF_SOLVER}  dt={REF_DT}  t_max={T_MAX}")
            try:
                sim_ref, P = build_simulate(cell, REF_SOLVER, REF_DT, T_MAX)
                p = jnp.broadcast_to(
                    jnp.asarray(_default_params(cell), dtype=jnp.float64), (1, P)
                )
                v_ref = np.asarray(sim_ref(p))
                reference_traces[cell] = _extract_soma_trace(v_ref)
            except Exception:
                traceback.print_exc()
            jax.clear_caches()

    for cell in cells:
        for solver in solvers:
            print(f"\n=== {cell} × {solver} ===", flush=True)
            cold_s = None
            note = ""
            try:
                t0 = time.time()
                sim, P = build_simulate(cell, solver, DT, T_MAX)
                p_def = jnp.asarray(_default_params(cell), dtype=jnp.float64)

                # First B=1 call  → tracing + compile.
                p1 = jnp.broadcast_to(p_def, (1, P))
                v1 = sim(p1)
                v1_np = np.asarray(v1)
                cold_s = time.time() - t0

                plot_traces[(cell, solver)] = _extract_soma_trace(v1_np)

                if not np.all(np.isfinite(v1_np)):
                    print(f"  UNSTABLE (NaN/Inf in trace), cold={cold_s:.2f}s")
                    rows.append(dict(
                        cell=cell, solver=solver, B=1,
                        cold_s=f"{cold_s:.3f}", warm_s="", sims_per_sec="",
                        note="UNSTABLE",
                    ))
                    continue

                for B in batches:
                    try:
                        pB = jnp.broadcast_to(p_def, (B, P))
                        fastest, mean = time_warm(sim, pB, n_warm=1, n_timed=3)
                        sps = B / mean
                        print(
                            f"  B={B:4d}  cold={cold_s or 0:6.2f}s  "
                            f"warm(min)={fastest*1000:7.1f}ms  "
                            f"warm(mean)={mean*1000:7.1f}ms  "
                            f"{sps:8.2f} sims/s",
                            flush=True,
                        )
                        rows.append(dict(
                            cell=cell, solver=solver, B=B,
                            cold_s=f"{cold_s:.3f}" if B == batches[0] else "",
                            warm_s=f"{mean:.6f}",
                            sims_per_sec=f"{sps:.3f}",
                            note=note,
                        ))
                    except Exception as e:
                        traceback.print_exc()
                        rows.append(dict(
                            cell=cell, solver=solver, B=B,
                            cold_s="", warm_s="", sims_per_sec="",
                            note=f"ERROR {type(e).__name__}: {e}",
                        ))
            except Exception as e:
                traceback.print_exc()
                rows.append(dict(
                    cell=cell, solver=solver, B=0,
                    cold_s="", warm_s="", sims_per_sec="",
                    note=f"BUILD-ERROR {type(e).__name__}: {e}",
                ))
            finally:
                jax.clear_caches()

    return rows, plot_traces, reference_traces


# ─────────────────────────────────────────────────────────────────────────
# output
# ─────────────────────────────────────────────────────────────────────────

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cell", "solver", "B", "cold_s", "warm_s", "sims_per_sec", "note"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_text_summary(rows, path):
    lines = []
    lines.append(f"{'cell':<16s} {'solver':<15s} {'B':>4s}  {'warm_ms':>9s}  {'sims/s':>10s}  note")
    lines.append("-" * 74)
    for r in rows:
        warm = f"{float(r['warm_s'])*1000:.1f}" if r["warm_s"] else "-"
        sps = f"{float(r['sims_per_sec']):.2f}" if r["sims_per_sec"] else "-"
        lines.append(
            f"{r['cell']:<16s} {r['solver']:<15s} {str(r['B']):>4s}  "
            f"{warm:>9s}  {sps:>10s}  {r['note']}"
        )
    path.write_text("\n".join(lines) + "\n")


def save_trace_plots(plot_traces, reference_traces, out_dir, t_max, dt):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (cell, solver), trace in plot_traces.items():
        fig, ax = plt.subplots(figsize=(9, 3.2))
        n = len(trace)
        t_axis = np.linspace(0.0, t_max, n)
        ax.plot(t_axis, trace, label=f"{solver} dt={dt}", lw=1.2)

        if cell in reference_traces:
            ref = reference_traces[cell]
            tr = np.linspace(0.0, t_max, len(ref))
            ax.plot(tr, ref, label=f"ref (bwd_euler dt={REF_DT})", lw=0.9, ls="--", alpha=0.8)

        ax.set_title(f"{cell}  ·  {solver}  ·  dt={dt} ms  t_max={t_max} ms  (single-thread CPU)")
        ax.set_xlabel("t (ms)"); ax.set_ylabel("V (mV)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cell}_{solver}.png", dpi=110)
        plt.close(fig)


def save_bars(rows, out_path, batch_of_interest=16):
    entries = [r for r in rows if str(r["B"]) == str(batch_of_interest) and r["warm_s"]]
    if not entries:
        print(f"  (no B={batch_of_interest} rows — skipping bar chart)")
        return
    cells = sorted({r["cell"] for r in entries})
    solvers = sorted({r["solver"] for r in entries})
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cells))
    width = 0.8 / max(len(solvers), 1)
    for i, s in enumerate(solvers):
        vals = []
        for c in cells:
            m = [r for r in entries if r["cell"] == c and r["solver"] == s]
            vals.append(float(m[0]["warm_s"]) * 1000 if m else 0.0)
        ax.bar(x + (i - (len(solvers) - 1) / 2) * width, vals, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_ylabel(f"warm time / batch, B={batch_of_interest} (ms)")
    ax.set_title(f"Jaxley solver timing · dt={DT} ms · t_max={T_MAX} ms · single-thread CPU")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# modes
# ─────────────────────────────────────────────────────────────────────────

def cmd_main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = args.cells or CELLS_DEFAULT
    solvers = args.solvers or SOLVERS_DEFAULT
    batches = args.batches or BATCHES_DEFAULT

    print(f"jax devices: {[d.platform for d in jax.devices()]}")
    print(f"env: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')} "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}")
    print(f"cells   = {cells}")
    print(f"solvers = {solvers}")
    print(f"batches = {batches}")
    print(f"dt={DT} ms  t_max={T_MAX} ms  ->  steps={int(round(T_MAX/DT))}\n", flush=True)

    rows, plot_traces, reference_traces = run_matrix(
        cells, solvers, batches, build_reference=not args.skip_reference
    )

    save_csv(rows, OUT_DIR / "summary_timing.csv")
    save_text_summary(rows, OUT_DIR / "summary_timing.txt")
    save_trace_plots(plot_traces, reference_traces, OUT_DIR / "traces", T_MAX, DT)
    save_bars(rows, OUT_DIR / "summary_bars.png")

    print(f"\nwrote results to {OUT_DIR}")


def cmd_scaling(args):
    """Run one (cell, solver, B) config and append timing to scaling_log.csv.

    Meant to be called repeatedly from a shell loop with different
    OMP_NUM_THREADS values — the shell records the thread count in env and
    this script reads it back for logging.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cell = args.scaling_cell
    solver = args.scaling_solver
    B = args.scaling_batch

    t0 = time.time()
    sim, P = build_simulate(cell, solver, DT, T_MAX)
    p_def = jnp.asarray(_default_params(cell), dtype=jnp.float64)
    pB = jnp.broadcast_to(p_def, (B, P))
    _ = np.asarray(sim(pB))  # warm / compile
    _ = np.asarray(sim(pB))  # warm 2
    cold = time.time() - t0

    ts = []
    for _ in range(5):
        t_s = time.time()
        _ = np.asarray(sim(pB))
        ts.append(time.time() - t_s)
    warm = float(np.mean(ts))

    threads = os.environ.get("OMP_NUM_THREADS", "?")
    line = (
        f"threads={threads:>3}  cell={cell}  solver={solver}  B={B}  "
        f"cold={cold:.2f}s  warm={warm*1000:.1f}ms  sims/s={B/warm:.2f}"
    )
    print(line)
    log_path = OUT_DIR / "scaling_log.csv"
    header_needed = not log_path.exists()
    with log_path.open("a") as f:
        if header_needed:
            f.write("threads,cell,solver,B,cold_s,warm_ms,sims_per_sec\n")
        f.write(f"{threads},{cell},{solver},{B},{cold:.3f},{warm*1000:.3f},{B/warm:.3f}\n")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--solvers", nargs="*")
    ap.add_argument("--batches", nargs="*", type=int)
    ap.add_argument("--skip-reference", action="store_true",
                    help="skip the dt=0.025 bwd_euler reference trace")
    ap.add_argument("--only-scaling", action="store_true",
                    help="run one config and append to scaling_log.csv")
    ap.add_argument("--scaling-cell", default="ball_and_stick")
    ap.add_argument("--scaling-solver", default="bwd_euler")
    ap.add_argument("--scaling-batch", type=int, default=16)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.only_scaling:
        cmd_scaling(args)
    else:
        cmd_main(args)


if __name__ == "__main__":
    main()
