"""GPU bench for the jaxley voltage-loss path.

Forward + forward+backward timing for ball_and_stick (control) and L5TTPC
(target), across `bwd_euler` and `crank_nicolson`, batch sizes in
`[1, 4, 16, 64, 128]`, `dt=0.1`, `t_max=100`. Built directly on
`jx.integrate` (NOT through JaxleyBridge) so the timing reflects the pure
ODE cost without torch <-> jax round-trip overhead.

The loss is `sum(v**2)` — a scalar; gradients flow through the integrator
back to the trainable parameter list. This mirrors what the Phase 2
HybridLoss path will do (sum-of-squared-error against a target trace,
gradient back to CNN params via the bridge), modulo the torch wrapper.

Usage:

    salloc -A m2043_g -q interactive -C gpu -t 04:00:00 -N 1 \\
           --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=4
    module load conda
    conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
    cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
    bash scripts/bench_gpu.slr

Output: $NEUINV_GPU_BENCH_OUT or
$PSCRATCH/tmp_neuInv/jaxley_gpu_bench/bench_gpu.csv

CSV columns: cell, solver, B, mode, cold_s, warm_ms, p5_ms, p95_ms,
sims_per_sec, peak_mem_mb, note
"""

import argparse
import csv
import os
import time
from pathlib import Path

# Prefer GPU; fall back to CPU.  The wrapper SLURM script sets this anyway.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from toolbox import jaxley_cells, jaxley_utils
from toolbox.tests.bench_jaxley_cells import _default_params_tensor


CELLS_DEFAULT   = ["ball_and_stick", "L5TTPC"]
SOLVERS_DEFAULT = ["bwd_euler", "crank_nicolson"]
BATCHES_DEFAULT = [1, 4, 16, 64, 128]
MODES_DEFAULT   = ["fwd", "fwd_bwd"]
DT              = 0.1
T_MAX           = 100.0

OUT_DIR_DEFAULT = Path(os.environ.get(
    "NEUINV_GPU_BENCH_OUT",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench",
))


def _build_loss_fn(cell_name: str, solver: str, checkpoint_lengths=None):
    """Build a `loss_one(flat_phys)` closure: P-vector → scalar loss."""
    import jaxley as jx
    spec = jaxley_cells.get(cell_name)
    cell, idx_map = spec.build_fn()

    stim_path = Path(spec.stim_dir) / f"{spec.default_stim_name}.csv"
    stim_csv  = jaxley_utils.load_stim_csv(stim_path)
    stim_up   = jaxley_utils.upsample_stim(stim_csv, spec.dt_stim, DT, T_MAX)
    stim_jnp  = jnp.asarray(stim_up[np.newaxis, :])
    data_stim = spec.stim_attach_fn(cell, stim_jnp)

    default_params = cell.get_parameters()
    keys           = [next(iter(e.keys())) for e in default_params]
    shapes         = [e[k].shape for e, k in zip(default_params, keys)]

    if len(default_params) != len(idx_map):
        raise RuntimeError(
            f"{cell_name}: {len(idx_map)} CNN indices vs {len(default_params)} "
            "trainable entries — spec/build mismatch."
        )

    def loss_one(flat_phys):
        params = []
        for k, idx, shape in zip(keys, idx_map, shapes):
            val = jnp.broadcast_to(flat_phys[idx:idx + 1], shape)
            params.append({k: val})
        v = jx.integrate(
            cell,
            params=params,
            delta_t=DT,
            t_max=T_MAX,
            data_stimuli=data_stim,
            solver=solver,
            checkpoint_lengths=checkpoint_lengths,
        )
        return jnp.sum(v ** 2)

    return loss_one


def _default_phys_jnp(cell_name: str, batch: int):
    p = _default_params_tensor(cell_name, batch=batch, dtype=torch.float32)
    return jnp.asarray(p.numpy())


def _time_fn(fn, x, n_warm: int = 2, n_timed: int = 5):
    for _ in range(n_warm):
        out = fn(x)
        jax.block_until_ready(out)
    ts = []
    for _ in range(n_timed):
        t0 = time.time()
        out = fn(x)
        jax.block_until_ready(out)
        ts.append(time.time() - t0)
    ts.sort()
    return ts[0], float(np.mean(ts)), ts[-1]


def _peak_mem_mb() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        return float(stats.get("peak_bytes_in_use", 0)) / 1e6
    except Exception:
        return -1.0


def _reset_peak_mem() -> None:
    try:
        d = jax.devices()[0]
        if hasattr(d, "memory_allocator_reset_peak_memory_stats"):
            d.memory_allocator_reset_peak_memory_stats()
    except Exception:
        pass


def run_combo(cell: str, solver: str, batches, mode: str, writer,
              checkpoint_lengths=None):
    label_extra = f" ckpt={checkpoint_lengths}" if checkpoint_lengths else ""
    print(f"\n=== {cell} × {solver} × mode={mode}{label_extra} ===", flush=True)
    try:
        loss_one = _build_loss_fn(cell, solver, checkpoint_lengths=checkpoint_lengths)
    except Exception as e:
        print(f"  build failed: {e}", flush=True)
        for B in batches:
            writer.writerow([cell, solver, B, mode, "", "", "", "", "", "",
                             f"BUILD_ERROR: {str(e)[:120]}"])
        return

    if mode == "fwd":
        fn = jax.jit(jax.vmap(loss_one))
    elif mode == "fwd_bwd":
        fn = jax.jit(jax.vmap(jax.value_and_grad(loss_one)))
    else:
        raise ValueError(mode)

    for B in batches:
        try:
            x = _default_phys_jnp(cell, B)
            _reset_peak_mem()

            t0  = time.time()
            out = fn(x)
            jax.block_until_ready(out)
            cold_s = time.time() - t0

            p5, mean_s, p95 = _time_fn(fn, x)
            mem_mb          = _peak_mem_mb()
            sims_per_sec    = B / mean_s
            writer.writerow([
                cell, solver, B, mode,
                f"{cold_s:.3f}", f"{mean_s*1000:.3f}",
                f"{p5*1000:.3f}", f"{p95*1000:.3f}",
                f"{sims_per_sec:.2f}", f"{mem_mb:.1f}", "",
            ])
            print(
                f"  B={B:4d}  cold={cold_s:7.2f}s  "
                f"warm={mean_s*1000:9.2f}ms  sims/s={sims_per_sec:8.2f}  "
                f"mem={mem_mb:7.0f}MB",
                flush=True,
            )
        except Exception as e:
            note = f"ERROR: {type(e).__name__}: {str(e)[:120]}"
            writer.writerow([cell, solver, B, mode, "", "", "", "", "", "", note])
            print(f"  B={B:4d}  {note}", flush=True)
    jax.clear_caches()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells",   nargs="+", default=CELLS_DEFAULT)
    ap.add_argument("--solvers", nargs="+", default=SOLVERS_DEFAULT)
    ap.add_argument("--batches", nargs="+", type=int, default=BATCHES_DEFAULT)
    ap.add_argument("--modes",   nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--l5ttpc-ncomp", type=int, default=None,
                    help="Override L5TTPC _NCOMP (C1 ablation). Default: file value (4).")
    ap.add_argument("--checkpoint-lengths", nargs="+", type=int, default=None,
                    help="Pass to jx.integrate for grad checkpointing (e.g. 10 100).")
    args = ap.parse_args()

    if args.l5ttpc_ncomp is not None:
        import toolbox.jaxley_cells.l5ttpc as _l5
        _l5._NCOMP = args.l5ttpc_ncomp
        print(f"override L5TTPC _NCOMP = {_l5._NCOMP}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    devs = jax.devices()
    print(f"jax devices: {[d.platform for d in devs]}  count={len(devs)}", flush=True)
    print(f"out dir:     {out_dir}", flush=True)

    csv_path = out_dir / "bench_gpu.csv"
    txt_path = out_dir / "bench_gpu.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cell", "solver", "B", "mode",
            "cold_s", "warm_ms", "p5_ms", "p95_ms",
            "sims_per_sec", "peak_mem_mb", "note",
        ])
        for cell in args.cells:
            for solver in args.solvers:
                for mode in args.modes:
                    run_combo(cell, solver, args.batches, mode, writer,
                              checkpoint_lengths=args.checkpoint_lengths)
                    f.flush()

    with open(csv_path) as f:
        lines = f.readlines()
    txt_path.write_text("".join(lines))
    print(f"\nwrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
