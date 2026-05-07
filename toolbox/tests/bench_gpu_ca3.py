"""GPU bench for CA3 Pyramidal — t_max=500 ms apples-to-apples vs NEURON.

Sweeps batch size in fp32 (forward + forward+backward) and adds two fp64
checks at the largest batch we know the device can hold.

Output: <out>/bench_ca3_gpu.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import torch

CELL_DEFAULT = "ca3_pyramidal"
T_MAX = 500.0
DT    = 0.1


def _build(solver: str, fp64: bool, cell_name: str = CELL_DEFAULT):
    if fp64:
        os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    if fp64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jaxley as jx
    from toolbox import jaxley_cells, jaxley_utils

    spec = jaxley_cells.get(cell_name)
    cell, idx_map = spec.build_fn()

    stim_path = Path(spec.stim_dir) / f"{spec.default_stim_name}.csv"
    stim_csv  = jaxley_utils.load_stim_csv(stim_path)
    stim_up   = jaxley_utils.upsample_stim(stim_csv, spec.dt_stim, DT, T_MAX)
    stim_jnp  = jnp.asarray(stim_up[np.newaxis, :], dtype=jnp.float64 if fp64 else jnp.float32)
    data_stim = spec.stim_attach_fn(cell, stim_jnp)

    default_params = cell.get_parameters()
    keys           = [next(iter(e.keys())) for e in default_params]
    shapes         = [e[k].shape for e, k in zip(default_params, keys)]

    import importlib
    cell_mod = importlib.import_module(f"toolbox.jaxley_cells.{cell_name}")
    default_row = jnp.asarray(
        [cell_mod._DEFAULTS[k] for k in cell_mod.PARAM_KEYS],
        dtype=jnp.float64 if fp64 else jnp.float32,
    )

    def loss_one(flat_phys):
        params = []
        for k, idx, shape in zip(keys, idx_map, shapes):
            val = jnp.broadcast_to(flat_phys[idx:idx + 1], shape)
            params.append({k: val})
        v = jx.integrate(
            cell, params=params, delta_t=DT, t_max=T_MAX,
            data_stimuli=data_stim, solver=solver,
        )
        return jnp.sum(v ** 2)

    return loss_one, default_row, jax


def _time(fn, x, n_warm=2, n_timed=5):
    import jax
    for _ in range(n_warm):
        out = fn(x); jax.block_until_ready(out)
    ts = []
    for _ in range(n_timed):
        t0 = time.time(); out = fn(x); jax.block_until_ready(out)
        ts.append(time.time() - t0)
    ts.sort()
    return ts[0], float(np.mean(ts)), ts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell",    default=CELL_DEFAULT,
                    help="Registered jaxley cell name (default: %(default)s)")
    ap.add_argument("--solver",  default="bwd_euler")
    ap.add_argument("--batches", nargs="+", type=int,
                    default=[1, 16, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--out-dir", default="/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_gpu_bench")
    ap.add_argument("--fp64-batch", type=int, default=256)
    ap.add_argument("--out-name", default=None,
                    help="output CSV filename stem (default: bench_<cell>_gpu)")
    ap.add_argument("--skip-fp64", action="store_true",
                    help="skip the fp64 spot check")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = args.out_name or f"bench_{args.cell}_gpu"
    csv_path = out_dir / f"{out_stem}.csv"

    rows = []
    rows.append(["dtype", "mode", "B", "cold_s", "warm_ms", "p5_ms", "p95_ms",
                 "ms_per_trace", "traces_per_sec", "note"])

    # ── fp32 forward sweep ─────────────────────────────────────────────────
    loss_one, default_row, jax = _build(args.solver, fp64=False, cell_name=args.cell)
    print(f"[bench] cell={args.cell}  solver={args.solver}  t_max={T_MAX} ms  dt={DT} ms", flush=True)

    for mode in ("fwd", "fwd_bwd"):
        if mode == "fwd":
            fn = jax.jit(jax.vmap(loss_one))
        else:
            fn = jax.jit(jax.vmap(jax.value_and_grad(loss_one)))
        for B in args.batches:
            try:
                x = jax.numpy.broadcast_to(default_row, (B, default_row.shape[0]))
                t0 = time.time(); out = fn(x); jax.block_until_ready(out); cold = time.time() - t0
                p5, mean_s, p95 = _time(fn, x)
                ms_per_trace = mean_s * 1000.0 / B
                traces_s     = B / mean_s
                rows.append(["fp32", mode, B,
                             f"{cold:.2f}", f"{mean_s*1000:.2f}",
                             f"{p5*1000:.2f}", f"{p95*1000:.2f}",
                             f"{ms_per_trace:.3f}", f"{traces_s:.1f}", ""])
                print(f"[fp32 {mode}] B={B:5d}  cold={cold:6.2f}s  warm={mean_s*1000:8.2f}ms  "
                      f"per-trace={ms_per_trace:8.4f}ms  traces/s={traces_s:9.1f}", flush=True)
            except Exception as e:
                rows.append(["fp32", mode, B, "", "", "", "", "", "",
                             f"ERROR: {type(e).__name__}: {str(e)[:80]}"])
                print(f"[fp32 {mode}] B={B:5d}  ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)
        jax.clear_caches()

    if args.skip_fp64:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerows(rows)
        print(f"\nwrote {csv_path}", flush=True)
        return

    # ── fp64 spot check (just one batch size) ──────────────────────────────
    print("\n--- fp64 spot check ---", flush=True)
    loss_one, default_row, jax = _build(args.solver, fp64=True, cell_name=args.cell)
    for mode in ("fwd", "fwd_bwd"):
        if mode == "fwd":
            fn = jax.jit(jax.vmap(loss_one))
        else:
            fn = jax.jit(jax.vmap(jax.value_and_grad(loss_one)))
        try:
            x = jax.numpy.broadcast_to(default_row, (args.fp64_batch, default_row.shape[0]))
            t0 = time.time(); out = fn(x); jax.block_until_ready(out); cold = time.time() - t0
            p5, mean_s, p95 = _time(fn, x)
            ms_per_trace = mean_s * 1000.0 / args.fp64_batch
            traces_s     = args.fp64_batch / mean_s
            rows.append(["fp64", mode, args.fp64_batch,
                         f"{cold:.2f}", f"{mean_s*1000:.2f}",
                         f"{p5*1000:.2f}", f"{p95*1000:.2f}",
                         f"{ms_per_trace:.3f}", f"{traces_s:.1f}", ""])
            print(f"[fp64 {mode}] B={args.fp64_batch:5d}  cold={cold:6.2f}s  warm={mean_s*1000:8.2f}ms  "
                  f"per-trace={ms_per_trace:8.4f}ms  traces/s={traces_s:9.1f}", flush=True)
        except Exception as e:
            rows.append(["fp64", mode, args.fp64_batch, "", "", "", "", "", "",
                         f"ERROR: {type(e).__name__}: {str(e)[:80]}"])
            print(f"[fp64 {mode}] B={args.fp64_batch:5d}  ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"\nwrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
