#!/usr/bin/env python
"""Forward + backward throughput across all jaxley cells in the repo.

Measures the *training-step* cost (forward sim + gradient of a voltage loss
w.r.t. the trainable params) by backpropping through `JaxleyBridge.simulate_batch`
— the exact autograd path `HybridLoss` uses during training.  Reports sims/sec
(traces per second) for forward-only and forward+backward, at a batch size.

Cells use their **default** physical parameters (data already in the repo via
`bench_jaxley_cells._default_params_tensor`), so nothing needs to be generated.

Run on a GPU node inside the conda env:

    salloc -C gpu -q interactive -t1:00:00 --gpus-per-task=1 -A m2043_g \
           --ntasks-per-node=1 -N 1
    module load conda && conda activate \
           /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
    python toolbox/tests/bench_fwd_bwd.py --batch 16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import jax

from toolbox import JaxleyBridge
from toolbox.tests.bench_jaxley_cells import _default_params_tensor


# All cells implemented in the repo + the two L5 cells under comparison.
DEFAULT_CELLS = [
    "ball_and_stick",       # HH soma + passive dendrite (cheapest)
    "ball_and_stick_bbp",   # BBP-channel ball & stick
    "ca3_pyramidal",        # single-compartment CA3 soma
    "L5PC_jaxley",          # existing jaxley morphology + jaxley_mech channels
    "L5TTPC",               # NEURON-converted BBP L5_TTPC1 morphology
]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, n_warm, n_timed):
    for _ in range(n_warm):
        fn(); _sync()
    ts = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn(); _sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return float(np.mean(ts)), ts[0], ts[-1]


def bench_cell(cell_name, batch, n_warm, n_timed, checkpoint=None):
    print(f"\n=== {cell_name} ===", flush=True)
    try:
        p_cpu = _default_params_tensor(cell_name, batch=batch, dtype=torch.float32)
    except Exception as e:
        print(f"  [skip] no default params: {e}"); return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = p_cpu.to(dev)
    ckpt = tuple(checkpoint) if checkpoint else None
    if ckpt:
        print(f"  (gradient checkpointing: checkpoint_lengths={ckpt}, batch={batch})")

    # --- forward only ---
    def fwd():
        with torch.no_grad():
            JaxleyBridge.simulate_batch(p, cell_name, checkpoint_lengths=ckpt)
    try:
        f_mean, f_lo, f_hi = _time(fwd, n_warm, n_timed)
    except Exception as e:
        print(f"  [skip] forward failed: {e}"); return None

    # --- forward + backward (voltage MSE-style loss) ---
    pg = p.clone().requires_grad_(True)

    def fwd_bwd():
        if pg.grad is not None:
            pg.grad = None
        v = JaxleyBridge.simulate_batch(pg, cell_name, checkpoint_lengths=ckpt)
        loss = v.pow(2).mean()      # cheap scalar; exercises full vjp
        loss.backward()
    try:
        b_mean, b_lo, b_hi = _time(fwd_bwd, n_warm, n_timed)
    except Exception as e:
        print(f"  [skip] backward failed: {e}"); b_mean = b_lo = b_hi = float("nan")

    out_shape = tuple(JaxleyBridge.simulate_batch(p, cell_name, checkpoint_lengths=ckpt).shape)
    f_sps = batch / f_mean
    b_sps = batch / b_mean if b_mean == b_mean else float("nan")
    print(f"  out shape:   {out_shape}")
    print(f"  forward    : {f_mean*1e3:8.1f} ms/batch -> {f_sps:8.1f} sims/s "
          f"(p5={f_lo*1e3:.0f} p95={f_hi*1e3:.0f})")
    print(f"  fwd+bwd    : {b_mean*1e3:8.1f} ms/batch -> {b_sps:8.1f} sims/s "
          f"(p5={b_lo*1e3:.0f} p95={b_hi*1e3:.0f})")
    if b_mean == b_mean:
        print(f"  bwd/fwd ratio: {b_mean/f_mean:.2f}x")
    return dict(name=cell_name, batch=batch, f_sps=f_sps, b_sps=b_sps,
                f_ms=f_mean*1e3, b_ms=b_mean*1e3, out_shape=out_shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="comma-separated checkpoint_lengths, e.g. 50,100 (for large cells)")
    args = ap.parse_args()
    ckpt = [int(x) for x in args.checkpoint.split(",")] if args.checkpoint else None

    plat = ", ".join(d.platform for d in jax.devices())
    print(f"jax devices: {len(jax.devices())} x {plat} | torch.cuda={torch.cuda.is_available()}")
    print(f"batch={args.batch}  warm={args.warm}  timed={args.timed}  dtype=fp32")

    rows = [r for r in (bench_cell(c, args.batch, args.warm, args.timed, ckpt) for c in args.cells) if r]

    print("\n" + "=" * 84)
    print(f"{'cell':<20}{'fwd ms':>10}{'fwd sims/s':>13}{'fwd+bwd ms':>13}"
          f"{'fwd+bwd sims/s':>16}{'bwd/fwd':>10}")
    print("-" * 84)
    for r in rows:
        ratio = r["b_ms"] / r["f_ms"] if r["b_ms"] == r["b_ms"] else float("nan")
        print(f"{r['name']:<20}{r['f_ms']:>10.1f}{r['f_sps']:>13.1f}"
              f"{r['b_ms']:>13.1f}{r['b_sps']:>16.1f}{ratio:>10.2f}")
    print("=" * 84)
    print(f"(batch={args.batch}; sims/s = traces per second; fwd+bwd = full training-step cost)")


if __name__ == "__main__":
    main()
