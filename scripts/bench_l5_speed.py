#!/usr/bin/env python
"""Time the exact TRAINING path (JaxleyBridge.simulate_batch fwd + backward of a voltage
loss) for the L5 cell at the current L5TTPC_NCOMP, over batch sizes and solver time steps.
Precision = JAX_ENABLE_X64 (true -> float64 params, as HybridLoss fp64: True).

    L5TTPC_NCOMP=2 JAX_ENABLE_X64=true python scripts/bench_l5_speed.py \\
        --batches 64,128,256 --dts 0.1 --ckpt 100,10 --out docs/model_ladder/speed.csv
Appends one CSV row per config: ncomp, precision, dt, t_max, B, ckpt, compile_s, fwd_ms,
fwd_bwd_ms, gpu_s_per_sim (fwd_bwd), sims_per_s, peak_GB.
"""
import argparse, os, sys, time, csv
os.environ.setdefault("JAX_PLATFORMS", "cuda"); os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
import numpy as np
sys.path.insert(0, os.getcwd())
import torch, jax
from toolbox import JaxleyBridge
from toolbox.jaxley_cells import l5ttpc

def peak_gb():
    try: return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
    except Exception: return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="64,128"); ap.add_argument("--dts", default="0.1")
    ap.add_argument("--ckpt", default="100,10", help="checkpoint_lengths or 'none'")
    ap.add_argument("--tmax", type=float, default=500.0); ap.add_argument("--stim", default="5k50kInterChaoticB")
    ap.add_argument("--iters", type=int, default=3); ap.add_argument("--out", required=True)
    ap.add_argument("--fwd-only", action="store_true")
    a = ap.parse_args()
    x64 = jax.config.jax_enable_x64; dtype = torch.float64 if x64 else torch.float32
    prec = "fp64" if x64 else "fp32"; ncomp = l5ttpc._NCOMP
    dev = torch.device("cuda")
    defaults = torch.tensor([[l5ttpc._DEFAULTS[k] for k in l5ttpc.PARAM_KEYS]], dtype=dtype)
    new = not os.path.exists(a.out)
    w = csv.writer(open(a.out, "a"))
    if new: w.writerow(["ncomp", "precision", "dt_ms", "t_max_ms", "B", "ckpt", "compile_s", "fwd_ms", "fwd_bwd_ms", "gpu_s_per_sim", "sims_per_s", "peak_GB"])
    for dt in [float(x) for x in a.dts.split(",")]:
        l5ttpc._DT = dt; l5ttpc._T_MAX = a.tmax; JaxleyBridge.clear_cache()
        steps = int(round(a.tmax / dt))
        for B in [int(x) for x in a.batches.split(",")]:
            ckpt = None if a.ckpt == "none" else tuple(int(x) for x in a.ckpt.split(","))
            if ckpt is not None:            # outer*inner must equal the step count
                inner = ckpt[1]; outer = int(np.ceil((steps + 1) / inner)); ckpt = (outer, inner)   # jaxley needs prod >= n_steps+1
            p = defaults.repeat(B, 1).to(dev).requires_grad_(True)
            try:
                t0 = time.perf_counter()
                v = JaxleyBridge.simulate_batch(p, "l5ttpc", a.stim, checkpoint_lengths=ckpt)
                torch.cuda.synchronize(); tgt = v.detach() * 0.9
                if not a.fwd_only:
                    ((v - tgt) ** 2).mean().backward(); torch.cuda.synchronize()
                comp = time.perf_counter() - t0
                fwd = []; fb = []
                for _ in range(a.iters):
                    p.grad = None; t1 = time.perf_counter()
                    v = JaxleyBridge.simulate_batch(p, "l5ttpc", a.stim, checkpoint_lengths=ckpt); torch.cuda.synchronize()
                    t2 = time.perf_counter(); fwd.append(t2 - t1)
                    if not a.fwd_only:
                        ((v - tgt) ** 2).mean().backward(); torch.cuda.synchronize()
                    fb.append(time.perf_counter() - t1)
                fwd_ms = 1000 * float(np.median(fwd)); fb_ms = 1000 * float(np.median(fb))
                row = [ncomp, prec, dt, a.tmax, B, a.ckpt, f"{comp:.1f}", f"{fwd_ms:.0f}", f"{fb_ms:.0f}", f"{fb_ms/1000/B:.4f}", f"{B/(fb_ms/1000):.2f}", f"{peak_gb():.1f}"]
            except Exception as e:
                msg = str(e).split("\n")[0][:80]
                row = [ncomp, prec, dt, a.tmax, B, a.ckpt, "", "", f"ERR {msg}", "", "", f"{peak_gb():.1f}"]
                JaxleyBridge.clear_cache()
            print("[bench]", row, flush=True); w.writerow(row)
            del p; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
