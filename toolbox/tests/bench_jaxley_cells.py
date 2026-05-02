"""Correctness + throughput benchmark for the Phase 1 jaxley cells.

Runs each registered cell with its **default** physical parameters on the
`5k50kInterChaoticB` stim, compares the soma voltage trace against the
pre-computed reference in
`/pscratch/sd/k/ktub1999/Neuron_Jaxley/results{,_detailed_morphology}/`,
and then times `simulate_batch` at several batch sizes.

Usage:

    module load conda
    conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
    python -m toolbox.tests.bench_jaxley_cells                   # both cells
    python -m toolbox.tests.bench_jaxley_cells ball_and_stick    # one cell

Writes a short report to stdout and to `docs/phase1/bench.txt` (overwrite).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# The bridge gates platform via JAX_PLATFORMS.  Prefer GPU when available;
# leave it to jaxlib to fall back to CPU otherwise.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax
import torch

# For fp64 sanity — cheaper jax overhead on CPU, and reference traces are
# fp32/fp64 mixed.  fp32 is fine for throughput; we only go fp64 for the
# correctness diff.
from toolbox import JaxleyBridge, jaxley_cells


REF_ROOT = Path("/pscratch/sd/k/ktub1999/Neuron_Jaxley")
REF_STIM = "5k50kInterChaoticB"
REF_PATHS = {
    "ball_and_stick": REF_ROOT / "results" / f"jaxley_{REF_STIM}.npz",
    "L5TTPC":         REF_ROOT / "results_detailed_morphology" / f"jaxley_{REF_STIM}_L5_sample_0.npz",
}

# Where to save the text report.  Default lives on $PSCRATCH so we don't
# eat the user's home quota.  Override with --report.
REPORT_DEFAULT = Path(os.environ.get(
    "NEUINV_BENCH_REPORT",
    "/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_bench.txt",
))


def _print(msg, log):
    print(msg)
    log.append(msg)


def _default_params_tensor(cell_name: str, batch: int, dtype: torch.dtype) -> torch.Tensor:
    """Return (batch, P) tensor filled with each cell's default phys values."""
    if cell_name == "ball_and_stick":
        from toolbox.jaxley_cells.ball_and_stick import _DEFAULTS, PARAM_KEYS
        row = [_DEFAULTS[k] for k in PARAM_KEYS]
    elif cell_name == "ball_and_stick_bbp":
        from toolbox.jaxley_cells.ball_and_stick_bbp import _DEFAULTS, PARAM_KEYS
        row = [_DEFAULTS[k] for k in PARAM_KEYS]
    elif cell_name in ("L5TTPC", "L5_TTPC1cADpyr0"):
        # Use values from the Neuron_Jaxley reference
        # biophysics.hoc defaults (see sim_jaxley_L5TTPC1.py build_cell).
        # Order must match l5ttpc.PARAM_KEYS.
        row = [
            0.026145,   # gNaTs2_tbar_NaTs2_t_apical
            0.004226,   # gSKv3_1bar_SKv3_1_apical
            0.000143,   # gImbar_Im_apical
            0.000080,   # gIhbar_Ih_dend (basal+apical base)
            3.137968,   # gNaTa_tbar_NaTa_t_axonal
            0.089259,   # gK_Tstbar_K_Tst_axonal
            0.006827,   # gNap_Et2bar_Nap_Et2_axonal
            0.007104,   # gSK_E2bar_SK_E2_axonal
            0.000990,   # gCa_HVAbar_Ca_HVA_axonal
            0.973538,   # gK_Pstbar_K_Pst_axonal
            0.008752,   # gCa_LVAstbar_Ca_LVAst_axonal
            3e-5,       # g_pas_axonal
            1.0,        # cm_axonal
            0.303472,   # gSKv3_1bar_SKv3_1_somatic
            0.983955,   # gNaTs2_tbar_NaTs2_t_somatic
            0.000333,   # gCa_LVAstbar_Ca_LVAst_somatic
            3e-5,       # g_pas_somatic
            1.0,        # cm_somatic
            -75.0,      # e_pas_all
        ]
    else:
        raise KeyError(cell_name)
    return torch.tensor([row] * batch, dtype=dtype)


def _count_spikes(v_trace: np.ndarray, threshold: float = 0.0) -> int:
    """Count upward threshold crossings — proxy for spike count."""
    above = v_trace > threshold
    return int(np.sum(above[1:] & ~above[:-1]))


def check_against_reference(cell_name: str, log):
    """Run one forward with default params and compare to the reference .npz."""
    ref_path = REF_PATHS.get(cell_name)
    if ref_path is None or not ref_path.exists():
        _print(f"[{cell_name}] SKIP reference comparison: {ref_path} not found", log)
        return None

    p = _default_params_tensor(cell_name, batch=1, dtype=torch.float32)
    t0 = time.time()
    v = JaxleyBridge.simulate_batch(p, cell_name).cpu().numpy()[0]
    t_first = time.time() - t0
    # v shape: (n_rec, T_out)
    v_soma = v[0]  # first recorded compartment is soma by construction

    ref = np.load(ref_path)
    v_ref = ref["v_soma"]
    # Align lengths — jaxley may return T_sim/step; reference is 5001.
    n = min(len(v_soma), len(v_ref))
    v_soma = v_soma[:n]
    v_ref = v_ref[:n]

    diff = v_soma - v_ref
    max_abs = float(np.max(np.abs(diff)))
    rmse    = float(np.sqrt(np.mean(diff ** 2)))
    spikes_me  = _count_spikes(v_soma)
    spikes_ref = _count_spikes(v_ref)

    _print(f"[{cell_name}] reference = {ref_path}", log)
    _print(f"[{cell_name}] one forward (incl. jit compile): {t_first:.2f} s", log)
    _print(f"[{cell_name}] v range        mine=[{v_soma.min():+.2f}, {v_soma.max():+.2f}]  "
           f"ref=[{v_ref.min():+.2f}, {v_ref.max():+.2f}]  (mV)", log)
    _print(f"[{cell_name}] diff vs ref    max|Δ|={max_abs:.3g}  rmse={rmse:.3g}  (mV)", log)
    _print(f"[{cell_name}] spike count    mine={spikes_me}  ref={spikes_ref}", log)

    # Ball-and-stick: we use the same stim file, dt, v_init as the ref.  Tiny
    # drift (< 1 mV typical) is acceptable; large drift means the port broke
    # something in biophysics.  We classify:
    #   OK       : max|Δ| < 1.0  mV
    #   CLOSE    : max|Δ| < 10   mV  and spike counts match
    #   DIVERGE  : otherwise
    if max_abs < 1.0:
        grade = "OK"
    elif max_abs < 10.0 and spikes_me == spikes_ref:
        grade = "CLOSE"
    else:
        grade = "DIVERGE"
    _print(f"[{cell_name}] grade = {grade}", log)
    return dict(max_abs=max_abs, rmse=rmse, spikes_me=spikes_me, spikes_ref=spikes_ref,
                grade=grade, t_first=t_first)


def bench_throughput(cell_name: str, log, batches=(1, 4, 16, 64), n_warm=2, n_timed=5):
    """Time `simulate_batch` at several batch sizes."""
    _print(f"[{cell_name}] throughput:  batch_size  ->  sims/sec  (mean t, p5-p95)", log)
    dtype = torch.float32
    p1 = _default_params_tensor(cell_name, batch=1, dtype=dtype)
    # Prime the jit cache once.
    for _ in range(n_warm):
        _ = JaxleyBridge.simulate_batch(p1, cell_name)

    for B in batches:
        p = _default_params_tensor(cell_name, batch=B, dtype=dtype)
        # Warmup at this batch size — if batch size changes the vmap'd
        # shape, JAX may recompile once.
        for _ in range(n_warm):
            v = JaxleyBridge.simulate_batch(p, cell_name)
            # block on the result so we time compute, not dispatch
            v.cpu()
        ts = []
        for _ in range(n_timed):
            t0 = time.time()
            v = JaxleyBridge.simulate_batch(p, cell_name)
            v.cpu()
            ts.append(time.time() - t0)
        ts.sort()
        mean = float(np.mean(ts))
        p5   = ts[0]
        p95  = ts[-1]
        sps  = B / mean
        _print(f"[{cell_name}]    B={B:4d}  ->  {sps:7.2f} sims/sec  "
               f"(t={mean*1000:7.1f} ms  p5={p5*1000:5.1f}  p95={p95*1000:5.1f})", log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=["ball_and_stick"],
                    help="Which cells to benchmark.  L5TTPC is expensive.")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--report", default=str(REPORT_DEFAULT),
                    help="where to write the text report")
    args = ap.parse_args()

    log: list = []
    platform = ", ".join(d.platform for d in jax.devices())
    device_cnt = len(jax.devices())
    torch_cuda = torch.cuda.is_available()
    _print(f"jax devices: {device_cnt} x {platform}  |  torch.cuda: {torch_cuda}", log)
    _print("", log)

    for cell in args.cells:
        _print(f"=== {cell} ===", log)
        if not args.skip_reference:
            check_against_reference(cell, log)
        if not args.skip_throughput:
            bench_throughput(cell, log)
        _print("", log)

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"wrote {rep}")


if __name__ == "__main__":
    main()
