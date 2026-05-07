# CA3 Pyramidal — GPU benchmark

Hardware: 1× NVIDIA A100-SXM4-80GB on Perlmutter (`m2043_g`, interactive
queue, node `nid008508`).  Cell: `ca3_pyramidal` (single-comp soma, 6
channels).  Stim: `5k50kInterChaoticB`, t_max = 500 ms, dt = 0.1 ms,
solver = `bwd_euler`.

CSV: `/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_gpu_bench/bench_ca3_gpu.csv`

## Forward only (fp32)

| B | warm time / step (ms) | per trace (ms) | traces/s |
|---:|---:|---:|---:|
| 1    | 106.95 | 106.95   | 9.4 |
| 16   | 87.06  | 5.44     | 183.8 |
| 64   | 94.88  | 1.48     | 674.5 |
| 128  | 106.96 | 0.836    | 1 196.7 |
| 256  | 106.46 | 0.416    | 2 404.7 |
| 512  | 108.43 | 0.212    | 4 721.8 |
| 1024 | 111.79 | 0.109    | 9 160.0 |
| **2048** | **112.70** | **0.0550** | **18 172.9** |

Per-step time is essentially constant from B=64 to B=2048 (87 → 113 ms,
+30%) — the ODE integrator is bottlenecked on the time loop, not on the
batched RHS, so per-trace cost falls almost linearly with B.  Throughput
scales 1953× from B=1 to B=2048 (almost ideal: 2048/1 = 2048×).

## Forward + backward (fp32, value_and_grad)

| B | warm (ms) | per trace (ms) | traces/s |
|---:|---:|---:|---:|
| 1    | 1 354.1 | 1 354.1 | 0.7 |
| 16   | 1 358.6 | 84.9   | 11.8 |
| 64   | 1 274.5 | 19.9   | 50.2 |
| 128  | 1 293.4 | 10.1   | 99.0 |
| 256  | 1 362.6 | 5.32   | 187.9 |
| 512  | 1 370.2 | 2.68   | 373.7 |
| 1024 | 1 417.7 | 1.38   | 722.3 |
| **2048** | **1 460.5** | **0.713** | **1 402.2** |

Backward is **~12.6× slower than forward** at large B (consistent with the
deep adjoint chain through 5000 stiff bwd_euler steps).  At B=2048 the
training-relevant cost is **0.71 ms per (forward+backward) trace** —
which is the relevant number for HybridLoss data throughput.

## fp64 spot check (B=256)

| Mode | fp32 (ms) | fp64 (ms) | fp64/fp32 |
|---|---:|---:|---:|
| Forward | 106.5 | 85.5 | **0.80×** |
| Forward+Backward | 1362.6 | 1375.5 | **1.01×** |

Surprisingly fp64 is *not* slower than fp32 here — XLA seems to fuse the
single-comp loop the same way regardless of dtype, and the A100's fp64
units are well-utilised at this small per-trace footprint.  Memory at
B=256 is still in the few-MB range (~10 MB fp64 fwd_bwd vs ~5 MB fp32),
so we have ~80 GB of headroom on this cell.

CA3 fp32 was numerically stable for both fwd and fwd_bwd over the full
500 ms span — none of the L5TTPC-style backward NaN'ing.  That's
expected: 6 channels × 1 comp × ~20% the gate-state dimensionality of
L5TTPC.  fp64 is therefore *optional* for CA3, not required.

## Apples-to-apples vs NEURON

| Stack | Hardware | Per-trace cost (500 ms) |
|---|---|---:|
| NEURON (CVODE off, dt=0.025) | 1× CPU thread | **118.7 ms** |
| Jaxley CPU (B=1, warm) | 1× CPU thread | 571 ms |
| Jaxley GPU fwd (B=2048) | 1× A100 | **0.055 ms** |
| Jaxley GPU fwd+bwd (B=2048) | 1× A100 | **0.71 ms** |

So at B=2048 on a single A100:
- Forward is **2 158× faster per trace than NEURON** (118.7 ms / 0.055 ms).
- Forward+backward is **167× faster per trace than NEURON forward alone**.

Equivalent way of looking at it: one A100 fwd at B=2048 = **~18 200 traces/s**,
which is **~2 158 NEURON-CPU-cores** worth of throughput.

## Practical implications for CA3 training

* For data-gen (forward only): batch 1024–2048, fp32, expect **~10⁴ traces/s/GPU**.
  500k samples (the L5TTPC pack scale) generated in **~30 s** on a single A100.
  With 16 GPUs (the Phase 2 setup) → **~3 s** for 500k samples.
* For training-step backward: at B=128/GPU and 16 GPUs, expect ~**1 ms/trace**
  combined → ~2 s/step for global B=2048.  Phase 2 saw ~5 s/step on
  ball_and_stick_bbp (slightly heavier biophysics); CA3 should be ~2–3× faster.
* fp64 is unnecessary for CA3 — keep fp32 for the training bridge.
* No memory-driven batch-size cap up to B=2048 on A100-80GB.

## Reproducing

```bash
salloc -A m2043_g -q interactive -C gpu -t 1:00:00 -N 1 \
       --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=4 --no-shell
JOB=<jobid>
srun --jobid=$JOB -n1 bash -c '
  export PYTHONPATH=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
  export PYTHONNOUSERSITE=1
  export JAX_PLATFORMS=cuda,cpu
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/ca3_bench500
  mkdir -p $JAX_COMPILATION_CACHE_DIR
  cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
  /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python \
    toolbox/tests/bench_gpu_ca3.py --batches 1 16 64 128 256 512 1024 2048
'
```

The original t_max=100 ms run (using the existing `bench_gpu_l5ttpc.py`
harness with `--cells ca3_pyramidal`) is at
`/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_gpu_bench/bench_gpu.csv`.
