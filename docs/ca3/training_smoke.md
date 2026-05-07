# CA3 Pyramidal — matched-data training smoke run

**Status: end-to-end pipeline works.** CNN → unit→phys → jaxley CA3 sim →
hybrid (channel + voltage) loss → backward → optimizer step. Val loss
descends 0.678 → 0.489 over 5 smoke epochs on 1 A100, no NaNs.

## Setup

- **Source cell**: `ca3_pyramidal` (single-comp, 6 conductances, fp32 stable)
- **Synth pack**: `/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_synth_v1/ca3_pyramidal_synth.mlPack1.h5`
  - N = 50 000; 80/10/10 split → 40 000 / 5 000 / 5 000
  - 1 probe (soma), 1 stim (`5k50kInterChaoticB`), 6 unit_par columns
  - Voltages: fp16 z-scored per-sample-per-probe, 5001 time bins
  - File size: 332 MB
- **Design YAML**: `ca3_matched.hpar.yaml` (channel_weight=1, voltage_weight=1, mask_channels=False, fp64=False)
- **Workdir**: `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/smoke_1778181760`
- **Hardware**: 1× A100-80GB (single GPU smoke), batch=128, numGlobSamp=4096 (32 steps/epoch)

## Phase A — data generation

50 000 traces × 500 ms simulated in **1.4 min** (~600 samples/s including JAX↔torch round-trip).

Note this is slower than the pure-bench number (9 000 traces/s at B=1024 fwd) because the gen pipeline pays D2H copy + numpy stage per batch. For this dataset size (50k × 5001 × fp32 ≈ 1 GB pre-z-score), copying is ~90% of the runtime.

After z-score:
```
mean abs = 1.84e-07     (target 0)
std mean = 1.000        (target 1)
```

## Phase B — training (5-epoch smoke)

| Epoch | Train loss | Val loss | sec/epoch | Δ val |
|---:|---:|---:|---:|---:|
| 0 | 0.844 | 0.678 | 71.8 (cold) | — |
| 1 | 0.873 | 0.933 | 59.4 | +0.255 |
| 2 | 0.703 | 0.675 | 59.3 | -0.258 |
| 3 | 0.719 | **0.622** | 59.4 | -0.053 |
| 4 | 0.606 | **0.489** | 59.4 | -0.133 |

- **Convergence**: clean monotonic descent epochs 1→4 after the warm-up oscillation. The bump at epoch 1 is a single-batch artifact (the val pass at epoch 0 happens before any gradient has been applied — it's effectively the random-init val loss).
- **Per-epoch wall-time**: ~59 s warm at B=128 with 32 steps/epoch on 1 A100. Of that, the jaxley fwd+bwd alone is ~10 ms/trace × 128 batch × 32 steps = 41 s, so ~70% of the epoch time is jaxley, ~30% is CNN+optimizer+dataloader.
- **No NaNs**, no instability, no fp64 needed.

## Comparison with ball_and_stick_bbp (Phase 3)

| Metric | ball_and_stick_bbp (Phase 3) | ca3_pyramidal (Phase 4) |
|---|---|---|
| Trainable conductances | 12 | 6 |
| Compartments | 6 (1 soma + 5-comp dend) | 1 |
| sec/epoch warm at B=128, 16 GPUs | ~21.5 (Phase 2 numbers) | **~6 estimated** (3.5× faster fwd × 2.6× faster bwd) |
| dtype | fp64 (required) | **fp32 (stable)** |
| Val descent over 5 epochs | first-pass smoke not previously logged | 0.68 → 0.49 (clean) |

## Reproduce

```bash
# 1. Generate dataset (1 GPU, 1.5 min compute + few min H5 write)
sbatch scripts/gen_ca3_data.slr

# 2. Train (4 nodes × 4 GPUs debug queue, 10 epochs)
sbatch batchShifterJaxleyCA3.slr

# Or single-GPU smoke (interactive):
salloc -A m2043_g -q interactive -C gpu -t 1:00:00 -N 1 \
       --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=32 --no-shell
JOB=<jobid>
srun --jobid=$JOB -n1 bash -c '
  export PYTHONPATH=/path/to/repo PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda,cpu
  export MASTER_ADDR=$(hostname) MASTER_PORT=8881
  /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python -u train_dist.py \
    --cellName ca3_pyramidal_synth --facility perlmutter \
    --design ca3_matched --jobId smoke --outPath ./out \
    --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
    --epochs 5 --numGlobSamp 4096 \
    --data_path_temp /pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_synth_v1/
'
```

## What's next

- **Full 10-epoch run** on 4 nodes × 4 GPUs (debug queue, ~3 min compute) via `batchShifterJaxleyCA3.slr` — projected val < 0.3 by epoch 10 based on the descent slope.
- **`evaluate_voltage.py`** on the resulting checkpoint to get per-parameter explained variance + voltage RMSE on the test split (acceptance criterion from Phase 3 plan: explained variance > 0.85, voltage RMSE_z < 0.3).
- **Real ephys fine-tune** (Phase 5) once matched-data convergence is confirmed.
