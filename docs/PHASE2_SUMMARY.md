# Phase 2 — Hybrid channel + voltage loss

**Status: shipped.** Hybrid-loss training pipeline works end-to-end at
production scale (16 A100s, fp64, t_max=500 ms, 0 NaN gradients,
14.9× DDP scaling). Phase 2 closed by validating the pipeline at 1 / 10 /
100 epochs and confirming the architectural ceiling that motivates Phase 3.

## Goal

Add a physics-grounded voltage-MSE term to the existing channel-MSE
training, computed by running the predicted parameters through a
differentiable jaxley simulation and comparing the resulting voltage to
the input trace. This unlocks fine-tuning on real ephys (Phase 4) where
ground-truth conductances don't exist.

## Deliverables

### Code

- **`toolbox/HybridLoss.py`** — `HybridLoss` (channel + voltage MSE) and
  `_ChannelOnlyAdapter`. Configurable knobs: `channel_weight`,
  `voltage_weight`, `mask_channels`, `clamp_unit_tanh`, `t_max_override`,
  `checkpoint_lengths`, `solver`, `fp64`. Factory `build_hybrid_loss(params)`
  reads voltage_loss block from the design YAML.
- **`toolbox/JaxleyBridge.py`** — extended with per-cell handle cache
  keyed by `(cell_name, stim_name, checkpoint_lengths, solver)`,
  configurable `solver` and `checkpoint_lengths` plumbed through
  `simulate_batch`, JAX device probe + per-rank pin in
  `_log_jax_devices_once`.
- **`toolbox/Trainer.py`** — additive hooks: criterion swap,
  `outputSize_override`, per-rank `set_device(SLURM_LOCALID)` + matching
  DDP `device_ids`, `verbose`-tolerant `ReduceLROnPlateau`, empty-
  `TperEpoch` guard for `add_histogram`, lazy ray imports.
- **`train_dist.py`** — lazy `RayTune` import (kept ray's transitive
  deps off the critical path), per-rank device pin before
  `dist.init_process_group`.
- **`evaluate_voltage.py`** — voltage-aware inference: load model, run
  CNN → unit→phys → jaxley on test split, dump per-sample voltage MSE
  CSV, histogram, CDF, overlay plots. Handles `t_max_override=auto`.
- **`toolbox/tests/test_hybrid_loss.py`** — 8 unit tests covering loss
  shape, autograd, mask_channels, factory, fp64 backward stability.
- **`toolbox/tests/phase2_demo.py`** — end-to-end "untrained CNN → loss"
  walk-through with prints at every stage.

### Configuration

- **`ballBBP_voltage_only.hpar.yaml`** — voltage-only design: 12-output
  CNN, `ball_and_stick_bbp` simulator, `mask_channels=True`, `fp64=True`,
  `t_max_override=auto`, `clamp_unit_tanh=True`.
- **`m8lay_vs3_jaxley.hpar.yaml`** — generic Phase-2 design template.

### SLURM

- **`batchShifterJaxley.slr`** — debug-queue 4-node × 4-GPU template
  with the full Perlmutter-conda-env NCCL/JAX environment, per-rank
  `JAX_COMPILATION_CACHE_DIR`, `bash -c` wrapper for `$SLURM_PROCID`
  expansion.
- **`batchShifterJaxley_100ep.slr`** — regular-queue long-run variant.

### Documentation

- **`docs/ARCHITECTURE.md`** — full pipeline map, per-stage code refs,
  multi-GPU layout with rationale for every env var, end-to-end
  gradient path.

## Numerical findings

### fp32 → fp64 was required (not optional)

Backward through `bwd_euler` over 5000 stiff steps with the BBP channel
set produces NaN gradients in fp32 because the cumulative
Jacobian-transpose product overflows the 23-bit mantissa near each spike.

| t_max (ms) | fp32 grad NaN | fp64 grad NaN |
|---:|---:|---:|
| 50  | 0/12 | 0/12 |
| 100 | 0/12 | 0/12 |
| 250 | 12/12 ❌ | 0/12 |
| 500 | 12/12 ❌ | 0/12 |

`crank_nicolson` solver and `jax.checkpoint` (both `[500, 10]` and
`[50, 100]`) did not help — same fp32 mantissa limit. Only fp64 fixes
it. Cost: ~2-3× forward, ~5× backward, 2× memory.

### Multi-GPU scaling

At B=128/GPU, t_max=500 ms, fp64, `ball_and_stick_bbp`:

| GPUs | epoch 0 (cold) | epoch 1+ (warm) | speedup vs 1 GPU |
|---:|---:|---:|---:|
| 1   | 337 s | 320 s | 1.0× |
| 4   | 107 s | 82.5 s | **3.88×** |
| 16  | 49.3 s | **21.5 s** | **14.9×** |

Per-step warm time stays ~5 s/step regardless of rank count → DDP
overhead is negligible, scaling is bound by independent jaxley sims
running on each GPU's local batch.

### Multi-GPU env (none of these are negotiable)

| Issue | Symptom | Fix |
|---|---|---|
| `--gpus-per-task=1` | NCCL `Cuda failure 101 'invalid device ordinal'` | `--gpus-per-node=4 --gpu-bind=none` |
| All ranks default to `cuda:0` for JAX | jaxley sims pile on GPU 0, no scaling | `jax.config.update("jax_default_device", devices[localid])` per rank |
| Shared compile cache | XLA `Buffer on cuda:N, replica assigned to cuda:0` | `JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/rank_$SLURM_PROCID` |
| `device_ids=[0]` in DDP | All ranks pile on GPU 0 | `DDP(model, device_ids=[self.device])` where `self.device = SLURM_LOCALID` |
| Trainer `current_device()` returns 0 even after set_device | Input on cuda:0, weight on cuda:N | Explicit `int(os.environ['SLURM_LOCALID'])` |

## Validation runs

| Job | Setup | Purpose | Result |
|---|---|---|---|
| 52364894 | 1 GPU, 2 epochs | end-to-end working | ✅ train 2.151 → 2.073, val 2.087 → 2.097 |
| 52371151 | 16 GPU, 2 epochs | DDP works | ✅ 14.9× scaling, ε NaN |
| 52371647 | 16 GPU, 10 epochs | trend | ✅ val 2.188 → 2.154 (-1.5%) |
| 52372480 | 16 GPU, 100 epochs, regular queue | convergence | ✅ val 2.177 → 2.156 (-1.0%), plateau by epoch 10 |

All 4 runs hit zero NaN gradients with `fp64=True`.

## Architectural ceiling — motivates Phase 3

Both 10-epoch and 100-epoch evaluations on `evaluate_voltage.py` show
the CNN **mode-collapsed to a constant predictor**: every test input
produces a bit-identical predicted voltage trace. Diagnostic from the
10-epoch run (200 test samples):

| metric | value | comment |
|---|---:|---|
| voltage RMSE_z mean | 1.43 | random baseline ≈ √2 = 1.41 |
| spike count diff (mean abs) | 10.6 | sim 13.4 vs data 4.5 |

The 100-epoch run plateaued at val=2.156 by epoch 10; epochs 20-99
are noise (max-min over that range = 0.001). LR collapsed to 5.9e-9
under ReduceLROnPlateau by epoch 50.

Cause: model mismatch. The 12-param `ball_and_stick_bbp` cannot
reproduce 19-param `L5_TTPC1` voltage traces, so the optimizer's
best move is a constant predictor that minimizes the average MSE.

## Phase 3 hand-off

A worktree is set up to address the mismatch:

- **branch**: `phase3/matched-bbp-data`
- **path**: `/global/u1/k/ktub1999/Neuron/neuron4/worktrees/phase3-matched-bbp-data/`
- **handoff doc**: `PHASE3_TASKS.md` in that worktree

Plan: generate training data with `ball_and_stick_bbp` *itself* (random
parameter sweeps via the same jaxley path), train the same CNN+HybridLoss
on the matched data with `mask_channels=False` (use both channel and
voltage supervision). With the simulator and the data being the same
model, the inverse problem becomes well-posed and the CNN should learn
input-conditional predictions that drive voltage MSE → 0. Acceptance:
per-parameter explained variance > 0.85, voltage RMSE_z < 0.3.
