# Phase 3 — Matched ball_and_stick_bbp training data

This worktree is where Phase 3 of the jaxley-CNN inverse-problem plan is built.
It branches from `CNN_Jaxley` after Phase 2 was validated at scale (16 GPUs,
fp64, t_max=500 ms, 0/1536 NaN gradients, near-linear DDP scaling).

You are a fresh Claude instance picking this up. Read the whole file before
editing. The codebase has lots of moving parts but the Phase 3 scope is
narrow and the upstream Phase 2 code is already working — **do not modify
HybridLoss / JaxleyBridge / Trainer / train_dist** unless something here
explicitly requires it.

## TL;DR what to build

A self-consistent training pipeline that uses **`ball_and_stick_bbp`** (the
12-param BBP-channel ball-and-stick cell) **as both the data generator and
the loss simulator**. With the simulator and the data matched, the inverse
problem becomes well-posed and the CNN can actually learn to recover input
parameters from voltage traces.

Three deliverables:

1. **`scripts/gen_ball_and_stick_data.py`** — generates an `.mlPack1.h5`
   file in the same layout `Dataloader_H5.py` reads, populated by jaxley
   simulations of `ball_and_stick_bbp` at random parameter draws.
2. **`plotJaxleyValidation.py`** — given a trained run, makes per-parameter
   pred-vs-true plots, voltage overlays, and a text summary. Reuses the
   existing `evaluate_voltage.py` patterns where possible.
3. **`ballBBP_matched.hpar.yaml`** — design YAML pointing the trainer at
   the new H5 pack. Channel + voltage hybrid loss (NOT mask_channels — we
   now have ground-truth params for each sample).

Acceptance bar: **per-parameter explained variance > 0.85** on the test
split, and **voltage RMSE_z < 0.3** (per `PLAN.md` §136). If you hit those,
Phase 3 is done.

---

## Why this phase exists — the Phase 2 finding

Phase 2 validated the loss + bridge end-to-end and trained a 12-output CNN
on `L5_TTPC1cADpyr0.mlPack1.h5` (a 19-param cADpyr data pack) with
`mask_channels=True` (voltage-only loss). After 10 epochs on 16 GPUs:

- val voltage MSE_z plateau ≈ 2.15 (random baseline ≈ 2.0)
- All 200 evaluated test samples got **bit-identical predicted voltage
  traces** (mode collapse) — the CNN converged to a constant predictor.

Cause: model mismatch. The 12-param `ball_and_stick_bbp` cannot reproduce
the full 19-param L5_TTPC1 voltage trace, so the optimizer's best move is
to ignore the input. Phase 3 fixes this by construction: the data and the
loss simulator are the same model.

Eval artefacts from that run:

```
/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_voltage_only/
    ballBBP_voltage_only/L5_TTPC1cADpyr0/52371647/out/eval/
    ├── summary.yaml
    ├── voltage_metrics.csv
    ├── voltage_loss_hist.png
    ├── voltage_rmse_cdf.png
    └── trace_overlay_00..08_*.png
```

A 100-epoch follow-up (`52372480`, regular queue) is/was running for
comparison; whatever it shows, it cannot fix the mismatch without Phase 3.

---

## What's already in the worktree (from Phase 2)

You have a working voltage-loss training stack:

- **`toolbox/HybridLoss.py`** — `HybridLoss(channel_weight, voltage_weight,
  mask_channels, clamp_unit_tanh, t_max_override, checkpoint_lengths,
  solver, fp64)` + `_ChannelOnlyAdapter` + `build_hybrid_loss(params)`.
- **`toolbox/JaxleyBridge.py`** — torch ↔ jax bridge with
  `simulate_batch(params_phys, cell_name, stim_name, checkpoint_lengths,
  solver)`. Per-cell handle cache keyed by all of those.
- **`toolbox/jaxley_cells/ball_and_stick_bbp.py`** — the 12-param cell
  spec. Already correctly handles fp64 when `JAX_ENABLE_X64=true`.
- **`toolbox/Trainer.py`** — `outputSize_override`, `use_voltage_loss`
  hook, DDP with per-rank `set_device(SLURM_LOCALID)`,
  `verbose`-tolerant `ReduceLROnPlateau`, empty-`TperEpoch` guard, lazy
  ray imports.
- **`train_dist.py`** — lazy `RayTune` import, per-rank device pin.
- **`toolbox/tests/test_hybrid_loss.py`** — 8 tests, all should pass.
- **`toolbox/tests/phase2_demo.py`** — end-to-end demo: untrained CNN →
  unit→phys → jaxley → loss. Re-run to verify the env.
- **`evaluate_voltage.py`** — runs CNN+jaxley on test split, dumps
  voltage RMSE/CDF/overlay plots. Will need a small extension for
  Phase 3 (also evaluate per-parameter recovery, not just voltage).
- **`batchShifterJaxley.slr`** — debug-queue 4-node × 4-GPU SLURM
  template with NCCL/JAX env, per-rank cache dir. Use as starting
  point.
- **`batchShifterJaxley_100ep.slr`** — regular-queue 100-epoch variant.
- **`ballBBP_voltage_only.hpar.yaml`** — Phase 2 design (voltage-only,
  12 outputs, `mask_channels=True`, `fp64=True`,
  `t_max_override=auto` -> 500 ms).

These files are uncommitted in the worktree (copied from the parent
checkout). **First action: commit them as the Phase 3 baseline** so any
future revert is clean. Suggested message:

```
phase 2 baseline: hybrid loss, jaxley bridge multi-GPU, eval script
```

`*yaml` is gitignored — use `git add -f ballBBP_voltage_only.hpar.yaml
m8lay_vs3_jaxley.hpar.yaml` if you want them tracked.

---

## Hard-won setup notes (don't re-discover these)

### Conda env, NOT shifter
The `CNN_Jaxley` line of work runs in
`/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley`. Bootstrap:

```bash
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1   # ~/.local has stale deps that break compute nodes
```

The env had to have `typing_extensions`, `ruamel.yaml<0.18`, `networkx`,
`requests`, `filelock`, `tensorboard`, `torchsummary`, `efel` pip-installed
into the env's `site-packages` (not `~/.local`). They should already be there.
Verify with:

```bash
PYTHONNOUSERSITE=1 /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python \
    -c "from toolbox.Trainer import Trainer; from toolbox.HybridLoss import build_hybrid_loss; print('OK')"
```

### Multi-GPU SLURM gotchas (already encoded in `batchShifterJaxley.slr`)
- **DO NOT** use `--gpus-per-task=1`. It hides peer GPUs from each rank's
  CUDA context, breaking NCCL P2P/SHM with `Cuda failure 101 'invalid
  device ordinal'`. Use `--gpus-per-node=4 --gpu-bind=none`.
- `Trainer.__init__` pins to `SLURM_LOCALID`. `train_dist.py` does the
  same before `dist.init_process_group`.
- Each rank must also pin **JAX** to its own GPU. `HybridLoss._log_jax_devices_once`
  does this via `jax.config.update("jax_default_device", devices[localid])`.
  Without it, all ranks pile their jaxley sims on cuda:0 and you get ~1×
  scaling (we measured 380 s / epoch for 4 GPU before fixing this).
- **JAX compilation cache must be per-rank**:
  `JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/rank_$SLURM_PROCID`. A shared
  cache loads rank 0's `cuda:0`-targeted binary on rank N → XLA
  "Buffer on cuda:N, but replica assigned to cuda:0".
- Required NCCL env (Perlmutter, no shifter):
  ```
  NCCL_NET_GDR_LEVEL=PHB
  FI_PROVIDER=cxi
  FI_CXI_DEFAULT_CQ_SIZE=131072
  ```
- The slr wraps the python launch in `bash -c "..."` so `$SLURM_PROCID`
  expands per-rank. Don't simplify back to a single `srun python ...`.

### fp32 NaN floor
Backward through `bwd_euler` over 5000 stiff steps with the BBP channel
set produces NaN gradients in fp32 (cumulative Jacobian-transpose product
overflows the mantissa near each spike). **Fix: `fp64=True` in the YAML +
`JAX_ENABLE_X64=true` in the env.** Costs ~2-3× forward, ~5× backward,
but eliminates all NaN. We measured 0/1536 NaN at B=128, t_max=500 ms.

`crank_nicolson` and `jax.checkpoint` did NOT help — same fp32 mantissa
limit. Don't rediscover this.

### Multi-GPU scaling, validated
On 16 A100s, `ball_and_stick_bbp` + B=128/GPU + t_max=500 ms + fp64:
- **5.4 s/step warm**, ~83 s/epoch warm at 32 768 samples/epoch
- **14.9× speedup** vs 1-GPU (out of 16× ideal)
- 100 epochs at 65 536 samples/epoch ≈ 4.6 hours

---

## Phase 3 work plan

### Step 1 — Commit Phase 2 baseline
As above. Don't skip this.

### Step 2 — Data generator: `scripts/gen_ball_and_stick_data.py`

Goal: write an `.mlPack1.h5` file with the same key layout
`Dataloader_H5.py` already reads, but the data is generated by
`ball_and_stick_bbp` at random parameter draws.

Required HDF5 structure (mirror what's in
`/pscratch/sd/k/ktub1999/Apr26ExcNoNoise_24733338/L5_TTPC1cADpyr0.mlPack1.h5`):

```
train_volts_norm  (N, T, P, S) fp16   - z-scored per-sample-per-probe
train_unit_par    (N, P_cnn) f32      - ground-truth unit params used to gen
valid_volts_norm  (Nv, T, P, S) fp16  - same
valid_unit_par    (Nv, P_cnn) f32
test_volts_norm   (Nt, T, P, S) fp16
test_unit_par     (Nt, P_cnn) f32
meta.JSON         (1,) string         - serialised dict; see fields below
5k50kInterChaoticB (0,)              - sentinel; stim CSV stored under same key
```

`meta.JSON` minimum:
```python
{
  "cell_name": "ball_and_stick_bbp_synth",
  "num_phys_par": 12,
  "num_probs": 1,                       # only soma; we don't simulate axon/dend traces
  "num_stims": 1,
  "num_time_bins": 5000,                # t_max/dt
  "parName": [...12 ball_and_stick_bbp PARAM_KEYS...],
  "phys_par_range": [[center, log_halfspan, "S/cm^2"]] * 12,
  "probe_names": ["soma"],
  "stim_names": ["5k50kInterChaoticB"],
  "timeAxis": {"step": 0.1, "unit": "(ms)"},
  "input_meta": { "phys_par_range": ... }   # build_hybrid_loss reads this
}
```

The dataloader expects `volts_norm.shape[2] >= max(probs_select)+1`. The
design YAML for Phase 3 should set `probsSelect="0"` (soma only).
`Trainer` calls `dataset.data_frames.shape[1:]` for `inputShape`, so this
must end up `[T_data, 1]` after dataloader reshape.

Implementation sketch:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=20000)         # total samples
    parser.add_argument("--split", nargs=3, type=int, default=[80, 10, 10])  # tr/va/te %
    parser.add_argument("--batch", type=int, default=128)        # jaxley batch
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 1. Build cell handle once via JaxleyBridge.get_handle("ball_and_stick_bbp")
    # 2. Sample unit ~ Uniform(-1, 1)^12 for N samples
    # 3. unit -> phys via centers · 10^(unit · log_halfspan)
    # 4. In batches, call JaxleyBridge.simulate_batch(phys_batch, "ball_and_stick_bbp")
    #    to get (B, 1, T_sim) soma traces in mV
    # 5. Per-sample z-score the soma trace -> (N, T_sim, 1) fp16
    # 6. Split 80/10/10 into train/valid/test
    # 7. Write HDF5 with the layout above + meta.JSON
```

Use `centers, logspans` from
`toolbox/jaxley_cells/ball_and_stick_bbp.py:_DEFAULTS` and the same
`log_halfspan=0.5` we used in Phase 2 (`ballBBP_voltage_only.hpar.yaml`).

Performance target: 20 k samples should take ~30-60 min on 1 A100 at
B=128 (~5-8 s/step, 156 steps).

Output paths: `$SCRATCH/synthetic_bbp_data/ballBBP_synth_v1/`,
inside it `ball_and_stick_bbp_synth.mlPack1.h5`.

### Step 3 — Phase 3 design YAML: `ballBBP_matched.hpar.yaml`

Start from `ballBBP_voltage_only.hpar.yaml` and change:

```yaml
data_path:
  perlmutter: /pscratch/sd/k/ktub1999/synthetic_bbp_data/ballBBP_synth_v1/

# probe set: only soma now (synthetic data has 1 probe)
# (the SLR will pass --probsSelect "0")

use_voltage_loss: True
voltage_loss:
    cell_name_for_sim: ball_and_stick_bbp
    channel_weight:    1.0          # USE channel loss now! we have ground-truth params
    voltage_weight:    1.0          # voltage as additional supervision
    mask_channels:     False        # NOT masked — we have true_unit
    clamp_unit_tanh:   True
    fp64:              True
    t_max_override:    auto
    soma_probe_index:  0
    # phys_par_range now read from H5 meta; can drop the inline list

model:
    outputSize_override: 12
    # rest same as Phase 2
```

Cell name in SLURM: use a synthetic cell-name like `ball_and_stick_bbp_synth`
that matches the H5 filename. Make sure
`/pscratch/.../ballBBP_synth_v1/ball_and_stick_bbp_synth.mlPack1.h5`
exists before submitting.

### Step 4 — Wire up + smoke test

Adapt `batchShifterJaxley.slr` → `batchShifterJaxleyMatched.slr`:
- `cellName=ball_and_stick_bbp_synth`
- `design=ballBBP_matched`
- `probsSelect="0"` (soma only)
- `epochs=10` for first smoke test, debug queue
- Add `evaluate_voltage.py` to `codeList` (it's already there in the
  Phase 2 slr).

Submit. Should converge to **train + val ≈ 0** (matched data, well-posed
problem) within 5-10 epochs.

### Step 5 — Phase 3 validation plotter: `plotJaxleyValidation.py`

Extends `evaluate_voltage.py` with:

1. Per-parameter pred vs true scatter / bar chart on test split:
   - x = `true_unit_par[:, k]`, y = `pred_unit[:, k]` for each k
   - Compute per-param Pearson R, RMSE, explained variance
   - Save `param_recovery_<param_name>.png` × 12
2. Existing voltage overlays (already in `evaluate_voltage.py`)
3. Text summary printing the acceptance numbers:
   - per-param explained variance (target > 0.85 on each)
   - voltage RMSE_z (target < 0.3)

### Step 6 — Full training run

If Step 4's 10-epoch smoke run is converging, submit a 100-epoch run on
regular queue (use `batchShifterJaxley_100ep.slr` as template). At
65 536 samples/epoch × 100 epochs × ~5 s/step / B=128 ≈ 4-5 hours on
16 GPUs.

After it finishes:
- Run `plotJaxleyValidation.py` on the checkpoint.
- Confirm acceptance criteria.
- Write `docs/phase3/summary.md` with the curves + final metrics.

### Step 7 (optional, if time) — Phase 4 prep

If Phase 3 acceptance hits, the matched-data-trained CNN is a meaningful
starting point for fine-tuning on real ephys. Phase 4 from `PLAN.md` §142.
Out of scope for this worktree.

---

## Useful paths and commands

**Data — Phase 2 reference pack:**
```
/pscratch/sd/k/ktub1999/Apr26ExcNoNoise_24733338/L5_TTPC1cADpyr0.mlPack1.h5
  shape: train_volts_norm (409 600, 4000, 4, 1) fp16
         train_unit_par   (409 600, 19) f32
```

**Stim CSV:**
```
/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/5k50kInterChaoticB.csv
  5000 samples × dt_stim=0.1 ms = 500 ms
```

**Run dirs (Phase 2 reference):**
```
/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/
  52364894/  # 1-GPU, 2-epoch, 11 min
  52371151/  # 16-GPU, 2-epoch, 1 min - 14.9× scaling
  52371647/  # 16-GPU, 10-epoch with eval/
  52372480/  # 100-epoch, regular queue (status TBD when you start)
```

**Submitting jobs:**
```bash
sbatch batchShifterJaxleyMatched.slr            # 4 nodes × 4 GPUs, debug 30 min
sbatch batchShifterJaxleyMatched_100ep.slr      # regular queue 5h+
```

**Interactive GPU for testing:**
```bash
salloc -N 1 -C gpu -q interactive -t 25:00 -A m2043_g \
       --gpus=4 --ntasks-per-node=4 --gpu-bind=none --no-shell
JID=<from output>
srun --jobid=$JID -N 1 --ntasks=4 --gpu-bind=none bash <wrapper>
```

`run_interactive_4gpu.sh` (under `/pscratch/sd/k/ktub1999/`) is the wrapper
template — adapt it.

**Test the env quickly:**
```bash
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python \
  -m toolbox.tests.test_hybrid_loss
# all 8 tests should pass
```

---

## What NOT to do

1. **Don't simplify the SLURM env**. The NCCL env vars + `--gpu-bind=none`
   + per-rank cache dir + `bash -c` wrapper were each found by hitting a
   real failure. `git log` on `batchShifterJaxley.slr` will show why each
   line is there.
2. **Don't drop `fp64=True`**. fp32 NaN'd at t_max ≥ 250 ms; checkpointing
   and crank_nicolson didn't help. Documented in
   `docs/phase1/L5TTPC.md` discussion.
3. **Don't change `_T_MAX` in the cell file**. Use `t_max_override: auto`
   in the YAML — it picks up `len(stim) × dt_stim`.
4. **Don't increase batch size beyond 128 per GPU** without re-checking
   memory. fp64 doubles memory vs fp32; 128 is the verified sweet spot.
5. **Don't try to use the `L5TTPC` cell** — it's heavier (492 apical
   compartments) and OOMs at B=4 fwd+bwd on a 40 GB A100. Phase 5
   problem, not Phase 3.

---

## Checklist for completion

- [ ] Phase 2 baseline committed
- [ ] `scripts/gen_ball_and_stick_data.py` written + tested locally
- [ ] Synthetic H5 pack generated (≥ 20 k samples) and inspected
      (no NaN, std > 0, etc.)
- [ ] `ballBBP_matched.hpar.yaml` created
- [ ] `batchShifterJaxleyMatched.slr` created and 10-epoch smoke run green
- [ ] `plotJaxleyValidation.py` written
- [ ] 100-epoch training run completed
- [ ] Per-param explained variance > 0.85 on test
- [ ] Voltage RMSE_z < 0.3 on test
- [ ] `docs/phase3/summary.md` written with curves + metrics

When done, report back: total samples generated, training wall-clock,
final per-param + voltage metrics, and any caveats found along the way.

---

Good luck. The hard parts (loss, bridge, multi-GPU, fp64, eval) are all
already solved. Phase 3 is a focused data-generation + cleanup task.
