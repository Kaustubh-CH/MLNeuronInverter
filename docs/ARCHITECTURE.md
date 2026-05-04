# Neuron-Inverter Architecture (with Jaxley voltage-loss path)

End-to-end map of how a voltage trace becomes a parameter prediction +
gradient signal in the `CNN_Jaxley` line of work. Reference for both
debugging and Phase 3+ planning.

## 0. The high-level question

> **Given a voltage trace from a neuron, recover the channel
> conductances that produced it.**

We solve this by training a CNN to predict the conductances, with the loss
having two complementary terms:

1. **Channel MSE** — supervises against ground-truth conductances when we
   have them (simulated data).
2. **Voltage MSE (physics loss)** — runs the predicted conductances
   through a differentiable jaxley simulation, compares the resulting
   voltage to the input trace. The physics loss is what enables fine-tune
   on real ephys (Phase 4) where ground-truth conductances don't exist.

## 1. Pipeline overview

```
                                ┌──────────────────────────────────────────┐
                                │ 1. NEURON simulation (external repo)     │
  channel-param sweep ────────► │   /pscratch/.../DL4neurons2              │
                                │   produces raw .h5 files                 │
                                └──────────────────────────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────────┐
                                │ 2. packBBP3/aggregate_*.py + format_*.py │
                                │   z-score per-sample-per-probe → fp16    │
                                │   train/valid/test 8/1/1 split           │
                                │   <cell>.mlPack1.h5                      │
                                └──────────────────────────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────────┐
                                │ 3. toolbox/Dataloader_H5.py              │
                                │   reads whole pack into RAM per rank     │
                                │   batch (B, T, probes) volts + true unit │
                                └──────────────────────────────────────────┘
                                                   │
                                                   ▼
   per training step ──►  ┌────────────────────────────────────────────────────┐
                          │ 4. CNN forward (toolbox/Model.py:MyModel)          │
                          │    Conv1d×3 ─► BatchNorm ─► FC×6                   │
                          │    input (B, C=probes, T) → output pred_unit (B,P) │
                          └────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                          ┌────────────────────────────────────────────────────┐
                          │ 5. HybridLoss.forward (toolbox/HybridLoss.py)      │
                          │   ┌─────────────────────────────────────┐          │
                          │   │ if mask_channels=False:             │          │
                          │   │   ch_loss = MSE(pred_unit,true_unit)│          │
                          │   └─────────────────────────────────────┘          │
                          │   ┌─────────────────────────────────────┐          │
                          │   │ pred_unit ─tanh─► unit_to_phys      │          │
                          │   │   pred_phys = center·10^(u·logspan) │          │
                          │   └─────────────────────────────────────┘          │
                          │                       │                            │
                          │                       ▼                            │
                          │   ┌─────────────────────────────────────┐          │
                          │   │ JaxleyBridge.simulate_batch          │          │
                          │   │   (torch ↔ JAX bridge, see §6)       │          │
                          │   │   → v_sim (B, n_rec, T_sim) mV       │          │
                          │   └─────────────────────────────────────┘          │
                          │                       │                            │
                          │                       ▼                            │
                          │   v_sim_z = zscore(v_sim_soma along time)          │
                          │   v_loss  = MSE(v_sim_z, true_volts_z)             │
                          │   total   = w_p·ch_loss + w_v·v_loss               │
                          └────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                          ┌────────────────────────────────────────────────────┐
                          │ 6. backward                                        │
                          │    torch autograd → _JaxleySimulate.backward       │
                          │    → captured JAX vjp closure (replays on grad_v)  │
                          │    → grad pred_phys → grad pred_unit → grad CNN.W  │
                          └────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                          ┌────────────────────────────────────────────────────┐
                          │ 7. DDP all-reduce + Adam.step()                    │
                          │    NCCL ring-allreduce of CNN params across ranks  │
                          │    optimizer updates; grad_scaler if AMP           │
                          └────────────────────────────────────────────────────┘
```

## 2. Stage-by-stage details

### Stage 1 — NEURON-side ground truth

Out of scope for this repo. Lives in `DL4neurons2`. Produces voltage
traces of full BBP cell models (e.g. `L5_TTPC1_cADpyr232_1`) under a
known channel parameter sweep. Each sample is `(T=4000, probes=4,
stims=1)` at dt_stim = 0.1 ms (10 kHz).

### Stage 2 — Data packing

`packBBP3/aggregate_*.py` aggregates 1000 raw `.h5` into one
`<cell>.simRaw.h5`. `packBBP3/format_bbp3_for_ML.py` then:

- per-sample-per-probe z-scores the voltages (mean/std over the time
  axis for each (sample, probe))
- splits train/valid/test 8/1/1 (409 600 / 51 200 / 51 200)
- casts voltages to fp16 to save RAM
- writes `<cell>.mlPack1.h5` with the layout:

```
train_volts_norm   (409 600, 4000, 4, 1)  fp16
train_unit_par     (409 600, 19)          fp32   ← ground-truth, [-1,1]
train_phys_par     (409 600, 19)          fp32   ← in S/cm² etc.
valid_*, test_*    same shape but smaller N
meta.JSON          (1,) string            ← parName, phys_par_range, etc.
5k50kInterChaoticB (0,)                   ← stim CSV stored as dataset
```

`unit_par` is in the unit-normalized range `[-1, 1]` such that
`phys = center · 10^(unit · log_halfspan)`. The `phys_par_range` in
`meta.JSON` lists `[center, log_halfspan, units]` per param.

### Stage 3 — Dataloader (toolbox/Dataloader_H5.py)

Reads the whole shard into RAM once per rank; no on-the-fly I/O. Per
rank, picks a contiguous slice of size `local_batch_size × steps`. With
`serialize_stims=True`, each stim becomes an independent sample (the
`unit_par` is replicated per stim, and the time axis stays at T=4000).

`__getitem__` returns `(X, Y)`:
- `X` shape: `(T, probes_after_select)` fp32 (e.g. `(4000, 3)` if
  `probsSelect="0 1 2"`)
- `Y` shape: `(P_data,)` fp32 = unit-normalized true params (P_data=19
  for cADpyr cells)

Trainer's batching → `(B, T, C)`. The dataloader sets
`params['model']['inputShape'] = [T, C]` and
`params['model']['outputSize'] = Y.shape[1]` (= 19 by default), but
**`outputSize_override` in the design YAML can force a different output
dim** — used when the loss simulator (e.g. `ball_and_stick_bbp`) has
fewer params than the data pack.

### Stage 4 — CNN (toolbox/Model.py)

Before forward, Trainer does `images = images.permute(0, 2, 1)` so the
`(B, T, C)` from the dataloader becomes `(B, C, T)` for `Conv1d`.

```
Conv1d(C → 30, k=4) → MaxPool1d(4) → ReLU → BatchNorm1d(30)
Conv1d(30 → 90, k=4) → MaxPool1d(4) → ReLU
Conv1d(90 → 180, k=4) → MaxPool1d(4) → ReLU
Flatten → BatchNorm1d(10980)
FC(10980 → 512) → ReLU → Dropout(0.04)   × 3
FC(512 → 256)   → ReLU → Dropout
FC(256 → 128)   → ReLU → Dropout
FC(128 → P_cnn) ← no activation; **unbounded output**
```

`P_cnn` = 19 by default (matches data pack), or 12 when
`outputSize_override: 12` (matches `ball_and_stick_bbp`). About 6.4 M
trainable parameters.

The output `pred_unit ∈ ℝ^(B × P_cnn)` is unbounded — no tanh in the
network. The clamp lives in HybridLoss.

### Stage 5 — HybridLoss (toolbox/HybridLoss.py)

Returns a scalar loss that drives the entire training:

```python
loss = w_p · MSE(pred_unit, true_unit)             # channel term
     + w_v · MSE(z(v_sim_soma), z(true_volts_soma)) # voltage term
```

Two important transformations sit between `pred_unit` and the voltage
simulator:

1. **`tanh` clamp** (when `clamp_unit_tanh: True`)
   - Squashes `pred_unit ∈ ℝ` to `[-1, 1]`
   - Required for voltage-only training (no channel anchor) — without
     it, the unbounded last-layer output would drive jaxley to
     unphysical conductances and NaN the integrator
2. **unit→phys mapping** (mirrors `toolbox/unitParamConvert.py` but in
   torch so gradients flow):
   ```
   pred_phys = center · 10^(pred_unit · log_halfspan)
   ```
   With `log_halfspan = 0.5` and the tanh clamp, phys ranges over
   `[center / √10, center · √10]` ≈ 3.16× the default conductance.

The voltage term then z-scores the simulated soma trace per sample
(matching how the data was z-scored in stage 2) and computes MSE
over `min(T_sim, T_data)` time bins.

`mask_channels=True` skips the channel term entirely — used in Phase 4
fine-tuning on real data (no ground-truth params).

`fp64=True` casts `pred_unit` to fp64 before unit→phys; required for
stable backward through the BBP channel set at full t_max=500 ms (see
§6.4).

### Stage 6 — Jaxley bridge (toolbox/JaxleyBridge.py)

This is where torch's autograd graph crosses into JAX. It's the hot path.

#### 6.1 Per-cell handle (built once per process)

```python
class _CellHandle:
    cell_name        : str            # e.g. "ball_and_stick_bbp"
    spec             : CellSpec       # registry entry
    cell             : jx.Cell        # built once
    default_params   : list[dict]     # snapshot of trainable structure
    simulate_batch   : Callable       # jax.jit(jax.vmap(_simulate_one))
    downsample_step  : int            # output decimation factor
    sim_len, out_len : int            # T_sim before/after decimation
    v_init           : float

_CELL_CACHE : dict[(cell_name, stim_name, ckpt_key, solver), _CellHandle]
```

The cache key includes `checkpoint_lengths` and `solver` because each
distinct combination produces a different jit-compiled function.

#### 6.2 Forward — `_JaxleySimulate.forward`

```python
@staticmethod
def forward(ctx, params_phys, cell_name, stim_name, ckpt, solver):
    handle = get_handle(cell_name, stim_name, ckpt, solver)
    params_j = _torch_to_jax(params_phys)             # zero-copy DLPack
    v_j, vjp_fn = jax.vjp(handle.simulate_batch, params_j)
    ctx.vjp_fn = vjp_fn                                # captured closure
    return _jax_to_torch(v_j, params_phys.device, params_phys.dtype)
```

`handle.simulate_batch` is `jax.jit(jax.vmap(_simulate_one))`. Each
`_simulate_one(flat_phys)`:

```python
def _simulate_one(flat_phys):
    params = []
    for entry, idx, shape in zip(default_params, entry_to_cnn_idx, shapes):
        val = jnp.broadcast_to(flat_phys[idx:idx+1], shape)
        params.append({key: val})
    v = jx.integrate(cell, params=params, delta_t=spec.dt,
                     t_max=spec.t_max, data_stimuli=data_stim,
                     solver=solver, checkpoint_lengths=ckpt)
    return v[:, ::step]    # downsample to 10 kHz
```

`vmap` over the batch axis lets jaxley simulate the entire batch in
parallel on one GPU. `jit` compiles the whole thing once per cache
key — ~25-70 s cold, then ~5 s/step warm at B=128 fp64 t_max=500 ms.

#### 6.3 Backward — `_JaxleySimulate.backward`

```python
@staticmethod
def backward(ctx, grad_out):
    grad_j = _torch_to_jax(grad_out)
    (dparams_j,) = ctx.vjp_fn(grad_j)        # replays JAX vjp
    return _jax_to_torch(dparams_j, ...), None, None, None, None
```

The captured `vjp_fn` knows how to reverse-mode-differentiate
`simulate_batch` w.r.t. its input — i.e. produce
`∂v_sim/∂params_phys`. torch then chains this into the rest of its
graph (CNN → unit→phys → bridge → MSE → loss).

#### 6.4 Why fp64 is required

`bwd_euler` is an **implicit** integrator — at each step it solves
`(I − dt·J) x = b`. Reverse-mode AD through this chain involves the
product of N Jacobian-transpose matrices over N=5000 timesteps for
t_max=500 ms / dt=0.1 ms.

The BBP channel set has fast spike-time gating with V-dependent rates:
`dα/dV` near threshold is huge, so each spike contributes a large
"kick" to the cumulative product. With ~6-15 spikes per 500 ms trace,
the product overflows fp32's 23-bit mantissa → NaN gradients. fp64
(52-bit mantissa) absorbs the cumulative growth.

We measured this empirically:

| t_max (ms) | fp32 grad NaN | fp64 grad NaN |
|---|---|---|
| 50 | 0/12 | 0/12 |
| 100 | 0/12 | 0/12 |
| 250 | 12/12 ❌ | 0/12 |
| 500 | 12/12 ❌ | 0/12 |

Forward simulation stays finite at all t_max — only backward NaNs.
`crank_nicolson` solver and `jax.checkpoint` also NaN'd in fp32; only
fp64 fixes it.

### Stage 7 — Optimizer + DDP (toolbox/Trainer.py)

```
loss.backward()
DDP automatic ring-allreduce of CNN gradients across ranks (NCCL/NVLink)
self.grad_scaler.step(self.optimizer)   # Adam, lr=3e-4
self.grad_scaler.update()
```

Each rank's HybridLoss runs on its own batch, on its own GPU, in parallel.
At `loss.backward()`, DDP synchronizes the **CNN** gradients across ranks
(jaxley's gradient is computed locally per rank because each rank's
inputs are independent). The all-reduce is fast on NVLink: ~50 ms for
6.4 M params on 16 GPUs intra-node.

We measured 14.9× speedup on 16 GPUs vs 1 GPU at B=128/GPU and
t_max=500 ms — near-linear scaling because per-step time stays ~5 s
regardless of rank count.

## 3. Cell registry (toolbox/jaxley_cells)

Each registered cell is a `CellSpec`:

```python
@dataclass
class CellSpec:
    build_fn          : Callable    # () -> (jx.Cell, entry_to_cnn_idx)
    param_keys        : List[str]   # CNN-output order
    stim_attach_fn    : Callable    # (cell, stim_jnp) -> data_stimuli
    record_fn         : Callable    # (cell) -> sets up voltage record
    dt, dt_stim       : float       # ms
    t_max             : float       # ms
    v_init            : float       # mV
    default_stim_name : str
    stim_dir          : Path
```

Registered cells (this branch):

| name | params | geometry | use case |
|---|---|---|---|
| `ball_and_stick` | 4 | HH 1-comp soma + 5-comp passive dend | bench / Phase 1 |
| `ball_and_stick_bbp` | 12 | BBP channels on 1-comp soma + 5-comp dend | **Phase 2/3** |
| `L5TTPC` | 19 | full BBP L5_TTPC1 morphology, ~712 comps | Phase 5 (heavy) |
| `single_comp` | — | single soma HH | unit tests |

The cell name in the YAML controls which registry entry the loss simulator
uses. The data pack's filename is independent — they can mismatch (and in
Phase 2 they did, intentionally).

## 4. Configuration surface

Single source of truth: `<design>.hpar.yaml`. The trainer overlays CLI
overrides on top.

Key sections:

```yaml
data_path:                   { perlmutter: <abs path to dir of .mlPack1.h5> }
data_conf:                   { serialize_stims, append_stim, parallel_stim, num_data_workers }
max_epochs, batch_size, const_local_batch
train_conf.optimizer:        [adam, initLR]
train_conf.LRsched:          { plateau_patience, reduceFactor }

use_voltage_loss: True/False           # gates HybridLoss
voltage_loss:                           # only read when use_voltage_loss=True
    cell_name_for_sim: <registry key>
    channel_weight:    1.0
    voltage_weight:    1.0
    mask_channels:     False/True       # Phase 4 ⇒ True
    clamp_unit_tanh:   True             # required for voltage-only
    fp64:              True             # required for t_max ≥ 250 ms
    t_max_override:    auto / <ms>      # auto = dt_stim · len(stim_csv)
    solver:            bwd_euler
    checkpoint_lengths: null            # gradient memory optimization
    soma_probe_index:  0                # which probe in probsSelect is soma
    phys_par_range:    [...]            # optional inline; else read from H5 meta
    stim_name:         5k50kInterChaoticB

model:
    outputSize_override: 12             # forces CNN P_cnn ≠ data's P
    num_cnn_blocks, conv_block, fc_block, ...
```

CLI flags in `train_dist.py` override a subset (cellName, probsSelect,
stimsSelect, initLR, epochs, data_path_temp, do_fine_tune).

## 5. Multi-GPU layout (Perlmutter)

The hard-won settings (encoded in `batchShifterJaxley.slr`):

```
SBATCH:
  -N4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none

env:
  module load python; source activate <env>
  PYTHONNOUSERSITE=1
  JAX_PLATFORMS=cuda
  JAX_ENABLE_X64=true
  NCCL_NET_GDR_LEVEL=PHB
  FI_PROVIDER=cxi
  FI_CXI_DEFAULT_CQ_SIZE=131072

per-rank (inside srun's bash -c):
  JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/rank_$SLURM_PROCID

per-process pinning:
  train_dist.py: torch.cuda.set_device(SLURM_LOCALID)
  Trainer.__init__: same, plus DDP(device_ids=[self.device], output_device=...)
  HybridLoss.build: jax.config.update("jax_default_device", devices[localid])
```

Why each is non-negotiable:

- `--gpu-bind=none` (NOT `--gpus-per-task=1`): otherwise NCCL fails with
  `Cuda failure 101 'invalid device ordinal'` because each rank can only
  see its own GPU and NCCL P2P/SHM transports need cross-rank visibility.
- per-rank `JAX_COMPILATION_CACHE_DIR`: shared cache loads rank 0's
  cuda:0-targeted binary on rank N → XLA `Buffer on cuda:N, replica
  assigned to cuda:0` error.
- JAX `jax_default_device` per rank: without it, all ranks default to
  cuda:0 → all jaxley sims pile on GPU 0, **DDP scaling vanishes** (we
  measured 380 s/epoch on 4 GPUs without this fix vs 107 s with).
- `bash -c` wrapper around python: needed for `$SLURM_PROCID` to expand
  per-rank inside the srun command.

## 6. End-to-end gradient path (one tensor follows the arrow)

```
true_volts (B, T, C) fp16/fp32       <─── from H5
        │
        ▼ (permute to channel-first)
images (B, C, T) fp32                <─── CNN input
        │
        ▼ Conv → BN → FC
pred_unit (B, P_cnn) fp32            <─── CNN output, unbounded
        │
        ▼ tanh()
pred_unit_clipped (B, P_cnn) ∈ [-1, 1]
        │
        ▼ .double() if fp64=True
pred_unit_fp64 (B, P_cnn) fp64
        │
        ▼ phys = center · 10^(unit · log_halfspan)
pred_phys (B, P_cnn) fp64
        │
        ▼ DLPack zero-copy
params_j (B, P_cnn) fp64 jax array
        │
        ▼ jax.vjp(simulate_batch, params_j)
        │    └── inside JAX:
        │        for each sample:
        │            broadcast unit-vector to per-trainable shape
        │            jx.integrate(cell, params, dt, t_max, stim, solver,
        │                         checkpoint_lengths)
        │            v[:, ::step]   ← decimate to 10 kHz
v_j (B, n_rec, T_out) fp64 jax array
        │
        ▼ DLPack
v_sim (B, n_rec, T_out) fp64 torch
        │
        ▼ [:, 0, :], zscore along time
v_sim_z (B, T_out) fp64
        │
        ▼ truncate to min(T_sim, T_data); MSE vs v_true_z
v_loss (scalar) fp64
        │
        ▼ .float() back to fp32 graph
loss (scalar) fp32                   <─── total = w_p·ch + w_v·v
        │
        ▼ loss.backward()
        │    └── reverses every step above, with the captured
        │        ctx.vjp_fn replaying the forward in JAX
grad CNN.weight (...)
        │
        ▼ DDP all-reduce (NCCL/NVLink)
sync'd grad
        │
        ▼ Adam.step()
updated CNN weights
```

## 7. Where the artefacts land

```
$SCRATCH/tmp_neuInv/jaxley_voltage_only/<design>/<cell>/<jobid>/
├── train_dist.py + toolbox/ + slr + .yaml    ← frozen code snapshot
├── log.train                                  ← stdout
├── out/
│   ├── blank_model.pth                        ← whole nn.Module
│   ├── checkpoints/ckpt.pth                   ← state_dict + optimizer
│   ├── sum_train.yaml                         ← per-run metadata
│   └── tb_logs/                               ← TensorBoard
└── out/eval/                                  ← evaluate_voltage.py output
    ├── summary.yaml
    ├── voltage_metrics.csv
    ├── voltage_loss_hist.png
    ├── voltage_rmse_cdf.png
    └── trace_overlay_NN_*.png
```

## 8. What the architecture forbids today (Phase 3+ scope)

1. **Model mismatch is fatal for inversion.** If the data was generated
   by a different cell than the loss simulator, the optimizer collapses
   to a constant predictor (Phase 2 finding, 200-sample eval on
   `ball_and_stick_bbp` predicting from `L5_TTPC1` data). Phase 3 fixes
   this by generating synthetic data **with `ball_and_stick_bbp`
   itself**.
2. **No real ephys path yet.** Stage 2 assumes simulated data with
   ground-truth params. Real ephys lacks `unit_par`, so:
   - `mask_channels=True` makes HybridLoss skip the channel term
   - the data H5 still needs the same shape/keys (or we add a thin
     adapter); the experimental dirs under
     `/global/homes/k/ktub1999/ExperimentalData/PyForEphys/` use a
     different format → preprocessing required for Phase 4
3. **L5TTPC simulator is not yet tractable.** OOMs at B=4 fwd+bwd on a
   40 GB A100. Needs `_NCOMP=2`, `checkpoint_lengths=[100, 10]`, and
   batch padding (T1 from `docs/phase1/L5TTPC_gpu_plan.md`) before it
   can drive a training run.

## 9. Code map (one-line summary of every file in the loop)

```
train_dist.py                            entrypoint; DDP env, parses args, builds Trainer
toolbox/Trainer.py                       train loop, optimizer, DDP wrap, criterion swap
toolbox/Dataloader_H5.py                 reads .mlPack1.h5 into RAM, serves batches
toolbox/Model.py                         1D-CNN + FC (default, channel-only path)
toolbox/Model_Multi.py / _Stim.py        parallel-stim variants
toolbox/HybridLoss.py                    channel + voltage MSE; tanh clamp; t_max override
toolbox/JaxleyBridge.py                  torch ↔ JAX, jit+vmap simulate, autograd.Function
toolbox/jaxley_utils.py                  unit↔phys helpers (numpy + jax twins)
toolbox/jaxley_cells/__init__.py         CellSpec registry
toolbox/jaxley_cells/<cell>.py           one cell builder per file
toolbox/Plotter.py                       result plotting (channel-only path)
toolbox/Util_IOfunc.py                   read_yaml / write_yaml
toolbox/unitParamConvert.py              CSV-side unit→phys (for predicted CSVs)
predict.py                               channel-only inference (existing tool)
predictExp.py                            inference on experimental directories
evaluate_voltage.py                      voltage-aware inference (Phase 2 addition)
batchShifterJaxley.slr                   debug-queue 4-node SLURM, full env setup
batchShifterJaxley_100ep.slr             regular-queue long run variant
<design>.hpar.yaml                       config
toolbox/refresh_structure.py             regenerates structure.md auto-appendix
toolbox/tests/test_jaxley_bridge.py      Phase 1 bridge tests
toolbox/tests/test_hybrid_loss.py        Phase 2 loss tests
toolbox/tests/phase2_demo.py             end-to-end "untrained CNN → loss" demo
```
