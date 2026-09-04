# Jaxley in the Neuron-Inverter pipeline — presentation write-up

Everything needed to build a talk: what Jaxley is doing for us, how it plugs into the ML
stack, the models (both senses) we run, how we measure identifiability, and the headline
CA3 chaoticRamp result.

All numbers below are pulled from run artifacts on Perlmutter (`out/eval/summary.yaml`,
`log.train`, the ledger CSV) or from the code, and are cited to their source. Nothing here is
estimated.

**Contents**
1. [Architecture](#1-architecture)
2. [The models we work with](#2-the-models-we-work-with)
3. [Sensitivity analysis](#3-sensitivity-analysis)
4. [Best result — CA3 chaoticRamp](#4-best-result--ca3-chaoticramp)
5. [Appendix: figures, commands, caveats](#5-appendix)

---

## 0. The one-slide version

We solve an inverse problem: **given a voltage trace, recover the ion-channel conductances
that produced it.** The original pipeline does this with plain supervised learning — a 1-D CNN
regresses conductances against ground-truth labels from NEURON simulations.

Jaxley is a compartmental neuron simulator written in JAX, which means **a simulation is a
differentiable function**. That lets us close the loop:

> CNN predicts conductances → Jaxley re-simulates a voltage trace *from those predictions* →
> compare the simulated trace to the observed one → backpropagate the mismatch all the way
> through the ODE solve and into the CNN weights.

The payoff: **this loss needs no parameter labels at all** — only the voltage trace you already
measured. That is what makes fine-tuning on real patch-clamp data possible, where ground-truth
conductances do not exist and never will.

The two extremes bracket the whole research programme, both measured on CA3:

| mode | what supervises it | mean channel R² | cost |
|---|---|---:|---|
| Supervised parameter-MSE (`use_voltage_loss: False`) | ground-truth labels | **0.995** | ~4 min, no ODE solve in the loop |
| Voltage-only (`channel_weight: 0`, `mask_channels: True`) | the trace alone | **0.763** | 9 h on 32 GPUs; every step pays a full batched ODE solve + adjoint |

The gap between 0.763 and 0.995 *is* the research question: it measures how much information
about conductances is actually extractable from a soma voltage trace, given a particular loss
and a particular stimulus. Section 3 shows the information is genuinely there (the inverse
problem is well-conditioned), so the gap is an **objective** problem, not a data or
architecture problem.

---

## 1. Architecture

### 1.1 Why a differentiable simulator changes the problem

Conventional supervised training needs `(trace, conductances)` pairs. You can only ever get
those from simulation, so the model learns the simulator's world and has no mechanism to
correct itself on real data.

With a differentiable simulator, the physics itself becomes the supervisor. The loss is

```
L = ‖ V_observed − Simulate(CNN(V_observed)) ‖
```

which is self-supervised: the CNN's own *input* is the target. That is the single most
important structural fact about the design, and it shows up literally in the code — the loss is
called as `self.criterion(outputs, labels, images)` and the third argument is the CNN's input
tensor.

### 1.2 The stack

Five layers, deliberately additive. The original `Model.py` / `Trainer.py` / `Dataloader_H5.py`
keep their shapes; the integration is a **single `if`** in the trainer.

```
  toolbox/jaxley_cells/          biophysics    — what cell to simulate
        ↑
  toolbox/JaxleyBridge.py        plumbing      — torch ⇄ JAX, autograd, jit+vmap, caching
        ↑
  toolbox/jaxley_utils.py        spine         — unit⇄phys, voltage normalization, stim IO
        ↑
  toolbox/HybridLoss.py          objective     — what "close enough" means
   + soft_dtw / soft_efel / trace_metrics
        ↑
  toolbox/Trainer.py             one-line hook — self.criterion swap
```

Plus a sixth piece running the same stack **in the opposite direction**: `scripts/gen_*.py`
generate synthetic training packs using the very same cell builders the loss later uses as its
simulator. That symmetry is not cosmetic — see §1.9.

### 1.3 The cell registry — the single integration point

Every simulable cell is a `CellSpec` registered under a string name
(`toolbox/jaxley_cells/__init__.py:26-92`). Adding a cell requires **no changes to the loss,
the bridge, or the training loop**.

```python
@dataclass
class CellSpec:
    build_fn:          Callable   # () -> (jx.Cell, entry_to_cnn_idx)
    param_keys:        List[str]  # CNN-output order — MUST match sum_train.yaml's parName
    stim_attach_fn:    Callable   # (cell, stim_jnp) -> data_stimuli
    record_fn:         Callable   # (cell) -> sets up voltage recording
    dt, dt_stim:       float      # ms
    t_max:             float      # ms
    v_init:            float      # mV
    default_stim_name: str
    stim_dir:          Path
```

The subtle field is `entry_to_cnn_idx`. A Jaxley cell can have more trainable *entries* than
the CNN has *outputs*, because one conductance name may appear in several sections.
`entry_to_cnn_idx[i]` says which CNN output feeds trainable entry `i`. Several entries may
share one index — L5TTPC's `gIhbar_Ih_dend` drives both basal and apical, so index 3 appears
twice and the cell has 20 entries for 19 CNN outputs. On the forward pass the scalar fans out;
on the backward pass JAX sums the partials back into one gradient. **Correct chain rule for
free.**

### 1.4 The bridge — where torch meets JAX

`toolbox/JaxleyBridge.py` is the hot path. Three things matter for the talk:

**Handle cache.** A `_CellHandle` holds the built `jx.Cell` and, crucially,
`simulate_batch = jax.jit(jax.vmap(_simulate_one))`. It is built once per process and cached
under `(cell_name, stim_name, checkpoint_lengths, solver)`, because each combination compiles
to a different XLA binary. **Cold compile 25–70 s; warm dispatch ~5 s/step at B=128, fp64,
t_max = 500 ms.** Changing batch size does *not* recompile (vmap owns the batch axis); changing
solver or checkpoint lengths does.

**Forward** is a `torch.autograd.Function` that captures JAX's VJP closure:

```python
def forward(ctx, params_phys, cell_name, stim_name, ckpt, solver):
    handle   = get_handle(cell_name, stim_name, ckpt, solver)
    params_j = _torch_to_jax(params_phys)              # zero-copy via DLPack
    v_j, vjp_fn = jax.vjp(handle.simulate_batch, params_j)
    ctx.vjp_fn = vjp_fn
    return _jax_to_torch(v_j, ...)
```

**Backward** just replays it:

```python
def backward(ctx, grad_out):
    (dparams_j,) = ctx.vjp_fn(_torch_to_jax(grad_out))
    return _jax_to_torch(dparams_j, ...), None, None, None, None
```

The design is clean because JAX never sees torch and torch never sees JAX — the only contract
is a DLPack tensor and a captured closure.

### 1.5 The spine — two conversions that must agree on both sides

`toolbox/jaxley_utils.py` is 90 lines and load-bearing.

**Unit ⇄ physical.** The CNN works in a normalized log space; Jaxley wants S/cm². Mirrored in
numpy and JAX so gradients flow:

```
phys = center · 10^(unit · log_halfspan)
```

with `[center, log_halfspan, unit_string]` per parameter, read from
`sum_train.yaml['input_meta']['phys_par_range']`. With `log_halfspan = 0.5` and a tanh clamp,
physical values span `[center/√10, center·√10]`.

**Voltage normalization — a fixed global scale, not a per-trace z-score.**

```python
VOLT_NORM_MEAN = -60.0951997   # mV
VOLT_NORM_STD  =  18.95055671  # mV
```

The reason is a real insight worth a slide: a per-trace z-score throws away *absolute* voltage
information — resting potential, spike height, subthreshold amplitude — which is exactly what
distinguishes one conductance set from another. A fixed scale preserves it. The constraint is
that the data generator and the loss must use the **same** transform, or the MSE compares two
different spaces.

### 1.6 The objective — a configurable battery, not one loss

`toolbox/HybridLoss.py` (~780 lines). Everything below is read from the `voltage_loss:` block of
the design YAML.

**Preprocessing applied to the CNN output before it reaches the simulator:**

1. `clamp_unit_tanh` — squashes the unbounded final linear layer to `[-1, 1]`. Not optional in
   practice for voltage-only training: without a parameter anchor the unbounded output drives
   Jaxley to unphysical conductances and NaNs the integrator.
2. `fp64` — cast to float64 before unit→phys (see §1.8).
3. unit→phys via the mapping above.

**The loss terms**, each weighted independently:

| term | keys | what it measures |
|---|---|---|
| channel MSE | `channel_weight`, `mask_channels` | parameter-space error; `mask_channels: True` skips it entirely — **the real-ephys mode** |
| voltage MSE | `voltage_weight`, `mse_weight` | pointwise fixed-scale trace error |
| soft-DTW | `dtw_weight`, `dtw_gamma`, `dtw_band_ms`, `dtw_n_points` | differentiable dynamic time warping — tolerant of spike-timing jitter, which pointwise MSE punishes savagely |
| soft-eFEL | `efel_weight`, `efel_features`, `efel_k`, `efel_thr`, `efel_nmax` | differentiable surrogates for electrophysiology features (spike rate, AP amplitude, AHP depth, ISI stats, upstroke/downstroke dV/dt, subthreshold levels) |
| multiscale blur | `blur_weight`, `blur_sigmas_ms` | van-Rossum-style Gaussian-smoothed MSE at several timescales |
| trace low-pass | `lowpass_ms` | moving-average filter applied to *both* traces before the MSE |
| randomized smoothing | `smooth_sigma`, `smooth_samples` | averages the voltage loss over Gaussian jitter of `pred_unit` — smooths the **parameter-space** landscape, not the trace |
| range penalty | `range_penalty_weight`, `range_penalty_margin` | soft alternative to the tanh clamp: `mean(relu(|pred_unit| − margin)²)` |

**Two orthogonal mechanisms on top:**

- `grad_precond` — per-channel gradient reweighting, either explicit weights or derived from a
  measured sensitivity vector. Motivated by wanting to boost channels the voltage loss barely
  sees. **Empirically it does not work** (§4.5).
- `schedule` — ramps a weight over N epochs. This turned out to matter enormously: soft-eFEL as
  a *primary* loss at weight 1.0 diverges, but ramped in as a small auxiliary (0.05 → 0.2 over
  20 epochs) under a soft-DTW primary is **the single best lever found** (§4.3).

**Three robustness mechanisms that are easy to miss but matter operationally:**

- **Batch padding.** The final mini-batch of an epoch is usually short, and a different batch
  shape triggers a fresh XLA compile. `pad_batch_size` pads up before the call and slices back
  after.
- **NaN quarantine, twice.** Non-finite simulated rows are dropped from the batch before the
  loss, and the bridge's backward runs `torch.nan_to_num` on the incoming gradient. Both exist
  because **under DDP a single bad sample's NaN propagates to every rank through the
  all-reduce**.
- **Time alignment.** `sim_t_skip_ms / sim_dt_ms` gives a bin offset to discard simulator
  warm-up; both traces are then truncated to `min(T_sim, T_data)`.

### 1.7 The differentiable feature libraries

- **`toolbox/soft_dtw.py`** — soft-DTW with a `softmin` recursion and a Sakoe-Chiba band. The
  band width is a real hyperparameter: **8 ms carries the residual firing-rate signal, 4 ms did
  not help** (measured: band4 mean R² 0.595 vs baseline 0.593 at 40k).
- **`toolbox/soft_efel.py`** (~630 lines) — the interesting one. Real eFEL features are step
  functions of the trace (count spikes, find peaks) and therefore have **zero gradient almost
  everywhere**. This module rebuilds them from smooth primitives: `_softplus_max`/`_softplus_min`
  for peaks, a sigmoid-based `_spike_times` detector at threshold `thr` with sharpness `k` and a
  cap of `nmax` spikes, and `_soft_select` for windowed picks.

  It ships `STRONG_FEATURES` — the six validated against real eFEL at Pearson r ≥ 0.79:
  `time_to_first_spike`, `mean_frequency`, `inv_first_ISI`, `AHP_depth_abs_slow`, `ISI_values`,
  `AP_amplitude`. Three surrogates were **dropped** from production configs after measuring
  r = 0.07 (`spike_half_width`), −0.14 (`fast_AHP_change`) and 0.16 (`AHP_slow_time`) against
  their real counterparts — a badly-correlated surrogate injects pure gradient noise.

  Two implementation details make it trainable at all: `only=` prunes which features are
  computed (the unpruned version materializes `(B, nmax, T)` window tensors that dominate the
  backward graph), and the whole forward path is free of `.item()`, boolean indexing and numpy,
  so the autograd graph never breaks.
- **`toolbox/trace_metrics.py`** — whole-trace distances: Gaussian kernels, multiscale blurred
  MSE, decimation, a second soft-DTW, smoothed ensemble std, `parameter_explained_var`.

### 1.8 The end-to-end gradient path

One tensor, disk to weight update:

```
true_volts (B, T, C)              from the mlPack1 H5
   ↓ permute to channel-first
images (B, C, T) fp32             CNN input
   ↓ Conv1d ×3 → BatchNorm → FC ×6
pred_unit (B, P) fp32             CNN output — unbounded, no activation
   ↓ tanh()                       clamp_unit_tanh
   ↓ .double()                    fp64
   ↓ phys = center · 10^(unit · log_halfspan)
pred_phys (B, P) fp64
   ↓ DLPack, zero-copy
params_j (B, P) jax array
   ↓ jax.vjp(jit(vmap(simulate_one)), params_j)
   ↓     broadcast each scalar to per-compartment shape
   ↓     jx.integrate(cell, params, dt, t_max, stim, solver, checkpoint_lengths)
   ↓     decimate to the 10 kHz grid the CNN was trained on
v_j (B, n_rec, T_out) fp64
   ↓ DLPack
v_sim (B, n_rec, T_out) torch
   ↓ take soma probe, fixed-scale normalize, truncate to min(T_sim, T_data)
   ↓ MSE / soft-DTW / soft-eFEL / blur, weighted sum
loss (scalar)
   ↓ loss.backward()   — ctx.vjp_fn replays the JAX adjoint
grad CNN.weight
   ↓ DDP ring-allreduce (NCCL over NVLink)
   ↓ Adam.step()
```

**DDP synchronizes only the CNN gradients.** Each rank's Jaxley adjoint is computed locally,
because each rank's samples are independent.

**Inside `jx.integrate`** — worth one slide, because it is where all the time and memory go.
The time loop is a `nested_checkpoint_scan` over T steps. Each step: (1) every channel's
`update_states` computes α/β rates from the current voltage and takes an exponential-Euler step
`m_{t+1} = m∞ + (m_t − m∞)·e^{−dt/τ}`; (2) each channel returns `(linear, const)` so that
`i(v) = g·(v − E)` is written `linear·v + const`, keeping the voltage solve linear;
(3) the stimulus current is added at the soma; (4) an **implicit** solve
`(I + dt·A)·v_{t+1} = v_t + dt·const` where `A` is the tridiagonal **Hines matrix**;
(5) record. `vmap` runs all of that for B independent neurons at once.

The **backward pass costs 6–9× the forward and ~200× the memory** — reverse-mode AD over a
length-T scan must retain every intermediate. Measured peak GPU memory ≈ 98 MB at B=64 and
≈ 196 MB at B=128, versus ≈ 0 in forward. `checkpoint_lengths=[20,50]` rematerializes inner
segments, trading ≈2× compute for ≈√T memory.

Two counter-intuitive measurements worth mentioning: **compartment count is nearly free** at
these sizes (6 → 81 compartments cost ~0% on an A100 — kernel-launch overhead dominates, not the
tridiagonal solve), and **for small cells CPU beats GPU at every batch size up to 128**, because
a 6-compartment cell is too small to amortize dispatch.

### 1.9 Why fp64 is not optional

`bwd_euler` is implicit: each step solves `(I − dt·J)x = b`. Reverse-mode AD through N = 5000
such steps multiplies N Jacobian-transpose matrices. The BBP channel set has fast, steeply
voltage-dependent gating, so `dα/dV` near threshold is enormous and every spike contributes a
large kick to the cumulative product. With 6–15 spikes in a 500 ms trace, that product overflows
fp32's 23-bit mantissa and **the gradient becomes NaN**:

| t_max | fp32 grad NaN | fp64 grad NaN |
|---|---|---|
| 50 ms | 0/12 | 0/12 |
| 100 ms | 0/12 | 0/12 |
| 250 ms | **12/12** | 0/12 |
| 500 ms | **12/12** | 0/12 |

The forward pass stays finite throughout — **only the adjoint blows up**. `crank_nicolson` and
`jax.checkpoint` were both tried and neither helped, because neither addresses the mantissa
limit. The fix is `fp64: True` plus `JAX_ENABLE_X64=true` in the environment (which must be set
*before* JAX is imported, hence its position in the `.slr`), at ~2–3× forward and ~5× backward
cost.

Caveat for accuracy: this is a property of the **stiff BBP channel set**, not of Jaxley.
Single-compartment CA3 is documented as fp32-stable; CA3 runs still use fp64 out of caution.

### 1.10 Multi-GPU on Perlmutter — four settings, each found by hitting a real failure

All encoded in `batchShifterJaxley*.slr`; none should be "simplified".

1. **`--gpus-per-node=4 --gpu-bind=none`, never `--gpus-per-task=1`.** Per-task binding hides
   peer GPUs from each rank's CUDA context, and NCCL's P2P/SHM transports need cross-rank
   visibility. Symptom: `Cuda failure 101 'invalid device ordinal'`.
2. **Per-rank JAX compilation cache**, `JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/rank_$SLURM_PROCID`.
   A shared cache serves rank 0's `cuda:0`-targeted binary to rank N and XLA dies with
   "Buffer on cuda:N, but replica assigned to cuda:0". **This one broke a whole CA3 ablation
   series (A2–A5) before it was diagnosed.**
3. **Pin JAX per rank**, `jax.config.update("jax_default_device", devices[localid])` inside
   `HybridLoss`. Torch is pinned via `SLURM_LOCALID`, but JAX defaults to `cuda:0`
   independently — without this every rank piles its Jaxley solve onto GPU 0 and DDP scaling
   silently vanishes (**measured 380 s/epoch on 4 GPUs vs 107 s with the fix**).
4. **Wrap the launch in `bash -c`** so `$SLURM_PROCID` expands per rank.

Environment: `PYTHONNOUSERSITE=1`, `JAX_PLATFORMS=cuda`, `JAX_ENABLE_X64=true`,
`NCCL_NET_GDR_LEVEL=PHB`, `FI_PROVIDER=cxi`, `FI_CXI_DEFAULT_CQ_SIZE=131072`. Runtime is a
**conda env, not shifter**: `/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley`.

**Scaling reality (good slide).** 16 GPUs gave **14.9×** over 1 GPU at B=128 — near-linear. But
**32 GPUs gave no speedup at all** over 16 at a pinned global batch of 2048 (220.8 vs ~247
s/epoch), because the fp64 Jaxley solve is dominated by the **sequential 5001-step time
integration**, which parallelizes over neither batch nor GPUs. Past 16 GPUs the right move is a
longer wall-clock, not more nodes.

### 1.11 Data generation — the same stack, run forwards

`scripts/gen_ca3_sharded.py` (and `gen_ball_and_stick_data.py`, `gen_multistim_data.py`) use the
**loss simulator itself** to manufacture training packs. The sharded CA3 generator: 16 ranks
each pin to a local GPU, draw a contiguous slice of `Uniform(-1,1)^P` unit parameters, simulate
under every requested stimulus, and write raw un-normalized shards; a single-rank merge phase
concatenates in rank order, applies `normalize_volts_fixed` (**the same transform HybridLoss
uses**), shuffles, splits 80/10/10, and writes an `mlPack1.h5` byte-compatible with what
`Dataloader_H5.py` already reads. Determinism is preserved by drawing the full parameter array
once from `default_rng(seed)` in both phases.

Why this symmetry matters: see §4.6, Finding "model mismatch is fatal and fails silently".

One non-obvious constraint: `simulate_batch` always constructs a `jax.vjp`, **even inside
`torch.no_grad()`**. Generation therefore pays for the backward tape whether it wants it or not,
which is why generators pass `checkpoint_lengths` for anything larger than CA3.

### 1.12 The trainer hook

Exactly one conditional:

```python
if params.get('use_voltage_loss'):
    self.criterion = build_hybrid_loss(params)
```

`_ChannelOnlyAdapter` wraps a plain MSE in the same call signature, so the call site never moves
and `voltage_weight: 0` reproduces the old path bit-for-bit (there is a regression test for
exactly that).

Three smaller additions round it out: `outputSize_override` lets the CNN emit a different number
of parameters than the data pack carries (needed whenever the loss simulator has fewer channels
than the pack — e.g. a 6-param `ca3_pyramidal` reading a 19-param pack);
`train_conf.clip_grad_norm`, added specifically for this path; and `set_epoch`, called once per
epoch to drive the weight schedules.

---

## 2. The models we work with

"Model" means two different things in this project. Both belong in the talk.

### 2.1 (a) The ML models

#### The standard CNN — `toolbox/Model.py`, class `MyModel`

A 1-D CNN → flatten → MLP regressor. Everything is built from the YAML `model:` block; nothing
is hard-coded. Input shape is injected by the dataloader (`Dataloader_H5.py:54-55`), not written
by hand.

Per conv block the order is **Conv1d → MaxPool1d → ReLU**, with normalization inserted **by list
position**, not per layer (`batch_norm_cnn_slot: 3` means after the first Conv/Pool/ReLU triple
only; `instance_norm_slot: -9` is a sentinel that never matches, so InstanceNorm is off
everywhere). Stride is hard-fixed to 1. The flatten dimension is computed *empirically* by
pushing two zero examples through the conv stack.

**The final Linear is bare — there is no output activation.** `pred_unit ∈ ℝ^(B×P)` is
unbounded; any clamping (`clamp_unit_tanh`) happens downstream in `HybridLoss`, not in the
network. Worth saying out loud in a talk, because it is why the tanh clamp is mandatory for
voltage-only runs.

Every design YAML in the repo — `m8lay_vs3`, all the CA3 ones, `ballBBP_voltage_only`,
`l5ttpc_supervised` — uses the **identical backbone**:

```
filter [30, 90, 180]   kernel [4, 4, 4]   pool [4, 4, 4]
FC dims [512, 512, 512, 256, 128]   dropFrac 0.04
```

Concretely, for the CA3 packs (T = 5001 bins, C = 1 soma probe, P = 6):

```
Conv1d(1  → 30,  k=4, s=1) → MaxPool1d(4) → ReLU → BatchNorm1d(30)
Conv1d(30 → 90,  k=4, s=1) → MaxPool1d(4) → ReLU
Conv1d(90 → 180, k=4, s=1) → MaxPool1d(4) → ReLU
Flatten(180 × 77 = 13860) → BatchNorm1d(13860)
FC(13860→512) → ReLU → Dropout(0.04)
FC(512→512)   → ReLU → Dropout(0.04)
FC(512→512)   → ReLU → Dropout(0.04)
FC(512→256)   → ReLU → Dropout(0.04)
FC(256→128)   → ReLU → Dropout(0.04)
FC(128→6)     ← no activation
```

Time-axis arithmetic (k=4 s=1 → T−3; pool 4 → ⌊/4⌋): 5001→4998→1249→1246→311→308→**77**.
Total ≈ **7.89 M** weights, dominated by the 13860→512 first FC (7.10 M).

For the BBP case (T = 4000, C = 3 probes, P = 19): 4000→…→**61**, flat = 180×61 = 10980,
≈ **6.41 M** weights.

#### The variants

| class | when it is selected | notes |
|---|---|---|
| `Model.py` | default | the workhorse; everything above |
| `Model_Multi.py` | `data_conf.parallel_stim: True` | builds one **independent CNN tower per stimulus**, concatenates their flattened outputs along the channel dim, then one shared BatchNorm + FC head. No `use_manual_features` support. |
| `Transformer_Model.py` | `model_type: Transformers` | sinusoidal positional encoding + `nn.TransformerEncoder`. Hyperparameters are **hard-coded at the call site** (`Trainer.py:154-160`: `d_model=2, nhead=2, num_encoder_layers=6, dim_feedforward=64, max_seq_len=4000`) and the output head is hard-wired to 19. Selected by `MultuStimTrans.hpar.yaml`. |
| `Model_Multi_Stim.py` | never | abandoned refactor — `def _init_` (single underscores) means `__init__` is never called. Do not present it as live. |
| `Model2d.py` | never | legacy |

Selection order (`Trainer.py:149-168`): `do_fine_tune` > `model_type == "Transformers"` >
`data_conf.parallel_stim` > default `Model.py`.

#### `use_manual_features` (off by default)

Enables a second input tensor of hand-engineered features (e.g. AP count from `efel`)
concatenated **after the CNN flatten**, so the extras bypass the conv stack and enter at the
first FC layer. Features are precomputed per domain, z-scored with **train-split statistics**,
and `np.repeat`ed by `numStim` when `serialize_stims` is on so rows stay aligned after the
shuffle.

⚠️ Caveat if anyone asks: this path is essentially unexercised. `Model.py:96` does
`x = x.view(-1, self.flat_dim)` unconditionally before the feature branch re-views to
`cnn_flat_dim`; with the flag on, `flat_dim` already includes the extras, so it will throw for
essentially any batch size. `extra_features_dim` appears in no YAML.

#### Representative hyperparameters

| | `m8lay_vs3` (BBP) | `ca3_vo_chaoticramp_dtw` | `ca3_supervised_chaoticramp` |
|---|---|---|---|
| conv filter / kernel / pool | `[30,90,180] / [4,4,4] / [4,4,4]` | same | same |
| FC dims / dropFrac | `[512,512,512,256,128]` / 0.04 | same | same |
| `outputSize_override` | — (uses data P) | **6** | **6** |
| `batch_size` (per GPU) | 512 | 128 | 128 |
| `max_epochs` | 12 | 100 | 200 |
| optimizer / initLR | `[adam, 0.005]` | `[adam, 1e-4]` | `[adam, 3e-4]` |
| LR schedule | plateau, patience 8, factor 0.11 | plateau, patience 10, factor 0.3 | plateau, patience 10, factor 0.3 |
| `clip_grad_norm` | absent | **1.0** | 0 |
| `use_voltage_loss` | n/a | True (`dtw 1.0`) | **False** (pure param-MSE) |

Adam is the only accepted optimizer name (`exit(99)` otherwise).

### 2.2 (b) The biophysical cell models

Seven registered names across six distinct cells (`l5ttpc.py` registers three aliases for one
spec). All share `dt = dt_stim = 0.1 ms`, `t_max = 500 ms`,
`default_stim_name = "5k50kInterChaoticB"`, and
`stim_dir = /pscratch/sd/k/ktub1999/main/DL4neurons2/stims`.

| registry name | file | CNN params | trainable entries | geometry | probes | v_init | status |
|---|---|---:|---:|---|---:|---:|---|
| `single_comp` | `soma_only.py` | 3 | 3 | 1 comp HH, L=20 µm r=10 µm | 1 | −65 | bench / unit tests |
| `ball_and_stick` | `ball_and_stick.py` | 4 | 4 | HH soma + passive dendrite | **2** (soma, dend) | −65 | Phase-1 gradcheck cell |
| `ball_and_stick_bbp` | `ball_and_stick_bbp.py` | 12 | 12 | BBP channels, soma + 5-comp dendrite | 1 | −75 | Phase 2/3 workhorse |
| **`ca3_pyramidal`** | `ca3_pyramidal.py` | **6** | 6 | **single compartment**, L = diam = 50 µm | 1 | −65 | **current research vehicle** |
| `L5TTPC` / `l5ttpc` | `l5ttpc.py` | 19 | **20** | `jx.read_swc(L5_TTPC1.swc)`, ncomp = `$L5TTPC_NCOMP` (default 4) | 1 | −75 | heavy; OOMs at B=4 fwd+bwd on 40 GB A100 |
| `l5ttpc_multiprobe` | `l5ttpc_multiprobe.py` | 19 | 20 | identical biophysics | **4** (soma, axon, apical, dend) | −75 | multi-probe observability experiment |
| `L5PC_jaxley` | `l5pc_jaxley.py` | 19 | 20 | vendored SWC + `jaxley_mech` Hay-2011 channels | 1 | −75 | self-contained — **no NEURON, no `/pscratch` dependency** |

#### `ca3_pyramidal` in detail — the one that works

Ported from the NEURON model in `DL4neurons2/Adapting CA3 Pyramidal Neuron/`.

- **Geometry, verbatim from `morphology_mechanisms.hoc`:** single soma branch, `ncomp=1`,
  L = 50 µm, radius = 25 µm (diam 50 µm), `capacitance = 1.41 µF/cm²`,
  `axial_resistivity = 150 Ω·cm`, `celsius = 34 °C`, `v_init = −65 mV`.
- **The 6 trainable parameters** (identity `entry_to_cnn_idx = [0..5]` — no fan-out):

  | idx | `PARAM_KEYS` name | jaxley key | default (S/cm²) |
  |---|---|---|---:|
  | 0 | `CA3_g_leak` | `Leak_CA3_g` | 3.9417e−5 |
  | 1 | `CA3_gbar_na3` | `Na3_gbar` | 0.04 |
  | 2 | `CA3_gkdrbar_kdr` | `Kdr_ca1_gkdrbar` | 0.01 |
  | 3 | `CA3_gkabar_kap` | `Kap_rox_gkabar` | 0.04 |
  | 4 | `CA3_gbar_km` | `Km_ca3_gbar` | 5.2e−4 |
  | 5 | `CA3_gkdbar_kd` | `Kd_ca3_gkdbar` | 2.5e−4 |

- **Channels** are hand-transcribed `.mod` ports in `toolbox/jaxley_channels/ca3_channels.py`:
  `Leak_CA3` (no gates), `Na3` (**three** gates `m, h, s` — `s` is slow inactivation),
  `Kdr_ca1` (`n`), `Kap_rox` (`n, l`), `Km_ca3` (`m`), `Kd_ca3` (`n`), with a 34 °C temperature
  factor and a singularity-safe `_trap0`. `cacum` (the Ca buffer) is deliberately **not** ported
  — the hoc inserts no Ca channels, so `ica = 0` always.
- **Reversals:** `ena = +55`, all four K channels `ek = −90`, and `Leak_CA3_e = +93.9115 mV`,
  taken verbatim from the source hoc. It looks like an upstream typo, but `g_leak` is small
  enough (3.94e−5) that the strong resting K_d current pulls rest to −65 mV anyway — **and both
  the NEURON reference and the Jaxley port use the same value, so the comparison is valid.**

#### The CA3 NEURON validation — why CA3 became the vehicle

**All 7 tests passed on the first try.** This is the slide that earns trust in everything
downstream.

| # | Setup | Metric | NEURON | Jaxley | Δ | threshold | pass |
|---|---|---|---:|---:|---:|---|---|
| 1 | Rest, I=0, 500 ms | mean V, last 100 ms | −65.000 | −65.000 | **0.000 mV** | < 0.5 mV | ✅ |
| 2 | +0.05 nA, 100→400 ms | RMSE | — | — | **0.0016 mV** | < 1.5 mV | ✅ |
| 2 | same | spike count | 0 | 0 | 0 | — | ✅ |
| 3 | +0.30 nA, 100→400 ms | peak \|ΔV\| | — | — | **0.00016 mV** | < 5 mV | ✅ |
| 3 | same | mean spike-time \|Δ\| | — | — | **0.0 ms** | < 1 ms | ✅ |
| 4 | f-I sweep, amps `[0, .05, .1, .2, .3, .5, 1.0]` | spike counts | `0/0/0/0/0/0/25` | `0/0/0/0/0/0/25` | 0 | ≤ 1 | ✅ |
| 5 | 50 random ḡ × U(10^−0.5, 10^0.5) | corr(spike count) | — | — | **ρ = 0.9945** | > 0.95 | ✅ |
| 6 | AP shape, +1.0 nA, 200 ms | peak V | 44.544 | 42.711 | **1.833 mV** | < 3 mV | ✅ |
| 6 | same | ½-width | 1.30 | 1.40 | **0.10 ms** | < 0.3 ms | ✅ |
| 6 | same | AHP depth | −14.699 | −14.065 | **0.634 mV** | < 3 mV | ✅ |
| 7 | wall-time, 100 traces × 500 ms, CPU | ms/trace | **118.7** | **571.3** | +452.6 | informational | ℹ |

Three honest caveats worth pre-empting from the audience:

- **No spikes at 0.30 nA in either stack.** The model has unusually strong resting K_d outward
  current (~6× the leak depolarizing drive), so threshold sits between 0.5 and 1.0 nA. NEURON
  and Jaxley agree — the "0 spikes" rows are physics, not a broken port.
- **AP peak is 1.83 mV lower in Jaxley.** Within tolerance, but the *signed* bias suggests
  `bwd_euler` slightly underestimates the upstroke vs NEURON's Crank-Nicolson. Not chased, since
  the f-I curve and spike timing match exactly.
- **Jaxley is ~4.8× slower than NEURON for a *single* trace on CPU.** This is the wrong
  benchmark and you should say so: Jaxley wins by **batching on GPU** (14.9× at 16 GPUs, B=128),
  and by being differentiable at all. Single-trace wall-time does not reflect the data-gen /
  training use case.

Figures: `docs/ca3/test1_rest_overlay.png` … `test6_apshape_overlay.png` (six overlays, one per
test) and `docs/ca3/summary.csv`.

#### The other cells, honestly

- **`ball_and_stick_bbp`** works and is well characterized, but its geometry is *pathological*
  for the standard stimuli: a 20 µm × 10 µm soma has R_in ≈ 26 MΩ, so a stimulus peaking at
  +6.8 nA mathematically forces **±180 mV passive swings**. Those are **not** solver artifacts —
  dt = 0.1 and dt = 0.025 traces overlay to within 0.5 mV. It also shows a depolarization plateau
  at ~−25 mV driven by persistent Na (`Nap_Et2`); dropping that default ~7× restores normal
  spike-recovery cycling.
- **`l5ttpc` is the unsolved one.** It OOMs at B=4 forward+backward on a 40 GB A100. Documented
  fidelity backlog: dt = 0.1 vs NEURON's 0.025 (worth ~0.5–2 ms of spike drift over 500 ms),
  uniform `ncomp = 4` vs NEURON's `d_lambda` rule, unverified Q10 factors, and the apical Ih
  gradient. Current divergence from the NEURON reference is **max|Δ| ≈ 94 mV with 2 spikes
  versus 4** — most of it plausibly dt-driven.
- **The L5TTPC apical Ih gradient is a genuine open bug.** `_apply_apical_ih_gradient` writes
  492 distinct per-compartment values implementing BBP's
  `gIh(d) = max(0, −0.8696 + 2.087·e^{0.0031 d})·8e−5`, and then
  `cell.apical.make_trainable("Ih_gbar")` **averages them into one scalar** which the bridge
  broadcasts back uniformly — proximal apical then gets ~13× too much Ih and distal ~4× too
  little. The gradient survives only at default state. The fix (an `entry_multipliers` vector
  giving `val_i = s·m_i`, so one CNN scalar drives a *shaped* spatial profile) exists in the
  `l5ttpc-ih-fix` worktree. **Know which behaviour your checkout has.**
- **`L5PC_jaxley`** trades fidelity for independence: no NEURON, no BBP hoc template, no
  `/pscratch` tree. Channel kinetics are `jaxley_mech` (Hay 2011), **not guaranteed
  bit-identical** to `bbp_channels_jaxley`, and the morphology is a *different* L5PC from BBP
  `L5_TTPC1`. So it is a faster self-contained option, **not** a free drop-in for a CNN already
  trained against BBP L5_TTPC1.

---

## 3. Sensitivity analysis

### 3.1 Why we do it

Voltage-only training can only recover the conductances the trace is actually *sensitive* to.
Before blaming the optimizer, you have to know whether the information is present at all. Every
tool below measures a property of the **inverse problem**, independent of any trained CNN —
which is exactly why the conclusions are trustworthy.

The framing that turned out to matter most: **observability ≠ trainability.** A stimulus can
maximize how much the trace moves when you wiggle a conductance and still train *worse*, because
the loss landscape it produces is harder to descend.

### 3.2 `sensitivity_analysis.py` — the CNN-independent identifiability probe

Everything is computed in the **unit-parameter space the CNN predicts** (the `[-1,1]` log-scaled
space the data was generated in).

1. **Jacobian** `J = ∂(z-scored V)/∂(unit θ)` by **central finite differences** — deliberately
   *not* autodiff, because it only needs the already-proven forward simulation. Step
   `eps = 0.02` in unit space (= 0.01 decades in log-conductance). The K = 24 operating points
   are **real parameter vectors read out of the pack's test split**, not random draws, so the
   analysis describes the distribution the model actually sees. Traces are z-scored per row
   before differencing, to match the training loss:

   ```
   J[k,:,p] = (Vz[k, +eps on p, :] − Vz[k, −eps on p, :]) / (2·eps)
   ```

2. **Fisher information** `F_s = (1/KT)·Σ_k J_kᵀJ_k` per stimulus. Information from independent
   stimuli **adds**: `F_multi = Σ_s F_s`. This is the formal justification for multi-stim.
3. **Marginal sensitivity** `sqrt(diag(F))` — how much each channel moves the trace.
4. **Cramér-Rao bound** `CRB_p = σ·sqrt((F⁻¹)_pp)` with a ridge `λ = 1e-9·tr(F)/P` because `F` is
   deliberately near-singular. `σ = 0.7`, chosen to match the models' own voltage RMSE_z.
5. **Identifiability index** `ident_idx = CRB_multi / prior_std`. **`ident_idx ≥ 1` ⟹
   unidentifiable** — the voltage cannot beat simply guessing the prior. This is the actual
   pass/fail criterion.
6. **Collinearity**: correlation matrix of the estimator covariance `F⁻¹`. Pairs with
   `|corr| > 0.8` are flagged as degenerate — channels the trace cannot separate.
7. **Eigenspectrum of `F_multi`** — the smallest-eigenvalue eigenvector is printed as the
   "least-observable direction", i.e. the parameter *combination* the voltage does not see.

Rankings, the single-vs-multi CRB ratio, collinearity and eigenvectors are all **independent of
the assumed noise σ**; σ only sets the absolute CRB scale.

Outputs: `sensitivity_bars.png/.csv`, `crb_bars.png` + `crb_by_stim.csv`, `collinearity.png`,
`eigenspectrum.png`, `summary.yaml`.

**The headline result:** CA3 is **well-conditioned — the voltage inverse is NOT
information-degenerate.**

- Fisher condition number **3–34** across stimuli.
- **All six channels identifiable**: `ident_idx ≤ 0.15`, i.e. an order of magnitude below the
  `≥ 1` unidentifiable threshold.
- **No collinear pair** — no two conductances trade off in a way the trace cannot separate.
- The least-observable Fisher direction is consistently **na3 + kdr**, the two fast spike
  channels, which trade off mildly.
- **Conditioning depends strongly on stimulus:** step/ramp give cond **2–5** (smooth, high and
  uniform sensitivity); chirp/chaotic give **10–34**. Notably, the stimulus that was in use for
  single-stim training at the time (`5k50kInterChaoticB`) came out **worst on every axis** — a
  concrete result that changed what we trained on.

So the comfortable explanation — "the inverse problem is just ill-posed" — is ruled out. The real
difficulty is that an ε-sweep (0.005 / 0.02 / 0.05) shows **marginal sensitivity scaling as
1/ε**: `V(θ)` is non-smooth because spike timing moves discontinuously. **The loss landscape is
cliffy, not flat.** That is why plain voltage-MSE fails, and it is what motivates every smoothing
choice in the objective — soft-DTW, soft-eFEL, randomized smoothing.

The decisive confirmation came from the supervised run: parameter-MSE with labels recovers all
six channels at R² 0.995 in about four minutes. **The information is in the data; the voltage
objective is what fails to extract it.**

> ⚠️ **Provenance flag before you put per-channel CRBs on a slide.** The raw output of this
> analysis no longer exists on disk — no surviving `crb_by_stim.csv`, `collinearity.png`,
> `eigenspectrum.png` or `sensitivity/summary.yaml` anywhere in the repo or on scratch, and no
> log containing the printed table. The numbers above are reconstructed from contemporaneous
> notes taken when the analysis was run (2026-07-04), not from a file you can point at. They are
> defensible as stated, but if the talk needs a **figure** (the collinearity heatmap or the
> eigenspectrum) or exact per-channel CRB values, **re-run the script** — it is roughly 70 s on
> one GPU for 4 stimuli × 24 operating points, so this is a coffee-break fix, not a project.

### 3.3 One-at-a-time stimulus ranking (`sensitivity_variation.py`)

For each channel, 500 cells with **only that conductance** varied over its full range (others
pinned at unit 0) are simulated per stimulus — the same draws reused across every stimulus — and
the **variation across those 500 traces** is measured. Six independent metric families are
available: raw voltage variation (mV), eFEL-feature variation, blurred van-Rossum distance,
soft-DTW distance, and two chaos-robust ones (`sysvar`, a degree-5 polynomial regression of the
trace on the swept parameter; `smoothvar`, a 20 ms low-passed ensemble std) added specifically
because the distance metrics saturate on chaotic stimuli.

Tooling around it: `sensitivity_variation_merge.py` stitches the per-GPU chunks and re-derives
the normalization over the full stimulus set; `sensitivity_variation_compare.py` cross-checks two
runs (e.g. native vs interpolated stimuli); `run_sensitivity_salloc.sh` (1 node / 4 GPUs) and
`run_sensitivity_metrics_salloc.sh` (4 nodes / 16 GPUs, full metric set, with a hard preflight
check) are the launchers; `batchSensitivityVariation.slr` is the 30-minute debug-queue version.
Note these all drive `sensitivity_variation.py` — **`sensitivity_analysis.py` (§3.2) has no shell
wrapper** and is run directly.

Selected rows from `variation_matrix.csv` (units mV; higher = channel more visible):

| stimulus | leak | na3 | **kdr** | kap | km | kd |
|---|---:|---:|---:|---:|---:|---:|
| `BBP_Exp_Step1000` | 16.30 | **10.86** | **11.89** | **11.30** | **11.49** | **17.29** |
| `BBP_Exp_Step600` | 15.39 | 5.04 | 3.71 | 6.40 | 6.96 | 16.26 |
| `5kChaoticRamp` | 11.52 | 4.90 | 5.10 | 4.72 | 6.19 | 10.76 |
| `5k0chaotic4` | 15.53 | 1.26 | 0.39 | 1.42 | 0.91 | 11.86 |
| `5k50kInterChaoticB` | 15.03 | 0.56 | 0.23 | 0.74 | 0.74 | 11.38 |
| `chirp23a` | **17.24** | 0.33 | 0.07 | 1.41 | 0.88 | 11.61 |
| ~~`ramp_500`~~ | 3.46 | 0.10 | ~~146.16~~ | 0.80 | 7.89 | 2.15 |

Two things to draw out:

- **`BBP_Exp_Step1000` dominates `5kChaoticRamp` on all six channels**, and by a factor of ~2.3
  on kdr (11.89 vs 5.10 mV). Yet chaoticRamp trains far better (§4). That is the
  observability ≠ trainability result in one table.
- **Subthreshold-heavy stimuli see leak and kd and nothing else.** `chirp23a` has the best leak
  sensitivity of any stimulus (17.24 mV) and essentially zero kdr (0.07 mV). This is the same
  split that shows up in the trained models (§4.4).

**Overall coverage** (mean over channels of the column-max-normalized variation), recomputed on
the **64 clean stimuli only** after excluding the pA-contaminated ones of §3.7:

| rank | by voltage variation | | by eFEL variation | |
|---:|---|---:|---|---:|
| 1 | `BBP_Exp_Step1000` | **0.991** | `BBP_Exp_Step600` | **0.892** |
| 2 | `BBP_Exp_Step800` | 0.896 | `chaotic4` | 0.786 |
| 3 | `BBP_Exp_Step600` | 0.630 | `5k0chirp` | 0.751 |
| 4 | `chirp_damp` | 0.611 | `5k0chaotic4` | 0.718 |
| 5 | `chirp_damp_8k` | 0.565 | `chirp16a` / `chirp_damp_16k_v1` | 0.659 |
| 7 | `5kChaoticRamp` | 0.521 | | |

Cleaning **strengthens** the headline rather than weakening it: `BBP_Exp_Step1000` goes from
0.779 (as originally published) to **0.991** — near best-in-set on every channel once the
artifact is removed.

Two methodological points worth a sentence each:

- **The voltage and eFEL metrics genuinely disagree.** Voltage variation favours long steps and
  ramps; eFEL variation favours chaotic and step stimuli that scatter spike timing (via
  `inv_first_ISI`, `AP_amplitude`). Neither is wrong — they answer different questions, and the
  disagreement is itself information.
- **The measurement is robust to stimulus resampling.** Comparing native-length against
  interpolated-to-4000 stimuli over the 65 common stimuli: Spearman ρ of the per-stimulus ranking
  is **0.917–0.957** per channel, and **the #1 stimulus changed for none of the six**. So the
  ranking is a property of the cell, not of the sampling grid.

#### Measuring sensitivity on a *chaotic* stimulus needs a different statistic

Worth one slide if the audience is technical, because it is a genuine methodological result.
Distance metrics (MSE, blur, DTW) **saturate at the attractor diameter** on a chaotic stimulus —
once two traces have decorrelated, making the parameters more different does not make the
distance larger, so the metric cannot grade sensitivity at all. `chaoticramp_variance_probe.py`
compared nine candidate statistics on `5kChaoticRamp` (N = 64 per channel) and scored each by its
**discrimination** — the max/min spread across the six channels:

| statistic | discrimination (max/min across channels) |
|---|---:|
| smoothed ensemble std @ 40 ms | **6.61** |
| smoothed ensemble std @ 20 ms | 6.33 |
| parameter-explained std, smoothed | 6.41 |
| blurred distance to reference | 4.33 |
| raw ensemble std | 2.31 |
| parameter-explained std, raw | 2.30 |
| spike-rate std | 1.26 |
| raw peak | 1.22 |

**Smoothing the ensemble std separates channels ~3× better than the raw std** (6.6× vs 2.3×
spread), and the obvious candidates — spike-rate std, raw peak — are useless discriminators at
≈1.2×. Also reassuring: a degree-5 polynomial regression of the trace on the swept parameter
shows **74–86 % of the ensemble spread is genuinely parameter-driven** on chaoticRamp (0.80 for
kdr specifically), so the chaos is not drowning the signal — it is only defeating the *naïve*
way of measuring it.

### 3.4 Feature-space sensitivity (`feature_channel_sensitivity.py`)

The same idea in *feature* space rather than trace space:

```
S[f, p] = RMS | ∂ soft_feature_f / ∂ θ_p |
```

normalized by the same `FEATURE_SCALES` the loss uses, plus a **specificity** measure
`S[f,p] / Σ_p' S[f,p']` that finds each channel's most *diagnostic* feature rather than merely
its most sensitive one. It emits a ready-to-paste `grad_precond.sensitivity` vector.

Measured on `5kChaoticRamp`, 32 operating points, eps 0.03
(`ca3_ablation/sens_chaoticramp/feature_channel_sensitivity.csv`):

| feature | leak | na3 | **kdr** | kap | km | kd |
|---|---:|---:|---:|---:|---:|---:|
| `time_to_first_spike` | 5.416 | 3.710 | **0.002** | 4.578 | 3.725 | 5.069 |
| `mean_frequency` | 2.211 | 2.434 | **2.211** | 2.056 | 2.472 | 2.163 |
| `inv_first_ISI` | 10.472 | 9.089 | **3.008** | 9.782 | 7.819 | 16.788 |
| `AHP_depth_abs_slow` | 0.739 | 0.558 | 0.587 | 0.399 | 0.524 | 0.703 |
| `ISI_values` | 0.946 | 0.596 | 0.348 | 0.648 | 0.609 | 0.967 |
| `AP_amplitude` | 0.246 | 0.658 | **0.051** | 0.384 | 0.125 | 0.367 |

And the rolled-up sensitivity vectors it emits:

| aggregation | leak | na3 | **kdr** | kap | km | kd |
|---|---:|---:|---:|---:|---:|---:|
| `feature_l2` | 12.06 | 10.17 | **3.80** | 11.03 | 9.04 | 17.71 |
| `feature_max` | 10.47 | 9.09 | **3.01** | 9.78 | 7.82 | 16.79 |
| `voltage_mse` | 6.45 | 3.81 | **4.54** | 4.84 | 6.35 | 5.87 |

**This is the cleanest diagnostic finding of the campaign, and it deserves its own slide.**

- kdr is **blind to spike-shape features**: `AP_amplitude` 0.051 and `time_to_first_spike` 0.002
  are effectively zero, versus 0.25–0.66 and 3.7–5.4 for every other channel.
- In *feature* space kdr is ~3× less visible than any other channel (`feature_l2` 3.80 vs
  9.0–17.7).
- But in **raw voltage-MSE space kdr is perfectly ordinary** (4.54, right in the middle of the
  3.81–6.45 band).
- `best_feature_per_channel` picks `inv_first_ISI` for all five other channels and
  **`mean_frequency` for kdr alone** — kdr's only handle is *rate/ISI*, which it **shares with
  km and kd**. Three K currents, one handle ⟹ mutual confound.

#### The sharper version: purpose-built Kdr handles are blind to Kdr

A second, larger run on a clean 3-stimulus step battery
(`passiveReverse5k-50pA`, `5k0step_200`, `5k0step_500`; max |I| = 0.05 / 0.2 / 0.5 nA, so
**entirely uncontaminated**) extended the feature set to 12, including two surrogates that were
**designed specifically as Kdr handles**. This is the strongest slide in the sensitivity section.

| feature | leak | na3 | **kdr** | kap | km | kd |
|---|---:|---:|---:|---:|---:|---:|
| `ap_upstroke_dvdt` | 14.270 | 8.427 | 1.567 | 10.552 | 10.376 | 14.839 |
| **`ap_downstroke_dvdt`** *(built for Kdr)* | 6.190 | 5.008 | **0.054** | 5.515 | 5.620 | 6.204 |
| **`kdr_repol_slope`** *(built for Kdr)* | 6.769 | 5.470 | **0.227** | 6.034 | 6.151 | 6.787 |
| `AP_amplitude` | 14.686 | 7.337 | **0.295** | 11.082 | 11.043 | 14.731 |
| `time_to_first_spike` | 17.231 | 5.305 | **0.121** | 10.291 | 10.508 | 15.780 |
| `AHP_depth_abs_slow` | 8.823 | 3.893 | **0.524** | 5.223 | 5.290 | 7.638 |
| `inv_first_ISI` | 36.094 | 5.190 | 1.114 | 4.994 | 4.631 | 25.397 |
| `slow_ahp_deepening` | 19.202 | 10.601 | 11.239 | 12.253 | 10.928 | 20.826 |
| `mean_frequency` | 7.145 | 4.842 | 4.786 | 4.502 | 5.006 | 5.525 |
| `depol_fraction` | 22.794 | 21.226 | 22.345 | 20.165 | 20.567 | 20.929 |
| `isi_adaptation_slope` | 87.181 | 49.287 | 46.768 | 37.760 | 60.037 | 38.153 |
| `adaptation_index` | 106.398 | **706.751** | 37.866 | 175.134 | 520.163 | 189.496 |

- **`ap_downstroke_dvdt` was written to be the Kdr repolarization handle. It gives Kdr 0.054 —
  114× smaller than leak (6.190) and 92× smaller than na3 (5.008), and Kdr is the *minimum* of
  the row.** Kdr (`kdrca1`, "delayed", τ floored at 2 ms) is simply too slow to shape the fast
  downstroke; Na inactivation sets that, so the feature just re-reads Na.
- **`kdr_repol_slope`**, the fixed-voltage-band trick designed to decouple from Na peak height,
  gives Kdr **0.227** versus 5.47–6.79 for everything else — again the row minimum, ~30× below
  leak. **You cannot manufacture a Kdr signal that the trace shape does not contain.**
- The only features where Kdr is *not* crushed are the non-shape ones — `depol_fraction`
  (22.345, but essentially uniform across all six, so responsive with **zero specificity**),
  `isi_adaptation_slope` (46.768, but dominated by leak at 87.181), `mean_frequency` (4.786,
  near-uniform).
- The specificity criterion `handle_score = S · (S / Σ_p S)` finds **no channel-specific handle
  for five of six channels** — they all pick `adaptation_index`, and Kdr picks
  `isi_adaptation_slope`, a row leak wins outright (87.2 vs 46.8).

And the contrast that makes the point cleanly: in **voltage** space all six channels are within
1.2× of each other (13.6–16.3), but in **feature** space they span 11× (65 to 709). **The feature
summary throws away most of what the raw trace already sees for Kdr.**

This measurement is codified as a findings block inside `toolbox/soft_efel.py:140-160`, so nobody
re-derives it: *"on this CA3 model, spike-SHAPE features cannot see Kdr, and the Km handles are
not Km-specific."* The one clean per-channel win is `ap_upstroke_dvdt` → Na.

#### Diagnosis

**kdr is a timing/rate channel, not a shape channel.** Any loss built out of spike-shape
features cannot see it. This is why the eFEL auxiliary that helped na3 so much barely moved kdr
(§4.5) — and §4 confirms it experimentally.

Practical rule that came out of this: **validate feature handles with
`feature_channel_sensitivity.py` before training on them.** Two purpose-built Kdr surrogates were
written, shipped, and only *afterwards* measured to be blind. That check costs one GPU-hour.

### 3.5 Extrapolation — `ood_probe.py`

Sweeps each conductance out to unit ±1.5, well beyond the trained ±1 box (31 points, others held
at 0), and asks what the CNN does. `--mode joint` instead draws all six together from
`U(−1.5, 1.5)` — "realistic OOD". Crucially it normalizes with the **same fixed-scale constants
used in training**, not a per-sample z-score, so the comparison is honest. The error metric is
against the *best possible saturated answer*: `mean |pred − clip(true, −1, 1)|`.

Answer: predictions **saturate at the tanh boundary in the correct direction** for
well-identified channels — the model degrades gracefully rather than hallucinating. And the
obvious "fix" backfires: **widening the trained range makes things worse.** Raising
`LOG_HALFSPAN` from 0.5 to 1.0 dropped mean R² from 0.566 to 0.242, because spreading the same
sample budget over a 10× wider parameter box thins the density everywhere. (No saved plot from
this run survives; the finding is recorded in `JAXLEY_ARCHITECTURE.md` and reproduces in minutes
if you want the figure.)

### 3.6 The step-0 audit — `scripts/voltage_loss_bias_probe.py`

Evaluates the **exact training criterion at the true parameters**. If the loss is not ≈ 0 there,
no amount of optimization can succeed, and the residual measures the model-mismatch floor. The
question it answers is *"is the objective biased, or is the landscape just cliffy?"* — because
those have opposite fixes.

It makes four measurements: (1) the residual at true θ, both with and without the tanh clamp;
(2) a **per-parameter offset sweep** — hold every sample at true θ, add δ to one channel over a
21-point grid, and check whether the loss minimum actually sits at δ = 0; (3) whether the tanh
representation itself is biasing the answer; (4) the loss at the CNN's own prediction versus at
truth.

The one surviving run on CA3:

```
loss_true_physical:     0.3826      # no tanh
loss_true_config_tanh:  0.3630      # with the configured tanh clamp
argmin_delta_per_param: [0, 0, 0, 0, 0, 0]      # all six minima exactly at truth
min_off_truth_params:   []
```

Read that carefully, because the two halves point different ways. By the script's own rule the
verdict is **BIASED** (residual 0.383 ≫ 0.1 — the loss is *not* zero at the right answer). But
**the valley minimum sits exactly at truth for all six channels**, and the tanh clamp is not the
culprit (0.363 < 0.383). So the objective points the right way; it just sits on a floor.

That floor was traced to a **fp64-generation / fp32-loss solver mismatch** (correlation 0.82
between the two solvers' traces at identical parameters) — a whole plateau that looked like an
optimization failure was a numerics mismatch, and it is why `fp64: True` is now standard.

Strongly recommend keeping this in the talk as a methodological point: **audit your loss at the
answer before you tune anything.**

### 3.7 ⚠️ Stimulus hygiene — four waveforms are unusable, and it invalidated published results

`jaxley_utils.load_stim_csv` reads a stimulus CSV as **nA with no scaling**, but four waveforms in
`/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/` are written in **pA**. They inject **1000× too
much current** and drive the soma to **+1600 mV** — ohmic drive of a 50 × 50 µm cylinder, not a
neuron.

The four are `ramp_500` (max |I| = 500), `4k50kInterstep_500_50khz` (500),
`4k50kInterramp_50khz` (499.9) and `4k50kInterstep_200_50khz` (200). Counting their
interpolated `_i4k` twins as separate pack-level entries gives the "seven stimuli" figure used
elsewhere in the docs. **Every other stimulus is clean** — the next largest is
`BBP_Exp_Step1000_8.00x` at 8.0 nA, then `5kChaoticRamp` at 6.0 nA.

**What this invalidated:**

- `sensitivity_best_stims.md`'s headline claims **"Kdr's best stimulus is `ramp_500`"** (the
  146.2 mV "sensitivity" is the overdrive flipping the cell across a spiking bifurcation) and
  **"KA's best is `4k50kInterstep_500_50khz`"**, plus its TL;DR battery advice that a ramp leg is
  essential to see Kdr. **That file is still uncorrected on this branch** — do not present from
  it directly.
- kdr's **top four stimuli by voltage variation are all contaminated** (146.2 / 146.1 / 106.3 /
  74.1 mV) versus **11.9 mV** for the best clean one. A **12.3× gap** — kdr's apparent
  observability lived almost entirely in the artifact. The clean kdr ranking is
  `BBP_Exp_Step1000` 11.89 → `BBP_Exp_Step800` 10.12 → `BBP_Exp_Step1000_2.00x` 5.35.
- The blur / DTW / sysvar / smoothvar metrics all pick a contaminated stimulus as "best for kdr"
  too — `ramp_500` scores 27.25 DTW and 260 792 blurred-MSE against ~0.06 and ~1000 for the next
  clean stimulus, a 250–400× gap. **That is the artifact screaming**, and it is a good
  illustration that every metric agreed on a wrong answer because they shared a bad input.
- Because coverage is column-max-normalized, the contamination shifted **every** normalized value
  in the kdr and kap columns — so the whole published coverage table had to be recomputed
  (done, in §3.3).
- Two earlier multi-stim runs that recovered kdr at **R² 0.985 / 0.991** each contain a
  contaminated probe (verified in the packs: `ca3_best_efel_multi` probe 1 peaks at +1604 mV;
  `ca3_best_mse_multi` probes 2 and 3 at +1604 and +998 mV). "The long step recovers kdr" and
  "the overdriven probe recovers kdr" are **perfectly confounded** in both. Those two numbers
  must not appear on a slide as results.

**What survives** (most of it, and the headline gets *stronger*):

- Everything about leak, na3, kd and km — none of their top-3 stimuli is contaminated.
- The `BBP_Exp_Step1000` / `BBP_Exp_Step800` headline, which improves from 0.779 → **0.991** and
  0.719 → 0.896 coverage after cleaning.
- The eFEL headline `BBP_Exp_Step600`, 0.786 → **0.892**.
- **All of §3.4** — both feature-sensitivity runs used clean batteries (`5kChaoticRamp` at 6 nA;
  the step battery at 0.05/0.2/0.5 nA). The Kdr-blindness result stands untouched.
- The chaoticRamp variance probe and the step-0 bias audit — both clean.
- The native-vs-interpolated robustness check — a property of the method.

**Standing rule:** before using any stimulus, check `max|I|` — anything > 20 nA is pA-scaled.
Open remediation items: fix `load_stim_csv` at source or quarantine the bad CSVs, correct
`sensitivity_best_stims.md`, and re-run the OAT sweep restricted to clean stimuli. The corrective
action already taken is the replacement battery `ca3_joint4_v1` (§4.7), verified with no probe
over +53 mV.

---

## 4. Best result — CA3 chaoticRamp

### 4.1 Setup

| | |
|---|---|
| Cell | `ca3_pyramidal` — single compartment, 6 conductances, NEURON-validated (§2.2) |
| Stimulus | `5kChaoticRamp`, 5001 points, dt 0.1 ms, 500 ms |
| Data pack | `ca3_chaoticramp_v3` — **200 000 train / 25 000 valid / 25 000 test**, T = 5001, 1 soma probe |
| Generation | `scripts/gen_ca3_sharded.py`, 16 shards, fp64, seed 0, `log_halfspan = 0.5` for all 6 params |
| Supervision | **strictly voltage-only** — `channel_weight: 0`, `mask_channels: True`. Ion channels never enter the loss. |
| Metric | `channel_r2_overall` from `evaluate_voltage.py` = mean over params of `1 − MSE/Var` in unit space, on 200 held-out test samples |

One fact to keep straight when quoting the two numbers together: the **supervised ceiling was
reached on only 40k samples in 3.7 minutes on a single node**, while the voltage-only champion
needed **200k samples and 72 node-hours** to reach 0.763. The cost asymmetry between the two
supervision modes is roughly **1200×**, and it is entirely the in-loop Jaxley solve.

### 4.2 The champion configuration

`ca3_vo_chaoticramp_dtw_amp_8n.hpar.yaml` — **soft-DTW primary + a small ramped soft-eFEL
auxiliary**:

```yaml
use_voltage_loss: True
voltage_loss:
    cell_name_for_sim: ca3_pyramidal
    channel_weight:    0.0        # strictly voltage-only
    voltage_weight:    1.0
    mask_channels:     True
    stim_name:         5kChaoticRamp
    soma_probe_index:  0
    clamp_unit_tanh:   True
    t_max_override:    auto
    solver:            bwd_euler
    fp64:              True
    # PRIMARY = soft-DTW (timing)
    mse_weight:        0.0
    dtw_weight:        1.0
    dtw_gamma:         0.1
    dtw_n_points:      256
    dtw_band_ms:       8.0
    blur_weight:       0.0
    # AUX = 2 soft-eFEL features (level / amplitude)
    efel_weight:       0.2
    efel_features:     [voltage_base, AP_amplitude]
    schedule:
        efel_weight:  { start: 0.05, end: 0.2, epochs: 20 }

model:
    outputSize_override: 6
    num_cnn_blocks: 2
    conv_block: { filter: [30,90,180], kernel: [4,4,4], pool: [4,4,4] }
    batch_norm_cnn_slot: 3
    batch_norm_flat: True
    fc_block: { dims: [512,512,512,256,128], dropFrac: 0.04 }

train_conf:
    optimizer: [adam, 0.0001]
    LRsched:   { plateau_patience: 10, reduceFactor: 0.3 }
    clip_grad_norm: 1.0
batch_size: 64          # per GPU; const_local_batch: True → global 2048 on 32 GPUs
```

The design rationale, which is the story to tell: **DTW gets spike times right but tolerates
absolute level**, so the resting baseline and AP amplitude of the reconstruction drift. The fix
is to add the two differentiable surrogates that *are* those quantities — `voltage_base`
(resting potential, scale 6 mV) and `AP_amplitude` (spike height, scale 26 mV) — computed from
the voltage traces only, so the run stays strictly voltage-only. Kept **small** so DTW still
drives timing: the soft-DTW loss is length-normalized (~0.05) while the eFEL Huber is ~10×
larger, and `efel_weight: 1.0` at LR 1e-4 has diverged before.

**Measured cost** (job 55951029, `-N8 --time=12:00:00 -q regular`): 150 epochs,
**217.2 ± 0.02 s/epoch**, 32 GPUs / 8 nodes, 97 train + 12 valid steps per epoch, elapsed
**32 629 s ≈ 9.06 h ≈ 72.7 node-hours**, final val loss −0.0607 at LR 9e-6.

⚠️ The YAML says `max_epochs: 100`; **the run as executed used 150** — the launcher overrode it.
Quote 150.

Cost of the surrounding campaign, for context:

| run | nodes / GPUs | epochs | s/epoch | node-hours |
|---|---|---:|---:|---:|
| **dtw_amp_200k (champion)** | 8 / 32 | 150 | 217.2 | **72.7** |
| dtw_400k | 8 / 32 | 150 | 441.0 | 147.3 |
| dtw_amp_400k | 4 / 16 | 150 | 494.5 | 82.5 |
| dtw_ica_200k (InterChaoticB) | 8 / 32 | 150 | 222.3 | 74.5 |
| dtw_200k | 8 / 32 | 94/100 — **TIMEOUT at the 6 h wall** | ~220.8 | 48.0 |
| baseline 40k | 4 / 16 | 100 | 48.7 | 5.4 |
| **supervised chaoticRamp** | **1 / 4** | 200 | **0.91** | **0.06** |
| data generation, 250k pack | 4 / 16 | — | — | ~0.2 (3.4 min) |

Two operational lessons in that table. **Data generation is nearly free** (~3.4 min for a 250k
pack) — the cost is entirely in training. And **8 nodes bought nothing**: at a pinned global
batch of 2048, 32 GPUs ran 220.8 s/epoch versus ~247 s/epoch on 16, because the fp64 solve is
dominated by the sequential 5001-step time integration. Future 200k+ runs should use 4 nodes and
a longer wall-clock.

### 4.3 The ledger — every voltage-only chaoticRamp arm

Sorted by mean R². All strictly voltage-only, all evaluated on 200 test samples.
Source: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.

| arm | recipe | data | **mean R²** | leak | na3 | **kdr** | kap | km | kd | `mse_z` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dtw_amp_200k** | **DTW + ramped soft-eFEL aux — BEST** | 200k | **0.763** | 0.898 | **0.870** | **0.366** | 0.866 | 0.754 | 0.825 | 2.030 |
| dtw_amp_400k | same recipe at 400k | 400k | 0.750 | 0.883 | 0.859 | 0.251 ⬇ | **0.904** | 0.771 | 0.830 | **1.897** |
| precondkdr5_ft_200k | DTW + grad-precond kdr×5, fine-tuned | 200k | 0.742 | 0.868 | 0.765 | 0.342 | 0.851 | 0.811 | 0.817 | 2.010 |
| dtw_400k | pure soft-DTW, 10× data | 400k | 0.736 | 0.885 | 0.800 | 0.219 ⬇ | 0.905 | 0.764 | 0.841 | 1.903 |
| precondkdr_200k | DTW + grad-precond kdr×2.5 | 200k | 0.731 | 0.869 | 0.768 | 0.266 ⬇ | 0.854 | 0.816 | 0.811 | 2.009 |
| dtw_200k | pure soft-DTW | 200k | 0.725 | 0.860 | 0.794 | 0.338 | 0.807 | 0.753 | 0.798 | 2.033 |
| dtw_80k | pure soft-DTW | 80k | 0.637 | 0.864 | 0.731 | 0.092 | 0.724 | 0.650 | 0.763 | 2.046 |
| precond2_80k | refined precond | 80k | 0.611 | 0.822 | 0.683 | 0.179 | 0.626 | 0.629 | 0.729 | 2.055 |
| precond | DTW + feature-sens precond (kdr×1.68) | 40k | 0.610 | 0.804 | 0.729 | 0.044 | 0.692 | 0.618 | 0.773 | 1.988 |
| precond2 | DTW + refined precond | 40k | 0.600 | 0.785 | 0.694 | 0.129 | 0.751 | 0.503 | 0.736 | 1.992 |
| band4 | DTW warp band 8 → 4 ms | 40k | 0.595 | 0.809 | 0.619 | 0.156 | 0.696 | 0.521 | 0.770 | 1.973 |
| **baseline** | pure soft-DTW | 40k | 0.593 | 0.788 | 0.642 | 0.155 | 0.753 | 0.483 | 0.738 | 1.984 |
| dtw_20k | pure soft-DTW | 20k | 0.489 | 0.719 | 0.493 | 0.167 | 0.478 | 0.381 | 0.697 | 1.995 |
| r1_dtwblur | DTW + van-Rossum blur | 40k | 0.479 | 0.763 | 0.537 | 0.115 | 0.286 | 0.419 | 0.751 | 2.015 |
| dtw_10k | pure soft-DTW | 10k | 0.440 | 0.592 | 0.500 | 0.165 | 0.417 | 0.341 | 0.628 | 2.012 |
| *dtw_ica_200k* | *pure DTW on `5k50kInterChaoticB` — different stimulus* | 200k | *0.219* | ***0.974*** | *−0.269* | *−0.120* | *0.158* | *−0.369* | ***0.937*** | ***0.970*** |
| *chaoramp_step* | *DTW, chaoticRamp + a step probe* | — | *0.126* | *0.437* | *0.167* | *−0.186* | *−0.315* | *−0.002* | *0.652* | *2.120* |

**Campaign arc: 0.593 → 0.763, +0.170 (≈ 29 % relative).**

#### Prehistory — the pre-DTW era, worth one backup slide

Before soft-DTW became the primary loss, chaoticRamp was run with plain voltage MSE and with
soft-eFEL. These are the runs that motivated the switch:

| run | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ca3_vo_blur_chaoticramp` | blur-MSE only (`blur_weight 1.0`) | 40k | 0.630 | **0.909** | 0.118 | 0.201 | 0.904 | 0.759 | **0.891** |
| `ca3_voltonly_mse_chaoticramp` | plain voltage MSE | 40k | 0.576 | 0.872 | **−0.148** | 0.206 | 0.882 | 0.783 | 0.859 |
| `ca3_voltonly_efel_chaoticramp` | MSE 1.0 + **soft-eFEL 1.0** | 40k | 0.566 | 0.767 | **0.263** | 0.094 | 0.825 | 0.645 | 0.802 |
| `ca3_voltonly_efel_chaoticramp_100k` | same, more data | 100k | 0.542 | 0.828 | 0.229 | −0.081 | 0.805 | 0.690 | 0.780 |
| `ca3_voltonly_efel_chaoticramp_wide` | same, `LOG_HALFSPAN` 0.5 → 1.0 | — | **0.242** | 0.761 | −0.049 | 0.103 | 0.061 | −0.173 | 0.748 |
| `ca3_vo_allfeat_chaoticramp` | **soft-eFEL only**, weight 0.25 | 40k | **−0.011** | −0.025 | −0.016 | −0.005 | −0.004 | −0.012 | −0.004 |

Three things this table establishes, all of which shaped the final recipe:

- **Plain MSE and soft-eFEL split on *which* channels they recover.** MSE wins overall (0.576 vs
  0.566) on the K currents and leak, but it is the **only** configuration where na3 goes
  *negative* (−0.148) — while soft-eFEL is the only one that recovers Na at all (+0.263). This
  is why the winning recipe is a **combination**, and why the eFEL contribution is specifically
  `AP_amplitude`.
- **soft-eFEL alone collapses the network to a constant predictor** (mean R² −0.011, every
  channel ≈ 0). The features cannot carry training on their own.
- **Widening the parameter range hurts badly** (0.242) — the same finding `ood_probe.py`
  reported independently (§3.5).

### 4.4 Champion vs supervised ceiling — the money slide

Same cell, same stimulus, same architecture, same evaluation protocol (200 test samples). The
only difference that matters is whether the loss is allowed to see the ground-truth
conductances — and note the supervised run got there on **less** data (40k, 200 epochs, one node,
3.7 minutes) than the voltage-only champion (200k, 150 epochs, 8 nodes, 9 h).

| | leak | na3 | **kdr** | kap | km | kd | **mean R²** | voltage `mse_z` | spike-count \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Supervised (labels — the ceiling) | 0.998 | 0.997 | **0.985** | 0.998 | 0.994 | 0.996 | **0.995** | 1.860 | 5.61 |
| **Voltage-only (best)** | 0.898 | 0.870 | **0.366** | 0.866 | 0.754 | 0.825 | **0.763** | 2.030 | 5.53 |
| gap | 0.100 | 0.127 | **0.619** | 0.132 | 0.240 | 0.171 | 0.232 | −0.170 | +0.08 |

Two things to say about this table:

1. **Five of six channels are within ~0.10–0.24 of the supervised ceiling. kdr alone accounts
   for most of the remaining gap** (0.619 of a total mean gap of 0.232 × 6 = 1.39, i.e. **45 %
   of all remaining error is one channel**).
2. **The supervised model does not reproduce the trace any better than the voltage-only model**
   — `mse_z` 1.86 vs 2.03, and spike-count error is comparable (5.61 vs 5.53) despite
   near-perfect parameter recovery. That is a strong, slightly counter-intuitive point: on a
   chaotic stimulus there is a **trace-reproduction floor around 1.9 that has nothing to do with
   parameter accuracy** — chaos amplifies infinitesimal differences, so even the right
   conductances do not give you the right trace point-by-point. Excellent trace overlap and good
   parameter recovery are *not* the same objective, and this is the cleanest evidence for it.

   *Fine print if challenged:* the supervised run was trained and scored on `ca3_chaoticramp_v1`
   (40k) and the champion on `ca3_chaoticramp_v3` (200k) — same cell, stimulus and generator
   settings, but different draws, so treat the voltage metrics as a like-for-like comparison of
   *magnitude*, not a paired one. The parameter-R² comparison is unaffected: both are `1−MSE/Var`
   within their own test split.

### 4.5 What moved the needle, and what did not

**Works:**

- **Soft-DTW as the primary loss.** Spike-timing tolerance matters more than pointwise accuracy.
- **A small, ramped soft-eFEL auxiliary — the single biggest lever of the campaign.**
  `voltage_base` + `AP_amplitude`, weight 0.05 → 0.2 over 20 epochs, on top of DTW:
  **0.725 → 0.763** at 200k, which **beats 400k of data (0.736) with half the samples.** An
  objective change outperforming a data change is the campaign's central result.
- **More data, up to a point.** 10k → 400k took the mean from 0.440 to 0.736. But the
  200k → 400k doubling bought only **+0.011** versus +0.088 for 80k → 200k — the curve has
  clearly bent over.

**Does not work:**

- **Gradient preconditioning.** Boosting a poorly-observed channel's own gradient makes it
  *worse*: kdr fell 0.338 → 0.266 under a ×2.5 boost. It helped only in the data-starved regime
  (40k) where it was compensating for scarcity. Closed line.
- **Van-Rossum blur.** A coarse blur is toxic to fast channels — kap 0.753 → 0.286. Smoothing
  the loss makes it *less discriminative*.
- **Narrower DTW band.** 8 ms → 4 ms: 0.593 → 0.595, i.e. nothing.
- **soft-eFEL as a *primary* loss** at weight 1.0 — diverges outright. The same features as a
  small ramped auxiliary win. Configs that lean on eFEL run at LR 5e-5 with
  `clip_grad_norm: 1.0`; at 3e-4 the loss peaks around epoch 5–8 and then blows up.
- **Architecture search.** RayTune, **130 trials on 8 nodes**, found nothing better than the
  baseline skeleton; larger and deeper networks diverge (loss → 2.0 = NaN). Architecture is not
  the lever.
- **Stacking the two best levers.** Taking the champion recipe to 400k gave **0.750, *below* the
  0.763 it was built from.** The entire deficit is kdr (0.366 → 0.251), while trace fit genuinely
  improved (`mse_z` 1.897 vs 2.030). Honest negative result — worth showing.

**The kdr story — the cleanest experiment in the ledger, and a good closing narrative:**

- kdr **does not track data**: 10k 0.165 → 20k 0.167 → 40k 0.155 → 80k 0.092 → 200k 0.338 →
  400k 0.219. That is noise around ~0.15–0.34, not a climb, and doubling to 400k *regressed* it.
- kdr **does not respond to gradient reweighting** (×2.5 → 0.266).
- The sensitivity analysis says why (§3.4): kdr is blind to every spike-shape feature and shares
  its only handle — firing rate / ISI / slow AHP — with km and kd. Three K currents, one handle.
- **Diagnosis: kdr is objective- and observability-limited, not data-limited.**
- Honest correction worth including: the eFEL auxiliary was a **na3 win, not a kdr win.** In
  error terms, na3 `(1−R²)` went 0.206 → 0.131 (**−37 %**) while kdr went 0.662 → 0.634
  (**−4 %**). `AP_amplitude` constrains Na-driven spike height, which is exactly what it should
  do. **kdr's actual handle — rate/ISI — is still not in the loss.**

### 4.6 Stimuli are complementary — the case for multi-stim

The sharpest observability result. Same recipe, same 200k samples, two different stimuli:

| | mean R² | leak | na3 | kdr | kap | km | kd | `mse_z` | spike \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `5kChaoticRamp` (spike-rich) | **0.725** | 0.860 | **0.794** | 0.338 | **0.807** | 0.753 | 0.798 | 2.033 | 6.13 |
| `5k50kInterChaoticB` (76 % near rest) | 0.219 | **0.974** | −0.269 | −0.120 | 0.158 | −0.369 | **0.937** | **0.970** | **0.99** |

These are **exact mirror images**:

- A mostly-subthreshold trace is easy to fit (`mse_z` 0.97, spike error 0.99 — both far better
  than chaoticRamp) and **perfectly constrains the passive and slow channels** — leak 0.974 and
  kd 0.937 are essentially at the supervised ceiling.
- But it contains almost no spikes, so it carries **no information about the spike-shaping (na3,
  kap) or rate (kdr, km) channels** — all four fail outright, with negative R².
- chaoticRamp is the reverse: lots of spikes ⟹ na3/kap recover, and chaos ruins the trace
  overlap.

This also **resolves the apparent paradox** that excellent trace overlap coexists with poor
parameter recovery: they are not in tension, they are the *same* fact seen twice. The chaos that
ruins chaoticRamp's overlap is precisely the spiking that makes na3/kap observable there.

Since Fisher information from independent stimuli **adds** (§3.2), observing both jointly should
recover all six far better than either alone — and it directly attacks the kdr confound by giving
the three K currents independent views. **This is the current front line.**

Worth stating explicitly, because it is the reason multi-stim rather than multi-probe:
**`ca3_pyramidal` is a single compartment**, so multi-*location* recording is impossible without
changing the morphology — which would break the NEURON ground-truth match that makes CA3 worth
using (§2.2). Multi-probe is available only on genuinely multi-compartment cells
(`l5ttpc_multiprobe`). For CA3 the only observability levers are **stimulus choice and loss
shaping**.

#### chaoticRamp is also the best stimulus for the *supervised* model

Useful cross-check, because it separates "this stimulus carries information" from "this
stimulus is easy to train on". Supervised parameter-MSE, same cell, same architecture:

| stimulus | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`5kChaoticRamp`** | **0.995** | 0.998 | **0.997** | **0.985** | 0.998 | 0.994 | 0.996 |
| `4k50kInterChaoticB` | 0.918 | 0.998 | 0.868 | 0.701 | 0.993 | 0.952 | 0.997 |
| `BBP_Exp_Step1000_i4k` | 0.829 | 0.996 | **0.566** | 0.609 | 0.828 | **0.981** | 0.995 |

So chaoticRamp is not merely the stimulus our voltage-only recipe happens to like — it is the
**most informative single stimulus of the three**, and the only one under which kdr is close to
fully recoverable even with labels (0.985 vs 0.70/0.61). The complementarity argument of the
previous table therefore stands *on top of* a genuinely good primary stimulus, not instead of
one.

### 4.7 Where it stands right now

Two jobs are queued on Perlmutter testing exactly this, both on a **clean 4-stimulus battery**
(`ca3_joint4_v1`: `5kChaoticRamp`, `5k0chaotic4`, `BBP_Exp_Step1000_i4k`, `chirp23a_i4k`; 80k
draws; no probe over +53 mV, so the pA/nA contamination of §3.7 is excluded by construction):

- **Variant A** (job 56379085) — the four stimuli as **CNN input channels**; the network sees all
  four traces of the same cell at once. Prediction on record: should approach the **union** of
  what each stimulus observes.
- **Variant B** (job 56379086) — **pooled**: each sample is still a single trace, tagged with its
  stimulus id, and `HybridLoss._pooled_voltage_loss` splits the batch by id and simulates each
  group under its own stimulus, weighted per-sample so a rare stimulus cannot outvote a common
  one. Prediction: should approach the **average** — but its real payoff would be
  **protocol-agnostic inference on experimental data.**

Both 4 nodes, 24 h wall, measured ~15 h (A) and ~18 h (B) for 100 epochs. Both were still
`PENDING` at the time of writing.

**The headline question they answer:** does kdr survive on a *clean* battery under the champion
objective? Given the OAT gap (11.9 mV clean vs 74–146 mV contaminated), a weak kdr here is real
evidence that the earlier 0.985/0.991 kdr recoveries were the overdrive artifact — **a publishable
negative, not a failure.**

**Still open and explicitly identified:** no rate/ISI term is in the loss, which is kdr's actual
handle. The named candidates are `mean_frequency`, `inv_first_ISI`, `AHP_depth_abs_slow`.

---

## 5. Appendix

### 5.1 Figures available for slides

| figure | what it shows |
|---|---|
| `docs/ca3/test1_rest_overlay.png` … `test6_apshape_overlay.png` | the six NEURON ↔ Jaxley validation overlays (§2.2) |
| `docs/ca3/summary.csv` | the validation numbers, full precision |
| `synth_plots/ca3_traces_overlay.png`, `ca3_traces_grid.png` | CA3 synthetic traces |
| `synth_plots/ca3_default_trace.png` | CA3 at default conductances |
| `synth_plots/ca3_multistim_perpage.pdf` | multi-stimulus trace book |
| `synth_plots/efel_vs_channel*.png/pdf` | eFEL feature response vs each channel |
| `docs/phase3/param_recovery_grid.png` | per-parameter true-vs-predicted scatter |
| `docs/phase3/voltage_rmse_cdf.png`, `voltage_loss_hist.png` | trace-error distributions |
| `<run>/out/eval/*.png` | per-run recovery grid + trace overlays from `evaluate_voltage.py` |
| `tmp_neuInv/.../sensitivity_variation/.../summary_heatmap.png`, `ranking_per_channel.png`, `efel_summary_heatmap.png` | OAT stimulus × channel heatmaps (§3.3) |
| `ca3_ablation/sens_chaoticramp/feature_channel_sensitivity.png` | the feature × channel heatmap on chaoticRamp (§3.4) |
| `.claude/worktrees/voltage-only-ca3/feature_sensitivity_stepbattery/feature_channel_sensitivity.png` | **the 12-feature step-battery heatmap — the Kdr-blindness slide** (§3.4) |
| `jaxley_ca3/smoke_1778181760/out/bias_probe/bias_sweep.png` | the step-0 loss-at-truth sweep (§3.6) |
| `tmp_neuInv/chaoticramp_probe/5kChaoticRamp_probe.pdf` | chaotic-stimulus variance-statistic comparison (§3.3) |

If you want one diagram built from scratch, build **§1.8's gradient path** — it is the single
picture that explains the whole project.

### 5.2 Suggested slide order

1. The inverse problem, and why labels run out on real data
2. Jaxley = differentiable simulator ⟹ the physics loss loop (§0)
3. The stack, five layers, one `if` in the trainer (§1.2, §1.12)
4. The gradient path, disk to weight update (§1.8)
5. The cell registry — add a cell, touch nothing else (§1.3)
6. The CNN (§2.1) — one backbone, unchanged across every experiment
7. The cell zoo + CA3 validated against NEURON on 7 tests (§2.2)
8. Sensitivity: Fisher says CA3 is well-conditioned — so it's the objective, not the problem (§3.2)
9. Sensitivity: **two surrogates purpose-built as Kdr handles are blind to Kdr** (§3.4) — the
   strongest single diagnostic slide
10. **Result: 0.593 → 0.763 voltage-only, vs 0.995 supervised** (§4.3, §4.4)
11. What worked / what didn't, incl. the honest negatives (§4.5)
12. Stimuli are complementary ⟹ multi-stim is the next lever (§4.6, §4.7)

Backup slides: fp64 NaN table (§1.9), DDP scaling and the 32-GPU wall (§1.10), the pA/nA
stimulus hygiene incident (§3.7), the step-0 loss-at-truth audit (§3.6), measuring variance on a
chaotic stimulus (§3.3), the pre-DTW MSE-vs-eFEL split (§4.3).

**If you only get three slides:** the loop diagram (§0), the champion-vs-supervised table (§4.4),
and the Kdr feature-blindness table (§3.4). Those three tell the whole story — the method works,
it recovers five of six channels, and we know exactly why the sixth fails.

### 5.3 Reproduction commands

```bash
# environment (always — conda, not shifter, for anything importing jaxley/jax)
module load conda
conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley

# generate a CA3 pack (16 ranks, ~5 min for 80k × 4 stims on 4 nodes)
sbatch scripts/run_ca3_gen.sh

# train the champion recipe
sbatch batchShifterJaxleyCA3_100ep.slr   # design: ca3_vo_chaoticramp_dtw_amp_8n

# score a run — writes out/eval/summary.yaml with channel_r2_overall
python evaluate_voltage.py --modelPath <run_dir>/out

# identifiability, CNN-independent (no shell wrapper — run directly, ~1 GPU)
python sensitivity_analysis.py --modelPath <run_dir>/out --numOp 24 --eps 0.02 --sigma 0.7
python ood_probe.py --mode sweep                       # extrapolation behaviour
python scripts/voltage_loss_bias_probe.py --modelPath <run_dir>/out   # step-0 audit

# feature-space sensitivity — emits a grad_precond.sensitivity vector
./run_feature_sensitivity_salloc.sh                    # 1 GPU salloc wrapper

# one-at-a-time stimulus ranking (these wrap sensitivity_variation.py, NOT sensitivity_analysis.py)
./run_sensitivity_salloc.sh                            # 1 node / 4 GPU, voltage + eFEL
./run_sensitivity_metrics_salloc.sh                    # 4 nodes / 16 GPU, all six metrics
sbatch batchSensitivityVariation.slr <cell> <stimDir> <nSamples> efel <tag>

# NEURON ↔ Jaxley validation
python toolbox/tests/test_ca3_neuron_vs_jaxley.py
```

### 5.4 Gotchas that have already cost time

1. **`.gitignore` swallows `*yaml`, `*h5`, `*csv`, `*txt`, `L*`.** New design YAMLs need
   `git add -f` or they silently do not exist for anyone else.
2. **SLURM scripts copy `$codeList` into a frozen `$wrkDir` snapshot.** A new Python file not in
   that list will not exist at runtime, and the failure looks like an import error rather than a
   packaging error.
3. **Resuming from a checkpoint resets the best-validation tracker**, so a resumed run can report
   a "new best" that is not one.
4. **Two independent soft-DTW implementations** exist — `toolbox/soft_dtw.py` (used by
   `HybridLoss`) and a second inside `toolbox/trace_metrics.py` (used by the offline variance
   probes). They are not guaranteed to agree; do not benchmark one against the other and
   conclude something about the loss.
5. **Multi-stim and multi-probe packs are stim-major, probe-inner.** The channel order in the
   data must line up with `stim_names_multi`, `probe_loss_indices` and `--probsSelect`. The
   launchers hard-code `probsSelect="0"`, so a two-stim run needs an edited launcher. **Silently
   training on the wrong channel is the failure mode here, and it does not raise.**
6. **`ball_synth_v1` has mismatched norm constants** — packed with mean −73.97 / std 42.65, not
   the current −60.095 / 18.95, so `HybridLoss`'s data-side de-normalization is wrong for that
   specific pack.
7. **Do not set `t_max_override: auto` on a battery containing `_i4k` stimuli** — it reads
   `len(stim_name.csv)` and would silently give 400 ms against 5001-point data.
8. **Do not run 800k chaoticRamp.** 200k → 400k bought −1.6 % mean error for 2× data and made
   kdr worse; ~186 node-hours for < 1 % expected gain.
9. **`dtw_200k`'s `sum_train.yaml` was hand-reconstructed** after that run hit the 6 h wall at
   epoch 94, and its `loss_valid` was byte-identical to `dtw_80k`'s to 17 digits until it was
   corrected (true value: epoch 94, −0.0571). The R² numbers in the ledger are unaffected —
   scoring reads the checkpoint, not the YAML — but do not quote loss values from reconstructed
   metadata without checking `log.train`.
10. **A launcher can override `max_epochs` in the YAML.** The champion's YAML says 100; the run
    executed 150. Read `sum_train.yaml` / `log.train` for what actually happened.

### 5.5 Source map

| claim | source |
|---|---|
| Ledger, all arms | `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv` |
| Champion per-channel + trace metrics | `.../ca3_vo_chaoticramp_dtw_amp_8n/ca3_pyramidal_synth/dtw_amp_200k/out/eval/summary.yaml` |
| Champion config | same dir, `ca3_vo_chaoticramp_dtw_amp_8n.hpar.yaml` |
| Champion cost (150 ep, 217.2 s/ep, 32 GPUs, 32 629 s) | same dir, `log.train` |
| Supervised ceiling | `.../ca3_supervised_chaoticramp/ca3_pyramidal_synth/super/out/eval/summary.yaml` |
| Pack composition, `log_halfspan`, `parName` | `ca3_chaoticramp_v3/ca3_pyramidal_synth.mlPack1.h5` → `meta.JSON` |
| OAT stimulus × channel variation | `tmp_neuInv/sensitivity_variation/ca3_pyramidal/salloc_55486075/interp4000/combined/variation_matrix.csv` |
| Feature × channel sensitivity (chaoticRamp, 6 features) | `tmp_neuInv/ca3_ablation/sens_chaoticramp/feature_sensitivity_summary.yaml` |
| Feature × channel sensitivity (step battery, 12 features — the Kdr-blindness table) | `.claude/worktrees/voltage-only-ca3/feature_sensitivity_stepbattery/feature_sensitivity_summary.yaml` |
| Step-0 bias audit | `tmp_neuInv/jaxley_ca3/smoke_1778181760/out/bias_probe/bias_summary.yaml` |
| Chaotic-stimulus variance statistics | `tmp_neuInv/chaoticramp_probe/5kChaoticRamp_probe.npz` |
| Native-vs-interpolated stimulus robustness | `tmp_neuInv/sensitivity_variation/ca3_pyramidal/salloc_55486075/compare/compare_summary.yaml` |
| Kdr-blindness findings block (codified) | `toolbox/soft_efel.py:140-160` |
| Fisher / CRLB numbers | **no surviving output** — prose only, `JAXLEY_ARCHITECTURE.md:577-583` |
| CA3 NEURON validation | `docs/ca3/summary.md`, `docs/ca3/summary.csv`, `docs/ca3/walltime.txt` |
| Findings narrative, negative results | `RESULTS_vo.md` (CA3 voltage-only ledger, `ca3-vo-dtwblur` worktree) |
| Architecture detail | `JAXLEY_ARCHITECTURE.md`, `docs/ARCHITECTURE.md` (Phase-3 vintage — loss section is stale) |
| Cell/channel definitions | `toolbox/jaxley_cells/`, `toolbox/jaxley_channels/ca3_channels.py` |
