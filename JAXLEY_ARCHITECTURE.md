# How the Jaxley Architecture Works

A synthesis of the design documents scattered across the ten worktrees, checked against the
code currently on `CNN_Jaxley`.

**Sources.** `docs/ARCHITECTURE.md` (the most complete single account, but written at Phase-3
vintage and now partly stale), `structure.md` (byte-identical hand-written sections in all ten
worktrees — only the auto-generated file inventory differs), `PLAN.md` (the original five-phase
plan), `PHASE3_TASKS.md`, the two branch-scoped `TASKS.md` files (`ball-stick-bbp`,
`l5ttpc-ih-fix`), `RESULTS_vo.md` (the CA3 voltage-only experiment ledger),
`STEP_STIM_PLAN.md`, `sensitivity_best_stims.md`, and `docs/phase1/`, `docs/phase3/`,
`docs/ca3/`. Where the docs and the code disagree, the code wins and the disagreement is
recorded in §11.

---

## 1. The idea

The base problem is an inverse problem: given a voltage trace, recover the ion-channel
conductances that produced it. The original pipeline solves it with plain supervised learning
— a 1-D CNN regresses 19 (or 6, or 12) conductances against ground-truth labels from NEURON
simulations, under an MSE loss in parameter space.

The Jaxley work adds a second, physically-grounded supervision signal. Jaxley is a
compartmental neuron simulator written in JAX, so a simulation is a differentiable function.
That makes the following loop possible:

> CNN predicts conductances → Jaxley re-simulates a voltage trace *from those predictions* →
> compare the simulated trace to the observed one → backpropagate the mismatch all the way
> through the ODE solve and into the CNN weights.

The payoff is that this loss needs **no parameter labels at all** — only the voltage trace you
already measured. That is what makes fine-tuning on real patch-clamp data possible, where
ground-truth conductances do not exist and never will. The parameter-space MSE and the
voltage-space physics loss are complementary, and the loss object supports any weighted
combination of the two, including either one alone.

The two extremes have both been run, and they behave very differently:

- **Supervised (labels, `use_voltage_loss: False`)** — solves CA3 essentially perfectly
  (R² ≈ 0.995 all six channels, ~4 minutes of training) because no in-loop ODE solve is needed.
  It is 1500–4000× faster per step.
- **Voltage-only (`channel_weight: 0`, `mask_channels: True`)** — the research frontier. Best
  CA3 result to date is mean R² 0.763. Every step pays for a full batched ODE solve plus its
  adjoint.

The gap between 0.763 and 0.995 is the entire research programme: it measures how much
information about conductances is *actually present* in a soma voltage trace, given a
particular loss and a particular stimulus.

---

## 2. The stack

Five components, deliberately additive — the original `Model.py` / `Trainer.py` /
`Dataloader_H5.py` keep their shapes, and the integration is a single `if` in the trainer.
This was ground rule #1 in `PLAN.md`.

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

Plus a sixth piece that runs the same stack in the opposite direction:
`scripts/gen_*.py` generate synthetic training packs *using the very same cell builders* that
the loss will later use as its simulator. That symmetry is not cosmetic — see §10.1.

---

## 3. Layer by layer

### 3.1 The cell registry (`toolbox/jaxley_cells/`)

Every simulable cell is a `CellSpec` registered under a string name. The registry is the single
integration point: adding a cell requires no changes to the loss, the bridge, or the training
loop.

```python
@dataclass
class CellSpec:
    build_fn:          Callable   # () -> (jx.Cell, entry_to_cnn_idx[, entry_multipliers])
    param_keys:        List[str]  # CNN-output order — MUST match sum_train.yaml's parName
    stim_attach_fn:    Callable   # (cell, stim_jnp) -> data_stimuli
    record_fn:         Callable   # (cell) -> sets up voltage recording
    dt, dt_stim:       float      # ms
    t_max:             float      # ms
    v_init:            float      # mV
    default_stim_name: str
    stim_dir:          Path
```

The subtle part is `entry_to_cnn_idx`. A jaxley cell may have more trainable *entries* than the
CNN has *outputs*, because one conductance name can appear in several sections. `entry_to_cnn_idx[i]`
says which CNN output feeds trainable entry `i`. Several entries may share one index — L5TTPC's
`gIhbar_Ih_dend` drives both basal and apical, so index 3 appears twice. On the forward pass the
scalar fans out; on the backward pass JAX sums the partials back into one gradient. You get the
correct chain rule for free.

The `l5ttpc-ih-fix` worktree extended this to a 3-tuple with an optional `entry_multipliers`
vector, so a single CNN scalar can drive a *shaped* spatial profile: `val_i = s · m_i`. That was
built to preserve BBP's distance-dependent apical Ih gradient
`gIh(d) = max(0, −0.8696 + 2.087·e^{0.0031 d}) · 8e−5` under CNN control. Without it,
`make_trainable` on a group calls `nanmean` and collapses 492 distinct per-compartment values
into one scalar — proximal apical then gets ~13× too much Ih and distal ~4× too little.

Cells currently registered (the set grew across worktrees, which is why the appendices differ):

| name | params | probes | geometry | status |
|---|---|---|---|---|
| `single_comp` (`soma_only.py`) | 3 | 1 | single-compartment HH | bench / unit tests |
| `ball_and_stick` | 4 | 2 (soma, dend) | HH soma + 5-comp passive dendrite | Phase-1 gradcheck cell |
| `ball_and_stick_bbp` | 12 | 1 | BBP channels, soma + 5-comp dendrite | Phase 2/3 workhorse |
| `ca3_pyramidal` | 6 | 1 | single compartment, L = diam = 50 µm, cm 1.41, Ra 150 | **the current research vehicle** |
| `L5TTPC` / `l5ttpc` | 19 | 1 | `jx.read_swc` on BBP `L5_TTPC1.swc`, ncomp from `L5TTPC_NCOMP` (default 4) | heavy; OOMs at B=4 fwd+bwd on 40 GB A100 |
| `l5ttpc_multiprobe` | 19 | 4 (soma, axon, apic, dend) | identical biophysics, three extra `.record()` | multi-probe observability experiment |
| `L5PC_jaxley` | 19 | 1 | vendored SWC + `jaxley_mech` Hay-2011 channels | self-contained — no NEURON or `/pscratch` dependency |

All cells default to `dt = dt_stim = 0.1 ms`, `t_max = 500 ms`, and
`default_stim_name = "5k50kInterChaoticB"` from
`/pscratch/sd/k/ktub1999/main/DL4neurons2/stims`.

CA3's channels are hand-ported `.mod` transcriptions in
`toolbox/jaxley_channels/ca3_channels.py` — `Leak_CA3`, `Na3` (three gates, including a slow
inactivation variable `s`), `Kdr_ca1`, `Kap_rox`, `Km_ca3`, `Kd_ca3`, with a 34 °C temperature
factor and a singularity-safe `_trap0`. The port was validated against NEURON on seven tests and
passed all of them first try: rest matches to 0.000 mV, sub-threshold RMSE 0.002 mV, f-I curve
identical, spike-count correlation 0.995 over a 50-draw random parameter sweep, AP peak within
1.83 mV. That validation is why CA3 became the vehicle — it is cheap and provably faithful.

### 3.2 The bridge (`toolbox/JaxleyBridge.py`, 282 lines)

This is where torch's autograd graph crosses into JAX and back. It is the hot path.

**Handle cache.** A `_CellHandle` holds the built `jx.Cell`, a snapshot of its trainable
structure, and — most importantly — `simulate_batch = jax.jit(jax.vmap(_simulate_one))`. It is
built once per process and cached under the key `(cell_name, stim_name, checkpoint_lengths,
solver)`, because each distinct combination compiles to a different XLA binary. Cold compile is
25–70 s; warm dispatch is ~5 s/step at B=128, fp64, t_max=500 ms. Changing batch size does *not*
recompile (vmap handles the batch axis); changing solver or checkpoint lengths does.

**Forward** is a `torch.autograd.Function`:

```python
def forward(ctx, params_phys, cell_name, stim_name, ckpt, solver):
    handle  = get_handle(cell_name, stim_name, ckpt, solver)
    params_j = _torch_to_jax(params_phys)              # zero-copy via DLPack
    v_j, vjp_fn = jax.vjp(handle.simulate_batch, params_j)
    ctx.vjp_fn = vjp_fn                                # capture the closure
    return _jax_to_torch(v_j, ...)
```

and `_simulate_one` for a single sample broadcasts each flat scalar into its per-compartment
shape, calls `jx.integrate(...)`, and decimates the output back to the 10 kHz grid the CNN was
trained on.

**Backward** simply replays the captured VJP closure:

```python
def backward(ctx, grad_out):
    (dparams_j,) = ctx.vjp_fn(_torch_to_jax(grad_out))
    return _jax_to_torch(dparams_j, ...), None, None, None, None
```

Torch then chains that into the rest of its graph. The design is nice because JAX never sees
torch and torch never sees JAX — the only contract is a DLPack tensor and a captured closure.

### 3.3 The spine (`toolbox/jaxley_utils.py`, 90 lines)

Small but load-bearing. It owns two conversions that *must* agree on both sides of the loss:

**Unit ⇄ physical.** The CNN works in a normalized log space; jaxley wants S/cm². The mapping,
mirrored in numpy and JAX so gradients flow, is

```
phys = center · 10^(unit · log_halfspan)
```

with `[center, log_halfspan, unit_string]` per parameter, read from
`sum_train.yaml['input_meta']['phys_par_range']` or from the pack's `meta.JSON`. With
`log_halfspan = 0.5` and a tanh clamp, physical values span `[center/√10, center·√10]`.

**Voltage normalization — and this is where the architecture has genuinely moved.** The docs
describe a *per-sample-per-probe z-score* (that is what `format_bbp3_for_ML.py` does for NEURON
packs). The current jaxley path instead uses a **fixed global scale**:

```python
VOLT_NORM_MEAN = -60.0951997   # mV
VOLT_NORM_STD  =  18.95055671  # mV
```

The reason is stated in the source and it is a real insight: a per-trace z-score throws away
absolute voltage information — resting potential, spike height, subthreshold amplitude — which
is exactly what distinguishes one conductance set from another. A fixed scale preserves it. The
constraint is that the data generator and the loss must use the *same* transform, or the MSE
compares two different spaces. (See §11 for a pack where this went wrong.)

### 3.4 The objective (`toolbox/HybridLoss.py`, 783 lines)

The loss has grown from the two terms in `PLAN.md` into a configurable battery. Everything below
is read from the `voltage_loss:` block of the design YAML; the defaults in parentheses are what
`build_hybrid_loss` uses when a key is absent.

**Preprocessing, applied to the CNN output before it reaches the simulator:**

1. `clamp_unit_tanh` (False) — squashes the unbounded final linear layer to `[-1, 1]`. Not
   optional in practice for voltage-only training: without a parameter anchor, the unbounded
   output drives jaxley to unphysical conductances and NaNs the integrator. It also bounds the
   supervised path.
2. `fp64` (False) — casts to float64 before unit→phys. See §6.
3. unit→phys via the mapping above.

**The loss terms**, each weighted independently:

| term | keys | what it measures |
|---|---|---|
| channel MSE | `channel_weight` (1.0), `mask_channels` (False) | parameter-space error; `mask_channels: True` skips it entirely — the real-ephys mode |
| voltage MSE | `voltage_weight` (0.0), `mse_weight` (1.0) | pointwise fixed-scale trace error |
| soft-DTW | `dtw_weight` (0.0), `dtw_gamma` (0.1), `dtw_band_ms` (8.0), `dtw_n_points` (200) | differentiable dynamic time warping — tolerant of spike-timing jitter, which pointwise MSE punishes savagely |
| soft-eFEL | `efel_weight` (0.0), `efel_features`, `efel_k`, `efel_thr` (−20 mV), `efel_nmax`, `efel_huber_delta` | differentiable surrogates for electrophysiology features (spike rate, AP amplitude, AHP depth, ISI statistics, upstroke/downstroke dV/dt, subthreshold levels) |
| multiscale blur | `blur_weight` (0.0), `blur_sigmas_ms` (default 8/4/2/1/0.5 ms), `blur_sigma_scale`, `blur_include_raw` | van-Rossum-style Gaussian-smoothed MSE at several timescales |
| trace low-pass | `lowpass_ms` (0 = off) | moving-average filter applied to *both* traces before the MSE |
| randomized smoothing | `smooth_sigma` (0.0), `smooth_samples` (1) | averages the whole voltage loss over Gaussian jitter of `pred_unit` — smooths the *parameter-space* landscape, not the trace |
| range penalty | `range_penalty_weight` (0.0), `range_penalty_margin` (1.0) | soft alternative to the tanh clamp: `mean(relu(|pred_unit| − margin)²)` |

**Two orthogonal mechanisms** sit on top:

- `grad_precond` — per-channel gradient reweighting, either explicit `weights` or derived from a
  measured `sensitivity` vector with an `exponent` and a `normalize` mode (`geomean` default).
  The motivation was to boost channels the voltage loss barely sees. Empirically it does not
  work (§10.4).
- `schedule` — ramps a weight over a given number of epochs. This turned out to matter a great
  deal: soft-eFEL as a *primary* loss at weight 1.0 diverges, but ramped in as a small auxiliary
  (0.05 → 0.2 over 20 epochs) under a soft-DTW primary it is the single best lever found.

**Simulation control:** `cell_name_for_sim`, `stim_name`, `stim_names_multi` (one stim per probe
channel, for multi-stim training), `probe_loss_indices`, `soma_probe_index` (0),
`t_max_override` (`auto` resolves to `dt_stim × len(stim_csv)`, mutates the cell module's
`_T_MAX` and clears the bridge cache), `solver` (`bwd_euler`), `sim_dt_ms` (0.1),
`sim_t_skip_ms` (0.0), `checkpoint_lengths`, `pad_batch_size`.

**Three robustness mechanisms** are easy to miss but matter operationally:

- **Batch padding.** The final mini-batch of an epoch is usually short, and a different batch
  shape would trigger a fresh XLA compile. `pad_batch_size` (taken from `local_batch_size` or
  `batch_size`) pads up before the call and slices back after.
- **NaN quarantine, twice.** Non-finite simulated rows are dropped from the batch before the
  loss, and the bridge's backward runs `torch.nan_to_num` on the incoming gradient. Both exist
  because under DDP a single bad sample's NaN propagates to every rank through the all-reduce.
- **Time alignment.** `sim_t_skip_ms / sim_dt_ms` gives a bin offset to discard simulator
  warm-up, after which both traces are truncated to `min(T_sim, T_data)`.

### 3.5 The differentiable feature libraries

- **`toolbox/soft_dtw.py`** (163 lines) — soft-DTW with a `softmin` recursion and a Sakoe-Chiba
  band. The band width is a real hyperparameter: 8 ms carries the residual firing-rate signal,
  4 ms did not help.
- **`toolbox/soft_efel.py`** (634 lines) — the largest of the three. Real eFEL features are
  step functions of the trace (count spikes, find peaks) and therefore have zero gradient
  almost everywhere. This module rebuilds them from smooth primitives: `_softplus_max` /
  `_softplus_min` for peaks, a sigmoid-based `_spike_times` detector at threshold `thr` with
  sharpness `k` and a cap of `nmax` spikes, and `_soft_select` for windowed picks. It ships
  `STRONG_FEATURES` (validated against real eFEL by Pearson correlation), `SUBTHRESHOLD_FEATURES`,
  and `FEATURE_SCALES` to put unlike features on a common footing. `STRONG_FEATURES` — the six
  that correlate at r ≥ 0.79 with real eFEL — are `time_to_first_spike`, `mean_frequency`,
  `inv_first_ISI`, `AHP_depth_abs_slow`, `ISI_values`, `AP_amplitude`. Three surrogates were
  dropped from production configs after measuring r = 0.07 (`spike_half_width`), −0.14
  (`fast_AHP_change`) and 0.16 (`AHP_slow_time`) against their real counterparts, because a
  badly-correlated surrogate injects pure gradient noise. `real_efel_features` is kept alongside
  as the non-differentiable reference, and `python toolbox/soft_efel.py` runs the correlation
  table plus a differentiability proof.

  Two implementation details make it trainable at all. The `only=` argument prunes which
  features are computed, because the unpruned version materializes `(B, nmax, T)` window tensors
  that would dominate the backward graph. And the whole forward path is free of `.item()`,
  boolean indexing, and numpy — everything is a soft gate, so the graph never breaks.

  A findings block inside the source (`soft_efel.py:128-181`) is worth reading before using it:
  as a *primary* training loss the feature set collapsed the CNN toward a constant predictor
  and lost to both plain z-MSE and soft-DTW. The one clean per-channel win is
  `ap_upstroke_dvdt` → Na; `ap_downstroke_dvdt` and `kdr_repol_slope` are, despite their names,
  blind to Kdr.
- **`toolbox/trace_metrics.py`** (195 lines) — whole-trace distances: Gaussian kernels,
  multiscale blurred MSE, decimation, a second soft-DTW implementation, smoothed ensemble std,
  and `parameter_explained_var`.

### 3.6 The trainer hook

Exactly what `PLAN.md` promised — one conditional:

```python
if params.get('use_voltage_loss'):
    self.criterion = build_hybrid_loss(params)
```

`_ChannelOnlyAdapter` wraps a plain MSE in the same call signature, so the call site never moves
and `voltage_weight: 0` reproduces the old path bit-for-bit (there is a regression test for
exactly that).

The call itself is `self.criterion(outputs, labels, images)`, and the third argument is the
neat trick: **the CNN's own input tensor doubles as the observed voltage target.** The physics
loss needs no new data at all — the trace you fed the network forward is the trace you compare
the re-simulation against. That is precisely why the objective survives the removal of labels.

Three smaller additions round it out: `outputSize_override` in the model block lets the CNN emit
a different number of parameters than the data pack carries (needed whenever the loss simulator
has fewer channels than the pack, e.g. a 6-param `ca3_pyramidal` reading a 19-param pack);
`train_conf.clip_grad_norm` was added specifically for this path; and `set_epoch` is called once
per epoch to drive the weight schedules.

### 3.7 The diagnostic toolchain

A set of tools built on the bridge but independent of the CNN. They are how every identifiability
claim in §10 was actually established, and they are the reason those claims are trustworthy —
each one measures a property of the *inverse problem*, not of a trained model.

- **`sensitivity_analysis.py`** — the CNN-independent identifiability probe. Computes the
  Jacobian `J = ∂(normalized V)/∂θ` by central finite differences at several operating points,
  then the Fisher information `F = mean(JᵀJ)/T` per stimulus, its additive multi-stimulus
  combination, marginal sensitivities, Cramér-Rao lower bounds on each parameter's achievable
  error, a collinearity matrix from `F⁻¹`, and the eigenspectrum. This is what established that
  CA3 is well-conditioned and therefore *not* degenerate (§10.3).
- **`feature_channel_sensitivity.py`** — the same idea in feature space:
  `S[f,p] = RMS|∂soft_feature_f/∂θ_p|`, normalized by the same `FEATURE_SCALES` the loss uses,
  plus a specificity measure `S[f,p]/Σ_p' S[f,p']` that finds each channel's most *diagnostic*
  feature rather than merely its most sensitive one. It emits a ready-to-paste
  `grad_precond.sensitivity` vector, and it is the tool that produced the "Kdr is invisible to
  spike-shape features" finding.
- **`ood_probe.py`** — extrapolation behaviour. Sweeps each conductance out to unit ±1.5,
  well beyond the trained ±1 box, and asks what the CNN does. Answer: predictions saturate at
  the tanh boundary in the correct direction for well-identified channels, and widening the
  trained range to compensate makes things *worse* (`LOG_HALFSPAN` 0.5 → 1.0 dropped mean R²
  from 0.566 to 0.242).
- **`scripts/voltage_loss_bias_probe.py`** — the "step-0 audit". Evaluates the exact training
  criterion at the *true* parameters. If the loss is not ≈0 there, no amount of optimization can
  succeed, and the gap measures the model-mismatch floor. This is how the fp64-generation /
  fp32-loss solver mismatch was caught (loss ≈ 0.38 at θ_true, correlation 0.82).
- **`sensitivity_variation.py`** and `chaoticramp_variance_probe.py` — one-at-a-time stimulus
  ranking ("which stimulus exposes which channel") and ensemble-variance measures for chaotic
  stimuli.
- **`evaluate_voltage.py`** and **`plotJaxleyValidation.py`** — the standard evaluation exits,
  producing `out/eval/summary.yaml` whose `channel_r2_overall` is the headline metric used
  throughout the experiment ledger.

---

## 4. The end-to-end gradient path

One tensor, followed from disk to weight update:

```
true_volts (B, T, C)              from the mlPack1 H5
   ↓ permute to channel-first
images (B, C, T) fp32             CNN input
   ↓ Conv1d ×3 → BatchNorm → FC ×6
pred_unit (B, P) fp32             CNN output — unbounded, no activation
   ↓ tanh()
pred_unit ∈ [-1, 1]
   ↓ .double() if fp64
   ↓ phys = center · 10^(unit · log_halfspan)
pred_phys (B, P) fp64
   ↓ DLPack, zero-copy
params_j (B, P) jax array
   ↓ jax.vjp(jit(vmap(simulate_one)), params_j)
   ↓     broadcast each scalar to per-compartment shape
   ↓     jx.integrate(cell, params, dt, t_max, stim, solver, checkpoint_lengths)
   ↓     decimate to 10 kHz
v_j (B, n_rec, T_out) fp64
   ↓ DLPack
v_sim (B, n_rec, T_out) torch
   ↓ take soma probe, fixed-scale normalize
   ↓ truncate to min(T_sim, T_data)
   ↓ MSE / soft-DTW / soft-eFEL / blur, weighted sum
loss (scalar)
   ↓ loss.backward()   — ctx.vjp_fn replays the JAX adjoint
grad CNN.weight
   ↓ DDP ring-allreduce (NCCL over NVLink)
   ↓ Adam.step()
updated weights
```

DDP synchronizes only the **CNN** gradients. Each rank's jaxley adjoint is computed locally
because each rank's samples are independent.

---

## 5. What actually happens inside `jx.integrate`

Worth understanding, because it is where all the time and memory go. Taking
`ball_and_stick_bbp` (6 compartments, ~49 channel states + 6 voltages ≈ 55 evolving scalars) as
the worked example:

**Setup, once per compiled function.** `build_init_and_step_fn` overwrites the trainable
scalars, runs each channel's `init_state` to relax gating variables to steady state at `v_init`,
and computes axial conductances from geometry, `Ra` and `cm`.

**The time loop** is a `nested_checkpoint_scan` over T steps (a plain `lax.scan` when
`checkpoint_lengths` is unset). Each step:

1. **Channel states** — every inserted channel's `update_states` computes α/β rates from the
   current voltage and takes an exponential-Euler step,
   `m_{t+1} = m∞ + (m_t − m∞)·e^{−dt/τ}`, vectorized across compartments.
2. **Currents** — each channel returns `(linear, const)` such that `i(v) = g·(v − E)` is written
   as `linear·v + const`, keeping the voltage solve linear.
3. **Stimulus** — the injected current for this timestep is added at the soma compartment.
4. **Implicit voltage solve** — `(I + dt·A)·v_{t+1} = v_t + dt·const`, where `A` is the
   tridiagonal Hines matrix (axial conductances off-diagonal, capacitance plus total channel
   conductance on the diagonal). Solved by a dendrogram-tailored LU sweep.
5. **Record** — the soma voltage is written into the scan output.

`vmap` runs all of this for B independent neurons at once, so the compiled program is one big
batched scan.

**The backward pass** costs 6–9× the forward and ~200× the memory, for two reasons. First, the
tape: reverse-mode AD over a length-T scan must retain every intermediate on the backward path —
voltages, gating states, currents, Hines factors at every step. Measured peak GPU memory ≈ 98 MB
at B=64 and ≈ 196 MB at B=128, versus ≈ 0 in forward. `checkpoint_lengths=[20, 50]` turns the
scan into nested scans and rematerializes the inner segments, trading ≈2× compute for ≈√T memory.
Second, the per-step VJP work: a transposed Hines solve, then gradients flowing back through
`i = ḡ·m^p·h^q·(v − E)` into `ḡ` (a trainable), into `m, h` (continuing up the state chain), and
into `v` (adding to the upstream voltage gradient).

A useful counter-intuitive measurement: **compartment count is nearly free** at these sizes.
Going from 6 to 81 compartments cost ~0% on an A100 — kernel-launch overhead dominates, not the
tridiagonal solve. And for small cells, **CPU beats GPU at every batch size up to 128**, because
a 6-compartment cell is too small for the GPU to amortize dispatch.

---

## 6. Why fp64 is not optional

`bwd_euler` is implicit: each step solves `(I − dt·J)x = b`. Reverse-mode AD through N=5000 such
steps multiplies N Jacobian-transpose matrices. The BBP channel set has fast, steeply
voltage-dependent gating, so `dα/dV` near threshold is enormous and every spike contributes a
large kick to the cumulative product. With 6–15 spikes in a 500 ms trace, that product overflows
fp32's 23-bit mantissa and the gradient becomes NaN. Measured:

| t_max | fp32 grad NaN | fp64 grad NaN |
|---|---|---|
| 50 ms | 0/12 | 0/12 |
| 100 ms | 0/12 | 0/12 |
| 250 ms | **12/12** | 0/12 |
| 500 ms | **12/12** | 0/12 |

The forward pass stays finite throughout — only the adjoint blows up. `crank_nicolson` and
`jax.checkpoint` were both tried and neither helped, because neither addresses the mantissa
limit. The fix is `fp64: True` in the YAML plus `JAX_ENABLE_X64=true` in the environment (which
must be set *before* JAX is imported, hence its position in the `.slr`), at a cost of ~2–3×
forward and ~5× backward. Under DDP this matters doubly: one rank's NaN propagates to every rank
through the all-reduce.

Two qualifications. First, this is a property of the **stiff BBP channel set**, not of jaxley:
single-compartment CA3 is documented as stable in fp32, and the CA3 launcher says so explicitly.
CA3 runs still use fp64 by default, but that is caution rather than necessity. Second,
`checkpoint_lengths=[outer, inner]` bounds the length of any single adjoint chain and so offers
partial relief in fp32 — it is a memory tool first and a numerics tool second.

---

## 7. Multi-GPU on Perlmutter

Four settings, each of which was found by hitting a real failure. All are encoded in
`batchShifterJaxley.slr` and none should be "simplified".

1. **`--gpus-per-node=4 --gpu-bind=none`, never `--gpus-per-task=1`.** Per-task binding hides
   peer GPUs from each rank's CUDA context, and NCCL's P2P/SHM transports need cross-rank
   visibility. Symptom: `Cuda failure 101 'invalid device ordinal'`.
2. **Per-rank JAX compilation cache**, `JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/rank_$SLURM_PROCID`.
   A shared cache serves rank 0's `cuda:0`-targeted binary to rank N, and XLA dies with
   "Buffer on cuda:N, but replica assigned to cuda:0". This one broke a whole CA3 ablation
   series (A2–A5) before it was diagnosed.
3. **Pin JAX per rank**, `jax.config.update("jax_default_device", devices[localid])` inside
   `HybridLoss`. Torch is pinned via `SLURM_LOCALID` in both `train_dist.py` and `Trainer`, but
   JAX defaults to `cuda:0` independently — without this every rank piles its jaxley solve onto
   GPU 0 and DDP scaling silently vanishes (measured 380 s/epoch on 4 GPUs versus 107 s with the
   fix).
4. **Wrap the launch in `bash -c`** so `$SLURM_PROCID` expands per rank.

Plus the environment: `PYTHONNOUSERSITE=1` (a stale `~/.local` breaks compute nodes),
`JAX_PLATFORMS=cuda`, `JAX_ENABLE_X64=true`, `NCCL_NET_GDR_LEVEL=PHB`, `FI_PROVIDER=cxi`,
`FI_CXI_DEFAULT_CQ_SIZE=131072`. And the runtime is a **conda env, not shifter** —
`/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley`.

**Scaling reality.** 16 GPUs gave 14.9× over 1 GPU at B=128 — near-linear. But 32 GPUs gave *no
speedup at all* over 16 at a pinned global batch of 2048 (220.8 vs ~247 s/epoch), because the
fp64 jaxley solve is dominated by the **sequential 5001-step time integration**, which
parallelizes over neither batch nor GPUs. Past 16 GPUs the right move is a longer wall-clock, not
more nodes.

---

## 8. Data generation — the same stack, run forwards

`scripts/gen_ca3_sharded.py`, `gen_ball_and_stick_data.py`, `gen_multistim_data.py` all use the
*loss simulator itself* to manufacture training packs. The sharded CA3 generator is the mature
one: 16 ranks each pin to a local GPU, draw a contiguous slice of `Uniform(-1,1)^P` unit
parameters, simulate them under every requested stimulus, and write raw un-normalized shards;
a single-rank merge phase concatenates in rank order, applies `normalize_volts_fixed` (the same
transform `HybridLoss` uses), shuffles, splits 80/10/10, and writes an `mlPack1.h5` whose layout
is byte-compatible with what `Dataloader_H5.py` already reads. Determinism is preserved by
drawing the full parameter array once from `default_rng(seed)` in both phases.

Multi-stim packs put K stimuli on the **probe axis** (`num_probs=K, num_stims=1`), which with
`serialize_stims: True` collapses them into CNN input channels. `stim_names_multi` in the loss
config must list them in the same order. `gen_multistim_data.py` does this for
`[5k50kInterChaoticB, 5k0step_500, 5k0chirp]`; the sharded CA3 generator takes an arbitrary
comma-separated `--stims` list.

One non-obvious constraint: `simulate_batch` always constructs a `jax.vjp`, even when the caller
is inside `torch.no_grad()`. Generation therefore pays for the backward tape whether it wants it
or not, which is why the generators pass `checkpoint_lengths` for anything larger than CA3 — the
forward-only pass will otherwise OOM.

---

## 9. Cell status and performance envelope

`ca3_pyramidal` is the only cell where the full loop is comfortable: one compartment, six
parameters, NEURON-validated, ~221 s/epoch at 200k samples on 16 GPUs.

`ball_and_stick_bbp` works and is well-characterized (see `docs/phase1/ball_and_stick_bbp_perf.md`
for a genuinely excellent forward/backward walkthrough), but note its geometry is pathological
for the standard stimuli: a 20 µm × 10 µm soma has R_in ≈ 26 MΩ, so a stim peaking at +6.8 nA
mathematically forces ±180 mV passive swings. Those excursions are *not* a solver artifact —
dt=0.1 and dt=0.025 traces overlay to within 0.5 mV. It also shows a depolarization plateau at
~−25 mV driven by persistent Na (`Nap_Et2`); dropping that default ~7× restores normal
spike-recovery cycling.

`l5ttpc` is the unsolved one. It OOMs at B=4 forward+backward on a 40 GB A100, and it has a
documented fidelity backlog: dt=0.1 versus NEURON's 0.025 (worth ~0.5–2 ms of spike drift over
500 ms), uniform `_NCOMP=4` versus NEURON's `d_lambda` rule, unverified Q10 temperature factors,
and the apical Ih gradient question. Current divergence from the NEURON reference is max|Δ| ≈ 94 mV
with 2 spikes versus 4 — most of it plausibly dt-driven.

---

## 10. What the architecture has actually taught us

This is the part the design docs cannot tell you, and it is the real content of the ten
worktrees. The infrastructure works; the science is about what a voltage trace can and cannot
reveal.

### 10.1 Model mismatch is fatal, and fails silently

Phase 2 trained a 12-parameter `ball_and_stick_bbp` simulator against a 19-parameter L5TTPC data
pack. The voltage loss plateaued at ≈2.15 against a random baseline of ≈2.0, and **all 200
evaluated test samples produced bit-identical predicted traces** — total mode collapse. When the
simulator cannot reproduce the data no matter what parameters it is given, the optimizer's best
move is to ignore the input entirely and emit a constant. Phase 3 fixed this by construction:
generate the data with the same cell the loss simulates. That is why the generators live in the
same repo and share the cell registry.

### 10.2 Matching the trace is much easier than recovering the parameters

Phase 3, with data and simulator matched, hit voltage `rmse_z` = 0.113 against a 0.30 bar — a
clean pass — while failing per-parameter explained variance on 11 of 12 parameters. The voltage
term saturated at ≈0.02 within ten epochs and thereafter carried almost no gradient; the channel
term plateaued at ≈0.25 because the unidentifiable parameters had no consistent gradient
direction. `ReduceLROnPlateau` dutifully drove the LR to 6e-9.

The pattern in *which* parameters recovered is the informative part: somatic Na/K parameters
recovered; apical parameters and small somatic conductances sat at EV ≈ 0. From a single soma
probe, the distal dendritic state is behind a cable filter and simply is not observable. The same
result reappeared on L5TTPC (R² 0.33 overall; somatic channels 0.75–0.93, distal < 0.15). This is
an observability limit, not a training failure — which is why multi-probe and multi-stim variants
exist.

### 10.3 CA3 is not degenerate — the loss landscape is just cliffy

A Fisher-information analysis found CA3 well-conditioned (condition number 3–34, all six
channels identifiable), which rules out the comfortable explanation that the inverse problem is
simply ill-posed. The difficulty is instead that sensitivity scales as 1/ε: `V(θ)` is non-smooth
because spike timing moves discontinuously. Chaotic stimuli are the worst for this, steps and
ramps the best. And the decisive confirmation came from the supervised run — parameter-MSE with
labels recovers all six channels at R² 0.995 in about four minutes. **The information is in the
data; the voltage objective is what fails to extract it.**

### 10.4 What moves the needle, and what does not

From the CA3 voltage-only ledger (baseline 0.593 → best 0.763 mean R², against a supervised
ceiling of 0.995):

**Works.** Soft-DTW as the primary loss — spike-timing tolerance matters more than pointwise
accuracy. More data, up to a point: 10k → 400k took the mean from 0.440 to 0.736, but the
200k→400k doubling bought only +0.011, so returns have clearly bent over. And best of all, a
*small, ramped* soft-eFEL auxiliary (`voltage_base` + `AP_amplitude`, weight 0.05→0.2 over 20
epochs) on top of soft-DTW: mean 0.763 at 200k, beating 400k of data with half the samples. That
is an objective change outperforming a data change, which is the campaign's central result.

**Does not work.** Gradient preconditioning — boosting a poorly-observed channel's gradient makes
it *worse* (kdr fell 0.338 → 0.266 under a ×2.5 boost), and it helped only in the data-starved
regime where it was compensating for scarcity. Van-Rossum blur — a coarse 48 ms blur is toxic to
fast channels (kap 0.75 → 0.29); smoothing the loss makes it less discriminative. Step stimuli as
the sole drive — soft-DTW is rate-invariant on tonic firing, so a step gives no rate gradient and
its large envelope drags the CNN into a bad basin. Architecture search — 130 RayTune trials found
nothing better than the baseline skeleton; larger networks diverge. And soft-eFEL as a *primary*
loss at weight 1.0 diverges outright; the same features as a small ramped auxiliary win.

**The kdr story is the cleanest experiment in the ledger.** The delayed-rectifier K conductance
was stuck at R² ≈ 0.15 for most of the campaign. It does not track data (0.165 at 10k, 0.338 at
200k, back to 0.219 at 400k — noise, not a trend), and it does not respond to gradient
reweighting. It shares its only handle — firing rate / ISI / slow AHP — with two other K
currents, so the three are mutually confounded. The only thing that ever moved it was changing
the objective (the eFEL auxiliary, to 0.366). Diagnosis: kdr is objective- and
observability-limited, not data-limited.

### 10.5 Stimuli are complementary, which argues for multi-stim

The sharpest observability result. The same recipe at the same 200k data on two different stimuli:

- **`5kChaoticRamp`** (spike-rich): recovers na3 and kap well, leak and kd less so, and its trace
  overlap is poor because chaos is hard to reproduce.
- **`5k50kInterChaoticB`** (76% near rest, few spikes): mean R² only 0.219, but leak = 0.974 and
  kd = 0.937, both essentially at the supervised ceiling — while na3, kdr, km and kap all fail
  outright. Its trace overlap is *excellent* (`voltage_mse_z` 0.97 versus ~1.9).

These are exact mirror images. A mostly-subthreshold trace is easy to fit and perfectly
constrains the passive and slow channels, but contains no spikes and therefore no information
about spike-shaping or rate channels. This also resolves an apparent paradox: excellent trace
overlap and poor parameter recovery are not in tension — they are the *same* fact seen twice.
Observing both stimuli jointly should recover all six far better than either alone, and it
directly attacks the kdr confound by giving the three K currents independent views.

Related and important: **observability ≠ trainability.** Stimuli ranked as maximally sensitive
by a one-at-a-time sweep train *worse* than a moderate chaotic stimulus (Step600 eFEL 0.328,
Step1000 0.371, versus chaotic 0.288).

---

## 11. Drift, gotchas, and things to distrust

Collected because each one has already cost time.

1. **`docs/ARCHITECTURE.md` is Phase-3 vintage.** Its loss section describes only channel MSE
   plus voltage MSE. The real loss now has seven weighted terms plus preconditioning and
   scheduling (§3.4). Its cell table is missing `ca3_pyramidal`, `l5ttpc_multiprobe` and
   `l5pc_jaxley`. Everything it says about the bridge, fp64, and the multi-GPU layout is still
   accurate.
2. **Normalization is documented as per-sample z-score but implemented as a fixed global scale**
   in the jaxley path (§3.3). Both exist in the repo, for different packs. Know which one your
   pack used.
3. **`ball_synth_v1` has mismatched norm constants** — it was packed with mean −73.97 / std 42.65,
   not the current −60.095 / 18.95, so `HybridLoss`'s data-side de-normalization is wrong for
   that specific pack.
4. **`sensitivity_best_stims.md` is partly invalid.** Seven CA3 stimulus CSVs contain values in
   pA that were injected as nA — a 1000× overdrive, with `ramp_500` and `InterramInterstep_50khz`
   reaching +1974 mV. That invalidates its headline claim that Kdr's best stimulus is `ramp_500`
   (the 146 mV "sensitivity" is the artifact), and the M1-battery kdr result derived from it.
   Treat the rest of that document's rankings with suspicion until re-measured.
5. **`.gitignore` swallows `*yaml`, `*h5`, `*csv`, `*txt`, `L*`.** New design YAMLs need
   `git add -f` or they silently do not exist for anyone else.
6. **SLURM scripts copy `$codeList` into a frozen `$wrkDir` snapshot.** A new Python file that is
   not in that list will not exist at runtime, and the failure looks like an import error rather
   than a packaging error.
7. **Resuming from a checkpoint resets the best-validation tracker**, so a resumed run can report
   a "new best" that is not one.
8. **Soft-eFEL diverges at LR 3e-4** — it peaks around epoch 5–8 and then blows up. Feature-heavy
   configs run at 5e-5 with `clip_grad_norm: 1.0`.
9. **`structure.md` is identical in all ten worktrees** apart from its auto-generated appendix.
   If you want to know what a branch actually added, diff the appendix (or just list
   `toolbox/jaxley_cells/` and `scripts/`), not the prose.
10. **L5TTPC's apical Ih gradient is silently overwritten during CNN-driven training.**
    `_apply_apical_ih_gradient` writes 492 distinct per-compartment values at build time, and
    then `cell.apical.make_trainable("Ih_gbar")` averages them into one scalar which the bridge
    broadcasts back uniformly. The gradient survives only at default state and in direct
    `jx.integrate` calls without `params=`. The `entry_multipliers` mechanism in the
    `l5ttpc-ih-fix` worktree is the fix; know which behaviour your checkout has.
11. **There are two independent soft-DTW implementations** — `toolbox/soft_dtw.py` (used by
    `HybridLoss`) and a second one inside `toolbox/trace_metrics.py` (used by the offline
    variance probes). They are not guaranteed to agree; do not benchmark one against the other
    and conclude something about the loss.
12. **`ca3_channels.py` ports `Leak_CA3_e = 93.9115 mV` verbatim from the source hoc.** It looks
    like an upstream typo (a leak reversal that positive is not physiological), but `g_leak` is
    small enough that the strong resting K_d current still pulls rest to −65 mV, and both the
    NEURON reference and the jaxley port use the same value — so the comparison is valid. If it
    is ever corrected, it must be corrected on both sides simultaneously.
13. **Multi-stim and multi-probe packs are stim-major, probe-inner.** The channel order in the
    data must line up with `stim_names_multi`, `probe_loss_indices`, and the `--probsSelect`
    flag. The launchers hard-code `probsSelect="0"`, so a two-stim run needs an edited launcher
    or a direct `train_dist.py` invocation — silently training on the wrong channel is the
    failure mode here, and it does not raise.

---

## 12. Where it stands

The infrastructure is finished and validated: the bridge is correct (gradcheck at fp64), the
CA3 cell reproduces NEURON on seven independent tests, DDP scales to 16 GPUs at 14.9×, and fp64
eliminates the NaN floor. Adding a cell requires touching only the registry.

The open problem is the objective. Supervised parameter-MSE reaches R² 0.995 on CA3; the best
voltage-only recipe reaches 0.763. Everything in §10 says the remaining gap is not a data
problem and not an architecture problem — it is that a single soma voltage trace under a single
stimulus, compared through a single differentiable distance, does not expose all six
conductances. The two levers with evidence behind them are **richer objectives** (differentiable
rate/ISI/AHP features targeting the confounded K currents) and **multi-stimulus observation**
(letting a subthreshold stimulus pin down leak and kd while a spike-rich one pins down na3 and
kap). Both are supported by the code as it stands.
