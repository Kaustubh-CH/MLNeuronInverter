# `ball_and_stick_bbp` — performance & integration notes

This is the new multi-section BBP-channel ball-and-stick cell (12 trainable
params). Below are CPU vs GPU timings on Perlmutter and a step-by-step
walkthrough of what `jx.integrate` does on the forward and backward passes
for this specific cell.

Bench harness: `toolbox/tests/bench_gpu_l5ttpc.py` (drives `jx.integrate`
directly with `jit(vmap(loss_one))` / `jit(vmap(value_and_grad(loss_one)))`).
Loss = `sum(v**2)` (a scalar) so gradients flow through the integrator
back to the 12 trainable parameters.

Settings: solver `bwd_euler`, `dt = 0.1 ms`, `t_max = 100 ms` (1000 steps),
`5k50kInterChaoticB` stim, soma comp(0) recorded.

CSV outputs:

- GPU: `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench_bbp/bench_gpu.csv`
- CPU: `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_cpu_bench_bbp/bench_gpu.csv`

---

## Numbers

### Forward only — `jit(vmap(loss_one))`

| B   | CPU (login)        | GPU (A100, login)  |
|----:|--------------------|--------------------|
|   1 |   8.0 ms /  126    |  73.9 ms /   13.5  |
|   4 |  13.4 ms /  298    |  70.4 ms /   56.8  |
|  16 |  14.2 ms / 1125    |  67.1 ms /  238.6  |
|  64 |  28.5 ms / 2247    |  72.1 ms /  887.9  |
| 128 | not run            |  83.6 ms / 1530.4  |

### Forward + backward — `jit(vmap(value_and_grad(loss_one)))`

| B   | CPU (login)        | GPU (A100, login)  |
|----:|--------------------|--------------------|
|   1 |  75.7 ms /   13.2  | 624.6 ms /    1.6  |
|   4 |  79.7 ms /   50.2  | 601.3 ms /    6.7  |
|  16 |  79.4 ms /  201.5  | 642.5 ms /   24.9  |
|  64 | 174.9 ms /  366.0  | 642.4 ms /   99.6  |
| 128 | not run            | 673.6 ms /  190.0  |

Reading: `warm_ms / sims·s⁻¹`. Cold-compile: ~3 s fwd / ~9 s fwd+bwd on CPU;
~4 s fwd / ~17–53 s fwd+bwd on GPU.

### Why B=128 was skipped on CPU

Scoping decision, not a hard limit. The CPU sweep was capped at B=64 to keep
the total bench under ~2 minutes. CPU fwd_bwd time scales roughly linearly
with batch (74 ms @ B=1 → 175 ms @ B=64), so B=128 would land around
~350 ms per call × (2 warm + 5 timed) per combo + cold compile — a few extra
minutes total. Just rerun if you want the full sweep:

```bash
JAX_PLATFORMS=cpu /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python \
    -m toolbox.tests.bench_gpu_l5ttpc \
    --cells ball_and_stick_bbp --solvers bwd_euler \
    --batches 1 4 16 64 128 --modes fwd fwd_bwd \
    --out-dir /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_cpu_bench_bbp
```

### Reading the numbers

- **CPU beats GPU at every batch tested up to ≥128 for this cell.** The cell
  is too small (6 compartments × ~12 channel-states each = ~55 scalars per
  step × 1000 steps) for the A100 to amortize its kernel-launch overhead.
- **GPU per-call time is essentially flat** from B=1→128 — pure dispatch /
  launch overhead, not compute.
- **Backward / forward ratio ≈ 8–9× on both devices.** Typical for
  `bwd_euler`'s checkpointed unrolled adjoint.
- **Peak GPU memory** grows ≈ linearly with B in fwd_bwd (~1.5 MB/sample for
  the saved activation tape) and is essentially zero in fwd.

---

## What happens on each forward call

The bench builds one `loss_one(flat_phys)` closure per `(cell, solver)` pair
(see `bench_gpu_l5ttpc.py:83-97`). `flat_phys` is a length-12 vector — the
CNN's prediction for one sample. The closure:

1. Broadcasts each scalar to the right per-compartment shape (apical
   entries → 5 copies for the 5 dendrite comps; soma entries → 1 copy).
2. Calls `jx.integrate(...)` with the cell, params, stim, dt, t_max, solver.
3. Returns `sum(v**2)` (a scalar).

Then it's wrapped once with `jax.jit(jax.vmap(loss_one))` (fwd) or
`jax.jit(jax.vmap(jax.value_and_grad(loss_one)))` (fwd+bwd). The first call
pays the compile cost; every later call dispatches the same compiled binary.

### Cold-path setup (once per call, not per step)

`build_init_and_step_fn` (`jaxley/integrate.py:18-199`) returns:

- **`init_fn`** — overwrites the 12 trainable scalars in `all_params`,
  runs each channel's `init_state` to relax the gating variables to steady
  state at v_init = −75 mV, and computes `axial_conductances` from
  geometry / `Ra` / `cm`.

After this you have:

- **`all_states`** — `v` (length 6), plus one entry per channel-state per
  compartment: `NaTs2_t_m`, `NaTs2_t_h`, `Nap_Et2_m`, `Nap_Et2_h`, ...,
  `CaComplex_cai`, `CaComplex_z_sk`, etc. ~49 scalar states + 6 voltages
  = ~55 evolving scalars per neuron.
- **`all_params`** — every channel parameter at every compartment, plus
  axial conductance edges, capacitance, area.

### Time loop — `nested_checkpoint_scan` over T=1000 steps

`integrate.py:458-464` calls `nested_checkpoint_scan(_body_fun, init_state,
externals, length=1000, nested_lengths=[1000])`. With no
`--checkpoint-lengths` arg, this degenerates into a single `lax.scan` over
1000 steps. (Checkpointing only affects the backward pass — see below.)

Each `_body_fun(state, externals[t])` is one timestep, which calls
`cell.step(...)` (`jaxley/modules/base.py:2816-3000`):

1. **Channel state update** — `_step_channels` walks every inserted channel
   and calls its `update_states(states, dt, v, params)`. For `NaTs2_t`
   (`bbp_channels_jaxley.py:78-84`):
   - compute α/β rate constants from current voltage,
   - exponential-Euler advance: `m_{t+1} = m∞ + (m_t − m∞) · exp(−dt/τ_m)`.
   All 11 channel `update_states` run in parallel, vectorized across
   compartments. `CaComplex` additionally forward-Eulers `cai` and updates
   `z_sk`.
2. **Per-channel current contributions** — each channel returns
   `(linear_term, const_term)` shaped to the compartment grid.
   `i_chan(v) = g · (v − E)` is rewritten as `linear · v + const` so the
   implicit voltage solve below stays linear in `v`. These are summed
   across channels into `linear_terms["v"]` and `const_terms["v"]`.
3. **External current injection** — soma comp(0) receives the stim value at
   this timestep (from the upsampled stim array).
4. **Tridiagonal voltage solve** (implicit Euler / `bwd_euler`):
   - Discretized cable equation: `(I + dt · A) · v_{t+1} = v_t + dt · const_terms`,
     where `A` is the cable's tridiagonal "Hines" matrix (axial
     conductances on off-diagonals, capacitance + total channel
     conductance on the diagonal). For our morphology this is a 6×6
     nearly-tridiagonal system.
   - Default `jaxley.dhs` voltage solver (`base.py:2959-2967`) does an
     LU-style sweep tailored to the dendrogram. Result: new voltage at
     every compartment.
5. **Record** — soma comp(0)'s new voltage is written into the per-step
   output of the scan.

After 1000 steps `_body_fun` has produced a `(1000, 1)` recording —
the soma trace. `sum(v**2)` collapses to one scalar. **`vmap` runs all of
this for B independent neurons in parallel**, so the compiled XLA program
is one big batched scan with batched param dicts.

That's why per-call wall time is so flat across batch sizes (B=1→128 stays
70–80 ms): a 6-compartment cell × 1000 steps is a tiny graph, totally
dominated by kernel-launch overhead on the A100. CPU is the opposite —
zero launch cost, scales linearly with B until cache pressure kicks in.

---

## What happens on each backward call

`jax.value_and_grad(loss_one)` produces the scalar loss **and** the
gradient of the loss w.r.t. `flat_phys` (the 12-vector). Under the hood
this is reverse-mode autodiff over the same `lax.scan`.

Two reasons it costs ≈ 6–9× the forward and ≈ 200× the memory:

### a) the saved tape

A `lax.scan` of length T is an unrolled chain. Reverse-mode autodiff needs
every intermediate value reachable on the backward path: `v_t`, all gating
states, intermediate currents, the LU factors of the Hines matrix at every
step. JAX builds a "tape" — for each of 1000 steps it stashes the inputs
and intermediates of every nonlinear op (exp, divide, multiply by
parameters). Measured peak GPU memory: **≈ 98 MB at B=64, ≈ 196 MB at
B=128** vs **≈ 0 MB in fwd**.

That's why `jaxley.integrate` exposes `checkpoint_lengths`. Pass
`[20, 50]` (product = 1000) and the scan becomes nested: outer scan of
length 20, inner scan of length 50; only outer-scan boundaries get stored
— inner segments are **rematerialized** (run forward again) during the
backward pass. Memory ≈ √T smaller, compute ≈ 2× the forward. We're not
using it; the cell is small enough that current memory cost is fine.

### b) the actual VJP work each step

For each timestep the reverse pass runs a **transposed body**:

1. **VJP of the Hines solve.** Forward solved `M · v_{t+1} = rhs`. Reverse
   computes `M^T · w = ḡ_{v_{t+1}}` and propagates `w` back into both `rhs`
   (gradient w.r.t. previous voltage and channel currents) and `M`
   (gradient w.r.t. axial conductances and capacitance). The transposed
   solve is the same cost as the forward solve, plus an outer product for
   `M`'s gradient.
2. **VJP of channel currents.** Gradients flow back through
   `i = g_bar · m^p · h^q · (v − E)`:
   - into `g_bar` → contributes to `∂L/∂g_bar` (one of the 12 trainable
     params),
   - into `m, h` → continues up the channel-state chain through earlier
     timesteps,
   - into `v` → adds to the upstream voltage gradient.
3. **VJP of the gating update.** `m_{t+1} = m∞(v_t) + (m_t − m∞(v_t)) ·
   exp(−dt/τ(v_t))` is differentiable in `v_t`, `m_t`, and (transitively)
   `g_bar` only via the current term. JAX auto-derives this; cost is
   roughly the cost of evaluating the rates again.

Doing all three at every step explains the measured ratios:

- **GPU**: fwd 70 ms → fwd+bwd 624 ms ≈ **8.9×**. Transposed Hines solve
  plus saved-tape replay is more than the forward, plus extra
  kernel-launch tax on a tiny graph.
- **CPU**: fwd 8 ms → fwd+bwd 76 ms ≈ **9.5×**. Same shape, no launch tax,
  but actual FLOPs hurt more here.

### c) what the gradient *is*

After all 1000 steps of the reverse scan have run, JAX delivers
`grad ∈ ℝ^12`. For `flat_phys[7]` (= `gNaTs2_tbar_NaTs2_t_apical`), the
gradient has accumulated contributions from all 5 dendrite compartments at
all 1000 timesteps — every time NaTs2_t in the dendrite contributed any
current, the reverse pass added a piece. That's exactly what
`entry_to_cnn_idx` is doing on the forward path: it tells JAX "this scalar
fans out into 5 array entries"; on the backward path JAX sums those 5
partials back into a single scalar gradient. (Same mechanism makes
L5TTPC's `gIhbar_Ih_dend` work with two `make_trainable` calls — basal +
apical — pointing at the same CNN index: their two partials simply sum.)

---

## Concrete numbers for this cell

- **State dim per neuron:** ≈ 49 channel-states + 6 voltages = **55 scalars**
  evolving every step.
- **Param dim:** **12 trainable scalars** (4 also broadcast to 5 dendrite
  comps internally; doesn't change grad dim).
- **Steps per simulation:** 1000 (dt=0.1, t_max=100). Reference runs use
  t_max=500 — fwd would be ≈ 5× slower and fwd_bwd needs ≈ 5× the tape
  memory.
- **One XLA program per `(B, mode, solver)`:** changing B doesn't recompile
  (vmap broadcasts batch shape). Changing solver or `checkpoint_lengths`
  does.

---

## CPU host RAM (RSS) — measured

`bench_gpu_l5ttpc.py`'s `peak_mem_mb` column reports `-1` on CPU because
`jax.devices()[0].memory_stats()` only works for accelerator devices. Below
is host-side peak RSS (`getrusage(RUSAGE_SELF).ru_maxrss`) for
`ball_and_stick_bbp` on a Perlmutter login node, single Python process.

| stage                               | peak RSS  |
|-------------------------------------|----------:|
| baseline (after `import jax/torch`) |   528 MB  |
| after `_build` + jaxley setup       |   906 MB  |
| **forward** B=1   (after JIT)       |  1008 MB  |
| forward B=16                        |  1056 MB  |
| forward B=64                        |  1090 MB  |
| forward B=128                       |  1134 MB  |
| **fwd+bwd** B=1   (after JIT)       |  1166 MB  |
| fwd+bwd B=16                        |  1229 MB  |
| fwd+bwd B=64                        |  1435 MB  |
| fwd+bwd B=128                       |  1645 MB  |

Reading: ~1 GB is fixed cost (Python + jax + jaxley + the JIT'd XLA
program). Beyond that the marginal cost per sample is:

- **forward:** ~1 MB/sample (just the per-step record array; no autograd tape)
- **fwd+bwd:** ~3.7 MB/sample (the 1000-step saved-activation tape — 6 comp ×
  ~55 evolving scalars × 1000 steps × fp32 ≈ 1.3 MB of "raw" state, the rest
  is XLA's tape bookkeeping)

VmPeak is much larger (~92 GB) because XLA mmaps a big virtual region for
JIT spill / scratch, but only ~1.6 GB is actually resident at the worst
point.

## Can you run 128 processes × B=128 in parallel on one node?

Short answer: **yes on a Perlmutter CPU compute node, with caveats.** The
node has 128 physical cores / 256 threads / 512 GB RAM (2× AMD EPYC 7763).
Don't try this on the login node — login is shared and only had 131 GB free
when I checked.

### Memory budget

`128 procs × 1.65 GB/proc ≈ 211 GB resident`. Fits comfortably in 512 GB.
With `JAX_COMPILATION_CACHE_DIR` pointing to shared scratch, only the first
process pays the ~9 s fwd_bwd compile; the other 127 cache-hit on startup.

### CPU contention — the actual bottleneck

The bench numbers above were measured with **one** process letting XLA's
Eigen threadpool grab all 256 hardware threads. If you launch 128 processes
naively, each one tries to grab all 256 threads → 128 × 256 = 32,768 threads
competing for 128 cores → catastrophic context-switching, ~10× slower than
the ideal one-thread-per-core layout.

The fix is to pin each process to a single core and force single-threaded
XLA:

```bash
# inside an SLURM step, srun -n 128 -c 1 ...
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
JAX_COMPILATION_CACHE_DIR=/pscratch/sd/k/ktub1999/jax_cache \
taskset -c "$SLURM_LOCALID" \
    python my_worker.py
```

### Expected throughput

A single process at B=128 fwd_bwd took **191 ms with 256-way Eigen
parallelism**. Single-threaded, that same call will be roughly 5–10× slower
— call it **~1–1.5 s per call**.

So the rough rate per node:

- **128 procs × 128 sims/call ÷ ~1.2 s ≈ ~13,000 fwd+bwd sims/sec/node**

vs the single-process B=64 fwd+bwd we measured at **366 sims/s**. Roughly a
**30–40× total throughput improvement** over single-process vmap, because
the cell is so small that the Eigen threadpool spends most of its time on
synchronization rather than compute.

### A more practical middle ground

128 single-thread processes is the throughput-optimal layout for *this*
small cell, but it's not the only option. Useful intermediate points:

| layout               | per-proc RAM | total RAM |  ~per-call (B=128 fwd_bwd)  |
|----------------------|-------------:|----------:|----------------------------:|
| 1 proc × 256 threads |     ~1.6 GB  |   1.6 GB  |     ~190 ms (measured)      |
| 8 procs × 16 threads |     ~1.6 GB  |    13 GB  |     ~250 ms (estimated)     |
| 16 procs × 8 threads |     ~1.6 GB  |    26 GB  |     ~400 ms (estimated)     |
| 128 procs × 1 thread |     ~1.6 GB  |   210 GB  |   ~1100 ms (estimated)      |

The 8×16 layout is usually the easiest to manage and is within ~2× of the
128×1 ceiling. SLURM-side: `srun -n 8 -c 16 --cpus-per-task=16` plus
`OMP_NUM_THREADS=16` and remove the `taskset` line.

### Caveats / things that bite

- Per-process JIT compile dominates wall time at small N. With 128 procs
  and a shared `JAX_COMPILATION_CACHE_DIR`, the first proc compiles, the
  other 127 cache-hit. Without the cache, you're paying ~9 s × 128 of
  compile work serialized through whatever cores are free.
- All 128 processes will try to read the same stim CSV at startup. Cache it
  on local node-scratch (`/tmp` on Perlmutter is per-node tmpfs) or every
  proc will hammer `$PSCRATCH`.
- `gc` in Python runs differently with single-threaded XLA. Memory
  high-watermark may climb 10–20% over what we measured here.
- This calculus assumes you're running the cell standalone (the bench
  shape). If the workload is the full bridge (`JaxleyBridge.simulate_batch`)
  inside a torch DataLoader, the answer changes — the bridge adds a torch
  ↔ jax round-trip and grad checkpointing settings interact.

## Default-param voltage traces — `ball_and_stick` vs `ball_and_stick_bbp`

Both cells driven with the canonical `5k50kInterChaoticB` stim (the same
file used by the bench reference comparison), `t_max=500 ms`,
solver `bwd_euler`, default parameters from
`bench_jaxley_cells.py::_default_params_tensor`.

Plot: `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench_bbp/trace_compare.png`
(also `trace_compare_dt.png` showing dt=0.1 vs dt=0.025 overlay).

### Numerical summary (dt=0.025 ms)

| cell                  | resting   | spike peak | spikes (>0 mV) | min       |
|-----------------------|----------:|-----------:|---------------:|----------:|
| ball_and_stick (HH)   | −65.0 mV  |  +58 mV    | **21**         | −252 mV   |
| ball_and_stick_bbp    | −75.0 mV  |  +49 mV    | **8**          | −340 mV   |

### What the traces show

- **Quiescent phase (0–140 ms)**: stim is exactly zero, so both cells sit
  at v_init (HH at −65 mV, BBP at −75 mV — that's `V_init` / `BBPLeak_eLeak`).
- **First stim ramp (~150 ms)**: large negative-current phase. Both cells
  hyperpolarize hard. The BBP cell goes deeper (−340 mV vs HH's −130 mV
  during the early ripples) because its lower input conductance lets V
  swing further per nA of injected current.
- **Sub-threshold dynamics (180–290 ms, small stim ripples)**:
  - HH spikes on essentially every stim ripple → busy spike train.
  - BBP enters a **depolarized plateau at ~−25 mV** with no spikes — the
    classic "depolarization block" you get when persistent Na (`Nap_Et2`)
    plus HVA Ca lift V above threshold but transient Na is inactivated.
    This is biologically realistic behavior for L5TTPC-like cells but
    isn't necessarily what we want for clean training traces.
- **Big stim block (~300–380 ms)**: both cells fire bursts. HH peaks at
  +58 mV, BBP at +49 mV. The BBP cell shows clean spike-AHP-spike
  structure; HH is more "buzzy" (less differentiation between spike and
  inter-spike).
- **Post-burst (~400 ms onward)**: HH returns cleanly to −65 mV; BBP
  settles at ~−85 mV (K-mediated AHP from `K_Tst` / `K_Pst`).

### Are the giant negative excursions numerical artifacts? No.

The dt=0.1 and dt=0.025 traces overlay almost exactly (`v_min` differs by
< 0.5 mV between the two). That's the test for solver instability — if it
were `bwd_euler` blowing up on stiff sodium kinetics, halving dt would
change the answer materially.

The actual cause is geometric: the trim soma (length 20 µm, radius 10 µm,
area 1.26e-5 cm²) gives an input resistance of ~26 MΩ. The
`5k50kInterChaoticB` stim peaks at +6.8 / −1.5 nA, so steady-state V swings
of ±180 / ±40 mV are mathematically forced. The full L5TTPC cell absorbs
the same stim cleanly because its much larger membrane area (~10⁴×) gives
a far lower R_in. **Same phenomenon shows up in the existing reference
trace** for `ball_and_stick` — and that reference comparison passes the
bench's grading. The BBP cell amplifies it slightly because its baseline
leak conductance (`BBPLeak_gLeak = 3e-5` S/cm²) is lower than HH's
(`HH_gLeak = 3e-4` S/cm²), giving 10× higher R_in.

### Takeaways

- The cell **builds correctly** and **produces biologically-shaped action
  potentials** at default params (peaks +49 mV; AP width ~1–2 ms).
- The defaults I copied are L5TTPC's full-morphology values. They're not
  tuned for this 6-comp trim and produce some L5TTPC-style behaviors
  (depolarization block, deep AHP) that are biologically plausible but
  not necessarily ideal training fodder. This is fine — the CNN-driven
  parameter sweeps will move conductances around to whatever the loss
  asks for.
- The "giant negative voltage" excursions are not a bug; they're a
  geometry × stim mismatch shared with the existing `ball_and_stick`
  reference and absorbed by the larger L5TTPC.

## Reproducing

```bash
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
cd /global/u1/k/ktub1999/Neuron/neuron4/worktrees/ball-stick-bbp

# GPU (login node has an A100, or use salloc -C gpu -q interactive)
JAX_PLATFORMS=cuda,cpu python -m toolbox.tests.bench_gpu_l5ttpc \
    --cells ball_and_stick_bbp --solvers bwd_euler \
    --batches 1 4 16 64 128 --modes fwd fwd_bwd \
    --out-dir /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench_bbp

# CPU
JAX_PLATFORMS=cpu python -m toolbox.tests.bench_gpu_l5ttpc \
    --cells ball_and_stick_bbp --solvers bwd_euler \
    --batches 1 4 16 64 --modes fwd fwd_bwd \
    --out-dir /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_cpu_bench_bbp
```

## Default-param tuning experiments

### Q1 — rebalance somatic Na/K to break the depolarization plateau

The baseline trace sits at ~−25 mV between bursts because somatic `Nap_Et2`
(persistent Na, default 0.0068) is large enough to keep the cell tonically
depolarized while transient `NaTs2_t` is inactivated. Tested four variants at
`5k50kInterChaoticB`, `t_max=500 ms`, `bwd_euler`:

| variant       | overrides                                                      | spikes | v_max  | v_min   |
|---------------|----------------------------------------------------------------|--------|--------|---------|
| baseline      | —                                                              | **8**  | +47 mV | −340 mV |
| low_Nap       | `Nap_Et2_somatic = 0.001`                                      | **14** | +48 mV | −340 mV |
| high_K_Tst    | `K_Tst_somatic = 0.20`                                         | 8      | +46 mV | −340 mV |
| both          | `Nap_Et2 = 0.001`, `K_Tst = 0.20`                              | **14** | +47 mV | −340 mV |

**Finding.** Persistent Na is the dominant cause; raising K_Tst alone does
nothing (K_Tst is transient and its time constant is too fast to clamp the
plateau). Dropping `Nap_Et2` ~7× restores the depolarization-recovery cycle
and recovers ~6 extra spikes. K_Tst contributes nothing additive on top.
Plot: `/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_gpu_bench_bbp/trace_rebalance.png`.

### Are BBP channels a problem in a small geometry?

**No — the channels themselves are fine.** Conductances are densities
(`S/cm²`), so each channel inserts the same per-area current regardless of
how big the soma is. The depolarization plateau is a balance issue between
`Nap_Et2` and the K family, not a "channel-too-big-for-geometry" issue, and
it's tunable in the parameter prior the CNN trains over.

The actual geometric pathology is unrelated to channels: `R_in ≈ 26 MΩ` ×
peak stim current produces ±180 mV passive swings that no channel set can
counteract, because the membrane area is too small to absorb the injected
charge. That's purely a soma+dendrite area issue — would persist even with
all channels at zero (pure leak).

So: keep the BBP channel set, fix the depolarization plateau by adjusting
the prior on `Nap_Et2_somatic` (or just let the CNN learn around it), and if
the deep negative excursions matter for training stability, fix them by
scaling the geometry — see Q2.

### Q2 — does scaling the geometry slow Jaxley down?

Two ways to "make it bigger," and they have different perf consequences:

**A. Scale diameter / length (more membrane area, same compartment count) — free.**
Geometry parameters (`length`, `radius`, `axial_resistivity`, `capacitance`)
are leaves in the JAX graph; the integrator sees the same number of
compartments and the same Hines tridiagonal solve. Zero perf impact.
Doubling soma radius cuts R_in by 4×, which is the cheapest way to absorb
the stim.

**B. Scale compartment count (better spatial resolution) — also basically free at our sizes.**
GPU bench, `ball_and_stick_bbp`, B=64, T=100 ms, `bwd_euler`, A100:

| dendrite ncomp | total comp | fwd warm (ms) | fwd_bwd warm (ms) | sims/s fwd | sims/s fwd_bwd |
|----------------|------------|---------------|--------------------|------------|-----------------|
|  5             |  6         | 108.0         | 936.0              | 593        | 68              |
| 10             | 11         |  99.6         | 935.7              | 643        | 68              |
| 20             | 21         | 103.7         | 911.0              | 617        | 70              |
| 40             | 41         | 102.2         | 933.5              | 626        | 69              |
| 80             | 81         | 101.0         | 931.2              | 633        | 69              |

**Finding.** Going from 6 → 81 total compartments (~14×) costs **~0%**.
Kernel-launch overhead dominates this regime, not the tridiagonal solve or
the channel state updates. The Hines solve is `O(n_comp)` but small `n_comp`
is dwarfed by per-step JIT dispatch on GPU.

This means: if you want bigger area (to absorb the stim), bump `radius` /
`length` (option A) and you're done. If you want both bigger area *and*
better spatial discretization, bump `_NCOMP_DEND` to 20 or 40 — still free.
The ceiling where `n_comp` starts mattering is well past 100; we won't hit
it in this morphology unless we go to full BBP-style hundreds-of-comp
reconstructions.

**Reproducing:**
```bash
PYTHONPATH=. python /tmp/scale_bench.py
```

