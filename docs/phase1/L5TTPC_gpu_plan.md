# L5TTPC GPU performance plan

The Phase 2 hybrid loss runs `jx.integrate` on the **GPU** as part of every
training step (or every k-th step). What we care about is **wall-clock per
training step** and **GPU memory headroom**, not standalone CPU sims/sec.

This doc supersedes `L5TTPC_runtime_plan.md` for anything Phase 2+. The CPU
plan stays as a reference for offline development on login nodes.

## Bench (`scripts/bench_gpu.slr`)

`toolbox/tests/bench_gpu_l5ttpc.py` — measures forward and forward+backward
through `jx.integrate` directly (no torch round-trip), at `dt = 0.1 ms`,
`t_max = 100 ms`, on 1× A100 (80 GB). Sweeps cells `{ball_and_stick,
L5TTPC}`, solvers `{bwd_euler, crank_nicolson}`, batches
`{1, 4, 16, 64, 128}`, modes `{fwd, fwd_bwd}`. Loss is `sum(v²)` — scalar,
exercises the full Jacobian-on-time-axis backward pass.

### Baseline numbers (NCOMP=4, on 80 GB A100)

Forward only, 100 ms / 1000 steps, warm:

| cell           | solver         | B   | warm (ms) | sims/sec | cold (s) |
|----------------|----------------|----:|---------:|---------:|---------:|
| ball_and_stick | bwd_euler      | 128 |  24      | 5,277    | 1.1 |
| ball_and_stick | crank_nicolson | 128 |  21      | 6,126    | 1.0 |
| L5TTPC         | bwd_euler      |   1 | 954      | 1.05     | 10.5 |
| L5TTPC         | bwd_euler      |  16 | 983      | 16.3     | 10.8 |
| L5TTPC         | bwd_euler      |  64 | 1,272    | 50.3     | 10.9 |
| L5TTPC         | bwd_euler      | 128 | 1,481    | 86.5     | 12.3 |
| L5TTPC         | crank_nicolson | 128 | 1,509    | 84.8     | 12.5 |

Forward + backward (the Phase 2 cost), warm:

| cell           | solver         | B   | warm (ms) | sims/sec | peak mem |
|----------------|----------------|----:|---------:|---------:|---------:|
| ball_and_stick | bwd_euler      | 128 | 182      | 705      | 54 MB |
| ball_and_stick | crank_nicolson | 128 | 182      | 705      | 54 MB |
| L5TTPC         | bwd_euler      |   1 | 4,765    | 0.21     | 1.48 GB |
| L5TTPC         | bwd_euler      |   4 | 5,487    | 0.73     | 5.97 GB |
| L5TTPC         | bwd_euler      |  16 | 6,206    | 2.58     | 23.86 GB |
| L5TTPC         | bwd_euler      |  64 | OOM (95 GB requested) |  | — |
| L5TTPC         | bwd_euler      | 128 | OOM (190 GB) |  | — |
| L5TTPC         | crank_nicolson |  16 | 6,268    | 2.55     | 23.92 GB |

Cold compile cost is the secondary surprise: **L5TTPC fwd_bwd takes
~64–70 s to JIT** the first time. That's 5× the forward-only compile and
~70× the ball_and_stick fwd_bwd compile. Persistent compilation cache (T2)
is therefore not optional — it's the difference between "1 minute of dead
time per job" and "1 second."

### Read-out

- **fwd_bwd / fwd ratio @ B=16, L5TTPC**: `6206 ms / 983 ms ≈ 6.3×`. This
  is normal for an implicit ODE integrator without checkpointing — every
  forward step's solve has to be reverse-mode-differentiated, ~5–8× the
  forward cost. Below 5× would mean jaxley is checkpointing aggressively;
  above 10× would mean it's rematerialising the forward inside backward.
  6.3× is healthy, **so memory not compute is what's hurting us**.
- **L5TTPC / ball_and_stick fwd_bwd ratio @ B=16**: `6206 / 173 ≈ 36×`.
  Geometry dominates; backprop scales with compartment count.
- **Largest B that fits on 80 GB A100**: **B = 16** for L5TTPC fwd_bwd at
  NCOMP=4. Theoretically B=32 should fit (~48 GB) — bench didn't probe it
  because the matrix jumped from 16 to 64. Worth checking later.
- **Memory per sim (fwd_bwd)**: ~1.49 GB/sim for L5TTPC at NCOMP=4. This
  is the budget the C1 / T3 optimizations have to crush.
- **Solver choice on GPU**: bwd_euler and crank_nicolson are
  indistinguishable in cost on both forward and backward. Pick on
  accuracy.
- **Memory readings note**: `peak_bytes_in_use` is monotonic and our reset
  call doesn't take effect (jax 0.x quirk). Trust only the first row of
  each (cell, mode) block; subsequent rows inherit the high-water mark.

### NCOMP=2 ablation (`bench_gpu_ncomp2`, 40 GB A100, partial)

| solver    | mode    | B  | warm (ms) | sims/sec | peak mem | vs NCOMP=4 |
|-----------|---------|---:|---------:|---------:|---------:|-----------|
| bwd_euler | fwd     |  1 |   628    |   1.59   | 1 MB     | **1.5× faster** |
| bwd_euler | fwd     | 16 |   662    |  24.2    | 3 MB     | **1.5× faster** |
| bwd_euler | fwd     | 64 |   779    |  82.1    | 13 MB    | **1.6× faster** |
| bwd_euler | fwd     |128 |   952    | 134.5    | 28 MB    | **1.6× faster** |
| bwd_euler | fwd_bwd |  1 | 3,285    |   0.30   | **0.74 GB** | **1.45× faster, 2× less mem** |
| bwd_euler | fwd_bwd |  4 | 3,610    |   1.11   | **2.99 GB** | 1.5× faster, 2× less mem |
| bwd_euler | fwd_bwd | 16 | 4,099    |   3.90   | **11.94 GB**| 1.5× faster, 2× less mem |
| bwd_euler | fwd_bwd | 64 | OOM (48 GB on 40 GB A100) |  |  | — |

NCOMP=2 confirms the C1 hypothesis: **half the compartments → exactly half
the memory, 1.5× the speed**. On an 80 GB A100, NCOMP=2 should fit
**B = 32 (≈ 24 GB)**, possibly B = 48 (≈ 36 GB). At B=64 (~48 GB) it
should also fit on 80 GB hardware — the OOM here was hardware-constrained,
not algorithmic.

### 4-way ablation (complete)

`scripts/bench_gpu_4way.slr` ran NCOMP=1, NCOMP=4 + ckpt[10,100], NCOMP=2 +
ckpt[10,100], and NCOMP=4 + ckpt[100,10] in parallel on 4× 40 GB A100s.
Final L5TTPC × bwd_euler × fwd_bwd numbers:

| config                    | B=16 ms | B=64 ms | B=128 ms | B=128 sims/s | B=128 peak mem |
|---------------------------|--------:|--------:|---------:|-------------:|---------------:|
| NCOMP=4 baseline          | 6,206   | OOM     | OOM      | —            | —              |
| NCOMP=2                   | 4,099   | OOM(40) | OOM(40)  | —            | —              |
| NCOMP=1                   | 2,848   | 3,506   | OOM(40)  | —            | —              |
| NCOMP=4 + ckpt[10,100]    | 7,262   | 9,001   | 11,236   | 11.4         | 20.4 GB        |
| NCOMP=4 + ckpt[100,10]    | 7,298   | 9,064   | 11,288   | 11.3         | **4.7 GB**     |
| NCOMP=2 + ckpt[10,100]    | 4,817   | 5,755   | 6,629    | **19.3**     | 10.2 GB        |

### Surprise finding: chunking direction matters for memory, not speed

`checkpoint_lengths=[10, 100]` (inner length 100) and `[100, 10]` (inner
length 10) take **virtually identical wall-time** (~11.3 s at NCOMP=4,
B=128) but use **4.4× different memory** (20.4 GB vs 4.7 GB).

The "inner" loop length is what gets re-materialised during backward, so
fewer timesteps per inner block → fewer activations stored per block.
**Always use `[outer, inner]` with the smaller value second.** Recommended:
`checkpoint_lengths=[100, 10]`.

A combined NCOMP=2 + ckpt[100,10] was not measured — predicted ~5 GB at
B=128 with the same ~19 sims/s as the [10,100] variant. **Worth a
single-config follow-up run before Phase 2 commits.**

## Cell-level optimizations (priority order)

### C1. NCOMP=2 — confirmed by ablation

Halves the per-sim memory and gives 1.5× speed. **One-line change**:
`_NCOMP = 2` in `toolbox/jaxley_cells/l5ttpc.py`. Validated by
`bench_gpu_ncomp2`.

**Trade-off**: spike-time accuracy in distal apical drops modestly. Run
`python -m toolbox.tests.bench_jaxley_cells L5TTPC` after to confirm
max|Δ| vs NEURON ref doesn't grow > 5 mV — current baseline is ~94 mV
(see `L5TTPC.md` item 1; mostly dt-driven, not NCOMP-driven).

### C2. Apical pruning — moderate, deferred

Pre-process the SWC: drop leaves shorter than 10 µm. Expected ~30–50 %
fewer apical compartments → another ~1.3–1.5× on top of C1. Modest
fidelity hit. Defer until C1 + checkpointing land — they may be enough.

### C3. Lock solver = bwd_euler

Bench confirms bwd_euler ≡ crank_nicolson in cost on GPU. Pin in Phase 2
design YAML (`solver: bwd_euler`) so config drift can't introduce
`exp_euler` (catastrophically slow on L5TTPC).

### C4. Keep dt = 0.1 ms

Coarsening would 2–4× the cost (every step is the work). Don't move until
fidelity work in `L5TTPC.md` is done and we *need* the timestep.

### C5. Record only soma — already true

`record_fn = cell.soma.comp(0).record()`. No change.

## Training-level optimizations (priority order)

These touch `toolbox/JaxleyBridge.py`, `toolbox/Trainer.py`, and the design
YAML — out of scope for cell files but the biggest wall-clock wins.

### T1. JIT cache reuse across batch sizes

If the training loop varies `B`, every distinct `B` triggers a fresh XLA
recompile — **~70 s per recompile** for L5TTPC fwd_bwd (bench data).
At 4 ranks × 1 short tail batch per epoch × 50 epochs = ~4 hours of
recompile overhead per run.

**Fix**: pad the last batch to the canonical `B` before
`JaxleyBridge.simulate_batch`. Trainer-level change. Verify by running
once with `JAX_LOG_COMPILES=1` — every recompile prints a line.

### T2. Persistent compilation cache

Set `JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc` in `batchShifter.slr`.
Drops 70 s cold compile to <1 s on every job restart. **No-brainer.**

Caveat for DDP: ranks must not race on cache writes. Standard pattern is
to populate the cache via a one-shot warmup job before the real training
job (T6).

### T3. Gradient checkpointing through `jx.integrate`

`jx.integrate` accepts `checkpoint_lengths=List[int]` (jaxley 0.13+).
With `t_max = 100 ms` and `dt = 0.1 ms` we have 1000 steps. Two-level
checkpointing `[10, 100]` saves ~10× memory at ~10–20% time cost.

**This is the lever that probably unlocks B=64+ for L5TTPC fwd_bwd.** The
4-way ablation will tell us by how much. If it works, T3 + T1 together
get us into the regime Phase 2 actually needs.

Implementation: thread an optional `checkpoint_lengths` arg through
`JaxleyBridge.simulate_batch` → `jx.integrate`. ~10-line change.

### T4. fp16 / bf16 forward, fp32 backward

`jax.config.update("jax_enable_x64", False)` is already implicit; question
is whether to push further to bf16 for forward integration. Expected:
1.5–2× speed, half memory. Risk: bwd_euler can be precision-sensitive
near spike threshold; could introduce subtle drift.

Defer until C1 + T1 + T2 + T3 are landed and we know we still need more.

### T5. Every-k-step jaxley loss

Biggest single training-time lever, but a Phase 2 design decision, not an
L5TTPC change. With k = 4, jaxley loss fires on 25 % of steps — 4× more
training throughput, 4× less voltage-loss gradient signal. Knob lives in
the design YAML.

### T6. DDP compile sharing via warmup job

With DDP across 4 GPUs, each rank compiles independently. Persistent cache
(T2) addresses this once populated, but cold first-run is still 4 × 70 s
in parallel. Add a one-line `compile_warmup.py` that imports + JITs the
configured cell, run it once before each new design's first real job.

## Suggested order (rev. after 4-way ablation)

1. **Bench first** — done (NCOMP=4 baseline + NCOMP=2 ablation + 4-way
   matrix complete).
2. **T3** (grad checkpointing): 1-hour bridge change, **biggest single
   win** — unlocks B=128 on a 40 GB A100. Use `[100, 10]` chunking, not
   `[10, 100]` (4.4× less memory, same speed).
3. **C1** (`_NCOMP = 2`): 5-minute cell change. On top of T3, gives
   another 1.5–2× speedup at B=128 (11.3 → 19.3 sims/s).
4. **T2** (persistent cache): 15-minute SLURM-script change — kills the
   90–180 s cold compile (yes, ckpt makes compile MUCH slower).
5. **T1** (pad last batch): 30-minute trainer change — kills per-epoch
   recompile tax.
6. **GPU re-bench** → measure NCOMP=2 + ckpt[100,10], the predicted best
   config. Update the table.
7. **T6** (warmup job for DDP): 30 minutes.
8. **C2** (apical pruning): if we still need more.
9. **T4** (bf16): only if everything above isn't enough.

## Validation checklist

For each landed change:

1. Re-run `bash scripts/bench_gpu.slr`. Update the baseline table.
2. Run `python -m toolbox.tests.bench_jaxley_cells L5TTPC` — fidelity
   reference comparison. Fail if max|Δ| vs NEURON grows > 5 mV.
3. For T1/T2: confirm `JAX_LOG_COMPILES=1` shows zero recompiles in
   steady state.
4. For T3: confirm `peak_mem_mb` drops as expected.
