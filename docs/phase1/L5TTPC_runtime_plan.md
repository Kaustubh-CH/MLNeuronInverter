# L5TTPC runtime improvement plan

## Baseline (from `scripts/bench_solvers.slr`, single-thread CPU)

| solver           | B=1     | B=4     | B=16    | B=64    |
|------------------|--------:|--------:|--------:|--------:|
| bwd_euler        | 5.24    | 5.39    | 5.36    | 7.35    |
| crank_nicolson   | 5.21    | 5.40    | 5.17    | 7.33    |
| exp_euler        | 0.41    | 0.34    | 0.33    | 0.33    |
| fwd_euler        | UNSTABLE | — | — | — |

(Units: sims/sec.) `dt = 0.1 ms`, `t_max = 100 ms`. Compile (cold) cost
~24 s per (cell, solver) combination.

Compared to ball_and_stick (~7000 sims/sec at B=64), L5TTPC is ~1000× slower.
Geometry dominates: 492 apical compartments + ~150 basal + ~60 axon + 1 soma
× 4 channels each = thousands of ODE state variables per sim.

Thread scaling on ball_and_stick is essentially flat (4404 → 4564 sims/sec
across 1 → 8 threads). L5TTPC almost certainly scales no better — XLA's CPU
backend is single-stream-bound on the implicit linear solve. So the path
forward is **batch parallelism**, **GPU**, and **reducing work per sim** —
not more CPU cores.

## Knobs, ranked by expected impact / effort

### A. Reduce compartment count (biggest lever, easy)

L5TTPC `_NCOMP = 4` for every branch. Apical tufts have many short terminal
branches that contribute almost nothing to the soma trace. Two complementary
sub-knobs:

- **A1. `_NCOMP = 2` everywhere** — halves the number of compartments
  globally. Expected ~1.8–2× speedup. Fidelity hit will show in distal-apical
  spikelet propagation, mostly invisible from soma. **Try first, measure.**
- **A2. Per-branch ncomp via the `d_lambda` rule** — 1–3 comps for short
  branches, 5–10 for long. Matches NEURON practice. Requires custom SWC
  pre-processing or a thin wrapper around `jx.read_swc`. ~1.5× speedup, with
  better fidelity than uniform `_NCOMP = 2`. Bigger code change.

### B. Apical-tuft pruning

L5TTPC apical tufts have many sub-µm terminal segments. Merging branches
shorter than e.g. 10 µm reduces compartment count 30–50 % with negligible
soma-trace impact. SWC pre-pass — write a small `prune_swc()` helper that
reads `_SWC_PATH`, removes leaves below a length threshold, writes a pruned
SWC alongside, and switch `_SWC_PATH` to it. ~1.3–1.5× on top of A.

### C. Lock in `bwd_euler` (defensive)

Bench shows `exp_euler` is 22× slower on L5TTPC. Already the default in
`JaxleyBridge` (`solver="bwd_euler"` hard-coded), but the design YAML for
Phase 2 should pin it explicitly so a future config change doesn't silently
regress.

### D. dt trade with the fidelity backlog

Current `_DT = 0.1`. Coarsening to 0.2 or 0.25 ms gets us 2–2.5× speedup but
worsens the spike-timing drift currently at 94 mV max|Δ|. **Don't move on
this until the fidelity items in `L5TTPC.md` are addressed** — going coarser
would mask the biophysics fixes.

### E. GPU + larger batch

CPU bench shows linear scaling with batch (5.36 sims/sec at B=16 →
7.35 sims/sec at B=64 — modest amortization of per-step overhead). On GPU
the picture changes completely: ball_and_stick at B=64 hit ~23 sims/sec on
1× A100 in the existing GPU bench (bench_jaxley_cells.py). Re-run the
solver-sweep bench on GPU once Phase 2 is scaffolded so we know the actual
L5TTPC-GPU speedup vs CPU. Expected 30–50× from batch parallelism alone.

### F. Custom solver (low priority, large code change)

`bwd_euler` does a tridiagonal-block solve per step. For HH-only single-cell
geometries jaxley uses Thomas algorithm; for branched morphologies it falls
back to a more general sparse solver. An L5TTPC-shaped sparsity pattern is
a tree, so Thomas-on-tree (Hines) is asymptotically optimal — if jaxley
isn't already using it, asking the jaxley devs about per-cell solver
selection is the cheap escalation.

## Suggested order of operations

1. **A1: `_NCOMP = 2`.** One-line change, smoke-test, re-bench. Expected
   ~13 sims/sec at B=64.
2. **B: apical pruning at 10 µm leaf threshold.** New helper, regenerate
   pruned SWC once. Expected ~17–20 sims/sec at B=64.
3. **GPU re-bench** of all (cell × solver × batch) combos. Confirms the
   CPU→GPU multiplier for L5TTPC specifically.
4. **A2: d_lambda-rule ncomp.** Defer until A1 + B fidelity is verified.
5. **F: solver questions to jaxley devs** if 3–4 don't get us into the
   1000+ sims/sec range we'd want for Phase 5.

## Verification

For each step above:

1. `bash scripts/bench_solvers.slr` (CPU; cheap reproducible single-thread bench).
2. `python -m toolbox.tests.bench_jaxley_cells L5TTPC` (GPU; throughput +
   reference comparison).
3. Update the baseline table at the top of this file with the new sims/sec.
4. Update the max|Δ| / rmse line in `L5TTPC.md` if the change touches
   fidelity (A2, B, D all do).
