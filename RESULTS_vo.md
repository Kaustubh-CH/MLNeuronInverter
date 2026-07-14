# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). Strictly voltage-only (`channel_weight 0`, `mask_channels True`) in
every arm. **Bar: mean R² 0.593, kdr 0.155, voltage_mse_z ≤ 1.98, no channel regresses.**
Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.

| arm | recipe | mean R² | leak | na3 | kdr | kap | km | kd | volt_mse_z | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **A precond** | DTW + feature-sensitivity grad-precond (kdr×1.68) | **0.610** | 0.804 | 0.729 | 0.044 | 0.692 | 0.618 | 0.773 | 1.99 | best mean; kdr+kap regress |
| B band4 | DTW warp band 8→4 ms | 0.595 | 0.809 | 0.619 | 0.156 | 0.696 | 0.521 | 0.770 | 1.97 | ≈ baseline; kdr flat |
| baseline | pure soft-DTW | 0.593 | 0.788 | 0.642 | 0.155 | 0.753 | 0.483 | 0.738 | 1.98 | (bar) |
| R1 blur | DTW + coarse→fine van-Rossum blur | 0.479 | 0.763 | 0.537 | 0.115 | 0.286 | 0.419 | 0.751 | 2.02 | ❌ over-smooths |

## Findings (2026-07-13)
- **Grad-precond = broad-spectrum win on mean (0.610).** Lifts na3 +0.09, km +0.14, kd,
  leak. Cost: kdr crashes, kap dips. The kap dip recurs under *every* perturbation
  (precond/band4/blur all < baseline's 0.753) → baseline DTW is near kap's optimum.
- **kdr is at a voltage-only ceiling (~0.155).** band4 flat (0.156), gradient-boost
  ×1.68 *crashed* it (0.044), blur hurt it. Sensitivity showed kdr's only handle is a
  non-specific `mean_frequency` → the objective can't isolate it. Supervised gets 0.986
  (info is there); the self-supervised voltage objective cannot extract it on this stim.
- **Blur is the wrong lever** (coarse blur toxic to fast channels kap/na3).

## Open forks (need priority steer)
1. Lock the broad win: refine precond to hold ~0.61 mean while protecting kap + not
   crashing kdr (cheap). 
2. Dedicated kdr attempt: randomized smoothing in PARAM space (untried; ~2.7 h, 2×
   cost, uncertain — smoothing already backfired once as blur).
3. Accept kdr ceiling; report precond as the broad-spectrum improvement.
