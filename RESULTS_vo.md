# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). STRICTLY voltage-only (`channel_weight 0`, `mask_channels True`) in EVERY
arm — ion channels never enter the loss. **Bar: mean 0.593, kdr 0.155.** Supervised ceiling
(uses labels, forbidden) = 0.995. Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.

## Ledger (sorted by mean R²)
| arm | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---|---|---|---|---|---|---|---|
| **precond** | DTW + feature-sens grad-precond (kdr×1.68) | 40k | **0.610** | 0.80 | 0.73 | 0.04 | 0.69 | 0.62 | 0.77 |
| **precond2** | DTW + refined precond (kdr-neutral, km↑) | 40k | 0.600 | 0.79 | 0.69 | 0.13 | 0.75 | 0.50 | 0.74 |
| band4 | DTW warp band 8→4 ms | 40k | 0.595 | 0.81 | 0.62 | 0.16 | 0.70 | 0.52 | 0.77 |
| **baseline** | pure soft-DTW | 40k | 0.593 | 0.79 | 0.64 | 0.16 | 0.75 | 0.48 | 0.74 |
| dtw_20k | pure DTW, half data | 20k | 0.489 | 0.72 | 0.49 | 0.17 | 0.48 | 0.38 | 0.70 |
| r1_dtwblur | DTW + van-Rossum blur | 40k | 0.479 | 0.76 | 0.54 | 0.11 | 0.29 | 0.42 | 0.75 |
| dtw_10k | pure DTW, quarter data | 10k | 0.440 | 0.59 | 0.50 | 0.16 | 0.42 | 0.34 | 0.63 |

## Findings
1. **Grad-precond is the broad-spectrum lever** — reweighting per-channel gradients lifts the
   mean. `precond` (0.610) maxes the mean but boosting kdr's gradient crashes its calibration
   (kdr R² 0.044, though Pearson r held ~0.41) and dips kap. `precond2` (kdr-neutral, km-boost,
   0.600) is the **balanced broad win**: protects kap (0.751≈baseline), holds kdr (0.129), lifts
   na3/km. Don't boost kdr.
2. **kdr is at a single-stim information ceiling (~r0.44 / R²~0.15).** Nothing moves it up; its
   only handle is a non-specific firing rate. Its real fix is a complementary stim (RUNNING).
3. **Blur is the wrong lever** — coarse blur toxic to fast channels (kap 0.75→0.29).
4. **Data scaling helps** (baseline DTW): 10k 0.440 → 20k 0.489 → 40k 0.593 (monotone) → not
   saturated at 40k. Testing 80k (RUNNING).

## In flight
- **Session 4** `dtw_80k` — baseline DTW @ 80k (v2 pack) → extend the data-scaling curve.
- **Step-stim** `ca3_vo_dtw_chaoramp_step` — 2-stim (chaoticRamp + sustained step) voltage-only,
  the dedicated kdr fix (adds a clean tonic-rate readout). Parallel salloc.
- **RayTune** (8-node/32-GPU, regular queue) — architecture + dataset-size HPO, 128 Optuna/ASHA
  trials, strictly voltage-only. Screens by voltage loss; winners get full channel-R² eval.
