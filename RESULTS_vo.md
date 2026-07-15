# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). STRICTLY voltage-only (`channel_weight 0`, `mask_channels True`) in EVERY
arm — ion channels never enter the loss. **NEW BEST: dtw_80k mean 0.637** (was 0.593 @ 40k).
Supervised ceiling (uses labels, forbidden) = 0.995. Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.

## Ledger (sorted by mean R²)
| arm | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---|---|---|---|---|---|---|---|
| **dtw_80k** | pure soft-DTW, 2× data | 80k | **0.637** | 0.86 | 0.73 | 0.09 | 0.72 | 0.65 | 0.76 |
| precond2_80k | refined precond @ 80k (protects kdr) | 80k | 0.611 | 0.82 | 0.68 | **0.18** | 0.63 | 0.63 | 0.73 |
| precond | DTW + feature-sens grad-precond (kdr×1.68) | 40k | 0.610 | 0.80 | 0.73 | 0.04 | 0.69 | 0.62 | 0.77 |
| precond2 | DTW + refined precond (kdr-neutral, km↑) | 40k | 0.600 | 0.79 | 0.69 | 0.13 | 0.75 | 0.50 | 0.74 |
| band4 | DTW warp band 8→4 ms | 40k | 0.595 | 0.81 | 0.62 | 0.16 | 0.70 | 0.52 | 0.77 |
| **baseline** | pure soft-DTW | 40k | 0.593 | 0.79 | 0.64 | 0.16 | 0.75 | 0.48 | 0.74 |
| dtw_20k | pure DTW, half data | 20k | 0.489 | 0.72 | 0.49 | 0.17 | 0.48 | 0.38 | 0.70 |
| r1_dtwblur | DTW + van-Rossum blur | 40k | 0.479 | 0.76 | 0.54 | 0.11 | 0.29 | 0.42 | 0.75 |
| dtw_10k | pure DTW, quarter data | 10k | 0.440 | 0.59 | 0.50 | 0.16 | 0.42 | 0.34 | 0.63 |

## Findings
1. **DATA is the dominant broad lever, and it beats every loss trick.** Baseline soft-DTW:
   10k 0.440 → 20k 0.489 → 40k 0.593 → **80k 0.637** (still climbing, slope flattening). At 80k
   plain DTW BEATS all 40k loss-engineering (precond 0.610, precond2 0.600). More data lifts
   **every channel except kdr**: km +0.17 (0.48→0.65, was data-limited), na3 +0.09, leak +0.07,
   kd/kap ≈flat. **Testing 160–200k (authorized ceiling).**
2. **Grad-precond REVERSES at scale.** It helped when data-limited (40k: precond2 0.600 > baseline
   0.593) but at 80k it HURTS the mean (0.611 < 0.637). It was compensating for data scarcity by
   rebalancing gradient attention; with abundant data all channels get enough signal and the
   reweighting just distorts. **Corollary: pick the lever by data regime — precond for small data,
   plain DTW for large.**
3. **kdr worsens with more data (0.16→0.09).** Confirms the ceiling at a deeper level: kdr's signal
   is so confounded that a better-fit net mis-calibrates it MORE. **precond2 is the only kdr
   protector** (0.179 @ 80k, best anywhere) but costs the mean — a genuine kdr-vs-mean tradeoff.
   If kdr specifically matters, precond2_80k; for broad mean, baseline dtw_80k.
4. **Blur is the wrong lever** — coarse blur toxic to fast channels (kap 0.75→0.29).
5. **Step-stim FAILED** (`chaoramp_step`, mean 0.126, kdr −0.19): soft-DTW is rate-invariant on
   tonic firing → the step gives kdr no rate gradient and its big envelope drags the CNN into a bad
   basin. Confirms "steps train worse".
6. **Architecture is NOT the lever** (RayTune 55903354, 130 trials, 8-node): top trials ≈ the
   baseline skeleton; bigger/deeper nets diverge (loss→2.0=NaN); LR 1e-5 ≫ 5e-4. No arch beat
   baseline on voltage loss; nothing promoted.

## In flight
- **200k data-scaling run** — generate a ~200k pack (authorized ceiling) + train baseline soft-DTW
  → the next data-scaling point (80k gave 0.637; is the curve still climbing or saturating?).
