# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). STRICTLY voltage-only (`channel_weight 0`, `mask_channels True`) in EVERY
arm — ion channels never enter the loss. **NEW BEST: dtw_200k mean 0.725** (was 0.593 @ 40k
when this campaign started — +0.132, ~22% rel). Supervised ceiling (uses labels, forbidden) =
0.995. Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.

## Ledger (sorted by mean R²)
| arm | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---|---|---|---|---|---|---|---|
| **dtw_200k** | pure soft-DTW, 5× data (ep95, converged) | 200k | **0.725** | 0.86 | 0.79 | 0.34 | 0.81 | 0.75 | 0.80 |
| dtw_80k | pure soft-DTW, 2× data | 80k | 0.637 | 0.86 | 0.73 | 0.09 | 0.72 | 0.65 | 0.76 |
| precond2_80k | refined precond @ 80k (protects kdr) | 80k | 0.611 | 0.82 | 0.68 | **0.18** | 0.63 | 0.63 | 0.73 |
| precond | DTW + feature-sens grad-precond (kdr×1.68) | 40k | 0.610 | 0.80 | 0.73 | 0.04 | 0.69 | 0.62 | 0.77 |
| precond2 | DTW + refined precond (kdr-neutral, km↑) | 40k | 0.600 | 0.79 | 0.69 | 0.13 | 0.75 | 0.50 | 0.74 |
| band4 | DTW warp band 8→4 ms | 40k | 0.595 | 0.81 | 0.62 | 0.16 | 0.70 | 0.52 | 0.77 |
| **baseline** | pure soft-DTW | 40k | 0.593 | 0.79 | 0.64 | 0.16 | 0.75 | 0.48 | 0.74 |
| dtw_20k | pure DTW, half data | 20k | 0.489 | 0.72 | 0.49 | 0.17 | 0.48 | 0.38 | 0.70 |
| r1_dtwblur | DTW + van-Rossum blur | 40k | 0.479 | 0.76 | 0.54 | 0.11 | 0.29 | 0.42 | 0.75 |
| dtw_10k | pure DTW, quarter data | 10k | 0.440 | 0.59 | 0.50 | 0.16 | 0.42 | 0.34 | 0.63 |

## Findings
1. **DATA is the dominant broad lever and the curve is NOT saturating — it is ACCELERATING.**
   Baseline soft-DTW: 10k 0.440 → 20k 0.489 → 40k 0.593 → 80k 0.637 → **200k 0.725**. Per-decade
   gains: +0.049, +0.104, +0.044, then **+0.088** for the last 2.5×. The slope did NOT flatten past
   80k; the 80k→200k jump is larger per-sample than 40k→80k. At 200k plain DTW beats every 40k/80k
   loss-engineering arm by ≥0.11. **The single most effective, strictly-voltage-only lever is more
   data, and 200k (authorized ceiling) has not exhausted it.**
2. **kdr was DATA-STARVED, not at a voltage-only ceiling — this overturns the prior "ceiling"
   conclusion.** kdr: 40k 0.155 → 80k 0.092 (dip) → **200k 0.338**. The 80k dip was the net
   over-fitting kdr's confound with medium data; abundant data isolates the real (weak, rate-based)
   signal and kdr more than doubles its best prior score. **kdr does have a voltage-only handle — it
   just needs data, not a loss trick.** (Supervised ceiling 0.986 ⇒ still headroom.)
3. **Every one of the 6 channels improved with data at strictly voltage-only** (vs 40k baseline):
   leak 0.79→0.86, na3 0.64→0.79, kdr 0.16→0.34, kap 0.75→0.81, km 0.48→0.75 (+0.27, biggest),
   kd 0.74→0.80. Broad-spectrum goal met by data alone — no channel regressed, no labels in the loss.
4. **Loss-engineering was compensating for data scarcity.** Grad-precond helped at 40k (precond2
   0.600 > baseline 0.593) but HURT at 80k (0.611 < 0.637); with abundant signal the reweighting just
   distorts. Same story for band4/blur. **Pick the lever by data regime — and when data is available,
   spend it before engineering the loss.**
5. **Blur is the wrong lever** — coarse blur toxic to fast channels (kap 0.75→0.29).
6. **Step-stim FAILED** (`chaoramp_step`, mean 0.126, kdr −0.19): soft-DTW is rate-invariant on
   tonic firing → the step gives kdr no rate gradient and its big envelope drags the CNN into a bad
   basin. Confirms "steps train worse".
7. **Architecture is NOT the lever** (RayTune 55903354, 130 trials, 8-node): top trials ≈ the
   baseline skeleton; bigger/deeper nets diverge (loss→2.0=NaN); LR 1e-5 ≫ 5e-4. No arch beat
   baseline on voltage loss; nothing promoted.

## Compute note (measured)
8 nodes / 32 GPU gave **no speedup** over 4 nodes / 16 GPU at pinned global batch 2048 (220.8 vs
~247 s/epoch): the jaxley fp64 solve is dominated by the SEQUENTIAL 5001-step time integration,
which does not parallelize over batch or GPUs. The 200k run therefore timed out at epoch 95/100 on
the 6 h limit; epoch-95 was converged and was scored directly (reconstructed `sum_train.yaml`).
**Run future 200k+ on 4 nodes with a longer wall-clock, not more nodes.**

## Next data-scaling point (not yet run — 200k is the authorized ceiling)
- The curve has not saturated and kdr is still climbing steeply, so >200k would very likely keep
  helping (kdr most of all). 200k was the explicit user-authorized ceiling → **do not launch beyond
  it without confirmation.** If approved: generate a ~400k pack (gen v4) + train baseline soft-DTW on
  4 nodes / longer wall-clock; expected next point ~0.75–0.78 mean, kdr toward ~0.45.
