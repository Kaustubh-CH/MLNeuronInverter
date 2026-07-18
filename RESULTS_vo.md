# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). STRICTLY voltage-only (`channel_weight 0`, `mask_channels True`) in EVERY
arm — ion channels never enter the loss. **NEW BEST: dtw_400k mean 0.736** (was 0.593 @ 40k
when this campaign started — +0.143, ~24% rel). Supervised ceiling (uses labels, forbidden) =
0.995. Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.
**BUT the mean now hides a split**: 200k→400k lifted 5/6 channels (kap 0.81→0.905!, kd, leak) yet
**kdr REGRESSED 0.338→0.219** — kdr does NOT track data (see Finding 2, corrected). And the
InterChaoticB run exposes a stimulus/observability trade-off (Finding 8) that motivates multi-stim.

## Transferable recipe — what worked for chaoticRamp (drop-in for InterChaoticB)
The winning voltage-only recipe is a small, portable set of choices. To move it to another
stim (e.g. `5k50kInterChaoticB`, which is the SAME 5001-pt / 500 ms / dt 0.1 ms length as
`5kChaoticRamp` ⇒ 1:1 transfer, same ~221 s/epoch), change ONLY `stim_name` + the data pack.
| # | change | value | why it worked |
|---|---|---|---|
| 1 | **loss = pure soft-DTW** | `dtw_weight 1`, `mse_weight 0`, `efel_weight 0`, `blur 0` | beats every loss trick; MSE/eFEL/blur all lost or diverged |
| 2 | DTW params | `dtw_gamma 0.1`, `dtw_band_ms 8`, `dtw_n_points 256` | 8 ms Sakoe-Chiba band carries the residual rate signal; band 4 didn't help |
| 3 | **DATA scaling** (dominant lever for 5/6) | 40k→400k, returns now diminishing | +0.14 mean; lifts leak/na3/kap/km/kd; **kdr does NOT track data** (Finding 2) |
| 4 | **150 epochs** | was 100 | 200k val loss still dropping at ep94 (−0.0540→−0.0571), LR 9e-6 |
| 5 | fp64 Jaxley solve | `fp64: True` | fp32 → NaN backward on high-gNa draws, NCCL propagates NaN across ranks |
| 6 | `clamp_unit_tanh` | `True` | bounds CNN param outputs into the trained range |
| 7 | global batch 2048 | `batch_size 64` ×32 GPU, `const_local_batch True` | matches the whole sweep; LR 1e-4 adam, `clip_grad_norm 1.0` |
| 8 | `serialize_stims: True` | probe axis → CNN channels | standard CA3 loader path |
| 9 | backbone (unchanged) | 2 CNN blocks [30,90,180] k4 p4; FC [512,512,512,256,128] drop 0.04 | RayTune (130 trials) found nothing better; bigger diverges |
| — | **strictly voltage-only** | `channel_weight 0`, `mask_channels True`, `voltage_weight 1` | hard constraint — ion channels NEVER in the loss |
| ✗ | do NOT: grad-precond at scale · van-Rossum blur · step-stim · bigger net · efel loss | — | precond helps only <80k; blur toxic to fast channels; steps poison DTW; efel diverges |

**InterChaoticB job** (submitted): `ca3_vo_interchaoticB_dtw_8n.hpar.yaml` (rows 1–9 verbatim,
`stim_name: 5k50kInterChaoticB`, pack `ca3_5kinterchaoticB_v1` 200k) + `gen_interchaoticB_v1.slr`
+ `train_interchaoticB_200k_8n.slr` (150 ep).

## Ledger (sorted by mean R²)
| arm | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---|---|---|---|---|---|---|---|
| **dtw_400k** | pure soft-DTW, 10× data (best mean, best trace) | 400k | **0.736** | 0.89 | 0.80 | **0.22** ⬇ | **0.91** | 0.76 | 0.84 |
| precondkdr_200k | DTW + grad-precond **kdr×2.5** (150ep) | 200k | 0.731 | 0.87 | 0.77 | **0.27** ⬇ | 0.85 | 0.82 | 0.81 |
| **dtw_200k** | pure soft-DTW, 5× data (ep95, converged) | 200k | **0.725** | 0.86 | 0.79 | **0.34** | 0.81 | 0.75 | 0.80 |
| dtw_80k | pure soft-DTW, 2× data | 80k | 0.637 | 0.86 | 0.73 | 0.09 | 0.72 | 0.65 | 0.76 |
| precond2_80k | refined precond @ 80k (protects kdr) | 80k | 0.611 | 0.82 | 0.68 | **0.18** | 0.63 | 0.63 | 0.73 |
| precond | DTW + feature-sens grad-precond (kdr×1.68) | 40k | 0.610 | 0.80 | 0.73 | 0.04 | 0.69 | 0.62 | 0.77 |
| precond2 | DTW + refined precond (kdr-neutral, km↑) | 40k | 0.600 | 0.79 | 0.69 | 0.13 | 0.75 | 0.50 | 0.74 |
| band4 | DTW warp band 8→4 ms | 40k | 0.595 | 0.81 | 0.62 | 0.16 | 0.70 | 0.52 | 0.77 |
| **baseline** | pure soft-DTW | 40k | 0.593 | 0.79 | 0.64 | 0.16 | 0.75 | 0.48 | 0.74 |
| dtw_20k | pure DTW, half data | 20k | 0.489 | 0.72 | 0.49 | 0.17 | 0.48 | 0.38 | 0.70 |
| r1_dtwblur | DTW + van-Rossum blur | 40k | 0.479 | 0.76 | 0.54 | 0.11 | 0.29 | 0.42 | 0.75 |
| dtw_10k | pure DTW, quarter data | 10k | 0.440 | 0.59 | 0.50 | 0.16 | 0.42 | 0.34 | 0.63 |
| dtw_ica_200k | pure DTW, **InterChaoticB** stim (≠ chaoticRamp) | 200k | 0.219 | **0.97** | −0.27 | −0.12 | 0.16 | −0.37 | **0.94** |

_dtw_ica_200k is a DIFFERENT stimulus — not comparable on mean. Its per-channel split is the point
(Finding 8): near-ceiling leak/kd, failed na3/kdr/km/kap._

## Findings
1. **DATA is the dominant broad lever for the MEAN and 5/6 channels — but returns are now
   DIMINISHING, not accelerating (corrected by 400k).** Mean soft-DTW: 10k 0.440 → 20k 0.489 →
   40k 0.593 → 80k 0.637 → 200k 0.725 → **400k 0.736**. The 200k→400k step (a full 2× data) added only
   **+0.011** — vs +0.088 for 80k→200k. So the earlier "accelerating" read was a 200k artifact; the
   curve is bending over. 400k still gives the **best mean AND best trace** (voltage_mse_z 1.90, spike
   diff 5.8 — both records) via kap (0.81→**0.905**), kd (0.80→0.84), leak (0.86→0.89). **More data
   still helps broadly, but the per-sample payoff past 200k is small — data alone won't reach the 0.995
   ceiling.**
2. **kdr does NOT track data — this CORRECTS the earlier "kdr was data-starved" claim.** kdr across
   data at fixed recipe: 10k 0.165 → 20k 0.167 → 40k 0.155 → 80k 0.092 → 200k **0.338** → 400k **0.219**.
   That is NOISE around ~0.15–0.34, not a climb; the 200k 0.338 was a favorable draw, and doubling to
   400k REGRESSED it to 0.219. **kdr is objective/confound-limited, NOT data-limited** — it shares its
   only handle (firing rate / ISI / slow-AHP) with km & kd, and neither more data NOR a gradient boost
   (precond ×2.5 → 0.266) breaks that confound. This reinstates a *reframed* ceiling: kdr's lever is the
   OBJECTIVE (add a differentiable rate/ISI term so DTW stops being rate-blind) or OBSERVABILITY
   (multi-stim, to separate the three K currents) — not data, not reweighting.
3. **Data lifts the OTHER five channels, kdr excepted** (40k → 400k best): leak 0.79→0.89, na3
   0.64→0.80, kap 0.75→**0.905** (biggest), km 0.48→0.76, kd 0.74→0.84 — all monotone-ish up. Only kdr
   fails to follow. The broad-spectrum goal is largely met by data for 5/6; kdr is the lone holdout and
   needs a targeted objective/observability fix.
4. **Loss-engineering was compensating for data scarcity.** Grad-precond helped at 40k (precond2
   0.600 > baseline 0.593) but HURT at 80k (0.611 < 0.637); with abundant signal the reweighting just
   distorts. Same story for band4/blur. **Pick the lever by data regime — and when data is available,
   spend it before engineering the loss.**
   - **CONFIRMED at 200k (55949843):** `precondkdr_200k` (kdr grad ×2.5) scored mean 0.731 — a +0.006
     sliver over plain dtw_200k — but **kdr itself REGRESSED 0.338 → 0.266**; the mean rose only because
     km (+0.06) and kap (+0.05) caught the redistributed gradient. Boosting the low-sensitivity channel's
     own gradient makes it *worse*, the identical signature as 40k (kdr×1.68 → kdr 0.044). **The kdr grad
     boost is a dead end at every data scale; kdr's lever is data (0.09→0.34), full stop.** Plain DTW
     remains the clean reference (higher kdr, one fewer knob).
5. **Blur is the wrong lever** — coarse blur toxic to fast channels (kap 0.75→0.29).
6. **Step-stim FAILED** (`chaoramp_step`, mean 0.126, kdr −0.19): soft-DTW is rate-invariant on
   tonic firing → the step gives kdr no rate gradient and its big envelope drags the CNN into a bad
   basin. Confirms "steps train worse".
7. **Architecture is NOT the lever** (RayTune 55903354, 130 trials, 8-node): top trials ≈ the
   baseline skeleton; bigger/deeper nets diverge (loss→2.0=NaN); LR 1e-5 ≫ 5e-4. No arch beat
   baseline on voltage loss; nothing promoted.
8. **InterChaoticB reveals a stimulus/observability trade-off — and the strongest case yet for
   MULTI-STIM.** Same recipe, same 200k, stim = `5k50kInterChaoticB` (76% near-rest, few spikes):
   mean 0.219, but the SPLIT is the point — leak **0.974** and kd **0.937** (near the 0.995 supervised
   ceiling!) while na3 **−0.27**, kdr **−0.12**, km **−0.37**, kap 0.16 all FAIL. Its trace overlap is
   *excellent* (voltage_mse_z **0.97** vs chaoticRamp's ~1.9; spike-diff **0.99** vs ~6) — because a
   mostly-subthreshold trace is easy to reproduce AND perfectly constrains the passive/slow channels
   (leak, kd), but has almost no spikes ⇒ the spike-shaping channels (na3, kap) and the rate channels
   (kdr, km) are UNOBSERVABLE. **This is the exact mirror of chaoticRamp** (lots of spikes → na3/kap
   recover, leak/kd weaker, trace overlap poor from chaos). The two stimuli are COMPLEMENTARY:
   InterChaoticB owns leak/kd, chaoticRamp owns na3/kap. Jointly observing both (multi-stim input)
   should recover all six far better than either alone — and directly attacks kdr's confound (Finding 2)
   by giving the three K currents independent views. **Resolves the user's earlier puzzle** (great trace
   overlap ↔ few spikes ↔ can't see the spiking channels; the chaos that ruins chaoticRamp's overlap is
   the very spiking that makes na3/kap observable there).

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
