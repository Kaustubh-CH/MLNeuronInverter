# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). STRICTLY voltage-only (`channel_weight 0`, `mask_channels True`) in EVERY
arm — ion channels never enter the loss. **NEW BEST: dtw_amp_200k mean 0.763** — pure soft-DTW +
a SMALL ramped soft-eFEL aux (`voltage_base`+`AP_amplitude`) at only 200k data (was 0.593 @ 40k when
this campaign started — +0.170, ~29% rel). Supervised ceiling (uses labels, forbidden) = 0.995.
Ledger CSV: `$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv`.
**The eFEL aux is the biggest single-lever win of the campaign** (Finding 9): +0.038 over dtw_200k and
it BEATS 400k data (0.736) with HALF the data — an OBJECTIVE change, not more data. It lifted na3
0.79→**0.87** (AP_amplitude directly constrains Na spike height) and gave the **best kdr yet (0.366)** —
the first thing to move kdr up, confirming kdr's lever is the objective, not data (Finding 2) or
gradient-reweighting. Data still helps 5/6 channels but with diminishing returns and NOT kdr; the
InterChaoticB split (Finding 8) motivates multi-stim as the next lever.

## Transferable recipe — what worked for chaoticRamp (drop-in for InterChaoticB)
The winning voltage-only recipe is a small, portable set of choices. To move it to another
stim (e.g. `5k50kInterChaoticB`, which is the SAME 5001-pt / 500 ms / dt 0.1 ms length as
`5kChaoticRamp` ⇒ 1:1 transfer, same ~221 s/epoch), change ONLY `stim_name` + the data pack.
| # | change | value | why it worked |
|---|---|---|---|
| 1 | **loss = soft-DTW primary + SMALL ramped eFEL aux** | `dtw_weight 1`, `mse_weight 0`, `blur 0`; `efel_weight 0.05→0.2` on `[voltage_base, AP_amplitude]` | DTW nails timing; the small eFEL aux anchors level/amplitude → BEST arm (0.763). eFEL as PRIMARY (weight 1.0) diverges; as a small ramped aux it wins. MSE/blur lost |
| 2 | DTW params | `dtw_gamma 0.1`, `dtw_band_ms 8`, `dtw_n_points 256` | 8 ms Sakoe-Chiba band carries the residual rate signal; band 4 didn't help |
| 3 | **DATA scaling** (dominant lever for 5/6) | 40k→400k, returns now diminishing | +0.14 mean; lifts leak/na3/kap/km/kd; **kdr does NOT track data** (Finding 2) |
| 4 | **150 epochs** | was 100 | 200k val loss still dropping at ep94 (−0.0540→−0.0571), LR 9e-6 |
| 5 | fp64 Jaxley solve | `fp64: True` | fp32 → NaN backward on high-gNa draws, NCCL propagates NaN across ranks |
| 6 | `clamp_unit_tanh` | `True` | bounds CNN param outputs into the trained range |
| 7 | global batch 2048 | `batch_size 64` ×32 GPU, `const_local_batch True` | matches the whole sweep; LR 1e-4 adam, `clip_grad_norm 1.0` |
| 8 | `serialize_stims: True` | probe axis → CNN channels | standard CA3 loader path |
| 9 | backbone (unchanged) | 2 CNN blocks [30,90,180] k4 p4; FC [512,512,512,256,128] drop 0.04 | RayTune (130 trials) found nothing better; bigger diverges |
| — | **strictly voltage-only** | `channel_weight 0`, `mask_channels True`, `voltage_weight 1` | hard constraint — ion channels NEVER in the loss |
| ✗ | do NOT: grad-precond at scale · van-Rossum blur · step-stim **under PURE DTW** · bigger net · efel as PRIMARY · **the 7 pA-scaled stims** · **>400k data** | — | precond helps only <80k; blur toxic to fast channels; a step poisons *pure* DTW (`chaoramp_step` changed 2 things at once, so this is NOT "steps are bad" — see Finding 12); efel@weight1.0 diverges (small ramped aux is fine — row 1); `ramp_500`/`4k50kInter{ramp,step}_*` inject 1000× current (see Stimulus hygiene); 400k costs kdr (Finding 11) |

**InterChaoticB job** (submitted): `ca3_vo_interchaoticB_dtw_8n.hpar.yaml` (rows 1–9 verbatim,
`stim_name: 5k50kInterChaoticB`, pack `ca3_5kinterchaoticB_v1` 200k) + `gen_interchaoticB_v1.slr`
+ `train_interchaoticB_200k_8n.slr` (150 ep).

## Ledger (sorted by mean R²)
| arm | recipe | data | mean R² | leak | na3 | kdr | kap | km | kd |
|---|---|---|---|---|---|---|---|---|---|
| **dtw_amp_200k** | DTW + soft-eFEL `voltage_base`+`AP_amplitude` (BEST) | 200k | **0.763** | 0.90 | **0.87** | **0.366** | 0.87 | 0.75 | 0.83 |
| dtw_amp_400k | DTW + eFEL aux **@400k** — the two best levers STACKED (Phase 20) | 400k | 0.750 | 0.88 | 0.86 | **0.251** ⬇ | **0.904** | 0.77 | 0.83 |
| precondkdr5_ft_200k | DTW + grad-precond **kdr×5.0**, fine-tune from ×2.5 ckpt | 200k | 0.742 | 0.87 | 0.76 | **0.34** | 0.85 | 0.81 | 0.82 |
| **dtw_400k** | pure soft-DTW, 10× data (best trace) | 400k | 0.736 | 0.89 | 0.80 | **0.22** ⬇ | **0.91** | 0.76 | 0.84 |
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
| **joint4_80k** | DTW + eFEL aux, **clean 4-stim battery** as CNN channels (Phase 21) | 80k×4 | 0.280 | **0.959** | 0.11 | **−0.277** | −0.32 | 0.31 | **0.897** |
| dtw_ica_200k | pure DTW, **InterChaoticB** stim (≠ chaoticRamp) | 200k | 0.219 | **0.97** | −0.27 | −0.12 | 0.16 | −0.37 | **0.94** |
| ~~pool4_80k~~ | pooled 4-stim — **VOID, training failed at epoch 4** (see Finding 14) | 80k×4 | ~~0.047~~ | — | — | — | — | — | — |

_dtw_ica_200k is a DIFFERENT stimulus — not comparable on mean. Its per-channel split is the point
(Finding 8): near-ceiling leak/kd, failed na3/kdr/km/kap._

_joint4_80k and pool4_80k are a DIFFERENT (4-stim) battery. Compare them to `dtw_80k` (0.637), which
is the same 80k sample count under one stimulus._

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
9. **A SMALL ramped soft-eFEL aux is the biggest single-lever win — and the FIRST thing to move kdr up
   (objective, not data).** `dtw_amp_200k` = pure DTW + `efel_weight 0.05→0.2` (ramped over 20 ep) on
   `[voltage_base, AP_amplitude]`, at 200k → **mean 0.763 (NEW BEST)**, +0.038 over dtw_200k and beating
   400k data (0.736) with HALF the data. Per-channel vs dtw_200k: na3 0.79→**0.87** (AP_amplitude
   directly constrains Na-driven spike height), kdr 0.338→**0.366** (best kdr of the campaign), leak
   0.86→0.90, kap 0.81→0.87, kd 0.80→0.83; km flat 0.75. Cost: voltage_mse_z 2.03 (a hair worse than
   400k's 1.90 — it trades a little pointwise fit for much better identifiability, the trade we want).
   **This VALIDATES the objective thesis (Finding 2): kdr's lever is the loss, not data/precond.** The
   old "efel diverges" caveat was about efel as PRIMARY at weight 1.0 (A7); as a small ramped aux under
   DTW it is the best recipe. **Next: add the rate/ISI/AHP features (`mean_frequency`, `inv_first_ISI`,
   `AHP_depth_abs_slow`) that target kdr's actual handle, and try amp@400k (stack the two winning levers).**
10. **Pushing the kdr grad-precond weight HIGHER did NOT crash kdr — my prediction was wrong (noted).**
    `precondkdr5_ft_200k` = fine-tune (50 ep, LR 3e-5) from the ×2.5 checkpoint with kdr weight 5.0
    (eff ~3.8) → mean 0.742, kdr **0.342** (RECOVERED from ×2.5's 0.266, ~= plain DTW 0.338). So higher
    weight didn't monotonically hurt kdr as the 40k/200k trend implied. BUT confounded: the fine-tune
    also added 50 epochs at a fresh LR, so the recovery may be the extra training, not the ×5.0 weight.
    Either way grad-precond still does NOT beat plain DTW on kdr and is well below the eFEL aux — the
    line stays closed as a kdr lever, but the "monotonically worse with weight" claim is retracted.

11. **Phase 20 — the two best levers do NOT stack: `amp@400k` is a NEGATIVE result.** Taking the best
    recipe (DTW + ramped eFEL aux, 0.763 @200k) to the best data point (400k) gave mean **0.750**, i.e.
    **below** the 200k version it was built from (−0.013), and it never became the champion. The whole
    deficit is kdr: **0.366 → 0.251**, while the four channels data helps (kap 0.866→**0.904**, km
    0.754→0.771, kd 0.825→0.830) and leak/na3 barely move (0.898→0.883, 0.870→0.859). Voltage fit
    genuinely improved — `voltage_mse_z` **1.897 vs 2.030**, the best of any amp arm — so the model
    reproduces traces better while identifying kdr worse. **This is Finding 2 again, now with the best
    objective in place: doubling data past 200k buys trace quality and kap, and costs kdr.** Two claims
    corrected while writing this up:
    - (a) the earlier "400k is simply better" read is wrong for the MEAN but right for the TRACE. On
      pure DTW, 400k beats 200k on 5 of 6 channels and on `mse_z`; with the eFEL aux it splits 3–3
      (kap/km/kd up, leak/na3/kdr down). In both pairs kdr is the channel that drags the mean down.
    - (b) **the eFEL aux was a na3 win, not a kdr win.** dtw_200k → dtw_amp_200k in error terms:
      na3 (1−R²) 0.206 → 0.131 = **−37%**, but kdr 0.662 → 0.634 = **−4%**. Finding 9's "first thing to
      move kdr up" overstated a 0.028 R² wobble; `AP_amplitude` constrains Na-driven spike height, which
      is exactly what it should do. kdr's handle (rate/ISI) is still not in the loss.
    **Do NOT run 800k chaoticRamp:** 200k→400k bought −1.6% mean error for 2× data *and* 150 vs 95
    epochs to prove itself, ~186 node-hours for <1% expected gain, and it makes kdr worse.
12. **The "step-stim FAILED" verdict (Finding 6) is over-scoped and is hereby narrowed to
    "step-stim under PURE DTW".** `chaoramp_step` changed TWO things at once — it added a stimulus AND
    that stimulus was a step — so it cannot separate "steps are bad" from "this battery/objective is
    bad". No step battery has ever been run under the champion objective (DTW primary + small ramped
    eFEL aux). That is the open experiment, not a closed door.

13. **Kdr does NOT survive a clean multi-stim battery — it goes NEGATIVE (−0.277).** This closes the
    question Finding 12 opened. `joint4_80k` ran the champion objective (DTW + ramped eFEL aux) over
    the clean 4-stim battery `[5kChaoticRamp, 5k0chaotic4, BBP_Exp_Step1000_i4k, chirp23a_i4k]`,
    Step1000 chosen as kdr's best UNCONTAMINATED stim (11.9 mV OAT). Training was healthy —
    val 0.810 → −0.0014 over a full 100 epochs, LR laddering 1e-4→3e-5→9e-6, best ckpt at epoch 96 —
    so this is not an optimization failure. Yet mean R² fell to **0.280 vs `dtw_80k`'s 0.637 at the
    same 80k sample count**, and kdr fell from +0.09 to **−0.28**.
    **Taken with the OAT gap (11.9 mV clean vs 74–146 mV contaminated), the Jul-4 kdr results
    (R² 0.985/0.991) are best explained as the 1000× overdrive artifact, not as a real "long step
    recovers kdr" effect.** Treat that pair of numbers as retracted.
    The per-channel split reproduces the `dtw_ica_200k` signature rather than the predicted UNION:
    leak **0.959** and kd **0.897** are the best values anywhere in this ledger, while every fast
    spiking channel collapsed (na3 0.73→0.11, kap 0.72→−0.32, kdr 0.09→−0.28). Trace fit degraded
    too (`mse_z` 2.20 vs 2.05, spike-count error 10.0 vs 6.9). Mechanism most consistent with the
    data: three of the four stims are comparatively quiet (chirp23a 4.7 spikes, 5k0chaotic4 7.6,
    Step1000 11.5, vs chaoticRamp's 24.2), so the loss — averaged over four traces — is dominated by
    subthreshold envelope matching, which is exactly what leak/kd are and what na3/kdr/kap are not.
    **Adding stimuli to the DTW loss dilutes spike information; it does not pool it.** The prediction
    on record ("joint should approach the union of what each stim observes") is falsified.
14. **`pool4_80k` is VOID — a training-dynamics failure, not a result about pooled multi-stim.** Its
    best checkpoint is **epoch 4**; zero of the remaining 95 epochs beat it, and `ckpt.pth` was last
    written 55 minutes into a 15.2-hour job. Validation thrashed early (0.725, 0.642, 0.824, 0.868,
    **0.217**, 0.263, 1.650, 1.002), the plateau scheduler treated the epoch-4 outlier as the
    permanent best, and LR collapsed 1e-4 → **2.19e-08** (4600×), freezing train loss at 0.080.
    **≈13.5 h / 54 node-hours produced nothing.** Root cause is a config mismatch: `valid_stims_select:
    [2]` validates on ONE stim while training pools all four, so the val signal is high-variance and
    not aligned with the objective being optimized — ideal conditions for a lucky early epoch to
    become unbeatable. **Do not score this arm.** Before any pooled re-run: validate on the pooled
    set (all four stims), and add an LR floor / `min_lr` so a bad early "best" cannot kill the run.

## Compute note (measured)
8 nodes / 32 GPU gave **no speedup** over 4 nodes / 16 GPU at pinned global batch 2048 (220.8 vs
~247 s/epoch): the jaxley fp64 solve is dominated by the SEQUENTIAL 5001-step time integration,
which does not parallelize over batch or GPUs. The 200k run therefore timed out at epoch 95/100 on
the 6 h limit; epoch-95 was converged and was scored directly (reconstructed `sum_train.yaml`).
**Run future 200k+ on 4 nodes with a longer wall-clock, not more nodes.**

**Multi-stim cost, measured (4 nodes, 80k×4 = 256k solves/epoch, 100 epochs):** joint **557.8 s/epoch
→ 15h33**; pooled **542.0 s/epoch → 15h10**. Epoch time is essentially linear in solve count — the
single-stim references are 48.65 s/epoch at 40k×1 (32k solves) and 98.83 at 80k×1 (64k solves), so an
8× solve increase predicts 389 s and the measured ~550 carries ~40% multi-stim overhead.
**Correction: pooled is NOT ~25% slower than joint** — that earlier claim came from compile-inflated
smoke epochs; at scale pooled is marginally *faster*. Budget ~16 h for a 4-stim 80k×100ep run.

## Data scaling — SETTLED, stop here (superseded; 400k has now run twice)
The prediction in this section ("expected ~0.75–0.78 mean, kdr toward ~0.45") was **half right**:
400k reached 0.736 pure-DTW / 0.750 with the eFEL aux, but kdr went the wrong way both times
(0.338→0.219 and 0.366→0.251). **Data scaling is closed as a kdr lever.** See Findings 2 and 11.
Do not launch 800k.

## Stimulus hygiene — SEVEN stims are unusable (pA injected as nA)
`jaxley_utils.load_stim_csv` reads a stim CSV as **nA** with no scaling, but seven CSVs in
`/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/` are written in **pA** — they inject 1000× too much
current and drive the soma to **+1600 mV**, which is ohmic drive of a 50×50 µm cylinder, not a neuron:
`ramp_500`, `4k50kInterramp_50khz`, `4k50kInterstep_500_50khz`, `4k50kInterstep_200_50khz` and their
`_i4k` twins. All other 69 stims are ≤6.8 nA and fine.

**This contaminates the two runs that motivated the whole multi-stim push.** The Jul-4 batteries that
recovered kdr at R² 0.985/0.991 each contain one or more of them — verified in the packs:
`ca3_best_efel_multi` probe 1 (`4k50kInterramp_50khz_i4k`) peaks at **+1604 mV**; `ca3_best_mse_multi`
probes 2 and 3 at **+1604** and **+998 mV**. So "the long step recovers kdr" and "the overdriven probe
recovers kdr" are **perfectly confounded** across both runs, and the OAT sweep favours the latter
reading: kdr's top four stimuli by voltage variation are ALL contaminated (146.2 / 146.1 / 106.3 /
74.1 mV) versus **11.9 mV** for the best clean stim, `BBP_Exp_Step1000`. Kdr's apparent observability
is concentrated almost entirely in the artifact.

Before using any stimulus, check `max|I|` — anything >20 nA is pA-scaled. Also note this invalidates
`sensitivity_best_stims.md`'s "Kdr best = `ramp_500`" and "KA best = `4k50kInterstep_500_50khz`" rows.
Raw OAT table: `tmp_neuInv/sensitivity_variation/ca3_pyramidal/salloc_55486075/interp4000/combined/`.
