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

## Experimental predictions — Roy/Paula Apr-2026 recordings (2026-08-24, first DTW-era exp runs)
Data: `exp_data_paula` 26422001–25 = 5 sweeps × 5 amplitudes of Roy's chaotic stim
(bit-for-bit `4k50kInterChaoticB` = `5k50kInterChaoticB[1000:5000]`; the 5k head is exactly
zero current, so a 5001-bin model window = 100 ms rest + the recording). **Measured rig→sim
scaling: recorded stim = sim waveform (corr 0.999, lag 0) × amp×0.2508/1000 + holding −0.5995 nA
⇒ Roy2000 = HALF the 6.82 nA training amplitude.** Packs: `RoyPaula6/` (6-par, fixed norm);
rig-faithful stims `Roy<amp>_ica5k.csv`; runner `plot_exp_overlay_roy.py` + `run_roy_exp.sh`
(job 57546673); outputs in each model's `out/exp_roy[_train]/`.

| model / sim amplitude | Roy100 mse_z (sp s/d) | Roy500 | Roy1000 | Roy1500 | Roy2000 |
|---|---|---|---|---|---|
| dtw_ica_200k @ rig | 0.547 (0/0) | 0.557 (0/0.6) | 0.454 (0/8.6) | 0.498 (0/12.4) | 0.573 (0/14.0) |
| **dtw_ica_200k @ train** | 0.550 (0/0) | 0.645 (4.0/0.6) | **0.538 (6.4/8.6)** | 0.509 (7.0/12.4) | 0.500 (8.0/14.0) |
| dtw_amp_200k @ rig | 0.512 (0/0) | 0.563 (0/0.6) | 0.508 (0/8.6) | 0.469 (0/12.4) | 0.457 (0/14.0) |
| dtw_amp_200k @ train (5kChaoticRamp — stim-mismatched overlay, firing bracket only) | 1.388 (17.2/0) | 1.497 (18.4/0.6) | 1.246 (21.8/8.6) | 1.214 (23.0/12.4) | 1.205 (23.4/14.0) |
| dtw_80k @ rig | 0.539 (0/0) | 0.561 (0/0.6) | **0.368** (0/8.6) | **0.322** (0/12.4) | **0.312** (0/14.0) |
| pool4v2_80k @ rig (added 2026-08-26, job 57627153) | 0.545 (0/0) | 0.558 (0/0.6) | 0.393 (0/8.6) | 0.375 (0/12.4) | 0.406 (0/14.0) |
| supervised_k128 @ rig (control, 2026-08-26, job 57630894) | 1.463 (0/0) | 0.587 (0/0.6) | 0.381 (0/8.6) | 0.512 (0/12.4) | 0.614 (0/14.0) |
| supervised_k128 @ train (protocol-matched; CORRECTED 400 ms window, job 57634916) | 1.438 (0.2/0) | 0.660 (5.4/0.6) | 0.662 (7.4/8.6) | **0.388** (10.0/12.4) | 0.623 (8.4/14.0) |
| **dtw_k128_80k @ rig** (2026-08-26, jobs 57628704/57633401) | 1.045 (0/0) | 0.578 (0/0.6) | 1.191 (**2.0**/8.6) | **0.298** (0/12.4) | **0.305** (0/14.0) |
| dtw_k128_80k @ train (5kChaoticRamp — stim-mismatched, firing bracket only) | 1.514 (26.0/0) | 1.803 (31.2/0.6) | 1.353 (36.6/8.6) | 1.250 (20.0/12.4) | 1.296 (17.4/14.0) |
| royexp_ft_57631428 @ rig v1 stims (exp FINE-TUNE, k128 base; 2026-08-26, job 57634651) | 0.497 (0/0) | 0.866 (0/0.6) | 0.992 (0/8.6) | 1.067 (0/12.4) | 1.115 (0/14.0) |
| royexp_ft @ its training stim Roy2000_icav2_5k (all amps) | 0.495 (0/0) | 0.834 (0.2/0.6) | 0.893 (1.0/8.6) | 0.887 (1.0/12.4) | 0.864 (1.0/14.0) |
| voefel_k128 @ rig (Jul-8 vo+eFEL model; 2026-08-26, job 57634715) | 0.505 (0/0) | 0.767 (0/0.6) | 0.774 (0/8.6) | 0.762 (0/12.4) | 0.769 (0/14.0) |
| voefel_k128 @ train (protocol-matched; CORRECTED 400 ms window, job 57634916) | 0.567 (4.4/0) | 0.693 (7.0/0.6) | 0.631 (7.0/8.6) | 0.583 (7.0/12.4) | 0.557 (8.0/14.0) |
| scratch_k128 @ rig (exp-data-ONLY, no pretraining; 2026-08-26, job 57635094) | 0.545 (0/0) | 1.018 (0/0.6) | 1.150 (0/8.6) | 1.164 (0/12.4) | 1.171 (0/14.0) |
| scratch_k128 @ Roy2000_icav2_5k (its training stim) | 0.541 (0/0) | 1.040 (1.0/0.6) | 1.068 (2.0/8.6) | 1.141 (2.0/12.4) | 1.168 (2.0/14.0) |
| 1stim specialists @ own icav2, held-out royv2 (ONE exp-only k128 model PER family; 2026-08-26, jobs 57636280 + 57636946–48; 18 sweeps/amp) | — (Roy100 dropped) | 0.604 (0/3.7) | 0.726 (0/8.4) | 0.803 (1.0/12.7) | 0.725 (1.8/15.3) |
| 1stim specialists @ rig v1 (each model scored at its own family's amplitude) | — | 0.873 (0/0.6) | 1.166 (0/8.6) | 1.105 (0/12.4) | 1.049 (0/14.0) |
| 1dtw specialists @ own icav2, held-out royv2 (DTW-loss twins of the row above; 2026-08-26, jobs 57637880/81 + salloc 57637882) | — (Roy100 dropped) | 0.597 (0/3.7) | 0.716 (0/8.4) | 0.838 (1.1/12.7) | 0.715 (1.7/15.3) |
| 1dtw specialists dtw_z @ own icav2 (timing-tolerant metric, same runs) | — | 0.234 | 0.223 | 0.278 | 0.232 |
| **icarec-ft** (FT twin retrained w/ TRUE recorded stims; 2026-08-26 job 57638114) @ icaRec held-out | 0.525 (0/0.1) | 0.578 (0/3.7) | 0.604 (0/8.4) | 0.591 (0.4/12.7) | 0.606 (1.3/15.3) |
| icarec-scr (scratch twin, job 57638113) @ icaRec held-out | 0.517 (0/0.1) | 0.638 (0/3.7) | 0.703 (0/8.4) | 0.769 (1.0/12.7) | 0.739 (2.0/15.3) |
| ft_champion re-scored @ icaRec (true drive; was icav2-trained) | 0.501 (0/0.1) | 0.600 (0/3.7) | 0.638 (0/8.4) | 0.672 (0.6/12.7) | 0.673 (1.8/15.3) |
| scratch_k128 re-scored @ icaRec | 0.555 (0/0.1) | 0.657 (0/3.7) | 0.706 (0/8.4) | 0.718 (1.0/12.7) | 0.714 (2.0/15.3) |
| **ftwide** (wide box, span 1.0; job 57638891) @ icaRec held-out | 0.495 (0/0.1) | 0.701 (0/3.7) | 0.591 (0/8.4) | 0.631 (**1.8**/12.7) | 0.683 (**4.8**/15.3) |
| ftwidedtw (wide box + pure DTW; job 57639887) @ icaRec held-out | 0.673 (0/0.1) | 0.610 (0/3.7) | 0.624 (0/8.4) | 0.639 (0.3/12.7) | 0.645 (1.6/15.3) |
| ftwidedtwefel (wide + DTW + eFEL; job 57641347) @ icaRec held-out | 0.596 (0/0.1) | 0.611 (0/3.7) | 0.641 (0/8.4) | 0.691 (0.8/12.7) | 0.668 (1.9/15.3) |
| polish (DTW+eFEL warm-started FROM ftwide's firing ckpt; job 57641930) @ icaRec | 0.501 (0/0.1) | 0.567 (0/3.7) | 0.640 (0/8.4) | 0.617 (0.3/12.7) | 0.695 (1.6/15.3) |
| **efel5** (wide + DTW + eFEL×5, rate-dominated; job 57642366) @ icaRec | 0.544 (0/0.1) | 0.718 (0/3.7) | 0.801 (**1.4**/8.4) | 0.924 (**6.6**/12.7) | 0.893 (**11.8**/15.3) |
| efel3 (wide + DTW + eFEL×3; job 57642626) @ icaRec — stayed in quiet basin | 0.573 (0/0.1) | 0.623 (0/3.7) | 0.786 (0.2/8.4) | 0.787 (0.6/12.7) | 0.708 (1.1/15.3) |
| efel8 (wide + DTW + eFEL×8; job 57642996) @ icaRec — rate-matched @2000 only | 0.488 (0/0.1) | 0.747 (0/3.7) | 0.848 (0.4/8.4) | 0.737 (2.3/12.7) | 0.972 (**15.9**/15.3) |
| efel5-stab (w=5, LR 2e-5, 80 ep, salloc; CONVERGED) @ icaRec | 0.577 (0/0.1) | 0.697 (0/3.7) | 0.931 (**8.7**/8.4) | 1.077 (28.6/12.7) | 1.065 (37.5/15.3) |
| **efel5-ema = ALL-ROUND CHAMPION** (w=5-stab + per-family EMA norm; salloc 57643643) @ icaRec | 0.567 (0/0.1) | 0.716 (**1.9**/3.7) | 1.041 (**5.9**/8.4) | 0.785 (**8.8**/12.7) | 0.795 (**10.9**/15.3) |

Findings: (1) **rheobase gap, model-independent** — under the actually-delivered current no
predicted parameter set EVER fires; the sim cylinder needs ~4× the real cell's current.
(2) **Best exp agreement of the campaign**: dtw_ica_200k re-simmed at training amplitude,
Roy1000 6.4 vs 8.6 spikes — but sim rate saturates ~6–8 while the real rate scales 0.6→14, and
predicted params DRIFT with amplitude (na3 0.83→0.04 unit) instead of being invariant for the
same cell: the model conflates drive with cell properties (never saw amplitude variation).
(3) **chaoticRamp models are OOD on this protocol**: dtw_80k posts the best envelopes
(0.312–0.368) but na3/kdr/km pin at the +1 tanh boundary — saturated, untrustworthy params
(synthetic-R²↔transfer anti-correlation again). (4) Levers before quantitative exp prediction:
amplitude-augmented training and/or an input-resistance-matched sim cell; multi-amplitude
consistency is the natural exp-side validation metric. Prior exp predictions (efel-era 4k
models, 2021 datasets): best trace interchaoticB_ft 0.708 under-firing 6.8/12.4; k128
rate-matched 12.4/12.4 @ 1.00.
(5) **Supervised-k128 control (2026-08-26, job 57630894)**: no NaN/extreme sims despite
`clamp_unit_tanh: False`. At rig amplitude 0 spikes like every model (rheobase gap is universal),
and the worst subthreshold envelope in the table (Roy100 1.463 vs 0.51–0.55 for voltage-only). At
its protocol-matched training amplitude it DOES fire — 7.4/10.0 spikes at Roy1000/1500 vs data
8.6/12.4, comparable to dtw_ica@train (6.4/7.0). [CORRECTED 2026-08-26, job 57634916: the first
@train scoring used a 500 ms window on the 4000-bin stim, plotting sim spikes ~100 ms early and
inflating mse_z to ~2.0. With the 400 ms window: mse_z 0.39–0.66 (Roy100 1.44) — COMPETITIVE with
dtw_ica@train, and Roy1500 0.388 is the best @train envelope in the table.] Jul-8 verdict revised:
on these recordings the supervised control is neither spike-blind nor envelope-poor at its
training amplitude; its weaknesses are the silent-sweep envelope (Roy100) and, like all
non-fine-tuned models, zero firing at rig current.
(6) **DTW-k128 (2026-08-26, train job 57628704, exp job 57633401)**: `dtw_80k` champion recipe with
conv kernel [4,4,4]→[128,128,128], same 80k chaoticramp_v2 pack, 100 ep @ 100.2 s/ep (+1.5% over k4
— the solve dominates), best val −0.0448 @ ep 94. Synthetic recovery DROPS as predicted: mean R²
0.515 vs 0.637, kap hit hardest (0.72→0.35), kdr 0.09→0.01, trace fit unchanged (mse_z 2.069 vs
2.046). But on the Roy recordings it is the **only model in the table that fires at the delivered
rig current** (2.0 spikes @ Roy1000; its Roy1000 mse_z 1.191 is worse than dtw_80k's 0.368 *because*
misplaced spikes cost more z-error than none) and posts the **best envelopes in the table** at
Roy1500/2000 (0.298/0.305 vs 0.312–0.322). The Jul-8 anti-correlation (synthetic accuracy vs
sim→real transfer) is confirmed under the soft-DTW objective. Verdict for the kernel decision:
k128 wins on the metric that matters for exp application (partially cracking the rheobase gap) and
is mixed on envelope (worse subthreshold Roy100: 1.045 vs 0.539). Supports repeating k128 on the
other arms (pool4 next, per the 08-26 plan) before any quantitative exp prediction.
(7) **k128 four-way closes the Jul-8 loop (2026-08-26, jobs 57634651/57634715; @train rows
corrected by 57634916 — see the stim-window bug below)**: same k128 backbone, four objectives, same
recordings. **The Jul-8 "vo+eFEL k128 = best exp transfer" does NOT reproduce on Roy**: at rig
voefel_k128 never fires and posts flat mid-table envelopes (0.76–0.77); at train amplitude
(corrected mse_z 0.56–0.69) it is competitive with dtw_ica@train but its spikes are FLAT (~7)
across a 20× amplitude range and it over-fires 4.4 on the silent Roy100 — the eFEL spike-forcing
signature. Supervised@train (corrected 0.39–0.66) is equally competitive with better spike
scaling (0.2→10). The July separation came from the old Exact dataset. On Roy the transfer
ranking within k128 is: dtw_k128 (only rig firing, best high-amp rig envelopes) > voefel ≈
supervised (@train: fire with competitive envelopes; neither fires at rig) with the **exp
fine-tune royexp_ft** a special case — it fires ONLY under its own
icav2 stim (~1 spike at 500–2000, mse_z 0.86–0.89) and not under the v1 rig stims (0 spikes),
having also lost the base's high-amp envelope edge (1.07–1.12). **The v1-vs-icav2 rig-stim
reconstruction now decides rheobase verdicts — resolve which is rig-faithful before reading any
firing result as real.** Champion for exp application after today: soft-DTW + k128 objective/
architecture, with amplitude augmentation and the stim-reconstruction question as the next levers.
(8) **Stim-window alignment bug in `plot_exp_overlay_roy.py` (found by user 2026-08-26, fixed +
re-scored job 57634916)**: the script forced `_T_MAX = 500 ms` for every sim; a 4000-bin training
stim (`4k50kInterChaoticB`, 400 ms) upsampled onto that grid gets its last value held for 100 ms
(`np.interp` clamps), and the last-`T_data` alignment then plots sim spikes ~100 ms EARLY, cuts
the first 100 ms of response, and appends a decay tail — inflating @train mse_z ~3× (2.0 → 0.6).
Only the two 4k-stim @train rows were affected (all other stims are 5000-bin); rig rows, dtw
models, and the fine-tune were verified clean (waveform cross-check: 4k stim == last 400 ms of
`Roy*_ica5k`, corr 1.000, lag 0). Fix: `_T_MAX` now follows the simulated stim's CSV length.
(9) **Exp-data-only ablation (2026-08-26, job 57635094)**: `ca3_royexp_scratch_k128` = the
fine-tune's exact twin (same 805-trace pack, per-family icav2 stims, MSE+eFEL loss, 40 ep, k128)
but FROM SCRATCH, LR 1e-4. Trained in ~11 min (16 s/ep, best val 1.483 @ ep 38). Held-out
per-family eval (royv2, 18 sweeps/amp, vs the fine-tune): **pretraining owns the envelope** —
scratch mse_fixed is worse at every amplitude (Roy100 0.0385 vs 0.0125, 3×; mse_z 0.55–0.79 vs
0.50–0.65) — **but the exp data alone teaches spiking**: scratch fires MORE than the FT at high
amps (held-out 1.0/2.0 spikes at 1500/2000 vs 0.6/1.1; standard pipeline 1–2 spikes under icav2 vs
FT's 0.2–1.0). Both remain far under the real rates (12–15). So 34 training neurons suffice for a
coarse rig-firing envelope model; sim pretraining refines subthreshold shape ~uniformly. Neither
fires under the v1 rig stims (the v1-vs-icav2 question from finding 7 stands).
(10) **Per-family specialists are NOT better (2026-08-26, salloc 57636280 [Roy500] + debug jobs
57636946/47/48 [Roy1000/1500/2000])**: four exp-only k128 models, one per stim family (Roy100
dropped from training AND evaluation per user), each trained on ONLY its family's sweeps
(~125 train / 16 valid / 18 test) with its own `Roy<amp>_icav2_5k` stim — the maximally
stim-matched configuration. Held-out, each specialist ≈ the pooled scratch model at its
amplitude (mse_z 0.60/0.73/0.80/0.73 vs scratch 0.65/0.70/0.79/0.75; spikes 0/0/1.0/1.8 vs
0/0/1.0/2.0) and the FT stays best on envelope everywhere (0.60/0.64/0.65/0.63). Training is
nearly flat: 1stim_500 val 1.98→1.97, 1stim_1000 val 3.25→**3.67 (worse)**, LRs plateau-decayed
to 3e-5–9e-6. Predicted params expose the mechanism: the high-amp specialists all push the
excitable corner (na3 +0.3–0.5, leak −0.5–0.9, all K down) with **gkdbar_kd pinned at the −1
tanh boundary for 100% of samples** (1stim_1000, scratch) yet still fire 0–1.8 vs data 8–15;
1stim_500 never leaves init (all params ~0). Verdict: stim specialization is not the binding
constraint — data volume (5× less per specialist, no cross-family signal) and the rheobase wall
are. The one-model-per-stim question is CLOSED; the actionable lever remains the stim
reconstruction (v1 vs icav2 holding current, 0.55 nA apart) and the parameter box, not the
training split.
(11) **v1-vs-v2 stim reconstruction RESOLVED against the recorded rig current (2026-08-26, desk
analysis) — both were wrong about holding, v1 catastrophically**: the exp packs carry the actual
recorded stimulus (`stim_Roy<amp>_pA` in RoyExpPack_ca3ft, `stim_pA` in RoyPaula6). Recorded
holding is **+0.0006 nA ≈ ZERO** — not −0.5995 nA (v1 `_ica5k`) and not −0.0496 nA (v2
`_icav2_5k`). Waveform shape is fine in both (corr 0.9993), but v1 has RMS error 0.60 nA (its
entire holding is phantom hyperpolarization) and v2 is exactly a constant −0.05 nA shift (RMS
0.050 nA; peak deficit 0.051–0.054 nA at every amplitude). Consequences: **every "silent under
rig stim" v1 verdict is void** — those sims were under-driven by 0.6 nA, which exceeds typical
CA3 rheobase; the rheobase-gap estimate shrinks accordingly. The icav2 held-out rows remain
approximately right (−0.05 nA bias, conservative on firing). New exact stims written:
`Roy<amp>_icaRec_5k.csv` (= recorded current, 1000-bin true-hold pre-pad) in the DL4neurons2
stim dir; `plot_exp_overlay_royv2.py` gained `--stimSuffix icaRec_5k` to re-score any model
under the true drive. Next: re-score champions + specialists under icaRec (does the extra
+0.05 nA close part of the firing gap?), then retrain best exp config with icaRec in-loop stims.
(12) **DTW-loss specialists: trains cleanly, same spikes (2026-08-26, jobs 57637880/81 + salloc
57637882)**: the four one-stim specialists retrained with the PURE soft-DTW champion loss
(`ca3_royexp_1stim_*_dtw`) instead of MSE+eFEL. Optimization is qualitatively healthier — val
falls ~15–35% and keeps falling (500: 0.081→0.070; 1000: 0.360→0.243; 1500: 0.761→0.512;
2000: 1.078→0.631) where the MSE+eFEL twins were flat or worsening. Held-out transfer, though,
is a wash: mse_z 0.597/0.716/0.838/0.715 vs twins 0.604/0.726/0.803/0.725, spikes identical
(0/0/1.1/1.7 vs 0/0/1.0/1.8). The objective lever does NOT move the spike deficit — more
evidence the binding constraint is drive/physiology (rheobase wall, finding 11 stim bias), not
the loss. New `dtw_z` column (timing-tolerant, training hyper-params) now in both scorers;
specialists all score dtw_z 0.22–0.28 held-out.
(13) **icaRec wave (2026-08-26, jobs 57638113/114 retrains + 57638309 rescore): true drive
helps at the margin; `icarec-ft` is the new best envelope model; the spike wall STANDS at
~2 vs 8–15**. (a) Re-scoring icav2-trained models under the exact recorded stimulus adds
firing everywhere near threshold (ft_champion Roy2000 1.1→1.8 spikes; 1dtw_1000 0→0.5;
1dtw_2000 1.7→2.3) — confirms the −0.05 nA icav2 bias mattered at rheobase. (b) Training
WITH icaRec in-loop: `icarec-ft` (fine-tune twin) posts the best held-out envelope of the
campaign under the true drive — mse_z 0.578/0.604/0.591/0.606 and dtw_z 0.177–0.224 at
500–2000, beating ft_champion@icaRec at every amplitude ≥1000 — but fires LESS (0.4/1.3 at
1500/2000); icarec-scr ≈ scratch (1.0/2.0). (c) NO model exceeds ~2.3 spikes vs data 8–15
even with exact drive, exact stim in training, and params saturating the excitable corner.
Excitability probe launched (job 57638766, `excitability_probe_icarec.py`): pushes params
1.5–3× PAST the box corner (na3 up to 31× center) under Roy1000/2000_icaRec — if nothing
fires ~10 spikes, the wall is the CELL MODEL (channels/geometry), not the box, and the next
lever is physiology (Ih/NaP/CaT, cell size), not training.
(14) **Excitability probe RESULT (job 57638766): the wall is the PARAMETER BOX, not the cell
model.** Under Roy2000_icaRec the pure excitable corner scaled 1.5× past the tanh boundary
(u = ±1.5 ⇒ phys 10^±0.75 ≈ 5.6× center) fires **30 spikes** (data 15.3) and ×2.0 fires **10**;
every in-box setting stays ≤3. Corner ×3 collapses to 1 spike (depol block) — the good region
is a shell at |u|≈1.5–2, just OUTSIDE the trained box. na3 alone does nothing (0–2 spikes even
at u_na3=3): firing needs the combined leak↓ + all-K↓ corner. At Roy1000 the best is 3 spikes
(corner ×2) vs data 8.4 — the model's f-I is steeper than the real cell's, so one global param
set can't match both amplitudes; per-sweep CNN predictions may interpolate. ACTION: wide-box
fine-tune `ca3_royexp_ft_icarec_wide` (log-span 0.5→1.0, so the firing shell sits inside tanh)
launched as job 57638891; scorer gained `--physFromModel` (REQUIRED for wide models — pack meta
still carries span 0.5). Also: dtw_k128 re-scored (padded input fix) — under the true drive its
z-shape at 1500/2000 is the best in the campaign (mse_z 0.42/0.41, dtw_z 0.127/0.116) but
mse_fixed 2.1–2.4 (huge DC/scale error) and ~0 spikes: great shape prior, wrong operating point.
(14j) **EMA balance completes it (salloc 57643643): the ALL-ROUND CHAMPION `efel5-ema`** —
w=5-stab + `pooled_stim_norm: ema` fires at EVERY supra-threshold family with a monotone,
data-shaped profile: 1.9/5.9/8.8/10.9 spikes vs data 3.7/8.4/12.7/15.3 (uniformly ~65–70% of
the real rate, NO overshoot). The stab run's 28.6/37.5 overshoot at 1500/2000 was a
loss-BALANCE artifact, not the model's f-I — per-family normalization redistributes the firing
pressure and the converged solution tracks the data's saturating curve. Envelope mid-pack
(mse_z 0.72–1.04 supra-threshold; the quiet models remain better on that axis). Remaining
uniform ~30% under-firing = the residual physiology/objective gap. Model:
`ca3_royexp_ft_icarec_wide_dtwefel5_ema/RoyExpChaotic/ft_icarec_wide_efel5_ema/out`
(wide box — score with `--physFromModel`). Recipe = supervised warm-start + icaRec true stims
+ wide box (span 1.0) + DTW 1.0 + eFEL 5.0 + per-family EMA norm + LR 2e-5 × 80 ep.
(14i) **Stabilized w=5 CONVERGES IN THE FIRING BASIN (salloc, LR 2e-5, 80 ep)**: val flat at
~8.7 (no divergence — the firing basin is stable at the lower LR), and it **rate-matches
Roy1000: 8.7 spikes vs data 8.4** — the family no model had ever fired at. But it OVER-fires
the high families (28.6/37.5 vs 12.7/15.3). Full picture across the ladder: the model's f-I
slope is ~2.5× the real cell's (data saturates 3.7→8.4→12.7→15.3; converged model goes
0→8.7→28.6→37.5) — the real CA3 cell has spike-frequency adaptation/saturation this 6-channel
soma model lacks. Since the CNN CAN emit different params per sweep, over-firing at high amps
is partly an objective-balance artifact → final run of the night: w=5-stab + per-family EMA
loss normalization (`pooled_stim_norm: ema`, the pool4-v2 machinery) to stop the high-amp
families dominating the averaged loss.
(14h) **eFEL×8 rate-MATCHES Roy2000 (job 57642996): 15.9 spikes vs data 15.3** — but
concentrates ALL firing there (1500: 2.3 vs efel5's 6.6; 1000: 0.4 vs 1.4) and diverges harder
(val 13.1→15.4, train 11.9→17.6). Full weight map: w=1→quiet, 3→quiet, 5→spread firing
(1.4/6.6/11.8), 8→rate-matched at 2000 only. The across-amplitude trade is the steep-f-I
misspecification (finding 14) surfacing in the objective: the model cannot satisfy every
family's rate at once, so the weight picks WHERE to spend the firing. Per-family loss
normalization or an f-I-aware term is the structural fix; efel5 remains the best all-round
firing model. Stability of the firing basin now rests on `efel5-stab` (salloc, LR 2e-5, 80 ep).
(14g) **The eFEL weight is a BASIN SWITCH, not a dial (job 57642626)**: w=3 settles quiet
(0.2/0.6/1.1 spikes at 1000–2000, val improving to 4.73 — the quiet optimum still wins the
total loss), while w=5 escapes to 11.8. Sweep: w=1→1.9, w=3→1.1, w=5→11.8. There is no smooth
Pareto interior in (1,3]; the transition sits in (3,5]. Far-side probe w=8 launched (job
57642996) — does firing climb toward the data's 15.3 or overshoot?
(14f) **eFEL×5 BREAKS THE SPIKE WALL (job 57642366)**: the rate-dominated objective (wide box +
DTW 1.0 + eFEL 5.0) fires **11.8 spikes at Roy2000** (data 15.3), **6.6 at Roy1500** (12.7), and
1.4 at Roy1000 — first model anywhere near experimental rates (2.5× the ftwide record). The
weight was the whole story: w=1→1.9 spikes, w=5→11.8. Cost: envelope degrades (mse_z 0.80–0.92,
dtw_z 0.37–0.41, mse_fixed 2.6–3.9 at 1000–2000) and val drifts up (8.2→8.8; scored ckpt =
best-val). Sweet-spot probe launched: efel_weight 3.0 (`ft_icarec_wide_dtwefel3`, job 57642626).
The spikes-vs-envelope Pareto front is now MAPPED by the weight — pick the operating point per
use case.
(14e) **Polish run REGRESSED (job 57641930)**: warm-started from ftwide's 4.8-spike checkpoint,
the stable DTW+eFEL objective actively UN-LEARNS the firing (Roy2000 4.8→1.6 spikes over 40 ep,
val 2.47→2.29 "improving"). The quiet envelope is the OPTIMUM of this loss family — the missing
spikes are an objective problem, not an optimization problem: matching the data's firing costs
more envelope error than the rate features (mean_frequency/inv_first_ISI/ISI_values, confirmed
IN the default STRONG_FEATURES set all along) recover. Last knob in the family: efel_weight
1.0→5.0 (`ft_icarec_wide_dtwefel5`, job 57642366) — if a rate-DOMINATED objective still settles
quiet, the loss family is exhausted and the residual gap is model misspecification (steep f-I,
finding 14: no single param set matches both Roy1000's 8.4 and Roy2000's 15.3).
(14d) **Wide + DTW + eFEL (job 57641347)**: stable (val 2.44→1.77) but only 0.8/1.9 spikes at
1500/2000 — every STABLE optimizer settles near the warm-start basin; ftwide's 4.8-spike
checkpoint came from the region its divergence explored. Final move of the night: `polish` run
(job 57641930) warm-starts from the FIRING ftwide checkpoint and fine-tunes with the stable
DTW+eFEL objective — start inside the firing region, clean up the envelope there.
(14c) **Wide + pure DTW (job 57639887)**: converges cleanly (val 0.38→0.21, no divergence) and
posts the campaign-best FIXED-space envelope (mse_fixed 0.20/0.51/0.67/0.88 at 500–2000, beating
icarec-ft's 0.23/0.55/0.74/0.98) — but retreats to near-silence (0.3/1.6 spikes at 1500/2000).
Confirms the decomposition: **eFEL is the spike-forcing term** (ftwide had it → 4.8 spikes;
July's "eFEL spike-forcing signature" again), **DTW is the stable envelope term**. Capstone
launched: `ft_icarec_wide_dtwefel` (job 57641347) = wide box + DTW 1.0 + eFEL 1.0, MSE 0.
(14b) **Wide-box fine-tune RESULT (job 57638891)**: spikes break through — Roy2000 **4.8**
(campaign record, was ≤2.3), Roy1500 1.8, dtw_z 0.181–0.192 ≈ tied with icarec-ft's best;
Roy1000 still 0 (steep f-I, finding 14). Costs: mse_fixed blows up (1.4–1.8; mistimed spikes +
DC error) and TRAINING DIVERGES under MSE+eFEL in the wide box (val 3.55→5.49 over 40 ep) —
the scored model is the best-val early checkpoint. Follow-up launched: `ft_icarec_wide_dtw`
(job 57639887) = wide box + PURE soft-DTW (the finding-12 stable optimizer) — combining all
four proven levers: warm-start, icaRec true drive, wide box, DTW loss.
(16) **Stim-as-channel A/B (2026-08-28, salloc 57694161; pack `RoyExpPack_stimch`, ch1 =
RECORDED stim in nA)**: cleanest information-vs-objective split of the campaign. Identical
recipe (efel5-ema loss, wide box, EMA norm, from scratch, LR 5e-5, 80 ep), only the input
differs. **2-ch (stim in)**: by far the best optimization (val 10.4→4.69, still falling) and
the best envelopes of any firing-recipe model (mse_fixed 0.016/0.23/0.61/0.89/1.19, dtw_z
0.17–0.27) — but nearly QUIET (0/0/0.1/0.9/2.5 spikes). **1-ch control**: campaign-best RATE
TRACKING — 0/0.1/6.9/11.2/13.2 vs data 0.1/3.7/8.4/12.7/15.3 (beats efel5-ema's 5.9/8.8/10.9)
— at much worse envelope (mse_z 1.02–1.12, val 10.4). Reading: the stim channel gives the
network the capacity to nail the envelope, and the total loss then PREFERS the quiet optimum
(4.69 < 10.4 — the objective scores the quiet 2-ch model far better than the firing 1-ch one).
Spikes-vs-envelope is an OBJECTIVE choice, not an information limit. Follow-up launched:
2-ch + eFEL 8.0 (`ca3_royexp_stimch_e8`) to re-balance rate pressure against the new envelope
capacity. Scorer: `plot_exp_overlay_stimch.py --channels`.
(15) **L5TTPC cross-model probe (job 57638445, `plot_exp_overlay_royv2_l5.py`, no-grad handle
path after the VJP OOM)**: the 19-par L5TTPC_jaxley_nc2 supervised model predicting + re-simming
the same recordings does NOT improve predictions overall. Faithful (unclamped): mse_z
0.62/0.61/1.25/0.79 at 500–2000, spikes ≤0.3 vs 3.7–15.3; tanh-clamped similar (0.62/0.61/
0.90/0.86, spikes ≤0.6). Curiosity: at Roy500–1000 its subthreshold envelope is competitive
with the CA3 champions (dtw_z 0.157–0.162 at Roy1000, best-in-campaign for that family), but it
collapses at 1500/2000. Cross-cell transfer is not the answer; CA3 + wider box is.

## pool4 v2 re-run (launched 2026-08-24, job 57548058)
Finding-14 fixes, all three implemented + unit-tested (21/21 pass): pooled validation over all
four stims (Dataloader_H5 flattens valid stim-major; the plateau scalar now averages every
protocol), `min_lr 1e-6` LR floor, and opt-in train-only per-stim EMA loss normalization
(`pooled_stim_norm: ema`, beta 0.98) so quiet stims cannot swamp spike information (the joint4
dilution). Same pack/batch 256/global 4096/60 steps/ep as v1 for a clean A/B; est ~578 s/epoch,
~16 h. Design `ca3_pool4_dtw_amp_v2_4n`, launcher `train_pool4_v2_4n.slr`.

**Outcome (COMPLETED 2026-08-25 08:47, 15:40:56; per-stim + exp scoring 2026-08-26, job 57627153).**
All three fixes engaged. LR laddered 1e-4 → 3e-5 (ep 17) → 9e-6 (42) → 2.7e-6 (55) → 1e-6 (68, floor
held); best val improved to the end, **0.0430 @ epoch 98** (v1: 0.2171 @ epoch 4, never beaten, LR
ratcheted to 2.19e-8). Steady-state 557.6 s/epoch vs v1's 539.5 — the pooled-validation overhead came
in at **+3.4%**, under the +7% estimate, and inside the joint4 band (557.8) ⇒ another
no-slowdown-at-scale datapoint for T3. **Reading the curves:** the v2 TRAIN loss is EMA-ratio-scaled —
each stim's loss ÷ its own running average — so it hovers near 1 with occasional ±5 transients and
admissible negative dips (soft-DTW < 0 over a small denominator). It is trendless BY CONSTRUCTION and
not comparable to any other run; judge optimization by the raw val curve only (grad-clip 1.0 contained
every train transient — val never blinked).

Mean R² **0.3671** (v1 void 0.0465; joint4 0.280; single-stim champion dtw_80k 0.637):
leak 0.740, kd 0.593, na3 0.452, km 0.358, kap 0.091, **kdr −0.032**. First valid pooled result:
pooling BEATS joint channel-stacking, still loses to the best single stim, kdr stays dead.

Per-stim scoring of the same model (`evaluate_voltage.py --stimIndex k`, 200 test samples each — the
protocol-agnostic CNN fed one protocol's trace at a time; ledger row = stim 0):

| eval stim | mean R² | leak | na3 | kdr | kap | km | kd | mse_z | spikes s/d |
|---|---|---|---|---|---|---|---|---|---|
| 5kChaoticRamp | **0.367** | 0.740 | 0.452 | −0.032 | 0.091 | 0.358 | 0.593 | 2.15 | 26.7/24.2 |
| 5k0chaotic4 | 0.164 | 0.912 | 0.133 | −0.277 | −0.395 | −0.168 | 0.780 | 0.99 | 6.1/7.6 |
| BBP_Exp_Step1000_i4k | −0.269 | −0.305 | 0.276 | −0.194 | −0.702 | 0.072 | −0.758 | 1.69 | 9.0/11.5 |
| chirp23a_i4k | 0.278 | 0.969 | 0.060 | −0.121 | −0.204 | 0.050 | 0.916 | 2.46 | 5.0/4.7 |

The per-stim split reproduces the observability map in-model: chaoticRamp is the ONLY stim where
na3/kap are positive, the quiet stims carry leak (0.91–0.97) and kd (0.78–0.92), and
**Step1000 recovers nothing** (mean −0.269) — the "kdr's best clean stim" premise fails again, now
inside a trained model. No stim sees kdr. On the Roy recordings the pooled model behaves like the
champions (row above): same rheobase gap (0 spikes at rig current), envelope mse_z 0.375–0.406
between dtw_80k and dtw_ica.

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

## 2026-09-03 wrap — Roy 2-param ladders, ms2ch, L5TTPC nc2 test scoring, cost tables
Written at the merge of `ca3-vo-dtwblur` into `CNN_Jaxley`. Numbers verified on disk today.

**Roy 2-param ladder (leak, kd only; `$SCRATCH/tmp_neuInv/jaxley_ca3/roy2k_vo2p/`, 2026-08-26..29).**
80k synthetic under the EXACT rig stimulus, 2-output CNN (`voltage_loss.param_subset`), then
fine-tunes on Paula's recordings. Held-out test MSE_fixed at Roy100/500/1000/1500/2000 (spikes at 2000
vs data 15.3):
| stage | Roy100 | 500 | 1000 | 1500 | 2000 | spikes@2000 |
|---|---|---|---|---|---|---|
| MSE base zero-shot (`out_vo`) | 0.011 | 0.25 | 0.80 | 1.19 | 1.78 | 4.2 |
| MSE ft Roy2000 (`out_ft`) | 0.048 | 0.23 | 0.57 | 0.74 | 1.02 | 1.0 |
| MSE ft Roy500-2000 (`out_ftall`) | **0.008** | **0.18** | **0.52** | **0.71** | 0.96 | 1.0 |
| DTW base zero-shot (`out_dtw_base`, ep100) | 0.006 | 0.20 | 2.50 | 3.94 | 5.48 | **34.3** |
| DTW ft Roy2000 (`out_dtw_ft2000`) | 0.008 | 0.18 | 0.53 | 0.73 | 0.97 | 1.0 |
| DTW ft Roy500-2000 (`out_dtw_ftall`) | 0.006 | 0.18 | 0.54 | 0.73 | **0.96** | 1.0 |
Conclusion: **capacity limit**. The DTW base fires HARD (11.7/24.1/34.3 spikes at 1000/1500/2000) and is
the best model at Roy100, yet every fine-tune collapses to ~1 spike and DTW-ftall ties MSE-ftall within
noise. With only leak+kd free at 0.497x drive the cell cannot do envelope AND spikes; fine-tuning forces
the trade and envelope wins. Loss shape is not the lever here.
DTW-ftall pathology (TensorBoard): val 0.382 at epoch 0 rose to 0.498 by 150; the eFEL aux ramps
0.05->0.2 over 20 epochs, which inflates the val metric the plateau scheduler watches, so LR was cut at
epochs 11 and 24 (5e-5 -> ~1e-8 by 115) before the ramp finished. Resumes also reset the best-val
tracker, so the scored ckpt is epoch 147. Fix for any future ramped-aux fine-tune: freeze the scheduler
during the ramp or schedule on the un-ramped term.

**ms2ch (stim as CNN input channel 2, leak+kd, `roy2k_ms2ch/`, jobs 57689690/91).** Base zero-shot is the
first synthetic-only 2-param model that FIRES at rig current: 2.2/2.6/4.5 spikes at 1000/1500/2000
(mse_fixed 1.27/1.43/1.63, mse_z 0.76/0.73/0.66). Its Roy2000 fine-tune collapses again (0/0.3/1.0,
mse_fixed 0.58/0.71/0.95). Stage M5 (ft Roy500-2000) never ran: the 08-30 relaunch died writing
`blank_model.pth` because pscratch hit 100.4% of quota (truncated 9.8 MB file). Needs one relaunch after
space is freed.

**L5TTPC ncomp=2 test scoring (2026-09-03, `l5ttpc_eval/` in the worktree; driver
`run_l5ttpc_eval_salloc.sh`, no-grad jaxley, 200 held-out samples each).** First time any of the nine
June/July L5 runs was scored:
| run | mean R² | Na soma | SKv3 soma | e_pas | Na axon | Ih dend | mse_z | spikes pred/true |
|---|---|---|---|---|---|---|---|---|
| paramonly_ft80k (supervised, 3 stims, 80k) | **0.428** | 0.94 | 0.91 | 0.86 | 0.86 | 0.57 | **0.27** | 3.3/9.4 |
| paramonly_3stim (supervised, 3 stims, 4k) | 0.335 | 0.93 | 0.90 | 0.79 | 0.77 | 0.51 | 0.83 | 3.5/6.5 |
| supervised_nc2 (1 stim, 16k) | 0.330 | 0.93 | 0.82 | 0.78 | 0.76 | 0.51 | 0.31 | 3.8/9.4 |
| hybrid_nc2 / hybrid_efel | −0.11 / −0.12 | ≤0 | ≤0 | ≤0 | ≤0 | ≤0 | 0.64 / 1.26 | 5.2 / 1.9 |
| multiprobe_nc2 (4 probes) | −0.36 | ≤0 | ≤0 | ≤0 | ≤0 | ≤0 | 0.54 | 1.5/9.4 |
| voltage_only_nc2 | −0.60 | ≤0 | ≤0 | ≤0 | ≤0 | ≤0 | 0.54 | 5.2/9.4 |
| multistim_nc2 / multistim_efel | −0.93 / −1.26 | ≤0 | ≤0 | ≤0 | ≤0 | ≤0 | 1.06 / 1.49 | 2.1 / 1.1 |
Only supervised recovers anything; every simulator-in-the-loop run (11-15 epochs, 4 h wall) is below
the prior mean on every channel. The 3-stim battery at 80k is the best L5 model (0.428, 4 min on 1 GPU);
apical SKv3/Im, somatic Ca-LVA and axonal K_Tst stay ~0 in every run (single-soma observability wall
persists; multiprobe never trained long enough to test it). All models under-fire 2-3x. Train/val curves:
`l5ttpc_eval/l5ttpc_train_val_curves.pdf`; composite overlays `l5ttpc_eval/l5ttpc_test_overlays_composite.pdf`.

**Cost tables.** `chaoticramp_runs_table.md` (every chaoticRamp run: knobs, samples, s/epoch, hours,
nodes, per-channel R²) and `all_models_time_table.md` (114 run dirs, GPU-s per sample). Totals: the 19
chaoticRamp single-stim runs = 85 train-h, 487 node-h; all 24 ledger runs = 143 train-h, 757 node-h.
Rule of thumb: CA3 in-loop solve 0.020 GPU-s/sample -> 100 epochs at N samples ≈ N x 3.5e-5 h on 16 GPUs;
L5 nc2 = 0.57 GPU-s/sample (x28), L5 3-stim = 2.2 (x110). The four 8-node jobs bought nothing over
4 nodes (221 vs 217-222 s/epoch at 200k).
