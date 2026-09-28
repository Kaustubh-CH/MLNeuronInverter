# Model ladder: physiological cell models for InterChaoticB and chaoticRamp

Date: 2026-09-03.  Branch `ca3-vo-dtwblur`.

Four rungs of model complexity, each with a jaxley cell that must respond
physiologically to the two training stimuli `5k50kInterChaoticB` and
`5kChaoticRamp`, at its default parameters AND across the parameter box the packs
sample (+-0.5 decade around the defaults).  CA3 (`ca3_pyramidal`) is the reference
rung the user already accepted and is NOT changed.

| rung | registry name | comps | params | what was wrong (physiology/cells_default_physiology.md) | fix |
|---|---|---|---|---|---|
| point process | `single_comp` (`soma_only.py`) | 1 | 3 (HH) | driven by the 6.8 nA L5-scale templates: Vmin -271 mV, 15 ms "spikes" | stim scaled x0.05 |
| ball and stick | `ball_and_stick` | soma + 1 stick comp | 4 (HH) | same overdrive: Vmin -251 mV | stim scaled x0.05 (same current as the point neuron) |
| ball and stick (BBP channels) | `ball_and_stick_bbp` | soma + stick | 12 -> 13 | one spike then a -25 mV plateau under EVERY stimulus (0.98 S/cm2 NaTs2_t against K_Tst/K_Pst only); Vmin -340 mV | + somatic **SKv3_1** (BBP Kv3 repolariser, trainable, 0.30 S/cm2) + fixed somatic Ih 8e-5; stim x0.07 |
| 2 comp (L5, ncomp=2/branch) | `l5ttpc`, `L5TTPC_NCOMP=2` | ~1000 | 19 | `_build` never set the BBP conductances, so the default cell (1e-5 placeholders) never fired; scan box also log-scaled e_pas/cm by +-0.5 decade (e_pas -24..-237 mV) | BBP `_DEFAULTS` set in `_build`; per-key ranges (cm x0.5-2, e_pas -85..-65 mV); stim x1.5 |
| 4 comp (L5, ncomp=4/branch) | `l5ttpc`, `L5TTPC_NCOMP=4` | ~2000 | 19 | same | same |

## How the stimulus scale works

`CellSpec.stim_scale` (new, default 1.0) multiplies the stim CSV once, inside
`JaxleyBridge._build_handle`.  Data generation (`scripts/gen_ca3_sharded.py`,
`scripts/gen_ball_and_stick_data.py`), the in-loop `HybridLoss`, `evaluate_voltage.py`
and every plotting script go through that handle, so they all see the identical
current.  Each cell module owns its value (`_STIM_SCALE`) and every pack records it in
`meta.simu_info.cell_spec._STIM_SCALE` (+ `_NCOMP`).  Explicit overrides
(`get_handle(..., stim_scale=x)`) exist only for sweeps.  The waveform SHAPE, which is
what the CNN sees, is unchanged; only the amplitude is matched to the cell's rheobase
(a 13 pF HH soma needs ~0.3 nA where a 500 pF L5 cell needs ~7 nA).

**Which scale a run simulates at (fixed 2026-09-05).** The module's `_STIM_SCALE` is only the
default for NEW generation; the generator records it in the pack meta as
`simu_info.cell_spec._STIM_SCALE`.  `build_hybrid_loss` and `evaluate_voltage.py` resolve the
training/eval scale as `voltage_loss.stim_scale` (design pin) > pack meta `_STIM_SCALE` > 1.0
(NEURON packs and the pre-ladder jaxley packs, all generated at the raw CSV amplitude) and warn
when that differs from the module (`HybridLoss.resolve_stim_scale`).  Before this fix every
legacy pack/model of the rescaled cells (the L5TTPC nc2 supervised R2 0.428 scoring, ball/ballBBP
synth v1, the NEURON packs behind `ballBBP_voltage_only`) would have been simulated at the
module's new scale.  The bridge cache key resolves None to the spec scale, so `get_handle(...)`
and `get_handle(..., stim_scale=<spec value>)` share one compiled handle.  Related fixes from the
same review: `HybridLoss.time_decim_factor` raises on a solver grid finer than the pack grid or a
non-integer ratio (was silently 1 / banker's-rounded); `evaluate_voltage.py --cellSim` takes the
solver dt and precision from the pack meta; its trace time axis uses the real grid (was 0.1 ms);
data probes are matched to cell recordings by name (pack `probe_names` vs module `PROBE_NAMES`);
every remaining unit->phys site (`plotJaxleyValidation`, `plot_exp_overlay*`,
`feature_channel_sensitivity`, `excitability_probe_icarec`) honours the linear ("lin") rows.
`toolbox/unitParamConvert*.py` are BBP-format only (`[lo, hi, unit]` rows) and must not be fed a
jaxley pack.

## PASS/FAIL criteria (`toolbox/physio_stats.py`)

Default trace: rest in [-85, -55] mV; >= 3 spikes; spike peak in [0, +55] mV; width at
half height 0.2-4 ms; AHP <= -45 mV; no depolarisation block (>= 50 ms above -20 mV);
Vmin >= -105 mV; Vmax <= +60 mV.
Box (128 uniform draws, +-0.5 decade): <= 40 % silent traces, <= 10 % block, <= 5 % of
traces out of range (> +60 or < -120 mV), 0 non-finite, median >= 2 spikes.
Reference: Paula's CA3 recordings (`physiology/ca3_packs_physiology.md`): rest -72, spikes
5 / 17 / 22 (med / p95 / max), peak +21, width 5 ms, AHP -68.

## Sweep (`scripts/stim_scale_scan.py`, GPU, fp64; tables `scan_<rung>.md`, figures `scan_<rung>.png`)

Chosen scale in bold.  "def" = default parameters, "box" = 128 draws.

| rung | scale | max I (nA) icb / cr | icb def spikes, peak, AHP, Vmin | icb box silent %, spikes med/p95, oor % | cr def spikes, peak, AHP | cr box silent %, spikes med/p95, block % | verdict |
|---|---|---|---|---|---|---|---|
| single_comp | 0.02 | 0.14 / 0.12 | 5, +38, -76, -76 | 35, 5/33, 0 | 6, +34, -76 | 34, 5/36, 0 | PASS (sparse) |
| single_comp | **0.05** | 0.34 / 0.30 | 8, +40, -76, -77 | 5, 8/34, 0 | 19, +30, -75 | 13, 15/39, 0 | **PASS** |
| single_comp | 0.10 | 0.68 / 0.60 | 12, +39, -77, -79 | 0, 12/35, 0 | 24, +25, -74 | 10, 22/42, 0 | PASS (Vmin -103 in box) |
| single_comp | 0.15 | 1.02 / 0.90 | 16, +38, -79, -88 | 0, 14/35, 2 | 27, +21 (2 ms wide) | 5, 24/45, 0 | PASS but box Vmin -123 |
| ball_and_stick | **0.05** | 0.34 / 0.30 | 8, +39, -76, -77 | 10, 8/33, 0 | 14, +29, -74 | 20, 14/38, 0 | **PASS** |
| ball_and_stick | 0.07 | 0.48 / 0.42 | 8, +40, -76, -77 | 5, 10/34, 0 | 20, +27, -74 | 16, 20/40, 0 | PASS |
| ball_and_stick_bbp | 0.05 | 0.34 / 0.30 | 2, +45, -89, -96 | 1, 1/3, 0 | 7, +44, -84 | 0, 5/11, 3 | FAIL icb (2 spikes) |
| ball_and_stick_bbp | **0.07** | 0.48 / 0.42 | 4, +45, -88, -100 | 0, 2/5, 0 | 8, +44, -83 | 0, 6/13, 3 | **PASS** |
| ball_and_stick_bbp | 0.10 | 0.68 / 0.60 | 5, +44, -90, -103 | 0, 3/6, 0 | 9, +45, -83 | 0, 6/15, 3 | PASS (box Vmin -116) |
| ball_and_stick_bbp | 0.15-0.2 | 1.0-1.4 | 6-7, Vmin -122..-136 | oor 51-80 % | 11-12 | 0, 8/18 | FAIL (hyperpolarisation) |
| l5ttpc nc2 | 1.0 (native) | 6.8 / 6.0 | 2, +30, -73, -92 | 4, 2/6, 0 | 9, +24, -62 | 2, 9/16, 0 | FAIL icb def (2 spikes); box PASS |
| l5ttpc nc2 | **1.5** | 10.2 / 9.0 | 8, +37, -79, -90 | 0, 7/11, 0 | 15, +26, -61 | 1, 14/23, 0 | **PASS** |
| l5ttpc nc2 | 2.0 | 13.6 / 12.0 | 10, +40, -81, -96 | 0, 11/13, 0 | 19, +27, -60 | 0, 17/29, 0 | PASS |
| l5ttpc nc4 | 1.0 (native) | 6.8 / 6.0 | 2, +34, -80, -92 | 0, 2/5, 0 | 10, +25, -61 | 0, 9/14, 0 | FAIL icb def (2 spikes); box PASS |
| l5ttpc nc4 | **1.5** | 10.2 / 9.0 | 7, +40, -82, -91 | 0, 7/11, 0 | 16, +27, -62 | 0, 14/20, 0 | **PASS** |
| l5ttpc nc4 | 2.0 | 13.6 / 12.0 | 10, +40, -82, -95 | 0, 10/13, 0 | 18, +29, -61 | 0, 17/26, 0 | PASS |
| l5ttpc nc1 (2026-09-04) | 1.0 | 6.8 / 6.0 | 3, +37, -77, -91 | 0, 3/9, 0 | 9, +26, -62 | 3, 10/19, 0 | PASS both (icb sparse) |
| l5ttpc nc1 | **1.5** | 10.2 / 9.0 | 9, +39, -79, -91 | 0, 8/12, 0 | 17, +26, -60 | 0, 15/26, 0 | **PASS** |
| ca3_pyramidal (reference, unchanged) | 1.0 | 6.8 / 6.0 | 6, +40, -80, -98 | 15, 7/26, 0 | 34, +23, -28, **BLOCK** (4.5 ms wide) | 0, 29/49, **73 % block** | icb PASS; **cr FAIL** |
| ca3_pyramidal | 0.25-0.5 | 1.7-3.4 / 1.5-3.0 | 0 spikes (icb) | 47-77 silent | 8-23, AHP -45..-36 | 35-7, block 5-20 % | no single scale passes both |

Notes.
* The BBP-channel cells (ball_and_stick_bbp, L5) are intrinsically sparse under
  InterChaoticB: 2 spikes at the native amplitude is the real BBP L5 behaviour
  (the old `l5ttpc_multistim` pack also has median 2).  The ladder scales are chosen so
  that every rung fires 4-8 spikes on InterChaoticB and 8-19 on chaoticRamp, which is
  what the CA3 reference does at 1.0.  To use the native BBP protocol for L5 set
  `l5ttpc._STIM_SCALE = 1.0` (one line).
* CA3 + chaoticRamp: the ramp ends at 4 nA DC; the default CA3 cell (50x50 um soma) sits
  in depolarisation block for the last ~150 ms and 72-75 % of every existing chaoticRamp
  pack is block (`physiology/ca3_packs_physiology.md`).  A ~0.3 x scale fixes the ramp but silences
  InterChaoticB, so a per-STIM scale (or the pA ladder in `stims/chaoticRamp_scaled/`) is
  the honest fix for CA3.  Left unchanged here because every CA3 model to date was trained
  at 1.0; flagged for the user.
* The old `synthetic_ball_data`, `synthetic_bbp_data` and `l5ttpc_jaxley_nc2_data` packs
  were per-trace z-scored (mean/std not stored), so `physiology/cells_default_physiology.md` de-normalised
  them with the wrong constants ("peak 61 mV, 70 % out of range" for L5 nc2 is that
  artifact, not the cell).  The scan now detects such packs and reports them as
  "mV NOT recoverable".  They are superseded by the ladder packs anyway (12-param BBP
  ball-and-stick, unscaled stims).

ncomp=1 L5 (`L5TTPC_NCOMP=1`, same cell file, x1.5) passes the same criteria and is the
cheapest L5 (`speed/README.md`), but its traces differ more from nc4 than nc2's do.

## Packs (`scripts/gen_model_ladder.slr`, `/pscratch/sd/k/ktub1999/model_ladder_data/<rung>_<stim>/`)

50 000 samples each (40k / 5k / 5k), fixed-scale normalisation (mean -60.1, std 19.0 mV),
+-0.5 decade box (L5: cm x0.5-2, e_pas -85..-65 mV), 5001 points at 10 kHz, fp64 solve.
`ball_and_stick` packs carry 2 probes (soma, dend).  Scan with
`LADDER_ONLY=1 python scripts/scan_cells_default_physiology.py docs/model_ladder/packs`.

**Two L5 sampling schemes.** The ladder L5 packs here (nc2/nc4, 2026-09-03/04) sample e_pas/cm
with 3-field rows (geometric centre + log span over -85..-65 mV and 0.5..2 uF/cm2, inverted
exponentially); the pilot pack `l5ttpc_nc2_icb4k_bbp_dt02` uses the DL4neurons2 run.py convention
(4th field "lin": e_pas -75 +- 10 mV, cm 1.25 +- 0.75, linear in u).  Each pack is self-consistent
with its own `phys_par_range` rows, but the two L5 generations do not sample the same distribution
inside the same bounds -- regenerate with the current code for a like-for-like comparison.  A pack
records its solver dt in `timeAxis.step` (0.2 for the dt02 pack), its stim scale in
`simu_info.cell_spec._STIM_SCALE` and its precision in `simu_info.fp64`.

### Pack scan (`packs.md`, 300 traces per pack, mV; `packs_pages/*.png`)

| pack | probes | rest mV | silent % | spikes med/p95/max | peak | width ms | AHP | block % | oor % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| single_comp / InterChaoticB (x0.05) | soma | -65.9 | 4 | 8 / 35 / 41 | +39 | 1.04 | -76 | 0 | 0 | PASS |
| single_comp / chaoticRamp (x0.05) | soma | -65.9 | 10 | 15 / 40 / 44 | +32 | 1.19 | -75 | 0 | 0 | PASS |
| ball_and_stick / InterChaoticB (x0.05) | soma, dend | -65.6 | 4 | 8 / 34 / 37 | +40 | 1.07 | -76 | 0 | 0 | PASS |
| ball_and_stick / chaoticRamp (x0.05) | soma, dend | -65.6 | 9 | 16 / 39 / 41 | +34 | 1.19 | -75 | 0 | 0 | PASS |
| ball_and_stick_bbp / InterChaoticB (x0.07) | soma | -76.7 | 0 | 2 / 5 / 6 | +44 | 0.35 | -87 | 9 | 0 | PASS (border) |
| ball_and_stick_bbp / chaoticRamp (x0.07) | soma | -76.7 | 0 | 4 / 14 / 19 | +43 | 0.35 | -83 | 10 | 0 | PASS (border) |
| l5ttpc_nc2 / InterChaoticB (x1.5, 25k) | soma | -74.5 | 0 | 7 / 11 / 18 | +36 | 0.49 | -80 | 0 | 0 | PASS |
| l5ttpc_nc2 / chaoticRamp (x1.5, 25k) | soma | -74.5 | 0 | 14 / 23 / 32 | +24 | 1.00 | -60 | 0 | 0 | PASS |
| l5ttpc_nc4 / InterChaoticB (x1.5, 25k) | soma | -74.5 | 0 | 7 / 10 / 19 | +37 | 0.50 | -81 | 0 | 0 | PASS |
| l5ttpc_nc4 / chaoticRamp (x1.5, 25k) | soma | -74.5 | 0 | 15 / 20 / 29 | +24 | 0.99 | -61 | 0 | 0 | PASS |

The 9-10 % "block" in the BBP ball-and-stick packs is the low-SKv3_1 corner of the box
(one spike, then the -25 mV plateau the stock cell always showed).  It is the cell family's
genuine failure mode inside a +-0.5-decade box; shrink the SKv3_1 span (e.g. 0.3) if those
traces are unwanted.  Note the packs are 40k/5k/5k after the 80/10/10 split.

All 10 packs PASS.  Full table: `packs.md`; example traces per pack: `packs_pages/`.

## Training (`scripts/make_ladder_designs.py` -> `ladder_<rung>_<icb|cr>_<sup|vo>.hpar.yaml`, `scripts/train_model_ladder.sh`)

* `sup`: supervised param-MSE with tanh-bounded outputs, 100 epochs, 1 node (minutes per
  run).  All 5 rungs x 2 stims.
* `vo`: the CA3 champion voltage-only recipe (soft-DTW + ramped soft-eFEL
  voltage_base/AP_amplitude aux, fp64, tanh, global batch 2048), 100 epochs, 4 nodes.
  HH / BBP-ball rungs only: in-loop L5 costs 0.57 GPU-s/sample (28x CA3) and only the
  supervised runs ever recovered L5 parameters (`RESULTS_vo.md` 2026-09-03).
Runs: `/global/homes/k/ktub1999/tmp_neuInv/model_ladder/<design>/<cellname>/<mode>_<stim>/out/eval/summary.yaml`
(HOME while pscratch is over quota; `NEUINV_TMP_ROOT`).  2-epoch smoke of both modes on the
point-neuron InterChaoticB pack passed end-to-end 2026-09-03 20:49 (train -> predict.py ->
evaluate_voltage.py; `sup` mean R2 0.395 after 2 epochs, `vo` sim/data spike count 14.1/13.8).
Launched 2026-09-03 20:55: `scripts/run_ladder_sup_salloc.sh` (all 10 supervised runs, one
interactive node, sequential) and the 6 `vo` 4-node sbatch jobs 57911619-57911624.

## Results — supervised param-MSE, 100 epochs, 200 test traces (`results.md`, 2026-09-03 21:00-21:46)

| rung | InterChaoticB mean R2 | chaoticRamp mean R2 | what is / is not recovered |
|---|---|---|---|
| single_comp (3 HH params) | **1.000** | **0.999** | everything (spike counts sim/data 13.8/13.8, 18.4/18.8) |
| ball_and_stick (4, soma+dend probes) | **0.926** | **0.888** | soma gNa/gK/gLeak 1.00; dendrite leak 0.71 / 0.56 |
| ball_and_stick_bbp (13) | **0.628** | **0.609** | all 8 somatic channels 0.92-0.99 (Ca_LVAst 0.60); the 5 dendritic ones ~0 (soma-only recording) |
| l5ttpc nc2 (19) | **0.655** | **0.493** | apical NaTs2 0.9, axonal NaTa/SK_E2/K_Pst 0.8-0.9, Ih 0.83, somatic block high; axonal K_Tst/Ca_HVA/Nap ~0 |
| l5ttpc nc4 (19) | **0.663** | **0.532** | same pattern as nc2 (nc4 vs nc2 differ by < 0.04) |

The old L5 nc2 supervised runs scored 0.33 (1 stim, 16k, +-1 decade box, native stim).  Full
per-parameter table below.  The six voltage-only 4-node jobs (57911619-57911624) were still
queued (Priority) when this was written; collect with `python scripts/collect_ladder_results.py`.

| rung | stim | mode | mean R2 | MSE_z mean / median | spikes sim / data | n | epochs | per-parameter R2 | run |
|---|---|---|---|---|---|---|---|---|---|
| ball_and_stick_bbp | chaoticRamp | sup | **0.609** | 1.05 / 1.05 | 5.5 / 6.7 | 200 | 99 | gNaTs2_tbar_NaTs2_t_so 0.98, gNap_Et2bar_Nap_Et2_so 0.96, gSKv3_1bar_SKv3_1_som 0.99, gK_Tstbar_K_Tst_som 0.93, gK_Pstbar_K_Pst_som 0.96, gCa_LVAstbar_Ca_LVAst_ 0.53, gCa_HVAbar_Ca_HVA_som 0.98, g_pas_som 0.90, gNaTs2_tbar_NaTs2_t_ap -0.01, gK_Pstbar_K_Pst_api -0.03, gImbar_Im_api -0.00, gIhbar_Ih_api 0.62, g_pas_api 0.10 | sup_cr |
| ball_and_stick_bbp | InterChaoticB | sup | **0.628** | 1.17 / 1.18 | 2.5 / 3.3 | 200 | 99 | gNaTs2_tbar_NaTs2_t_so 0.98, gNap_Et2bar_Nap_Et2_so 0.98, gSKv3_1bar_SKv3_1_som 0.99, gK_Tstbar_K_Tst_som 0.96, gK_Pstbar_K_Pst_som 0.97, gCa_LVAstbar_Ca_LVAst_ 0.60, gCa_HVAbar_Ca_HVA_som 0.97, g_pas_som 0.92, gNaTs2_tbar_NaTs2_t_ap -0.06, gK_Pstbar_K_Pst_api -0.03, gImbar_Im_api -0.05, gIhbar_Ih_api 0.88, g_pas_api 0.06 | sup_icb |
| ball_and_stick | chaoticRamp | sup | **0.888** | 0.79 / 0.40 | 17.1 / 17.4 | 200 | 99 | HH_gNa 1.00, HH_gK 1.00, HH_gLeak 1.00, Leak_gLeak 0.56 | sup_cr |
| ball_and_stick | InterChaoticB | sup | **0.926** | 0.55 / 0.32 | 12.6 / 12.7 | 200 | 99 | HH_gNa 1.00, HH_gK 1.00, HH_gLeak 1.00, Leak_gLeak 0.71 | sup_icb |
| l5ttpc_nc2 | chaoticRamp | sup | **0.493** | 0.11 / 0.08 | 14.4 / 18.0 | 200 | 99 | gNaTs2_tbar_NaTs2_t_ap 0.92, gSKv3_1bar_SKv3_1_api 0.39, gImbar_Im_api 0.59, gIhbar_Ih_dend 0.57, gNaTa_tbar_NaTa_t_ax 0.83, gK_Tstbar_K_Tst_ax -0.28, gNap_Et2bar_Nap_Et2_ax 0.12, gSK_E2bar_SK_E2_ax 0.87, gCa_HVAbar_Ca_HVA_ax -0.06, gK_Pstbar_K_Pst_ax 0.81, gCa_LVAstbar_Ca_LVAst_ 0.17, g_pas_ax 0.18, cm_ax 0.52, gSKv3_1bar_SKv3_1_som 0.98, gNaTs2_tbar_NaTs2_t_so 0.98, gCa_LVAstbar_Ca_LVAst_ -0.07, g_pas_som -0.06, cm_som 0.94, e_pas_all 0.96 | sup_cr |
| l5ttpc_nc2 | InterChaoticB | sup | **0.655** | 0.66 / 0.68 | 7.5 / 7.9 | 200 | 99 | gNaTs2_tbar_NaTs2_t_ap 0.94, gSKv3_1bar_SKv3_1_api 0.54, gImbar_Im_api 0.36, gIhbar_Ih_dend 0.83, gNaTa_tbar_NaTa_t_ax 0.92, gK_Tstbar_K_Tst_ax -0.14, gNap_Et2bar_Nap_Et2_ax 0.30, gSK_E2bar_SK_E2_ax 0.89, gCa_HVAbar_Ca_HVA_ax 0.21, gK_Pstbar_K_Pst_ax 0.82, gCa_LVAstbar_Ca_LVAst_ 0.37, g_pas_ax 0.64, cm_ax 0.89, gSKv3_1bar_SKv3_1_som 0.98, gNaTs2_tbar_NaTs2_t_so 0.98, gCa_LVAstbar_Ca_LVAst_ 0.54, g_pas_som 0.43, cm_som 0.97, e_pas_all 0.98 | sup_icb |
| l5ttpc_nc4 | chaoticRamp | sup | **0.532** | 0.12 / 0.10 | 14.2 / 17.2 | 200 | 99 | gNaTs2_tbar_NaTs2_t_ap 0.93, gSKv3_1bar_SKv3_1_api 0.47, gImbar_Im_api 0.43, gIhbar_Ih_dend 0.58, gNaTa_tbar_NaTa_t_ax 0.89, gK_Tstbar_K_Tst_ax -0.01, gNap_Et2bar_Nap_Et2_ax 0.51, gSK_E2bar_SK_E2_ax 0.89, gCa_HVAbar_Ca_HVA_ax 0.19, gK_Pstbar_K_Pst_ax 0.83, gCa_LVAstbar_Ca_LVAst_ 0.14, g_pas_ax 0.06, cm_ax 0.47, gSKv3_1bar_SKv3_1_som 0.97, gNaTs2_tbar_NaTs2_t_so 0.98, gCa_LVAstbar_Ca_LVAst_ 0.02, g_pas_som -0.12, cm_som 0.94, e_pas_all 0.96 | sup_cr |
| l5ttpc_nc4 | InterChaoticB | sup | **0.663** | 0.72 / 0.72 | 7.2 / 7.3 | 200 | 99 | gNaTs2_tbar_NaTs2_t_ap 0.88, gSKv3_1bar_SKv3_1_api 0.61, gImbar_Im_api 0.28, gIhbar_Ih_dend 0.84, gNaTa_tbar_NaTa_t_ax 0.91, gK_Tstbar_K_Tst_ax -0.01, gNap_Et2bar_Nap_Et2_ax 0.48, gSK_E2bar_SK_E2_ax 0.93, gCa_HVAbar_Ca_HVA_ax 0.27, gK_Pstbar_K_Pst_ax 0.89, gCa_LVAstbar_Ca_LVAst_ 0.35, g_pas_ax 0.59, cm_ax 0.83, gSKv3_1bar_SKv3_1_som 0.97, gNaTs2_tbar_NaTs2_t_so 0.97, gCa_LVAstbar_Ca_LVAst_ 0.55, g_pas_som 0.30, cm_som 0.94, e_pas_all 0.97 | sup_icb |
| single_comp | chaoticRamp | sup | **0.999** | 0.63 / 0.38 | 18.4 / 18.8 | 200 | 99 | HH_gNa 1.00, HH_gK 1.00, HH_gLeak 1.00 | sup_cr |
| single_comp | InterChaoticB | sup | **1.000** | 0.47 / 0.33 | 13.8 / 13.8 | 200 | 99 | HH_gNa 1.00, HH_gK 1.00, HH_gLeak 1.00 | sup_icb |
