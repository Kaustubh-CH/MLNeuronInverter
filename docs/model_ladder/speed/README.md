# L5 voltage-only (in-loop) speed study — 2026-09-04

Question: how to make the L5 nc2 / nc4 voltage-only runs (0.57 GPU-s per sample per epoch at
nc2 = 28x CA3; 20 h for 100 epochs x 20k on 16 GPUs) affordable, and how fast is an ncomp=1 L5.

Method: `scripts/bench_l5_speed.py` times the EXACT training path (`JaxleyBridge.simulate_batch`
forward + backward of a voltage loss, gradient checkpointing `[outer,10]`, `5k50kInterChaoticB`,
default BBP parameters, median of 2-3 warm iterations after compile) on one A100.  Passes 1-2 on
40 GB nodes (`timing_nc*.csv`, `timing_fp32.csv`, `timing_big.csv`), pass 3 on an 80 GB node
(`timing_80g.csv`).  Accuracy of the cheaper settings: `scripts/l5_accuracy_dt_ncomp.py`
(`accuracy.md`, 33 traces = default + 32 box draws, both stims, vs ncomp 4 / dt 0.1).
The ncomp=1 physiology sweep is `../scan_l5ttpc_nc1.md`.

## GPU-seconds per sample (forward + backward, 500 ms, fp64 unless noted)

| setting | nc1 | nc2 | nc4 | note |
|---|---|---|---|---|
| B=64, dt 0.1 (the old training config) | 0.41 | **0.575** | **0.93** | matches the measured 566 s/epoch of the old nc2 run |
| B=128, dt 0.1 | 0.23 | 0.34 | 0.62 (80 GB only) | wall time per step is nearly flat in B -> throughput ~ B |
| B=256, dt 0.1 | 0.14 | 0.24 (80 GB) | OOM | |
| B=512, dt 0.1 | 0.097 (80 GB) | OOM | OOM | |
| dt 0.2 | 0.117 @128, **0.048 @512** | 0.169 @128, **0.118 @256** | **0.313 @128** (80 GB) | 2.0x for free in wall time |
| dt 0.25 | 0.039 @512 | 0.095 @256 | not run | 2.5x |
| fp32, dt 0.1 | 0.19 @128 | 0.27 @128, 0.165 @256, **0.113 @512** | 0.45 @128 | 1.25x, halves memory (so 2x batch) |
| t_max 400 ms (4k stim), dt 0.1 | | 0.27 @128 | | 1.25x (the 5k stim's first 100 ms and last 50 ms are quiet) |
| no checkpointing | | OOM @128 fp32 | | checkpointing costs no time; keep `[outer, 10]` |

Peak memory (fp64, dt 0.1): nc1 14 GB @128, 29 GB @256, 58 GB @512; nc2 24 GB @128, 48 GB @256;
nc4 22 GB @64, 44 GB @128.  fp32 halves these.  Compile: 90-300 s per config (persistent
JAX cache matters).

Forward-only for comparison: nc2 B=128 = 5.4 s per batch (0.04 GPU-s/sample); backward is 7-8x
the forward, i.e. the cost is the reverse-mode pass through 5000 implicit steps, and it is
latency-bound: doubling the batch adds only ~10-20 % wall time.  That is why batch size and
step count are the levers, and why the loss function is irrelevant to speed.

## What the cheaper settings do to the traces (`accuracy.md`)

Reference = nc4, dt 0.1.  Chaotic stimuli amplify every perturbation into spike-time jitter,
so RMSE is dominated by shifted spikes, not by changed spike shape or count.

| config | InterChaoticB: spikes within +-1 / mean shift / RMSE | chaoticRamp: within +-1 / shift / RMSE |
|---|---|---|
| nc2 dt 0.1 (current 2-comp rung) | 97 % / 4.8 ms / 5.2 mV | 64 % / 2.4 ms / 7.1 mV |
| nc4 dt 0.2 | 97 % / 2.0 ms / 5.1 mV | 67 % / 2.4 ms / 6.9 mV |
| nc2 dt 0.2 | 97 % / 5.4 ms / 5.4 mV | 55 % / 4.6 ms / 7.1 mV |
| nc4 dt 0.25 | 94 % / 3.0 ms / 6.6 mV | 45 % / 3.7 ms / 8.3 mV |
| nc1 dt 0.1 | 27 % / 8.1 ms / 7.8 mV | 42 % / 4.0 ms / 9.3 mV |
| nc1 dt 0.2 | 52 % / 8.3 ms / 7.6 mV | 48 % / 5.7 ms / 8.6 mV |

Reading: dt 0.2 perturbs a trace about as much as going from nc4 to nc2 does (both ~5-7 mV RMSE,
2-5 ms spike shifts, spike counts within +-1 for 97 % on InterChaoticB) -- an accepted level of
change.  dt 0.25 is a little worse.  ncomp=1 is a genuinely different cell (27-42 % spike-count
agreement with nc4, 8 ms shifts), although its default-parameter physiology is fine
(`scan_l5ttpc_nc1.md`: 9 / 17 spikes at x1.5, +39 / +26 mV peaks, box PASS).  Whatever setting
is used, the pack must be generated with the SAME ncomp and dt as the in-loop simulator
(self-consistency); mixing them puts a 5-9 mV floor under the voltage loss.

## Recommendation

| rung | old | proposed | GPU-s/sample | 100 ep x 20k on 16 GPUs | speed-up |
|---|---|---|---|---|---|
| nc2 | B=64 fp64 dt 0.1, 40 GB | B=256 fp64 dt 0.2 on 80 GB nodes (`-C "gpu&hbm80g"`) | 0.118 | 4.1 h | 4.9x |
| nc2 | | + fp32 (B=512) | ~0.08 (est.) | ~2.8 h | ~7x |
| nc2 | | + 400 ms stims | ~0.065 (est.) | ~2.3 h | ~9x |
| nc4 | B=64 fp64 dt 0.1 | B=128 fp64 dt 0.2 on 80 GB | 0.313 | 10.9 h | 3.0x |
| nc4 | | + fp32 (B=256) + 400 ms | ~0.19 (est.) | ~6.6 h | ~5x |
| nc1 | -- | B=512 fp64 dt 0.2 on 80 GB | 0.048 | 1.7 h | 12x vs old nc2 |

fp32 needs a training pilot first: the reason the CA3 recipe went to fp64 was NaN backward
passes on high-gNa draws (the bridge now zeroes non-finite per-sample gradients, so a pilot is
cheap).  On 40 GB nodes the numbers are B=128 dt 0.2: nc2 0.169 (3.4x), nc4 needs B=64.

Plumbing added for this: `CellSpec`/bridge accept a solver dt coarser than the 10 kHz stim
grid (`handle.out_dt`), `voltage_loss.sim_dt_override: 0.2` in a design YAML sets it, and
`HybridLoss` / `evaluate_voltage.py` decimate the pack traces to the solver grid.  Global batch
2048 = 16 GPUs x 128; B=256 per GPU on 8 GPUs (2 nodes) keeps the same global batch.

## Pilot (submitted 2026-09-05): nc2, fp32, dt 0.2, 400 ms InterChaoticB, voltage-only, 100 epochs

Design `ladder_l5ttpc_nc2_icb4k_vo_fp32dt02.hpar.yaml`; pack
`/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/` (50k = 40k/5k/5k, generated at
dt 0.2 on the 400 ms `4k50kInterChaoticB`, stim x1.5, **DL4neurons2 run.py sampling**: u ~ U(-1,1),
conductances base*10^u (+-1 decade), e_pas -85..-65 mV and cm 0.5..2 uF/cm2 LINEAR -- the
phys_par_range rows carry a 4th field "lin", handled by `jaxley_utils.unit_to_phys_*`,
`HybridLoss._unit_to_phys` and `evaluate_voltage.py`; verified against run.py to machine precision).
Physiology of the pack: rest -74, 6 % silent, spikes 6/15/23, +33 mV peaks, 0 % block.
Job: `scripts/submit_pilot_nc2_fp32dt02.sh` -> 2 x 80 GB nodes (`-C "gpu&hbm80g"`), B=256/GPU
(global 2048), `NEUINV_JAX_X64=false`, 12 h wall.  Two 2-epoch smokes on 4x40 GB: 77 s/epoch for
4096 samples (0.065 GPU-s/sample train + validation), finite losses; the first smoke exposed a
double decimation of packs already stored on the 0.2 ms grid (fixed: the data grid now comes from
the pack's `timeAxis.step`).  Expected ~340 s/epoch on 8 GPUs -> ~9.5 h.
Run dir: `$SCRATCH/tmp_neuInv/model_ladder/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02/l5ttpc_nc2_bbp_synth/vo_fp32dt02_100ep/`.

**Review + precision-floor check (2026-09-05, job still queued, so it picks up the fixes):** a code
review of the session's changes found two real bugs on the eval / legacy-pack path -- the stim
scale was taken from the cell module instead of the pack (every pre-ladder pack or model of the
rescaled cells would have been simulated at the wrong current), and a finer-than-pack or
non-integer solver grid was silently accepted -- plus six smaller ones; all fixed, see
`docs/model_ladder/README.md` "How the stimulus scale works".  None of them changes this pilot
(pack scale 1.5 == module 1.5, grid ratio 1).  **fp32 floor:** through `build_hybrid_loss` on
this pack, loss(theta_true) = -0.12722 in fp64 and -0.12722 in fp32 (4 test samples, CPU);
z-MSE(sim@theta_true, pack) 0.0000 (one sample 7e-6), identical spike counts, max |dV| 1.8 mV at
one spike peak.  The CA3 fp64-gen/fp32-loss floor (0.38) does NOT apply to L5 nc2 bwd_euler
dt 0.2.  `scripts/run_ca3_gen.sh` now takes `GEN_X64=false` for an all-fp32 pack and records
`simu_info.fp64` truthfully; `HybridLoss` prints a NOTE when pack and loss precision differ.

### Pilot outcome (job 57948690, ran 2026-09-08 05:50-15:26, 9.6 h)

Speed as predicted: **340.5 +/- 0.4 s/epoch** on 8 x A100-80GB for 40k samples (0.068 GPU-s/sample
incl. validation), 100 epochs in 9.5 h, **no NaN or divergence in fp32** (train loss 0.56 -> 0.157,
val 0.53 -> 0.132; best val 0.109 at epoch 44).  The plateau scheduler (patience 10, factor 0.3)
cut the LR five times after the eFEL ramp finished (1e-4 -> 3e-5 @ep33 -> 9e-6 @ep53 -> ... ->
2.4e-7 @ep94), so nothing moved after epoch ~55; final val 0.132 is 0.26 above the measured
loss(theta_true) floor of -0.127.
**Recovery failed:** test channel R2 = **-0.88** (every one of the 19 parameters negative; the raw
CNN output spans [-5.4, 6.1] and the tanh-clamped predictions sit at +-1.00 for most parameters),
voltage MSE_z 0.90 mean / 0.85 median, spikes sim 6.1 vs data 9.5.  Same picture as the earlier
L5 in-loop runs (all < 0) and the supervised nc2 on this cell got 0.655 -- the CA3 voltage-only
recipe does not transfer to the 19-parameter L5 cell; the speed levers work, the objective does
not.  Next levers if this is pursued: freeze the LR scheduler during the aux ramp (memory:
Roy ladders), hybrid/supervised-anchored loss, or multi-probe supervision.

### Supervised twin on the pilot pack (job 58131043, interactive, 2026-09-09)

`ladder_l5ttpc_nc2_icb4k_sup.hpar.yaml` (param-MSE, same 50k pack, 100 ep, 2 min of training + eval):
**mean R2 0.556** (best val at ep 30), voltage MSE_z 0.70 mean / 0.69 median, spikes sim 5.7 vs data 9.5.
So the pack is learnable (0.556 vs 0.655 for the 500 ms dt-0.1 pack with the geometric box) and the
voltage-only pilot's -0.88 is the objective, not the data.  Note the supervised model under-fires by
the same ~4 spikes as the pilot -- the re-simulated spike count is very sensitive to small parameter
errors on this cell, so spike-count mismatch alone does not separate the two.

### 200k run (submitted 2026-09-09): 50 epochs, 8 x 80 GB nodes

Pack `l5ttpc_nc2_icb4k_bbp_dt02_250k` (200k/25k/25k, same recipe as the pilot pack, generated on
interactive job 58131379 by `scripts/run_200k_gen_salloc.sh`).  Design
`ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_200k.hpar.yaml` = the pilot recipe with batch 128/GPU (global
4096 on 32 GPUs; 2440 optimizer steps vs the pilot's 1950), 50 epochs, `plateau_patience: 25` so the
scheduler cannot repeat the pilot's LR collapse.  Job 58131394 (`scripts/submit_nc2_200k_fp32dt02.sh`,
`--dependency=afterok` on the generation), 8 x hbm80g, 12 h wall; estimate ~11-12 min/epoch ->
~9.5 h, 300 GPU-h.  Run dir
`/pscratch/sd/k/ktub1999/tmp_neuInv/model_ladder/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_200k/l5ttpc_nc2_bbp_synth/vo_fp32dt02_200k_50ep/`.

### Roy v2 experimental predictions with the pilot (2026-09-09, )

Zero-shot (interactive 58132371) and a voltage-only fine-tune on Paula's Roy v2 recordings (805 traces,
44 neurons; dt-0.2 twin pack `RoyExpPack_l5dt02`), scored on the 72 held-out test traces with
`plot_exp_overlay_royv2_l5dt02.py` (re-sim under each family's recorded current Roy<amp>_icaRec_5k at
scale 1.0, dt 0.2).  Fine-tune = the CA3 exp champion recipe on the 19-par L5 model
(`l5nc2_royexp_ft_dt02_efel5_ema.hpar.yaml`: DTW 1 + soft-eFEL x5 STRONG_FEATURES, per-family stimulus via
`stim_from_label` -- ported from CNN_Jaxley into this branch's HybridLoss -- fp32, LR 1e-4, 16 epochs,
9.3 min/epoch because every mixed-amplitude batch needs one L5 sim per family).  Val loss 10.15 -> 6.19,
still descending at the last epoch; the 40-epoch LR 5e-5 twin is job 58132377 (regular queue).

| family | n | spikes data | zero-shot | fine-tuned | mse_z zero-shot | fine-tuned | dtw_z zero-shot | fine-tuned |
|---|---|---|---|---|---|---|---|---|
| Roy500 | 18 | 3.7 | 0.0 | 0.7 | 0.61 | 1.41 | 0.19 | 0.97 |
| Roy1000 | 18 | 8.3 | 0.0 | 0.1 | 0.59 | 1.27 | 0.14 | 0.72 |
| Roy1500 | 18 | 12.5 | 0.0 | 2.8 | 0.47 | 1.26 | 0.11 | 0.68 |
| Roy2000 | 18 | 15.3 | 0.8 | 6.1 | 0.97 | 1.18 | 0.60 | 0.61 |

Zero-shot the pilot reproduces rest and the sub-threshold envelope but is silent (the quiet basin every
synthetic-only model started in on CA3 too).  Sixteen epochs of the rate-dominated fine-tune move it
toward the data's firing at the high amplitudes (6.1 of 15.3 at Roy2000, 2.8 of 12.5 at Roy1500) at the
price of the envelope (mse_z/dtw_z roughly double) -- the same trade the CA3 efel5 runs made before their
EMA/stab variants converged.  Figures: `exp/roy_before_after_l5dt02.png` (A zero-shot / B fine-tuned),
per-run `out/exp_royv2_l5dt02/` (per-family PDFs, CSVs, unit-param boxplots).
Gotcha: an exp fine-tune's `sum_train.yaml` carries the exp pack's 6-par CA3 meta, no `parName`;
the overlay script now takes the box from `voltage_loss.phys_par_range` and names from the cell module.
**Diagnosis from the overlays:** the fine-tuned re-sims sit at -40..-50 mV rest (data -65..-70) with broad
slow spikes and extra spikes in the quiet windows -- the rate-weighted eFEL set has nothing pinning the
resting level while e_pas_all/g_pas are free, so raising rest is the cheapest route to spikes.  The queued
40-ep twin was therefore re-submitted as `l5nc2_royexp_ft_dt02_efel5_vb` (voltage_base added to
efel_features; job 58132377 cancelled, replacement submitted 2026-09-09 evening).

### 200k run outcome (job 58131394, ran 2026-09-12 01:29-11:13, 8 x 80 GB nodes)

694 s/epoch on 32 GPUs (estimate was 660-720), 50 epochs in 9.7 h, LR held at 1e-4 the whole run
(patience 25 never fired).  Val 0.187 -> best **0.132 at epoch 27**, then drifted up (0.15-0.24 by
epochs 47-49; train 0.17 -> 0.23 in the last three epochs, i.e. late instability, not overfit).
Eval of the best checkpoint: **test channel R2 -0.81** (all 19 negative), voltage MSE_z 0.93, spikes
5.5 vs 9.6.  Five times the data, kernel 128 and no LR collapse reproduce the pilot's -0.88 almost
exactly -- the voltage-only objective, not data volume, schedule or receptive field, is what fails on
this cell (supervised on the same recipe of pack: 0.556).

### 40-epoch exp fine-tune with voltage_base (job 58136907, 2026-09-12, 6.1 h)

Val 9.35 -> best 8.77 at epoch 6 -> flat 9.29 from epoch ~16 on; the plateau scheduler cut LR 5e-5 ->
1.5e-5 -> 4.5e-6.  Adding voltage_base (scale 5.5 mV) removed the cheap "raise the rest" route the
16-epoch plain run took (val 6.19) and the model then found nothing else to improve -- it stayed near
the zero-shot solution.  Overlay/scores: `exp/zeroshot_ft_vb.log` and the run's
`out/exp_royv2_l5dt02/` (scored 2026-09-15).

40-ep +voltage_base twin, scored 2026-09-15 on the 72 test traces:

| family | scores |
|---|---|
| Roy500/1000/1500/2000 | spikes 0.0 / 0.0 / 0.0 / 1.3 (data 3.7/8.3/12.5/15.3) | mse_z 0.60 / 0.59 / 0.50 / 1.03 | dtw_z 0.19 / 0.15 / 0.14 / 0.53 |

200k k128 model ZERO-SHOT on the recordings (scored 2026-09-15, best-val ckpt ep27; `out/exp_royv2_l5dt02/` of the 200k run):

| family | scores |
|---|---|
| Roy500/1000/1500/2000 | spikes 0.0 / 0.0 / 0.0 / 0.0 (data 3.7/8.3/12.5/15.3) | mse_z 0.61 / 0.60 / 0.50 / 0.46 | dtw_z 0.19 / 0.16 / 0.14 / 0.13 |
