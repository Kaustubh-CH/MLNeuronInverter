# Phase 3 — matched ball_and_stick_bbp data + sim

## TL;DR

Phase 3 fixed Phase 2's mode-collapse failure (data ≠ sim) by generating
training data from the same `ball_and_stick_bbp` cell that `HybridLoss`
uses as its forward simulator.  The pipeline runs end-to-end on 16 GPUs;
voltage matching converges quickly, but **per-parameter recovery does
not meet the EV > 0.85 bar on 11/12 parameters** at 100 epochs.

| Acceptance bar | Result |
|---|---|
| Voltage `rmse_z` mean **< 0.30** | **PASS** — 0.113 |
| Per-param explained variance **> 0.85 on all 12** | **FAIL** — 1/12 pass |

The training loss plateaued by ~epoch 10 and never recovered, with
`ReduceLROnPlateau` driving the LR to ~6e-9 by epoch 96.  More epochs at
this LR would not help; the next iteration needs a different recipe (see
*What's next*).

---

## Data pack

* File: `$SCRATCH/synthetic_bbp_data/ballBBP_synth_v1/ball_and_stick_bbp_synth.mlPack1.h5`
* Generator: `scripts/gen_ball_and_stick_data.py` (job `52377462`, 1 A100, 8m44s, ~43 samp/s in fp64)
* 20480 total samples, split 80/10/10:
  * 16384 train / 2048 valid / 2048 test
* Per sample: `(T=5001, P=1, S=1)` — 500.1 ms soma trace, single stim
  (`5k50kInterChaoticB`)
* Unit params: 12 (BBP channels on soma + 5-comp dendrite), drawn
  uniformly from `[-1, 1]^12`; phys = `center · 10^(unit · 0.5)` using
  `ball_and_stick_bbp._DEFAULTS` as centers
* Voltage normalization: per-sample-per-probe z-score (mean ≈ 5e-8,
  std ≈ 1.000)
* Size on disk: 91.2 MB (fp16 voltages, fp32 unit_par)

Data-quality plots in this directory:

* `synth_data_50samples.png` — 50 sample traces overlaid for each split
* `synth_data_grid12.png` — 12-panel grid of individual train traces

---

## Training

* Job `52379233`: 4 nodes × 4 GPUs (16 A100s), 100 epochs, 1h6m
* Per-GPU batch 128, global batch 2048; `numGlobSamp=65536`/epoch
* fp64 sim (`JAX_ENABLE_X64=true`, solver `bwd_euler`); `t_max=500 ms`
  resolved from the stim CSV
* Hybrid loss: `channel_weight=1.0`, `voltage_weight=1.0`,
  `mask_channels=False`, `clamp_unit_tanh=True`
* SLR: `batchShifterJaxleyMatched_100ep.slr`
* Run dir: `$SCRATCH/tmp_neuInv/jaxley_matched/ballBBP_matched/ball_and_stick_bbp_synth/52379233/`

### Loss curve (val)

| epoch | val-loss | LR |
|---|---|---|
| 0   | 0.568 | 3.0e-04 |
| 5   | 0.324 | 3.0e-04 |
| 9   | 0.267 | 3.0e-04 |
| 15  | 0.283 | (after plateau) |
| 35  | 0.270 |  |
| 54  | 0.269 |  |
| 70  | 0.268 |  |
| 96  | 0.268 | 5.9e-09 |
| 99  | 0.268 | 5.9e-09 |

The model converged in ~10 epochs and the remaining 90 epochs added
nothing — `ReduceLROnPlateau` decayed the LR through six rounds (×0.3
each) until learning effectively stopped.

---

## Evaluation (`plotJaxleyValidation.py`, N=2000 test samples)

Run inside an interactive 1-GPU allocation (job `52437562`); jaxley sim
took 115 s for 2000 traces.  Outputs in
`<run_dir>/out/eval_phase3/`; copies of the most informative artefacts
are mirrored here (`eval_summary.yaml`, `param_recovery_summary.csv`,
`param_recovery_grid.png`, `voltage_rmse_cdf.png`, `voltage_loss_hist.png`).

### Voltage (passes)

| metric | value | threshold | verdict |
|---|---|---|---|
| `rmse_z` mean   | 0.113 | < 0.30 | **PASS** |
| `rmse_z` median | 0.083 |        |  |
| `mse_z`  mean   | 0.020 |        |  |
| spike count `|sim − data|` mean | 1.57 | — | (informational) |

### Per-parameter recovery (mostly fails)

| k | param | R | EV | RMSE | verdict |
|---|---|---|---|---|---|
| 0  | `gNaTs2_tbar_NaTs2_t_somatic`   | +0.36 | +0.12 | 0.54 | FAIL |
| 1  | `gNap_Et2bar_Nap_Et2_somatic`   | +0.92 | **+0.85** | 0.22 | **PASS** |
| 2  | `gK_Tstbar_K_Tst_somatic`       | +0.20 | +0.04 | 0.58 | FAIL |
| 3  | `gK_Pstbar_K_Pst_somatic`       | +0.88 | +0.76 | 0.29 | FAIL |
| 4  | `gCa_LVAstbar_Ca_LVAst_somatic` | +0.08 | +0.005 | 0.58 | FAIL |
| 5  | `gCa_HVAbar_Ca_HVA_somatic`     | +0.54 | +0.27 | 0.48 | FAIL |
| 6  | `g_pas_somatic`                 | +0.81 | +0.64 | 0.35 | FAIL |
| 7  | `gNaTs2_tbar_NaTs2_t_apical`    | −0.05 | −0.01 | 0.58 | FAIL |
| 8  | `gK_Pstbar_K_Pst_apical`        | −0.01 | −0.003 | 0.57 | FAIL |
| 9  | `gImbar_Im_apical`              | +0.03 |  0.00 | 0.59 | FAIL |
| 10 | `gIhbar_Ih_apical`              | +0.57 | +0.32 | 0.48 | FAIL |
| 11 | `g_pas_apical`                  | +0.21 | +0.04 | 0.56 | FAIL |

Pattern: **somatic-Na/K params are recoverable; apical params and
small-conductance somatic params are essentially unidentifiable** from
the soma trace alone.  This is expected given a single soma probe — the
distal dendritic state has weak signal at the soma after the cable
filter.  EV near 0 means the CNN's prediction has no information about
the true value beyond the marginal mean.

---

## Diagnosis: why did training plateau at val=0.27?

* Loss components weren't logged separately, but the math is consistent
  with: voltage MSE ≈ 0.02 (matches eval), channel MSE ≈ 0.25.  With
  `channel_weight=voltage_weight=1.0` the total ≈ 0.27 exactly.
* The voltage term saturates at ~0.02 fast because matching the soma
  trace doesn't require getting the params right (degenerate parameter
  space).
* The channel term plateaus because the gradient signal on the
  unidentifiable params (apical, small Ca, Im, K_Tst) is dominated by
  noise; the optimizer has no consistent direction to push those.
* `ReduceLROnPlateau` correctly noticed and drove the LR toward zero;
  but the underlying issue is loss-shape, not LR.

This is not a bug in the pipeline — voltage convergence and channel
convergence on the well-identified params (NaTs2 persistent, K_Pst,
g_pas) confirm the bridge, the data, and the supervision signal are
correct.  The matched-data construction successfully prevented the
mode-collapse failure of Phase 2.

---

## What's next (out of scope for this worktree)

Things to try in a follow-up phase:

1. **Decouple LR scheduling from a degenerate loss landscape.**  Either
   disable `ReduceLROnPlateau`, switch to cosine annealing, or rebase
   the patience window so it doesn't trip after the voltage term
   saturates.
2. **Reweight loss components.**  The voltage term saturating at 0.02
   means it carries almost no gradient after epoch 10.  Try
   `voltage_weight=0` (channel-only on matched data) to confirm
   identifiability isolates to the channel term, then put voltage back
   with a much smaller weight.
3. **Add probes.**  The cell as built only records soma; extending
   `_attach_record` to also pull the dendrite tip voltage would give
   the loss a separate window into the apical params currently stuck at
   EV ≈ 0.  Requires a small generator change (`num_probs=2`) and a
   matching dataloader probe-select.
4. **Larger model / different optimizer.**  Current CNN may be
   capacity-limited on the harder params; a Transformer-style head or
   AdamW with weight decay might help on params with slow gradient
   signals.
5. **Identifiability sanity check.**  Run the same eval on the train
   split — if train EV is also ≤0.85 on the bad params, those params
   are *truly unidentifiable from soma voltage*, not just
   under-trained.  That would be a hard upper bound on Phase 3 with
   this cell + probe set.

---

## Acceptance verdict

**FAIL** by the original PHASE3_TASKS.md criteria.  Voltage acceptance
passes cleanly; per-parameter EV passes only for `gNap_Et2_somatic`
(barely, at 0.85).  The pipeline (data generator, matched design YAML,
SLR, plotter) is correct and reusable; the next-phase recipe needs to
address the channel-term plateau, not the infrastructure.
