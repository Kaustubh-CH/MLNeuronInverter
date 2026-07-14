# CA3 voltage-only chaoticRamp — experiment ledger

Headline metric = `evaluate_voltage.py` `out/eval/summary.yaml` `channel_r2_overall`
(tanh applied). Strictly voltage-only (`channel_weight 0`, `mask_channels True`) in
every arm. **Bar to beat: mean R² 0.593, kdr 0.155, voltage_mse_z ≤ 1.98, no channel regresses.**

| arm | recipe | mean R² | leak | na3 | kdr | kap | km | kd | volt_mse_z | spike|diff| | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | pure soft-DTW (ca3_vo_chaoticramp_dtw) | **0.593** | 0.788 | 0.642 | 0.155 | 0.753 | 0.483 | 0.738 | 1.98 | 7.27 | (bar) |
| R1 | DTW + coarse→fine van-Rossum blur | 0.479 | 0.763 | 0.537 | 0.115 | 0.286 | 0.419 | 0.751 | 2.01 | 6.45 | ❌ REGRESSED |

## R1 finding (2026-07-13)
Blur is the **wrong lever**. It improved spike-rate match (|diff| 7.27→6.45) but
rate-matching ≠ kdr identifiability (kdr fell). Coarse 48 ms blur is toxic to
fast-timescale channels: **kap collapsed 0.75→0.29**, na3 dropped. Smoothing the loss
made it easier to minimize (val 0.055) but less discriminative. Abandon blur.

## Next: R3 — grad-preconditioner on the baseline DTW (no blur)
Reweight per-channel gradients (`grad_precond {sensitivity: <voltage ‖∂V/∂θ‖ vec>,
exponent: 0.5, normalize: geomean}`) to un-starve low-sensitivity channels (kdr, km)
without touching the discriminative DTW distance. Sensitivity vector from
`feature_channel_sensitivity.py --stims 5kChaoticRamp` (voltage_mse row).
