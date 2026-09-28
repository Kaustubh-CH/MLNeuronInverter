# Stimulus catalog (2026-09-24)

Source: `/pscratch/sd/k/ktub1999/main/DL4neurons2/stims/*.csv` (the `_STIM_DIR` of every
`toolbox/jaxley_cells/*` module). One current value per line, **nA**, **dt 0.1 ms**
(`jaxley_utils.load_stim_csv`, then `upsample_stim` → solver grid; `np.interp` holds the last
value past the end). Extra, HOME-hosted: `stims/chaoticRamp_scaled/` (10/50 kHz + pA ladder,
`scripts/make_chaoticramp_scaled.py`).

Per-file numbers (n, duration, min/max/mean/std, end value, flag, designs that name it):
**`stim_table.csv`** — regenerate with `python scripts/stim_catalog.py` (93 CSVs, 49 families
by name). One page per family: **`stim_gallery.pdf`** (`srun -n1 python scripts/stim_catalog.py --gallery`).
"Used by" = design yamls in this repo that list the stem under a `*stim*` key; generation
scripts (`gen_ca3_sharded.py --stims`, ladder packs) are not scanned.

## Before you use a stim

1. **`max|I| > 20 nA` ⇒ pA stored as nA (1000× overdrive).** Seven files, all flagged
   `PA_AS_NA` in the table: `ramp_500`, `ramp_500_i4k`, `4k50kInterramp_50khz{,_i4k}`,
   `4k50kInterstep_{200,500}_50khz`, `4k50kInterstep_500_50khz_i4k`. They drive the soma to
   +1000–2000 mV. `load_stim_csv` still has **no guard**. Every result from them is void
   (the Jul-4 kdr R² 0.985/0.991, "Kdr best = ramp_500" in `sensitivity_best_stims.md`).
2. **Length sets `t_max`.** `t_max_override: auto` reads `len(csv)`; `_i4k` = 4000 pts (400 ms),
   `5k*` = 5000 (500 ms), `BBP_Exp_Step*` (no suffix) = 6000, `chirp23a` = 23000. Mixing lengths in
   one battery needs an explicit `t_max_override`.
3. **Amplitude is an operating point, not a detail.** The L5 nc2 model's input resistance is 3× below
   the recordings' (48.7 vs 146 MΩ), so rig currents need `stim_scale ≈ 3` (README §6 of
   `docs/model_ladder/sensitivity/`). The synthetic packs use ×1.5 (CA3/L5 ICB).

## Families

| family (files) | shape | length | amplitude (nA) | purpose / status | used by |
|---|---|---|---|---|---|
| **InterChaoticB** `4k50kInterChaoticB`, `5k50kInterChaoticB` (+ `cahotic_50khz` 2 s parent, `ICB_hold_5k` = with −0.05 hold) | chaotic noise, 50 kHz-derived, decimated | 400 / 500 ms | −1.49 … 6.82 | **Main training stim.** L5 ladder (4k, ×1.5, all nc1/nc2 packs), CA3 ICB packs (5k); CA3 leak R² 0.97 / kd 0.94, poor on na3/kdr | 6 + 15 designs |
| **ChaoticRamp** `5kChaoticRamp`, `4kChaoticRamp` | InterChaoticB + a slow ramp to 6 nA | 500 / 400 ms | −0.40 … 6.00 | **CA3 voltage-only champion** (dtw 80k→400k, mean R² 0.64→0.74); na3/kap recovered, kdr ceiling. = ICB + 1-D ramp-bias (R² 0.98 decomposition) | 27 designs |
| **Roy _icaRec** `Roy{100,500,1000,1500,2000}_icaRec_5k` | the rig-**recorded** current, 100 ms lead pad | 500 ms | Roy2000: −0.75 … 3.40 | **Paula/Roy experimental protocol.** All five = `5k50kInterChaoticB × N/4022` (corr 0.9993) — one waveform, five amplitudes. Exp fine-tunes (CA3 + L5) | 15 designs |
| Roy _icav2 `Roy*_icav2_5k` | icaRec with −0.05 nA holding | 500 ms | Roy2000: −0.79 … 3.34 | CA3 1-stim ladders (08-26); holding ≈ 0 was circular (baseline-subtracted) | 9 designs |
| Roy _ica `Roy*_ica5k` | with −0.6 nA phantom hold | 500 ms | −1.35 … 2.82 | **void** (v1 "never fires" artefact) | — |
| `Roy2000_scale_5k` | Roy2000 rescale check | 500 ms | −0.74 … 3.39 | one-off | — |
| **chaotic (legacy BBP)** `chaotic3`, `chaotic4` (+`_i4k`), `5k0chaotic4`, `5k0chaotic5{A,B,C}`, `5k0chaotic_kevin{A,B}`, `chaotic_{1,2}`, `Updatedchaotic{3,4}` (160 ms), `4k50kInterchaotic_50khz` | chaotic noise, lower amplitude than ICB | 160–1000 ms | up to 3.0 (chaotic_1/2: 6.8) | DL4neurons/BBP-era; `5k0chaotic4` quiet (7.6 spikes on CA3) — in the joint4 battery that diluted spikes | 3 designs |
| **BBP_Exp_Step** `BBP_Exp_Step{200,800,1000}` (600 ms), `_i4k` 600/1000 (400 ms), `Step1000_{2,4,8}x` | single square step | 400–600 ms | 0.2 … 1.0 (8x: 8) | Step1000 dominates chaoticRamp on all 6 CA3 channels (kdr 11.9 vs 5.1 mV OAT) but trains worse (observability ≠ trainability) | 3 designs |
| step `step` (760 ms, 0.2), `step_{200,500}`, `5k0step_{200,500}`, `*_16.00x`, `Updatedstep_*` | square step | 160–760 ms | 0.2 / 0.5 (16x: 3.2 / 8) | early CA3 A7 ablations | 2–3 designs |
| ramp `ramp`, `5k0ramp`, `5k0rampScaled`, `Updatedramp`, `5kRamp8pA` | linear ramp | 160–500 ms | 0 … 0.5 (Scaled 4.0; 8pA 0.8) | legacy; **not** the pA files above | — |
| chirp `chirp`, `5k0chirp`, `5k0chirpScaled`, `Updatedchirp`, `4k50kInterchirp_50khz` | linear chirp | 160–500 ms | ±1 (Scaled ±3) | legacy | — |
| chirp_a / damp `chirp16a`, `chirp23a` (+`_i4k`), `chirp_05`, `chirp_damp{,_8k,_10k,_16k_v1}` | damped / offset chirp | 400–2300 ms | −1.38 … 2.34 (chirp_05: −0.5 … 1.5) | `chirp23a_i4k` in the joint4 battery (quiet, 4.7 spikes; drives CA3 to −190 mV) | 3 designs |
| passiveReverse `passiveReverse5k±{00..70}pA` | small ±step (pA-labelled, **correct nA values**) | 500 ms | ±0.01 … 0.07 | passive/R_in probes; `-50pA` in A7 | 2 designs |
| `he_1i_1` | see gallery | 760 ms | −0.40 … 1.67 | provenance unknown | — |
| `testSave` | same range/mean as chaotic4 (4150 pts) | 415 ms | −0.80 … 3.03 | scratch; ignore | — |

**Current (use these):** InterChaoticB (training, ×1.5), ChaoticRamp (CA3), Roy `_icaRec`
(experimental; `stim_scale` 3 for L5 nc2), BBP_Exp_Step1000 (observability reference),
passiveReverse (R_in). **Legacy:** chaotic/step/ramp/chirp `Updated*`/`5k0*` variants.
**Never:** the seven `PA_AS_NA` files and `Roy*_ica5k`.
