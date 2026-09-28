# c = 3 exp fine-tune of the 200k L5 nc2 model (2026-09-24)

Question: does fine-tuning the 200k voltage-only L5 nc2 model (`vo_fp32dt02_200k_50ep`, job 58131394) on the
Roy v2 recordings at `stim_scale 3` (the R_in-matched scale, `../README.md` §6) fix the under-firing that the
×1 fine-tune left in place?

Runs (1 node, 40 ep, LR 5e-5, recipe `l5nc2_royexp_ft_dt02_efel5_vb` = soft-DTW + soft-eFEL×5 incl. voltage_base):
| label | design | warm start | scale | job |
|---|---|---|---|---|
| ft200k_c3 | `l5nc2_royexp_ft_dt02_efel5_vb_c3` / `ft200k_40ep` | 200k | 3.0 | 58842880 |
| ft200k_x1 | `l5nc2_royexp_ft_dt02_efel5_vb` / `ft200k_40ep` | 200k | 1.0 | 58842882 (control) |
| pilotft_x1 | `l5nc2_royexp_ft_dt02_efel5_vb` / `ft_efel5vb_40ep` | 100-ep **pilot** | 1.0 | 09-09 |
| zs200k_{x1,c3} | — (zero-shot 200k) | — | 1 / 3 | — |

Launch `scripts/submit_l5_royexp_ft_200k.sh`; score `STAGE=ref|ft bash scripts/run_royexp_ft_compare_salloc.sh`
(overlay `plot_exp_overlay_royv2_l5dt02.py` → `roy_traces.npz`; table `scripts/compare_royexp_ft.py`).
Test split: 18 traces per family (Roy500–2000). R_in = `scripts/rin_fit.py` against the **unscaled rig current**
(MΩ per rig-nA, same metric as the recordings), median over fits with r² ≥ 0.5.

## References (compare_ref.csv)

| model | Roy500 spikes | Roy1000 | Roy1500 | Roy2000 | spike RMSE @2000 | mse_fixed @2000 | R_in med (500/1000/1500/2000) |
|---|---|---|---|---|---|---|---|
| recordings | 3.7 | 8.3 | 12.5 | 15.3 | — | — | 169 / 140 / 138 / 136 |
| pilotft_x1 | 0.0 | 0.0 | 0.0 | 1.3 | 14.4 | 1.47 | 36 / 39 / 39 / 41 |
| zs200k_x1 | 0.0 | 0.0 | 0.0 | 0.0 | 15.6 | 1.41 | 39 / 39 / 39 / 40 |
| zs200k_c3 | 0.0 | 1.3 | 3.9 | 7.8 | 8.0 | 0.86 | 116 / 125 / 101 / 89 |

- The ×1 exp fine-tune did not move R_in (≈ 39 MΩ, below even the BBP default 48.7) and stays silent through
  Roy1500: at ×1 the recordings are outside what the model can reach, as §6 predicted.
- **Only re-scaling, no training**, the 200k model at c = 3 already halves the Roy2000 spike error, cuts
  mse_fixed 40 %, and lands R_in at 89–125 (data 136–169). Remaining gap: fires about half the data's spikes, and
  R_in falls with amplitude faster than the data's (active conductances opening).
- Overlay (`overlay_zs200k_c3.png`): at c = 3 the model fires inside the big 190–280 ms burst window and
  tracks the sub-threshold envelope, but misses the EARLY spikes (80–170 ms) where the data fire off small
  depolarisations; its bumps peak near −50 mV → threshold/rheobase still too high. One Roy1000 trace ends in a
  depolarised plateau (≈ −45 mV) after the burst.
- mse_z / dtw_z are shape-only (z-scored) and hardly separate the models; mse_fixed and spikes do.

## First launch lost to the pscratch quota (2026-09-25/26)

58842880 (c3) / 58842882 (x1) / sub10 58809881 all FAILED in `torch.save` with pscratch at 102.5 % of quota.
c3 trained 11 epochs first (≈ 560 s/epoch on 4 GPUs): val 4.61 (ep 0, the saved ckpt) → 4.70, 4.91, 4.91, 5.13,
5.19, 5.45, 5.54, 4.92, 5.16, 5.07; train flat 4.9–5.3 — **no improvement past epoch 0 at LR 5e-5**.
Relaunched with every write in HOME (`scripts/submit_l5_royexp_ft_200k.sh` now defaults
`NEUINV_TMP_ROOT=/global/homes/k/ktub1999/tmp_neuInv`): c3 **58895462**, x1 **58895463**.

Epoch-0 ckpt scored (`compare_ep0.csv`, `overlay_ft200k_c3_ep0.png`; metadata reconstructed from the 200k
`sum_train.yaml` + the c3 voltage_loss block, `tmp_neuInv/royexp_ft_c3_ep0_model/`):

| model | spikes 500/1000/1500/2000 | spike RMSE @2000 | mse_fixed @2000 | R_in med |
|---|---|---|---|---|
| zs200k_c3 | 0 / 1.3 / 3.9 / 7.8 | 8.0 | 0.86 | 116 / 125 / 101 / 89 |
| ft200k_c3 ep 0 | 0 / 1.4 / 3.8 / 7.8 | 8.2 | 0.93 | 107 / 128 / 108 / 90 |

One epoch of exp fine-tuning ≈ the zero-shot c3 model (within noise; mse_fixed slightly worse). All of the
gain so far is the scale itself.

Relaunch 58895462/3 died at init (DDP FileStore in HOME: no flock → DistStoreError 524); rendezvous moved to node-local `/tmp`.

## Fine-tunes (compare.csv) — jobs 58902670 (c3) / 58902671 (x1), COMPLETED 2026-09-26

Both 40 ep, ≈ 6.2 h on 1 node (≈ 565 s/epoch).  Curves (train / val):
- c3: train 5.14 → 4.85, val 5.13 → 4.90 (noisy 4.28–5.84; best ckpt = ep 19, val 4.28 — a single-batch
  outlier, valid = 80 traces in 1 step); LR cut to 1.5e-5 at ep 30.
- x1: train 10.27 → 9.59, val 9.42 → 8.91 (best ep 8, 8.30); LR cut at ep 19 and 32.

| model | spikes 500/1000/1500/2000 | spike RMSE 1000/1500/2000 | mse_fixed @1500/@2000 | R_in med 500/1000/1500/2000 |
|---|---|---|---|---|
| recordings | 3.7 / 8.3 / 12.5 / 15.3 | — | — | 169 / 140 / 138 / 136 |
| **ft200k_c3** (ep 19) | 0 / 2.8 / 4.6 / 7.4 | 7.95 / 9.48 / 8.71 | 0.71 / 0.93 | 116 / 126 / 102 / 80 |
| zs200k_c3 | 0 / 1.3 / 3.9 / 7.8 | 7.99 / 9.53 / 8.03 | 0.68 / 0.86 | 116 / 125 / 101 / 89 |
| ft200k_c3 ep 0 | 0 / 1.4 / 3.8 / 7.8 | 8.28 / 9.69 / 8.21 | 0.79 / 0.93 | 107 / 128 / 108 / 90 |
| ft200k_x1 (ep 8) | 0 / 0 / 0 / 0 | 9.28 / 13.21 / 15.59 | 1.00 / 1.61 | 40 / 45 / 46 / 66 (3 fits) |
| pilotft_x1 | 0 / 0 / 0 / 1.3 | 9.28 / 13.21 / 14.42 | 1.02 / 1.47 | 36 / 39 / 39 / 41 |

**Verdict.**
1. **The c = 3 exp fine-tune is ≈ the zero-shot c = 3 model.** +1.5 spikes at Roy1000, +0.7 at Roy1500, −0.3 at
   Roy2000; spike RMSE and mse_fixed within ±0.1; R_in unchanged except Roy2000 (89 → 80, further from data).
   Everything c = 3 buys comes from the scale; 40 epochs of soft-DTW + soft-eFEL×5 on 805 traces add nothing
   measurable. Train loss moved 6 %.
2. **Not a box/tanh limit:** predicted unit params sit well inside [−1, 1] (`unit_params_ft200k_c3.png`, |u| mostly
   < 0.5). The Na conductances that would recruit the early spikes stay LOW (NaTa_t_axonal ≈ −0.5, somatic
   NaTs2_t ≈ −0.25) — the objective is not pushing them up, i.e. the gradient toward firing is weak/absent.
3. **×1 fine-tune from 200k is worse than useless:** silent everywhere, and at Roy2000 it enters a depolarisation
   block (≈ −25 mV plateau at 190–210 ms, mse_z 1.42, only 3/18 R_in fits valid). At ×1 the data are out of reach
   (§6), and the fine-tune finds the block as its best compromise for the burst envelope.
4. Shared failure (`overlay_ft200k_c3.png`): the early 80–170 ms spikes that the cells fire off small
   depolarisations are never produced (bumps top out near −50 mV), and several traces drift up after the burst
   (Ca/plateau tail).

**Next levers (not run):** (a) a fine-tune that can actually move — higher LR (1e-4–3e-4) and/or a spike-count /
time_to_first_spike term weighted up, since train loss barely moved; (b) retrain the SYNTHETIC model at c = 3
(Roy stims × 3 pack) so the prior already covers the recorded regime, instead of asking 805 traces to do it;
(c) threshold: the missed early spikes point at axonal Na / Nap / rest, where the box centre may be too
unexcitable (§6 "box-centre question").
