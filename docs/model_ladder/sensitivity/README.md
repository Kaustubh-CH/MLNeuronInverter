# Sensitivity / identifiability of the L5TTPC nc2 model under the Roy stimuli

Run 2026-09-15, job 58383215 (1 GPU, 280 s). Outputs: `l5nc2_roy_58383215/`
(`summary.yaml`, `crb_by_stim.csv`, `sensitivity_by_stim.csv`, `feature_channel_sensitivity.csv`,
6 plots, `run.log`). Raw copy incl. `arrays.npz`: `/pscratch/sd/k/ktub1999/tmp_neuInv/sensitivity/l5ttpc_nc2_roy_58383215/`.

Tool: `sensitivity_analysis.py` (ported from CNN_Jaxley, generalised; pure-forward jitted sims,
the training bridge's VJP forward OOMs a 40 GB GPU for the 19-par cell) via
`scripts/run_sensitivity_salloc.sh` (**the python call must be wrapped in `srun` — `salloc <script>`
runs the script on the login node**).

## Setup (= the exp fine-tune physics)

| item | value |
|---|---|
| cell | `l5ttpc`, `L5TTPC_NCOMP=2`, 19 params, BBP defaults, box from the dt02 pilot pack (lin rows for cm / e_pas) |
| stimuli | Roy100 / 500 / 1000 / 1500 / 2000 `_icaRec_5k`, 500 ms, **stim_scale 1.0** (real injected current) |
| solver | bwd_euler, dt 0.2 ms, fp64 (training solves fp32; identical to 5 dp at theta_true) |
| window | skip 99.8 ms (= `sim_t_skip_ms` of the fine-tune loss), fixed-z space (`VOLT_NORM_MEAN/STD`, the loss's space) |
| operating points | 32 draws from the pilot pack test split; central differences ±0.02 unit; 1248 sims / stim |
| CRB noise | sigma 0.7 z-units (ranking / ratios are sigma-free) |

## 1. What the battery evokes in the model box

| stim | peak I (nA) | op-points that fire | spikes/trace (model box) | spikes/trace (Roy data, 18 traces) |
|---|---|---|---|---|
| Roy100 | 0.17 | 1 / 32 | 0.06 | – |
| Roy500 | 0.85 | 1 / 32 | 0.06 | 3.7 |
| Roy1000 | 1.70 | 1 / 32 | 0.06 | 8.3 |
| Roy1500 | 2.55 | 6 / 32 | 0.91 (max 13) | 12.5 |
| Roy2000 | 3.40 | 9 / 32 | 1.78 (max 17) | 15.3 |

**The Roy currents are sub-threshold for the BBP L5 nc2 model over essentially the whole trained
parameter box** (`stim_traces.png`: Roy2000 peaks at −48 mV for the first four operating points; rest
−75 mV). The recorded neurons fire 4–15 spikes at the same currents. This is the excitability
(rheobase) misspecification behind the zero-shot "silent" predictions and behind the fine-tune
finding firing only by depolarising rest.

## 2. Identifiability (Fisher / Cramér-Rao, `crb_bars.png`)

Fisher condition number: Roy100 1.9e8, Roy500 3.9e7, Roy1000 9.7e6, Roy1500 2.2e5, Roy2000 5.8e3,
**combined 3.0e3** (CA3 under its chaotic stims: 3–34).

Channels with identifiability index CRB/prior_std < 1:

| stim | identifiable |
|---|---|
| Roy100 only | 1 / 19 (NaTs2_t_soma) |
| Roy500 only | 2 / 19 (NaTs2_t_soma, NaTs2_t_api) |
| Roy1000 only | 3 / 19 (+ pas_axo, e_pas_all) |
| Roy1500 only | 3 / 19 |
| Roy2000 only | 7 / 19 |
| **all five** | **13 / 19** |

Combined, worst first (ii = CRB / prior std):

| channel | sens_z | raw mV/unit | ii_ALL | |
|---|---|---|---|---|
| Ca_LVAst_axo | 0.21 | 4.0 | 6.05 | unidentifiable (null direction −0.89) |
| K_Tst_axo | 0.27 | 5.2 | 5.63 | unidentifiable (null direction −0.46) |
| Ih_dend | 0.97 | 18.4 | 1.58 | unidentifiable |
| Ca_HVA_axo | 2.54 | 48.2 | 1.42 | unidentifiable |
| Ca_LVAst_soma | 1.12 | 21.2 | 1.33 | unidentifiable |
| SK_E2_axo | 2.84 | 53.8 | 1.27 | unidentifiable |
| cm_soma | 2.76 | 52.3 | 0.99 | borderline |
| SKv3_1_api | 2.22 | 42.0 | 0.74 | |
| cm_axo | 2.15 | 40.8 | 0.72 | |
| K_Pst_axo | 6.07 | 115.1 | 0.53 | |
| Nap_Et2_axo | 5.62 | 106.5 | 0.50 | |
| pas_soma | 3.53 | 66.9 | 0.48 | |
| Im_api | 3.96 | 75.1 | 0.45 | |
| NaTa_t_axo | 4.50 | 85.4 | 0.44 | |
| pas_axo | 4.02 | 76.2 | 0.40 | |
| e_pas_all | 5.81 | 110.0 | 0.37 | |
| NaTs2_t_api | 4.25 | 80.6 | 0.33 | |
| SKv3_1_soma | 5.73 | 108.5 | 0.30 | |
| NaTs2_t_soma | 10.55 | 199.9 | 0.11 | best; visible even at Roy100 (sub-threshold Na activation) |

No |corr| > 0.8 pair in F⁻¹ (no two-channel trade-off); the degeneracy is the near-invisibility of
the axonal Ca_LVAst / K_Tst pair (4–5 mV per unit step: nothing to see, not a trade-off).

## 3. Feature handles (`feature_heatmap.png`)

With 0–2 spikes per trace the spike features of the fine-tune loss (time_to_first_spike,
mean_frequency, inv_first_ISI, AHP_depth_abs_slow, ISI_values, AP_amplitude) are driven by the
few firing operating points. `time_to_first_spike` is dominated by Nap_Et2_axo (18.9), e_pas_all
(18.8) and K_Pst_axo (14.5): the cheapest way to move it is to depolarise rest — exactly what the
16-epoch fine-tune did (rest −40..−50 mV). `voltage_base` is carried by NaTs2_t_soma (3.0),
e_pas_all (2.3) and Ih_dend (1.5). The sub-threshold pair `steady_state_voltage_stimend` /
`voltage_deflection` is the strongest handle for SKv3_1_soma (11.4), K_Pst_axo (11.3), Nap_Et2_axo
(10.6), e_pas_all (10.7) and NaTs2_t_soma (20.4) — i.e. the information that IS in these traces
sits in the sub-threshold envelope, which the spike-centric feature set does not read.

## 4. Take-aways

1. The primary misspecification of the L5 nc2 model on the Roy data is **excitability**, not
   the loss: the model needs ~2 nA to fire where the cells fire at 0.5 nA. Any voltage-only
   objective on these traces can only push the model toward firing by moving rest / leak.
2. Identifiability of the 19-par box from soma voltage under this battery is partial (13/19,
   condition 3e3) and comes almost entirely from Roy1500/2000; Roy100/500 carry information
   about NaTs2_t_soma and the passive set only.
3. If the Roy data are to constrain active channels, the model's rheobase must first be brought
   into the data's range (a calibrated box / default set, or a stimulus-scale sweep of the nc2
   model against the data's f–I) — then rerun this analysis at that operating point.

Reproduce: `salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g --gpus-per-node=1 --cpus-per-task=32 bash scripts/run_sensitivity_salloc.sh`
(env: `STIMS`, `H5`, `STIM_SCALE`, `SIM_DT`, `TSKIP`, `NUMOP`, `EPS`, `L5TTPC_NCOMP`).

## 5. One-at-a-time (OAT) voltage-variation sweep — `l5nc2_roy_oat_58385397/` (2026-09-15)

Same instrument as the CA3 `roy2000_decomp/combined/per_stim_variation.pdf`: hold every channel at
its default, sweep ONE channel over its box (N=500 uniform draws in unit space), and measure the
mean temporal std of the soma trace (mV). `sensitivity_variation.py` (ported from CNN_Jaxley;
pure-forward sims, `stem@scale` per-stim multiplier, lin rows) via `scripts/run_sensvar_salloc.sh`
(job 58385397, 1 GPU, 22 min, 9500 sims per stim, dt 0.2, fp64). Raw run incl. `saved_traces.npz`
(decimated fp16 sweeps, re-analysable without re-simulating) and a `fisher/` copy of section 2:
`/pscratch/sd/k/ktub1999/tmp_neuInv/sensitivity_variation/l5ttpc_nc2/roy_58385397/`.

Mean temporal std (mV) when sweeping the channel alone; Roy stims at the real current (x1.0),
the training stimulus 5k50kInterChaoticB at the pack's x1.5:

| channel | Roy100 | Roy500 | Roy1000 | Roy1500 | Roy2000 | ICB x1.5 | best |
|---|---|---|---|---|---|---|---|
| NaTs2_t_apical | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.58 | ICB |
| SKv3_1_apical | 0.05 | 0.06 | 0.06 | 0.07 | 0.06 | 0.24 | ICB |
| Im_apical | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.15 | ICB |
| Ih_dend | 1.81 | 1.77 | 1.72 | 1.67 | 1.88 | 1.67 | Roy2000 |
| NaTa_t_axonal | 0.04 | 0.04 | 0.05 | 0.89 | 1.25 | 1.22 | Roy2000 |
| K_Tst_axonal | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.14 | ICB |
| Nap_Et2_axonal | 0.00 | 0.00 | 0.00 | 0.00 | 5.42 | 4.67 | Roy2000 |
| SK_E2_axonal | 0.33 | 0.33 | 0.33 | 0.34 | 0.90 | 1.29 | ICB |
| Ca_HVA_axonal | 0.00 | 0.00 | 0.00 | 0.00 | 0.43 | 0.61 | ICB |
| K_Pst_axonal | 0.50 | 0.51 | 0.53 | 0.55 | 4.76 | 4.47 | Roy2000 |
| Ca_LVAst_axonal | 0.00 | 0.00 | 0.00 | 0.00 | 0.37 | 0.67 | ICB |
| g_pas_axonal | 0.28 | 0.27 | 0.25 | 0.24 | 0.63 | 0.24 | Roy2000 |
| cm_axonal | 0.01 | 0.01 | 0.02 | 0.02 | 0.93 | 0.34 | Roy2000 |
| SKv3_1_somatic | 0.30 | 0.32 | 0.36 | 0.40 | 1.02 | 1.85 | ICB |
| NaTs2_t_somatic | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 1.53 | ICB |
| Ca_LVAst_somatic | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.27 | ICB |
| g_pas_somatic | 0.01 | 0.03 | 0.05 | 0.07 | 0.75 | 0.14 | Roy2000 |
| cm_somatic | 0.00 | 0.01 | 0.01 | 0.02 | 0.42 | 0.62 | ICB |
| e_pas_all | 3.64 | 3.65 | 3.65 | 3.30 | 2.93 | 2.66 | Roy1000 |

Reading: **Roy100–Roy1500 expose only the passive set** — e_pas (3.6 mV), Ih_dend (1.8), K_Pst
(0.5), SK_E2 (0.3), g_pas_axonal (0.3); every sodium / calcium / K_Tst / Im / cm column is
< 0.1 mV, because the default cell does not fire until ~2 nA. **Roy2000 is the first amplitude
that reaches the spiking channels** (Nap_Et2 5.4, K_Pst 4.8, NaTa_t 1.25, cm_axonal 0.9,
Ca_HVA 0.4) and it then matches or beats the x1.5 training stimulus on the axonal/passive set.
The apical channels (NaTs2_t 0.00 vs 1.58, Im, SKv3_1) and somatic NaTs2_t / Ca_LVAst are only
reached by the stronger training stimulus. Consistent with section 2 (OAT box-wide spread vs
local Fisher): the Roy battery at real current probes the passive physics of this model plus,
at Roy2000, the axonal spike generator.

## 6. Why the Roy regime is insensitive, and what makes it sensitive (2026-09-23)

**All Roy stimuli are one waveform.** `Roy<N>_icaRec_5k` = `5k50kInterChaoticB` × N/4022
(corr 0.9993 for every N; residual std < 4 % of the signal). So the training stimulus
ICB × 1.5 is effectively **"Roy6000"** (peak 10.2 nA vs Roy2000's 3.4 nA), and every
difference between the Roy columns and the ICB column in §5 is stimulus **amplitude**
(operating point), not waveform.

**Mechanism.** A conductance changes the trace only where its gates are open. Below
threshold the Na / Ca / K_Tst / Kv3 / Im gates sit at their resting open probability (~0), so
sweeping g_bar over two decades multiplies ~0 → 0.00 mV (§5). Only channels that already
conduct at rest (leak, e_pas, Ih, some K_Pst / SK) register. Sensitivity switches on
channel by channel as the drive recruits spikes: axonal initiation zone first (Roy2000 ×1), then
somatic Na/Kv3 and the back-propagating apical Na (ICB ×1.5).

**Why the model stays sub-threshold where the cells fire: input resistance is 3× too low.**
The same V~I fit (`scripts/rin_fit.py`: V = V0 + R·(RC-filtered I) over sub-threshold samples)
on all 805 recordings versus the model (`scripts/l5_excitability_probe.py`, job 58809593,
BBP default + 32 box draws, dt 0.2 fp64):

| | R_in (MOhm) | spikes Roy500 / 1000 / 1500 / 2000 |
|---|---|---|
| recordings (Roy500–2000 fits, r² 0.8, n = 642) | **146** (IQR 88–218), tau 6–8 ms | 2.6 / 9.3 / 13.5 / 15.4 |
| model, BBP default, ×1 | **48.7** | box mean 0.0 / 0.2 / 0.4 / 1.2 (fire 0–41 %) |
| dendritic g_pas 3e-5 → 1e-6, cm 2 → 1 | 84.7 | 0.7 / 0.6 / 3.0 / 3.7 |
| stim ×2 (≡ ½ membrane area) | 94 per rig-nA | 0.2 / 1.2 / 3.2 / 3.8 |
| **stim ×3 (≡ ⅓ area)** | **135 per rig-nA** | 0.4 / 3.2 / 5.3 / 8.4 (best draw 0 / 9 / 12 / 16) |
| stim ×4 (≡ ¼ area) | 171 per rig-nA | 1.2 / 3.8 / 8.4 / 11.3 (best draw 2 / 10 / 11 / 15) |

- The basal+apical tree carries nearly all membrane area; its leak (3e-5) and cm (2) are
  **fixed**, which is also why the trainable g_pas_somatic / g_pas_axonal are insensitive
  (0.14 / 0.24 mV at ICB ×1.5). Knobs `L5TTPC_DEND_GPAS` / `L5TTPC_DEND_CM` (defaults = BBP)
  were added to `toolbox/jaxley_cells/l5ttpc.py` for this probe.
- Lowering dendritic leak saturates at ~85 MOhm (Ih at 8e-5 and the soma/axon then dominate)
  and still gives only ~3.7 spikes at Roy2000: not the right lever by itself.
- A uniform stim scale c is exactly equivalent to shrinking every membrane area by c
  (all currents per area and cm scale together, tau unchanged). **c ≈ 3–4 matches the
  recordings' R_in (135–171 vs 146 MOhm)** and puts real data inside the model's reach:
  5/32 box draws reproduce the recorded f–I within RMSE < 4 spikes (best 1.3–1.5) at c = 3 and 4,
  versus 0/32 at ×1.

**Sensitivity at the matched operating point** (OAT, N = 500, jobs 58809896 c = 3 / 58809810 c = 4;
raw runs `tmp_neuInv/sensitivity_variation/l5ttpc_nc2/roy_c{3,4}_*`, table
`tmp_neuInv/excitability_probe/oat_x1_x3_x4_compare.csv`), mean temporal std (mV):

| channel | ×1 R1000 | ×1 R2000 | ×3 R1000 | ×3 R2000 | ×4 R1000 | ×4 R2000 | ICB ×1.5 |
|---|---|---|---|---|---|---|---|
| NaTs2_t_apical | 0.00 | 0.00 | 0.70 | 1.68 | 1.03 | 1.39 | 1.58 |
| NaTs2_t_somatic | 0.00 | 0.24 | 0.53 | 1.53 | 0.88 | 1.36 | 1.53 |
| SKv3_1_somatic | 0.36 | 1.02 | 0.89 | 1.85 | 1.53 | 2.37 | 1.85 |
| NaTa_t_axonal | 0.05 | 1.25 | 0.99 | 1.26 | 1.09 | 0.58 | 1.22 |
| Nap_Et2_axonal | 0.00 | 5.42 | 4.65 | 4.69 | 5.32 | 4.42 | 4.67 |
| K_Pst_axonal | 0.53 | 4.76 | 4.56 | 4.53 | 5.25 | 4.15 | 4.47 |
| SK_E2_axonal | 0.33 | 0.90 | 0.79 | 1.31 | 1.31 | 0.93 | 1.29 |
| Ca_HVA_axonal | 0.00 | 0.43 | 0.47 | 0.62 | 0.60 | 0.14 | 0.61 |
| Ca_LVAst_axonal | 0.00 | 0.37 | 0.37 | 0.67 | 0.45 | 0.40 | 0.67 |
| Ca_LVAst_somatic | 0.00 | 0.00 | 0.00 | 0.30 | 0.01 | 0.26 | 0.27 |
| cm_somatic | 0.01 | 0.42 | 0.15 | 0.63 | 0.35 | 0.26 | 0.62 |
| Ih_dend | 1.72 | 1.88 | 1.79 | 1.68 | 1.99 | 1.36 | 1.67 |
| e_pas_all | 3.65 | 2.93 | 3.20 | 2.68 | 2.81 | 2.21 | 2.66 |
| channels ≥ 0.5 mV | 3 | 10 | 9 | 13 | 11 | 9 | 12 |

Sanity check: ×3 Roy2000 ≡ ICB × 1.49, and its column matches ICB ×1.5 to ±0.01 mV on
every channel. At c = 3 the recorded amplitudes span the same operating range the
training stimulus uses (Roy1000 ×3 ≈ ICB ×0.75 … Roy2000 ×3 ≈ ICB ×1.5), so **the stimulus
the synthetic pack trains on and the one the fine-tune sees become the same physics.**
At c = 4, Roy2000 overshoots (adaptation / partial block lowers apical-Na and NaTa_t).

**Recommendation.** Train and fine-tune at one current scale c ≈ 3 (a single rig→model
area correction, physically ≈ a cell with ⅓ the L5 TTPC membrane area). Generate the synthetic
pack on the Roy stimuli at ×3 (or reuse ICB-based packs, since Roy_N × 3 = ICB × 3N/4022),
and fine-tune on the recordings with `stim_scale 3`. The remaining f–I gap (box mean
under-fires at Roy1000/1500) is a box-centre question, not a threshold wall: some draws
already match. Figure: `excitability/excitability_probe.png`.
