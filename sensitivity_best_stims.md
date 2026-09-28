# CA3 Pyramidal — Best Stimuli for Conductance Sensitivity

Which stimulus best *exposes* each ion channel, by two independent metrics, from a one-at-a-time (OAT) sweep: for each channel, 500 cells with only that conductance varied over its full range (others at default) are simulated per stimulus, and the **variation across those 500 traces** is measured.

- **MSE / voltage variation** — mean over time of the across-sample std of the raw soma voltage (mV). Absolute how-much-does-the-trace-move.

- **eFEL variation** — across-sample std of the soft-eFEL features (`STRONG_FEATURES`, scaled by `FEATURE_SCALES` as in `HybridLoss`), averaged over features. How much the *spike/feature statistics* move.

Source: interpolated stim set (all 68 stims resampled to 4000 steps / 400 ms, equal footing). Native-length rankings agree (Spearman ρ ≈ 0.92–0.96 for voltage). Scratch files (`testSave`, `CompareTest`, `Updated*`) excluded from recommendations.

## TL;DR — best single stimulus

| Metric | Best single stim | Why |
|---|---|---|
| **MSE / voltage** | **`BBP_Exp_Step1000`** | highest mean coverage across all 6 channels; wins leak-adjacent Na/KM/KD outright |
| **eFEL features** | **`BBP_Exp_Step600`** | highest mean feature-coverage across channels (a chaotic/step; drives the most feature variation) |

> No single stimulus is best for *every* channel. If you can only run one, a **long depolarising step (`BBP_Exp_Step1000`)** is the strongest all-round choice; a **ramp (`ramp_500`)** is essential to see Kdr, and a **chirp (`chirp23a`)** best drives the passive leak. A 3-stim battery {step, ramp, chirp} covers all six.

## Best stimulus per channel

| Channel | MSE / voltage best | (mV) | eFEL best | (scaled) | eFEL top feature |
|---|---|---:|---|---:|---|
| **leak** | `chirp23a` | 17.2 | `5k0chaotic4` | 1.89 | inv_first_ISI |
| **Na (na3)** | `BBP_Exp_Step1000` | 10.9 | `5k0chaotic4` | 1.84 | inv_first_ISI |
| **Kdr** | `ramp_500` | 146.2 | `4k50kInterramp_50khz` | 2.79 | AP_amplitude |
| **KA (kap)** | `4k50kInterstep_500_50khz` | 17.4 | `5k0chaotic4` | 2.04 | inv_first_ISI |
| **KM (km)** | `BBP_Exp_Step1000` | 11.5 | `BBP_Exp_Step600` | 1.60 | inv_first_ISI |
| **KD (kd)** | `BBP_Exp_Step1000` | 17.3 | `chaotic3` | 1.82 | inv_first_ISI |

## Overall stimulus ranking (top 5, legitimate stims)

Coverage = mean over channels of the per-channel-normalised variation (1.0 = this stim is the best in the set for that channel).

**MSE / voltage**

| Rank | Stim | Coverage |
|---|---|---:|
| 1 | `BBP_Exp_Step1000` | 0.779 |
| 2 | `BBP_Exp_Step800` | 0.719 |
| 3 | `BBP_Exp_Step600` | 0.550 |
| 4 | `chirp_damp` | 0.535 |
| 5 | `4k50kInterstep_500_50khz` | 0.518 |

**eFEL features**

| Rank | Stim | Coverage |
|---|---|---:|
| 1 | `BBP_Exp_Step600` | 0.786 |
| 2 | `chaotic4` | 0.753 |
| 3 | `5k0chaotic4` | 0.714 |
| 4 | `chirp_damp_16k_v1` | 0.659 |
| 5 | `chirp16a` | 0.659 |

## Notes

- **The two metrics disagree, and that is informative.** Voltage variation is dominated by long **steps/ramps** (they hold the cell where slow K currents and Na shape a big absolute voltage envelope). eFEL variation favours **chaotic/step** stimuli that maximally scatter *spike-timing* features (`inv_first_ISI`, `AP_amplitude`) — a channel can move spike timing a lot while moving the mV envelope little, and vice-versa.

- **Kdr** is the standout: `ramp_500` gives ~146 mV voltage variation (>8× any other channel) because sweeping Kdr flips the cell across a spiking bifurcation along the ramp — a genuine but 'cliffy' observability signal.

- eFEL numbers are scaled (per-`FEATURE_SCALES`) and unitless; compare within the eFEL column only, not against the mV column.

