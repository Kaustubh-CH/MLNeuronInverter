| pack | date | N | stims (channels) | box ±log10 | varied | norm | rest mV | silent % | spikes med / p95 / max | peak mV | width ms | AHP mV | block % | non-finite % | >+60 or <−120 % | models trained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXPERIMENT (Paula, all families)** | 2026-04 | 635 | Roy100..2000 | — | — | raw mV | -72.0 | 33 | 5 / 17 / 22 | 21 | 5.10 | -68 | 1 | 0.0 | 17.2 | reference |
|   exp Roy100 |  | 129 | Roy100 | — | — | raw mV | -71.1 | 99 | 0 / 0 / 1 | 13 | 1.10 | -57 | 0 | 0.0 | 0.0 |  |
|   exp Roy1000 |  | 129 | Roy1000 | — | — | raw mV | -71.4 | 8 | 6 / 13 / 14 | 18 | 6.44 | -72 | 0 | 0.0 | 7.8 |  |
|   exp Roy1500 |  | 125 | Roy1500 | — | — | raw mV | -72.9 | 0 | 11 / 17 / 18 | 21 | 5.27 | -68 | 0 | 0.0 | 27.2 |  |
|   exp Roy2000 |  | 123 | Roy2000 | — | — | raw mV | -73.6 | 0 | 14 / 20 / 22 | 29 | 4.71 | -71 | 4 | 0.0 | 49.6 |  |
|   exp Roy500 |  | 129 | Roy500 | — | — | raw mV | -70.8 | 53 | 0 / 5 / 6 | 14 | 3.60 | -69 | 0 | 0.0 | 3.1 |  |
| ca3_4kchaoticramp_v1 |  | 40,000 | 4kChaoticRamp | 0.5 | 6 | fixed | -65.0 | 0 | 26 / 43 / 65 | 27 | 4.35 | -34 | 75 | 0.0 | 0.0 | ca3_voltonly_efel_4kchaoticramp, ca3_voltonly_efel_4kchaoticramp_k128 |
| ca3_4kinterchaoticB_v1 |  | 40,000 | 4k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 22 | 4 / 28 / 47 | 39 | 0.88 | -76 | 0 | 0.0 | 0.0 | ca3_efel_precond_interchaoticB, ca3_supervised_interchaoticB, ca3_supervised_int |
| ca3_5kinterchaoticB_v1 |  | 200,000 | 5k50kInterChaoticB | 0.5 | 6 | fixed | -66.6 | 23 | 4 / 25 / 59 | 41 | 0.88 | -78 | 0 | 0.0 | 0.2 | ca3_vo_interchaoticB_dtw_8n |
| ca3_best_efel_multi |  | 40,000 | 5k0chaotic4_i4k/5k0chaotic4_i4k | 0.5 | 6 | fixed | -65.0 | 55 | 0 / 30 / 49 | 41 | 1.07 | -57 | 1 | 0.0 | 0.0 |  |
|  |  | 0 | 4k50kInterramp_50khz_i4k/5k0chaotic4_i4k | 0.5 | 6 | fixed | -65.0 | 0 | 3 / 6 / 11 | 181 | 43.35 | -41 | 100 | 0.0 | 100.0 |  |
|  |  | 0 | BBP_Exp_Step600_i4k/5k0chaotic4_i4k | 0.5 | 6 | fixed | -65.0 | 56 | 0 / 27 / 64 | 40 | 1.15 | -41 | 8 | 0.0 | 0.0 |  |
|  |  | 0 | chaotic3_i4k/5k0chaotic4_i4k | 0.5 | 6 | fixed | -65.0 | 69 | 0 / 27 / 48 | 45 | 1.10 | -52 | 0 | 0.0 | 0.0 |  |
| ca3_best_efel_step600 |  | 40,000 | BBP_Exp_Step600_i4k | 0.5 | 6 | fixed | -65.0 | 56 | 0 / 27 / 64 | 40 | 1.15 | -41 | 8 | 0.0 | 0.0 | ca3_best_efel_step600 |
| ca3_best_mse_multi |  | 40,000 | chirp23a_i4k/chirp23a_i4k | 0.5 | 6 | fixed | -102.1 | 59 | 0 / 17 / 35 | 40 | 1.12 | -47 | 4 | 0.0 | 25.0 |  |
|  |  | 0 | BBP_Exp_Step1000_i4k/chirp23a_i4k | 0.5 | 6 | fixed | -65.0 | 33 | 4 / 31 / 60 | 35 | 1.17 | -38 | 25 | 0.0 | 0.0 |  |
|  |  | 0 | ramp_500_i4k/chirp23a_i4k | 0.5 | 6 | fixed | -65.0 | 0 | 3 / 6 / 11 | 181 | 43.35 | -41 | 100 | 0.0 | 100.0 |  |
|  |  | 0 | 4k50kInterstep_500_50khz_i4k/chirp23a_i4k | 0.5 | 6 | fixed | -65.0 | 0 | 1 / 4 / 9 | 420 | 300.00 | -84 | 100 | 0.0 | 100.0 |  |
| ca3_best_mse_step1000 |  | 40,000 | BBP_Exp_Step1000_i4k | 0.5 | 6 | fixed | -65.0 | 33 | 4 / 31 / 60 | 35 | 1.17 | -38 | 25 | 0.0 | 0.0 | ca3_best_mse_step1000, ca3_supervised_step1000 |
| ca3_chaoramp_step_v1 |  | 40,000 | 5kChaoticRamp/5kChaoticRamp | 0.5 | 6 | fixed | -65.0 | 0 | 26 / 48 / 75 | 27 | 4.26 | -34 | 75 | 0.0 | 0.0 | ca3_vo_dtw_chaoramp_step |
|  |  | 0 | 5k0step_500/5kChaoticRamp | 0.5 | 6 | fixed | -65.0 | 62 | 0 / 32 / 72 | 43 | 1.15 | -41 | 6 | 0.0 | 0.0 |  |
| ca3_chaoticramp_v1 |  | 40,000 | 5kChaoticRamp | 0.5 | 6 | fixed | -65.0 | 0 | 26 / 48 / 75 | 27 | 4.26 | -34 | 75 | 0.0 | 0.0 | ca3_supervised_chaoticramp, ca3_vo_blur_chaoticramp, ca3_vo_chaoticramp_dtw, ca3 |
| ca3_chaoticramp_v2 |  | 80,000 | 5kChaoticRamp | 0.5 | 6 | fixed | -65.0 | 0 | 25 / 53 / 78 | 29 | 3.97 | -35 | 76 | 0.0 | 0.0 | ca3_vo_chaoticramp_dtw, ca3_vo_chaoticramp_dtw_k128, ca3_vo_dtw_precond2_chaotic |
| ca3_chaoticramp_v3 |  | 200,000 | 5kChaoticRamp | 0.5 | 6 | fixed | -66.6 | 0 | 27 / 51 / 70 | 30 | 3.66 | -36 | 72 | 0.0 | 0.0 | ca3_vo_chaoticramp_dtw_8n, ca3_vo_chaoticramp_dtw_amp_8n, ca3_vo_chaoticramp_dtw |
| ca3_chaoticramp_v4 |  | 400,000 | 5kChaoticRamp | 0.5 | 6 | fixed | -64.8 | 0 | 26 / 50 / 86 | 28 | 4.01 | -34 | 75 | 0.0 | 0.0 | ca3_vo_chaoticramp_dtw_amp_v4_4n, ca3_vo_chaoticramp_dtw_v4_8n |
| ca3_chaoticramp_wide_v1 |  | 40,000 | 5kChaoticRamp | 1.0 | 6 | fixed | -66.0 | 22 | 15 / 52 / 151 | 29 | 3.64 | -39 | 42 | 0.0 | 1.2 | ca3_voltonly_efel_chaoticramp_wide |
| ca3_joint4_v1 |  | 64,000 | 5kChaoticRamp/5kChaoticRamp | 0.5 | 6 | fixed | -62.6 | 0 | 27 / 49 / 77 | 28 | 3.83 | -34 | 74 | 0.0 | 0.0 | ca3_joint4_dtw_amp_4n |
|  |  | 0 | 5k0chaotic4/5kChaoticRamp | 0.5 | 6 | fixed | -62.6 | 42 | 4 / 32 / 57 | 42 | 0.97 | -55 | 1 | 0.0 | 0.0 |  |
|  |  | 0 | BBP_Exp_Step1000_i4k/5kChaoticRamp | 0.5 | 6 | fixed | -62.6 | 26 | 6 / 35 / 91 | 38 | 1.13 | -39 | 24 | 0.0 | 0.0 |  |
|  |  | 0 | chirp23a_i4k/5kChaoticRamp | 0.5 | 6 | fixed | -104.1 | 50 | 0 / 22 / 45 | 42 | 1.06 | -47 | 4 | 0.0 | 29.2 |  |
| ca3_joint4_v1_pooled |  | 64,000 | soma/5kChaoticRamp | 0.5 | 6 | fixed | -62.6 | 0 | 27 / 49 / 77 | 28 | 3.83 | -34 | 74 | 0.0 | 0.0 | ca3_pool4_dtw_amp_4n, ca3_pool4_dtw_amp_v2_4n |
|  |  | 0 | soma/5k0chaotic4 | 0.5 | 6 | fixed | -62.6 | 42 | 4 / 32 / 57 | 42 | 0.97 | -55 | 1 | 0.0 | 0.0 |  |
|  |  | 0 | soma/BBP_Exp_Step1000_i4k | 0.5 | 6 | fixed | -62.6 | 26 | 6 / 35 / 91 | 38 | 1.13 | -39 | 24 | 0.0 | 0.0 |  |
|  |  | 0 | soma/chirp23a_i4k | 0.5 | 6 | fixed | -104.1 | 50 | 0 / 22 / 45 | 42 | 1.06 | -47 | 4 | 0.0 | 29.2 |  |
| ca3_multistim_v1 |  | 40,000 | 5k50kInterChaoticB/5k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 22 | 4 / 32 / 57 | 39 | 0.88 | -76 | 1 | 0.0 | 0.0 |  |
|  |  | 0 | 5k0step_500/5k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 62 | 0 / 32 / 72 | 43 | 1.15 | -41 | 6 | 0.0 | 0.0 |  |
|  |  | 0 | 5k0chirp/5k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 46 | 1 / 25 / 58 | 40 | 1.11 | -56 | 2 | 0.0 | 12.2 |  |
|  |  | 0 | 5k0ramp/5k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 70 | 0 / 29 / 63 | 45 | 1.11 | -44 | 3 | 0.0 | 0.0 |  |
| ca3_roy2k_ms2ch |  | 64,000 | soma/Roy100_icav2_5k | 0.5 | 3 | fixed | -65.2 | 71 | 0 / 18 / 32 | 42 | 0.87 | -50 | 0 | 0.0 | 0.0 | roy2k_ms2ch |
|  |  | 0 | stimulus/Roy100_icav2_5k | 0.5 | 3 | fixed | -63.9 | 0 | 10 / 16 / 16 | 104 | 0.90 | -92 | 0 | 0.0 | 77.8 |  |
| ca3_roy2k_vo2p |  | 64,000 | Roy2000_icav2_5k | 0.5 | 2 | fixed | -65.2 | 52 | 0 / 23 / 36 | 42 | 0.84 | -61 | 0 | 0.0 | 0.0 | roy2k_vo2p |
| ca3_stepbattery_v1 |  | 40,000 | passiveReverse5k-50pA/passiveReverse5k-50pA | 0.5 | 6 | fixed | -65.0 | 88 | 0 / 17 / 53 | 44 | 1.15 | -41 | 2 | 0.0 | 0.0 | ca3_vo_A7_hybrid, ca3_vo_A7_mse |
|  |  | 0 | 5k0step_200/passiveReverse5k-50pA | 0.5 | 6 | fixed | -65.0 | 82 | 0 / 27 / 61 | 44 | 1.14 | -42 | 2 | 0.0 | 0.0 |  |
|  |  | 0 | 5k0step_500/passiveReverse5k-50pA | 0.5 | 6 | fixed | -65.0 | 62 | 0 / 32 / 72 | 43 | 1.15 | -41 | 6 | 0.0 | 0.0 |  |
| ca3_synth_v1 |  | 40,000 | 5k50kInterChaoticB | 0.5 | 6 | fixed | -62.2 | 2 | 8 / 22 / 48 | 39 | 1.14 | -78 | 0 | 0.0 | 40.0 | ca3_matched, smoke_1778181760 |
| ca3_synth_v2 |  | 160,000 | 5k50kInterChaoticB | 0.5 | 6 | fixed | -62.2 | 1 | 9 / 22 / 39 | 37 | 1.10 | -78 | 0 | 0.0 | 31.8 | ca3_matched |
| ca3_synth_v3 |  | 40,000 | 5k50kInterChaoticB | 0.5 | 6 | fixed | -65.0 | 22 | 4 / 32 / 57 | 39 | 0.88 | -76 | 1 | 0.0 | 0.0 | ca3_matched, ca3_voltonly, ca3_voltonly_efel |
