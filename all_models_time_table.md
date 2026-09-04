| model (under tmp_neuInv/) | cell | loss | samples/epoch | GPUs | s/epoch | epochs run | measured h | est. h / 100 ep | GPU-s per sample | mean R² |
|---|---|---|---|---|---|---|---|---|---|---|
| ca3_ablation/ca3_vo_A0_mse/out | ca3_pyramidal_synth | MSE | 40,000 | 32 | 14 | 100 | 0.4 | 0.4 | 0.011 | -0.046 |
| ca3_ablation/ca3_vo_A1_fp64/out | ca3_pyramidal_synth | MSE | 40,000 | 32 | 14 | 100 | 0.4 | 0.4 | 0.011 | -0.126 |
| ca3_ablation/ca3_vo_A2_dtw/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 32 | 18 | 95 | 0.5 | 0.5 | 0.015 | -0.056 |
| ca3_ablation/ca3_vo_A3_dvdt/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 32 | 14 | 100 | 0.4 | 0.4 | 0.011 | 0.075 |
| ca3_ablation/ca3_vo_A4_smooth/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 54 | 100 | 1.5 | 1.5 | 0.022 | -0.172 |
| ca3_ablation/ca3_vo_A5_lowpass/out | ca3_pyramidal_synth | MSE | 40,000 | 32 | 14 | 100 | 0.4 | 0.4 | 0.011 | -0.137 |
| ca3_ablation/ca3_vo_A7_hybrid/out | ca3_pyramidal_stepbattery | MSE+eFEL | 40,000 | 16 | 137 | 100 | 3.8 | 3.8 | 0.055 | 0.140 |
| ca3_ablation/ca3_vo_A7_mse/out | ca3_pyramidal_stepbattery | MSE | 40,000 | 16 | 102 | 100 | 2.9 | 2.8 | 0.041 | 0.070 |
| ca3_ablation/ca3_vo_blur_chaoticramp/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 35 | 100 | 1.0 | 1.0 | 0.014 | 0.630 |
| ca3_ablation/ca3_vo_blur_interchaoticB/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 28 | 100 | 0.8 | 0.8 | 0.011 | -0.345 |
| ca3_ablation/ca3_vo_chaoticramp_dtw/out | ca3_pyramidal_synth | DTW | 40,000 | 16 | 49 | 100 | 1.4 | 1.3 | 0.019 | 0.593 |
| jaxley_ca3/ca3_best_efel_step600/ca3_best_efel_step600/55498766/out | ca3_best_efel_step600 | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.012 | 0.017 |
| jaxley_ca3/ca3_best_mse_step1000/ca3_best_mse_step1000/55497219/out | ca3_best_mse_step1000 | MSE | 40,000 | 16 | 28 | 100 | 0.8 | 0.8 | 0.011 | -0.478 |
| jaxley_ca3/ca3_best_mse_step1000/ca3_best_mse_step1000/55499133/out | ca3_best_mse_step1000 | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.011 | -0.112 |
| jaxley_ca3/ca3_efel_precond_interchaoticB/ca3_pyramidal_synth/run1/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.011 | 0.223 |
| jaxley_ca3/ca3_joint4_dtw_amp_4n/ca3_pyramidal_joint4/joint4_80k/out | ca3_pyramidal_joint4 | DTW+eFEL | 64,000 | 16 | 558 | 100 | 15.5 | 15.5 | 0.139 | 0.280 |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52652251/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 10 | 0.1 | 1.0 | 0.014 |  |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52656278/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 10 | 0.1 | 1.0 | 0.014 |  |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52661942/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 100 | 1.0 | 1.0 | 0.014 | 0.123 |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52662153/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 10 | 0.1 | 1.0 | 0.014 |  |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52670775/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 10 | 0.1 | 1.0 | 0.014 |  |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/52835681/out | ca3_pyramidal_synth | MSE+ch | 160,000 | 16 | 140 | 10 | 0.4 | 3.9 | 0.014 | 0.125 |
| jaxley_ca3/ca3_matched/ca3_pyramidal_synth/55465675/out | ca3_pyramidal_synth | MSE+ch | 40,000 | 16 | 35 | 118 | 1.1 | 1.0 | 0.014 |  |
| jaxley_ca3/ca3_pool4_dtw_amp_4n/ca3_pyramidal_pool4/pool4_80k/out | ca3_pyramidal_pool4 | DTW+eFEL | 64,000 | 16 | 534 | 100 | 15.1 | 14.8 | 0.134 | 0.047 |
| jaxley_ca3/ca3_pool4_dtw_amp_4n/ca3_pyramidal_pool4/smoke_B_56414351/out | ca3_pyramidal_pool4 | DTW+eFEL | 64,000 | 16 | 706 | 3 | 0.7 | 19.6 | 0.177 | 0.048 |
| jaxley_ca3/ca3_pool4_dtw_amp_v2_4n/ca3_pyramidal_pool4/pool4v2_80k/out | ca3_pyramidal_pool4 | DTW+eFEL | 64,000 | 16 | 555 | 100 | 15.6 | 15.4 | 0.139 | 0.367 |
| jaxley_ca3/ca3_royexp_1stim_1000/RoyExp1000/onestim_1000/out | RoyExp1000 | MSE+eFEL | 129 | 4 | 14 | 40 | 0.2 | 0.4 | 0.433 |  |
| jaxley_ca3/ca3_royexp_1stim_1000_dtw/RoyExp1000/onestim_dtw_1000/out | RoyExp1000 | DTW | 129 | 4 | 17 | 40 | 0.2 | 0.5 | 0.542 |  |
| jaxley_ca3/ca3_royexp_1stim_1500/RoyExp1500/onestim_1500/out | RoyExp1500 | MSE+eFEL | 125 | 4 | 13 | 40 | 0.1 | 0.4 | 0.405 |  |
| jaxley_ca3/ca3_royexp_1stim_1500_dtw/RoyExp1500/onestim_dtw_1500/out | RoyExp1500 | DTW | 125 | 4 | 16 | 40 | 0.2 | 0.4 | 0.499 |  |
| jaxley_ca3/ca3_royexp_1stim_2000/RoyExp2000/onestim_2000/out | RoyExp2000 | MSE+eFEL | 123 | 4 | 13 | 40 | 0.1 | 0.3 | 0.409 |  |
| jaxley_ca3/ca3_royexp_1stim_2000_dtw/RoyExp2000/onestim_dtw_2000/out | RoyExp2000 | DTW | 123 | 4 | 16 | 40 | 0.2 | 0.4 | 0.509 |  |
| jaxley_ca3/ca3_royexp_1stim_500/RoyExp500/onestim_500/out | RoyExp500 | MSE+eFEL | 129 | 4 | 14 | 40 | 0.2 | 0.4 | 0.435 |  |
| jaxley_ca3/ca3_royexp_1stim_500_dtw/RoyExp500/onestim_dtw_500/out | RoyExp500 | DTW | 129 | 4 | 18 | 40 | 0.2 | 0.5 | 0.545 |  |
| jaxley_ca3/ca3_royexp_ft_icarec/RoyExpChaotic/ft_icarec/out | RoyExpChaotic | MSE+eFEL | 635 | 4 | 16 | 40 | 0.2 | 0.5 | 0.102 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide/RoyExpChaotic/ft_icarec_wide/out | RoyExpChaotic | MSE+eFEL | 635 | 4 | 16 | 40 | 0.2 | 0.5 | 0.103 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtw/RoyExpChaotic/ft_icarec_wide_dtw/out | RoyExpChaotic | DTW | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel/RoyExpChaotic/ft_icarec_wide_dtwefel/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.127 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel/RoyExpChaotic/ft_icarec_wide_polish/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.129 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel3/RoyExpChaotic/ft_icarec_wide_efel3/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel5/RoyExpChaotic/ft_icarec_wide_efel5/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel5/RoyExpChaotic/ft_icarec_wide_efel5_stab/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 80 | 0.5 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel5_ema/RoyExpChaotic/ft_icarec_wide_efel5_ema/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 80 | 0.5 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_ft_icarec_wide_dtwefel8/RoyExpChaotic/ft_icarec_wide_efel8/out | RoyExpChaotic | DTW+eFEL | 635 | 4 | 20 | 40 | 0.2 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_scratch_icarec/RoyExpChaotic/scratch_icarec/out | RoyExpChaotic | MSE+eFEL | 635 | 4 | 16 | 40 | 0.2 | 0.4 | 0.102 |  |
| jaxley_ca3/ca3_royexp_scratch_k128/RoyExpChaotic/scratch_k128/out | RoyExpChaotic | MSE+eFEL | 635 | 4 | 16 | 40 | 0.2 | 0.4 | 0.101 |  |
| jaxley_ca3/ca3_royexp_stimch/RoyExpStimCh/stimch_1ch/out | RoyExpStimCh | DTW+eFEL | 635 | 4 | 20 | 80 | 0.5 | 0.6 | 0.128 |  |
| jaxley_ca3/ca3_royexp_stimch/RoyExpStimCh/stimch_2ch/out | RoyExpStimCh | DTW+eFEL | 635 | 4 | 20 | 80 | 0.5 | 0.6 | 0.127 |  |
| jaxley_ca3/ca3_supervised_chaoticramp/ca3_pyramidal_synth/super/out | ca3_pyramidal_synth | param-MSE | 40,000 | 4 | 1 | 200 | 0.1 | 0.0 | 0.000 | 0.995 |
| jaxley_ca3/ca3_supervised_interchaoticB/ca3_pyramidal_synth/super/out | ca3_pyramidal_synth | param-MSE | 40,000 | 16 | 1 | 200 | 0.0 | 0.0 | 0.000 | 0.909 |
| jaxley_ca3/ca3_supervised_interchaoticB_k128/ca3_pyramidal_synth/super/out | ca3_pyramidal_synth | param-MSE | 40,000 | 16 | 1 | 200 | 0.0 | 0.0 | 0.000 | 0.918 |
| jaxley_ca3/ca3_supervised_step1000/ca3_best_mse_step1000/super/out | ca3_best_mse_step1000 | param-MSE | 40,000 | 16 | 1 | 200 | 0.0 | 0.0 | 0.000 | 0.829 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_10k/out | ca3_pyramidal_synth | DTW | 10,000 | 16 | 12 | 100 | 0.3 | 0.3 | 0.019 | 0.440 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_20k/out | ca3_pyramidal_synth | DTW | 20,000 | 16 | 25 | 100 | 0.7 | 0.7 | 0.020 | 0.489 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_80k/out | ca3_pyramidal_synth | DTW | 80,000 | 16 | 99 | 100 | 2.7 | 2.7 | 0.020 | 0.637 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_8n/ca3_pyramidal_synth/dtw_200k/out | ca3_pyramidal_synth | DTW | 200,000 | 16 | 221 | 95 | 5.9 | 6.1 | 0.018 | 0.725 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_8n/ca3_pyramidal_synth/smoke8n/out | ca3_pyramidal_synth | DTW | 16,384 | 32 | 43 | 2 | 0.0 | 1.2 | 0.084 |  |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_8n/ca3_pyramidal_synth/dtw_amp_200k/out | ca3_pyramidal_synth | DTW+eFEL | 200,000 | 32 | 217 | 150 | 9.1 | 6.0 | 0.035 | 0.763 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_v4_4n/ca3_pyramidal_synth/dtw_amp_400k/out | ca3_pyramidal_synth | DTW+eFEL | 400,000 | 16 | 494 | 150 | 20.6 | 13.7 | 0.020 | 0.750 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k/out | ca3_pyramidal_synth | DTW | 80,000 | 16 | 100 | 100 | 2.8 | 2.8 | 0.020 | 0.515 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_precondkdr5_ft_8n/ca3_pyramidal_synth/precondkdr5_ft_200k/out | ca3_pyramidal_synth | DTW | 200,000 | 16 | 247 | 50 | 3.4 | 6.9 | 0.020 | 0.742 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_precondkdr_8n/ca3_pyramidal_synth/precondkdr_200k/out | ca3_pyramidal_synth | DTW | 200,000 | 32 | 222 | 150 | 9.3 | 6.2 | 0.036 | 0.731 |
| jaxley_ca3/ca3_vo_chaoticramp_dtw_v4_8n/ca3_pyramidal_synth/dtw_400k/out | ca3_pyramidal_synth | DTW | 400,000 | 32 | 441 | 150 | 18.4 | 12.3 | 0.035 | 0.736 |
| jaxley_ca3/ca3_vo_dtw_band4_chaoticramp/ca3_pyramidal_synth/band4/out | ca3_pyramidal_synth | DTW | 40,000 | 16 | 49 | 100 | 1.4 | 1.4 | 0.020 | 0.595 |
| jaxley_ca3/ca3_vo_dtw_chaoramp_step/ca3_pyramidal_chaoramp_step/chaoramp_step/out | ca3_pyramidal_chaoramp_step | DTW | 40,000 | 16 | 96 | 100 | 2.7 | 2.7 | 0.038 | 0.126 |
| jaxley_ca3/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2/out | ca3_pyramidal_synth | DTW | 40,000 | 16 | 49 | 100 | 1.4 | 1.4 | 0.019 | 0.600 |
| jaxley_ca3/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2_80k/out | ca3_pyramidal_synth | DTW | 80,000 | 16 | 99 | 100 | 2.8 | 2.8 | 0.020 | 0.611 |
| jaxley_ca3/ca3_vo_dtw_precond_chaoticramp/ca3_pyramidal_synth/precond/out | ca3_pyramidal_synth | DTW | 40,000 | 16 | 49 | 100 | 1.4 | 1.4 | 0.020 | 0.610 |
| jaxley_ca3/ca3_vo_dtwblur_chaoticramp/ca3_pyramidal_synth/r1_dtwblur/out | ca3_pyramidal_synth | DTW | 40,000 | 16 | 48 | 100 | 1.4 | 1.3 | 0.019 | 0.479 |
| jaxley_ca3/ca3_vo_interchaoticB_dtw_8n/ca3_pyramidal_synth/dtw_ica_200k/out | ca3_pyramidal_synth | DTW | 200,000 | 32 | 222 | 150 | 9.3 | 6.2 | 0.036 | 0.219 |
| jaxley_ca3/ca3_voltonly/ca3_pyramidal_synth/55467862/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 35 | 100 | 1.0 | 1.0 | 0.014 | -0.138 |
| jaxley_ca3/ca3_voltonly/ca3_pyramidal_synth/55473876/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 35 | 100 | 1.0 | 1.0 | 0.014 | 0.047 |
| jaxley_ca3/ca3_voltonly_efel/ca3_pyramidal_synth/55480692/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 36 | 100 | 1.0 | 1.0 | 0.014 | 0.196 |
| jaxley_ca3/ca3_voltonly_efel_4kchaoticramp/ca3_pyramidal_synth/pipe/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 28 | 100 | 0.8 | 0.8 | 0.011 | 0.524 |
| jaxley_ca3/ca3_voltonly_efel_4kchaoticramp_k128/ca3_pyramidal_synth/super/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.012 | 0.447 |
| jaxley_ca3/ca3_voltonly_efel_chaoticramp/ca3_pyramidal_synth/55615880/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 36 | 100 | 1.0 | 1.0 | 0.014 | 0.566 |
| jaxley_ca3/ca3_voltonly_efel_chaoticramp_100k/ca3_pyramidal_synth/55621298/out | ca3_pyramidal_synth | MSE+eFEL | 80,000 | 16 | 72 | 200 | 4.0 | 2.0 | 0.014 | 0.542 |
| jaxley_ca3/ca3_voltonly_efel_chaoticramp_100k/ca3_pyramidal_synth/55625889/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 38 | 200 | 2.1 | 1.1 | 0.015 | 0.494 |
| jaxley_ca3/ca3_voltonly_efel_chaoticramp_notanh/ca3_pyramidal_synth/55808210_ct/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 36 | 100 | 1.0 | 1.0 | 0.014 | 0.504 |
| jaxley_ca3/ca3_voltonly_efel_chaoticramp_wide/ca3_pyramidal_synth/pipe/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 36 | 100 | 1.0 | 1.0 | 0.014 | 0.242 |
| jaxley_ca3/ca3_voltonly_efel_interchaoticB_ft/ca3_pyramidal_synth/pipe/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 29 | 60 | 0.5 | 0.8 | 0.011 | 0.204 |
| jaxley_ca3/ca3_voltonly_efel_interchaoticB_k128/ca3_pyramidal_synth/super/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.012 | 0.088 |
| jaxley_ca3/ca3_voltonly_efel_interchaoticB_notanh/ca3_pyramidal_synth/55809843/out | ca3_pyramidal_synth | MSE+eFEL | 40,000 | 16 | 29 | 100 | 0.8 | 0.8 | 0.011 | 0.020 |
| jaxley_ca3/ca3_voltonly_mse_chaoticramp/ca3_pyramidal_synth/55615920/out | ca3_pyramidal_synth | MSE | 40,000 | 16 | 35 | 100 | 1.0 | 1.0 | 0.014 | 0.576 |
| jaxley_ca3/l5ttpc_supervised/L5TTPC_jaxley_nc2/super/out | L5TTPC_jaxley_nc2 | param-MSE | 16,000 | 16 | 1 | 200 | 0.0 | 0.0 | 0.001 |  |
| jaxley_ca3/roy2k_ms2ch/out_ms_base | ca3_roy2k_ms2ch | DTW+eFEL | 20,000 | 4 | 368 | 100 | 10.2 | 10.2 | 0.074 |  |
| jaxley_ca3/roy2k_ms2ch/out_ms_ft2000 | RoyExp2000 | DTW+eFEL | 123 | 1 | 16 | 150 | 0.7 | 0.4 | 0.126 |  |
| jaxley_ca3/roy2k_vo2p/out_dtw_base | ca3_roy2k_vo2p | DTW+eFEL | 20,000 | 4 | 107 | 100 | 3.0 | 3.0 | 0.021 |  |
| jaxley_ca3/roy2k_vo2p/out_dtw_ft2000 | RoyExp2000 | DTW+eFEL | 123 | 1 | 16 | 150 | 0.7 | 0.4 | 0.130 |  |
| jaxley_ca3/roy2k_vo2p/out_dtw_ftall | RoyExpNo100 | DTW+eFEL | 506 | 4 | 64 | 150 | 2.7 | 1.8 | 0.505 |  |
| jaxley_ca3/roy2k_vo2p/out_ft | RoyExp2000 | MSE+eFEL | 123 | 1 | 13 | 40 | 0.1 | 0.4 | 0.107 |  |
| jaxley_ca3/roy2k_vo2p/out_ft_dtw | RoyExp2000 | DTW+eFEL | 123 | 1 | 16 | 150 | 0.7 | 0.4 | 0.131 |  |
| jaxley_ca3/roy2k_vo2p/out_ftall | RoyExpNo100 | MSE+eFEL | 506 | 4 | 51 | 40 | 0.6 | 1.4 | 0.406 |  |
| jaxley_ca3/roy2k_vo2p/out_vo | ca3_roy2k_vo2p | MSE+eFEL | 20,000 | 4 | 79 | 100 | 2.2 | 2.2 | 0.016 |  |
| jaxley_ca3/royexp_ft_57631428/out | RoyExpChaotic | MSE+eFEL | 635 | 4 | 80 | 40 | 0.9 | 2.2 | 0.506 |  |
| jaxley_ca3/royexp_ft_smoke_57631428/out | RoyExpChaotic | MSE+eFEL | 256 | 4 | 61 | 2 | 0.0 | 1.7 | 0.954 |  |
| jaxley_ca3/smoke_1778181760/out | ca3_pyramidal_synth | MSE+ch | 4,096 | 1 | 59 | 5 | 0.1 | 1.6 | 0.014 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/52364894/out | L5_TTPC1cADpyr0 | MSE | 8,192 | 1 | 329 | 2 | 0.2 | 9.1 | 0.040 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/52371151/out | L5_TTPC1cADpyr0 | MSE | 8,192 | 16 | 35 | 2 | 0.0 | 1.0 | 0.069 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/52371647/out | L5_TTPC1cADpyr0 | MSE | 32,768 | 16 | 83 | 10 | 0.2 | 2.3 | 0.040 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/52372480/out | L5_TTPC1cADpyr0 | MSE | 65,536 | 16 | 165 | 100 | 4.6 | 4.6 | 0.040 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/interactive_52370021/out | L5_TTPC1cADpyr0 | MSE | 8,192 | 4 | 363 | 2 | 0.2 | 10.1 | 0.177 |  |
| jaxley_voltage_only/ballBBP_voltage_only/L5_TTPC1cADpyr0/interactive_52370981/out | L5_TTPC1cADpyr0 | MSE | 8,192 | 4 | 95 | 2 | 0.1 | 2.6 | 0.046 |  |
| jaxley_voltage_only/l5ttpc_jaxley_hybrid/L5TTPC_jaxley_nc2/salloc_55144257/out | L5TTPC_jaxley_nc2 | MSE+ch | 16,000 | 16 | 565 | 15 | 2.4 | 15.7 | 0.565 |  |
| jaxley_voltage_only/l5ttpc_jaxley_hybrid_efel/L5TTPC_multistim/reg15/out | L5TTPC_multistim | MSE+eFEL+ch | 16,000 | 16 | 752 | 15 | 3.2 | 20.9 | 0.752 |  |
| jaxley_voltage_only/l5ttpc_jaxley_ncomp2/L5TTPC_jaxley_nc2/55135593/out | L5TTPC_jaxley_nc2 | MSE | 16,000 | 16 | 566 | 15 | 2.4 | 15.7 | 0.566 |  |
| jaxley_voltage_only/l5ttpc_multiprobe/L5TTPC_multiprobe/salloc_55163753/out | L5TTPC_multiprobe | MSE | 8,000 | 16 | 268 | 15 | 1.2 | 7.4 | 0.536 |  |
| jaxley_voltage_only/l5ttpc_multistim/L5TTPC_multistim/salloc_55166043/out | L5TTPC_multistim | MSE | 8,000 | 16 | 1124 | 11 | 3.5 | 31.2 | 2.248 |  |
| jaxley_voltage_only/l5ttpc_multistim_efel/L5TTPC_multistim/salloc_55370141/out | L5TTPC_multistim | MSE+eFEL | 16,000 | 16 | 2209 | 15 | 9.3 | 61.4 | 2.209 |  |
| jaxley_voltage_only/l5ttpc_multistim_paramonly/L5TTPC_multistim/debug_55374316/out | L5TTPC_multistim | param-MSE | 16,000 | 4 | 1 | 40 | 0.0 | 0.0 | 0.000 |  |
| jaxley_voltage_only/l5ttpc_multistim_paramonly/L5TTPC_multistim/finetune_55375352/out | L5TTPC_multistim | param-MSE | 80,000 | 1 | 6 | 40 | 0.1 | 0.2 | 0.000 |  |
| jaxley_voltage_only/l5ttpc_singlestim_efel/L5TTPC_multistim/salloc_55371844/out | L5TTPC_multistim | MSE+eFEL | 16,000 | 16 | 754 | 15 | 3.2 | 20.9 | 0.754 |  |
| jaxley_voltage_only/l5ttpc_singlestim_paramonly/L5TTPC_multistim/debug_55374315/out | L5TTPC_multistim | param-MSE | 16,000 | 4 | 0 | 40 | 0.0 | 0.0 | 0.000 |  |
| jaxley_voltage_only/l5ttpc_singlestim_paramonly/L5TTPC_multistim/finetune_55375351/out | L5TTPC_multistim | param-MSE | 80,000 | 1 | 4 | 40 | 0.0 | 0.1 | 0.000 |  |
114 runs
