# chaoticRamp CA3 runs of record (generated 2026-09-03 by scripts/chaoticramp_run_table.py)

s/epoch = steady-state median of the last 10 epochs from tb_logs; train h = sum of epoch times; wall = SLURM allocation (several runs shared one salloc, so wall > train h for those). Global batch 2048, LR 1e-4, fp64, tanh clamp unless noted.

| run | date | job | nodes | samples | epochs | s/epoch | train h | wall | loss knobs | mean R² | leak | na3 | kdr | kap | km | kd | mse_z | spikes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised_chaoticramp | 2026-07-08 | 55691039 | 1n×4gpu | 40000 | 200 | 1 | 0.1 | 00:04:53 | param-MSE (supervised) | 0.995 | +1.00 | +1.00 | +0.98 | +1.00 | +0.99 | +1.00 | 1.86 | 27.0/22.9 |
| dtw_amp_200k | 2026-07-19 | 55951029 | 8n×32gpu | 200000 | 150 | 217 | 9.1 | 09:05:38 | DTW 1.0 g0.1 band8.0 n256; eFEL 0.2 ['voltage_base', 'AP_amplitude']; tanh | 0.763 | +0.90 | +0.87 | +0.37 | +0.87 | +0.75 | +0.83 | 2.03 | 27.4/25.1 |
| dtw_amp_400k | 2026-08-03 | 56227050 | 4n×16gpu | 400000 | 150 | 494 | 20.6 | 20:38:51 | DTW 1.0 g0.1 band8.0 n256; eFEL 0.2 ['voltage_base', 'AP_amplitude']; tanh | 0.750 | +0.88 | +0.86 | +0.25 | +0.90 | +0.77 | +0.83 | 1.90 | 27.9/24.7 |
| precondkdr5_ft_200k | 2026-07-19 | 56070201 | 4n×16gpu | 200000 | 50 | 247 | 3.4 | 03:27:55 | DTW 1.0 g0.1 band8.0 n256; precond {'normalize': 'geomean', 'weights': [1.0, 1.0, 5.0, 1.0, 1.0, 1.0]}; tanh | 0.742 | +0.87 | +0.76 | +0.34 | +0.85 | +0.81 | +0.82 | 2.01 | 26.6/25.1 |
| dtw_400k | 2026-07-17 | 55949891 | 8n×32gpu | 400000 | 150 | 441 | 18.4 | 18:25:04 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.736 | +0.89 | +0.80 | +0.22 | +0.91 | +0.76 | +0.84 | 1.90 | 27.0/24.7 |
| precondkdr_200k | 2026-07-17 | 55949843 | 8n×32gpu | 200000 | 150 | 222 | 9.3 | 09:18:24 | DTW 1.0 g0.1 band8.0 n256; precond {'normalize': 'geomean', 'weights': [1.0, 1.0, 2.5, 1.0, 1.0, 1.0]}; tanh | 0.731 | +0.87 | +0.77 | +0.27 | +0.85 | +0.82 | +0.81 | 2.01 | 26.6/25.1 |
| dtw_200k | 2026-07-14 | 55907527 | 4n×16gpu | 200000 | 95 | 221 | 5.9 | 04:00:02 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.725 | +0.86 | +0.79 | +0.34 | +0.81 | +0.75 | +0.80 | 2.03 | 26.9/25.1 |
| dtw_80k | 2026-07-14 | 55907527 | 4n×16gpu | 80000 | 100 | 99 | 2.7 | 04:00:02 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.637 | +0.86 | +0.73 | +0.09 | +0.72 | +0.65 | +0.76 | 2.05 | 25.8/22.5 |
| precond2_80k | 2026-07-14 | 55909694 | 4n×16gpu | 80000 | 100 | 99 | 2.8 | 04:00:16 | DTW 1.0 g0.1 band8.0 n256; precond {'enabled': True, 'normalize': 'geomean', 'weights': [0.85, 1.0, 1.0, 1.0, 1.15, 0.85]}; tanh | 0.611 | +0.82 | +0.68 | +0.18 | +0.63 | +0.63 | +0.73 | 2.05 | 25.8/22.5 |
| precond | 2026-07-13 | 55882823 | 4n×16gpu | 40000 | 100 | 49 | 1.4 | 02:46:40 | DTW 1.0 g0.1 band8.0 n256; precond {'enabled': True, 'exponent': 0.5, 'normalize': 'geomean', 'sensitivity': [10.4721, 9.0886, 3.0083, 9.7817, 7.8189, 16.7885]}; tanh | 0.610 | +0.80 | +0.73 | +0.04 | +0.69 | +0.62 | +0.77 | 1.99 | 25.5/22.9 |
| precond2 | 2026-07-14 | 55899724 | 4n×16gpu | 40000 | 100 | 49 | 1.4 | 02:27:19 | DTW 1.0 g0.1 band8.0 n256; precond {'enabled': True, 'normalize': 'geomean', 'weights': [0.85, 1.0, 1.0, 1.0, 1.15, 0.85]}; tanh | 0.600 | +0.79 | +0.69 | +0.13 | +0.75 | +0.50 | +0.74 | 1.99 | 24.7/22.9 |
| band4 | 2026-07-13 | 55882823 | 4n×16gpu | 40000 | 100 | 49 | 1.4 | 02:46:40 | DTW 1.0 g0.1 band4.0 n256; tanh | 0.595 | +0.81 | +0.62 | +0.16 | +0.70 | +0.52 | +0.77 | 1.97 | 24.7/22.9 |
| ca3_vo_chaoticramp_dtw | 2026-07-13 | 55873960 | 4n×16gpu | 40000 | 100 | 49 | 1.4 | 01:22:24 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.593 | +0.79 | +0.64 | +0.15 | +0.75 | +0.48 | +0.74 | 1.98 | 24.4/22.9 |
| voltonly_mse_chaoticramp | 2026-07-06 | 55615920 | 4n×16gpu | 40000 | 100 | 35 | 1.0 | 01:00:52 | MSE 1.0; tanh | 0.576 | +0.87 | -0.15 | +0.21 | +0.88 | +0.78 | +0.86 | 1.89 | 21.2/22.9 |
| voltonly_efel_chaoticramp | 2026-07-06 | 55615880 | 4n×16gpu | 40000 | 100 | 36 | 1.0 | 01:01:17 | MSE 1.0; eFEL 1.0 ; tanh | 0.566 | +0.77 | +0.26 | +0.09 | +0.82 | +0.64 | +0.80 | 1.93 | 23.4/22.9 |
| dtw_k128_80k | 2026-08-26 | 57628704 | 4n×16gpu | 80000 | 100 | 100 | 2.8 | 02:49:07 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.515 | +0.83 | +0.62 | +0.01 | +0.35 | +0.56 | +0.71 | 2.07 | 25.6/22.5 |
| dtw_20k | 2026-07-14 | 55899724 | 4n×16gpu | 20000 | 100 | 25 | 0.7 | 02:27:19 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.489 | +0.72 | +0.49 | +0.17 | +0.48 | +0.38 | +0.70 | 2.00 | 23.8/22.9 |
| r1_dtwblur | 2026-07-13 | 55880328 | 4n×16gpu | 40000 | 100 | 48 | 1.4 | 01:23:04 | DTW 1.0 g0.1 band8.0 n256; blur 1.0; tanh | 0.479 | +0.76 | +0.54 | +0.11 | +0.29 | +0.42 | +0.75 | 2.01 | 25.7/22.9 |
| dtw_10k | 2026-07-14 | 55899724 | 4n×16gpu | 10000 | 100 | 12 | 0.3 | 02:27:19 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.440 | +0.59 | +0.50 | +0.16 | +0.42 | +0.34 | +0.63 | 2.01 | 24.3/22.9 |
| pool4v2_80k | 2026-08-24 | 57548058 | 4n×16gpu | 80000 | 100 | 555 | 15.6 | 15:40:56 | DTW 1.0 g0.1 band8.0 n256; eFEL 0.2 ['voltage_base', 'AP_amplitude']; tanh | 0.367 | +0.74 | +0.45 | -0.03 | +0.09 | +0.36 | +0.59 | 2.15 | 26.7/24.2 |
| joint4_80k | 2026-08-08 | 56379085 | 4n×16gpu | 80000 | 100 | 558 | 15.5 | 15:33:16 | DTW 1.0 g0.1 band8.0 n256; eFEL 0.2 ['voltage_base', 'AP_amplitude']; tanh | 0.280 | +0.96 | +0.11 | -0.28 | -0.32 | +0.31 | +0.90 | 2.20 | 32.0/24.2 |
| dtw_ica_200k | 2026-07-17 | 55949893 | 8n×32gpu | 200000 | 150 | 222 | 9.3 | 09:18:43 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.219 | +0.97 | -0.27 | -0.12 | +0.16 | -0.37 | +0.94 | 0.97 | 8.8/8.7 |
| chaoramp_step | 2026-07-14 | 55901165 | 4n×16gpu | 40000 | 100 | 96 | 2.7 | 02:43:57 | DTW 1.0 g0.1 band8.0 n256; tanh | 0.126 | +0.44 | +0.17 | -0.19 | -0.31 | -0.00 | +0.65 | 2.12 | 27.2/22.9 |
| pool4_80k | 2026-08-08 | 56379086 | 4n×16gpu | 80000 | 100 | 534 | 15.1 | 15:09:54 | DTW 1.0 g0.1 band8.0 n256; eFEL 0.2 ['voltage_base', 'AP_amplitude']; tanh | 0.047 | +0.51 | -0.07 | -0.15 | -0.18 | -0.15 | +0.32 | 2.24 | 29.2/24.2 |

supervised_chaoticramp   lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_supervised_chaoticramp/ca3_pyramidal_synth/super/out
dtw_amp_200k             lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_8n/ca3_pyramidal_synth/dtw_amp_200k/out
dtw_amp_400k             lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_v4_4n/ca3_pyramidal_synth/dtw_amp_400k/out
precondkdr5_ft_200k      lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_precondkdr5_ft_8n/ca3_pyramidal_synth/precondkdr5_ft_200k/out
dtw_400k                 lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_v4_8n/ca3_pyramidal_synth/dtw_400k/out
precondkdr_200k          lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_precondkdr_8n/ca3_pyramidal_synth/precondkdr_200k/out
dtw_200k                 lr=None gbs=None stims=5kChaoticRamp state=TIMEOUT  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_8n/ca3_pyramidal_synth/dtw_200k/out
dtw_80k                  lr=None gbs=None stims=5kChaoticRamp state=TIMEOUT  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_80k/out
precond2_80k             lr=None gbs=None stims=5kChaoticRamp state=TIMEOUT  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2_80k/out
precond                  lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_precond_chaoticramp/ca3_pyramidal_synth/precond/out
precond2                 lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2/out
band4                    lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_band4_chaoticramp/ca3_pyramidal_synth/band4/out
ca3_vo_chaoticramp_dtw   lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/ca3_vo_chaoticramp_dtw/out
voltonly_mse_chaoticramp lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_voltonly_mse_chaoticramp/ca3_pyramidal_synth/55615920/out
voltonly_efel_chaoticramp lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_voltonly_efel_chaoticramp/ca3_pyramidal_synth/55615880/out
dtw_k128_80k             lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k/out
dtw_20k                  lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_20k/out
r1_dtwblur               lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtwblur_chaoticramp/ca3_pyramidal_synth/r1_dtwblur/out
dtw_10k                  lr=None gbs=None stims=5kChaoticRamp state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_10k/out
pool4v2_80k              lr=None gbs=None stims=['5kChaoticRamp', '5k0chaotic4', 'BBP_Exp_Step1000_i4k', 'chirp23a_i4k'] state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_pool4_dtw_amp_v2_4n/ca3_pyramidal_pool4/pool4v2_80k/out
joint4_80k               lr=None gbs=None stims=['5kChaoticRamp', '5k0chaotic4', 'BBP_Exp_Step1000_i4k', 'chirp23a_i4k'] state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_joint4_dtw_amp_4n/ca3_pyramidal_joint4/joint4_80k/out
dtw_ica_200k             lr=None gbs=None stims=5k50kInterChaoticB state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_interchaoticB_dtw_8n/ca3_pyramidal_synth/dtw_ica_200k/out
chaoramp_step            lr=None gbs=None stims=['5kChaoticRamp', '5k0step_500'] state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_chaoramp_step/ca3_pyramidal_chaoramp_step/chaoramp_step/out
pool4_80k                lr=None gbs=None stims=['5kChaoticRamp', '5k0chaotic4', 'BBP_Exp_Step1000_i4k', 'chirp23a_i4k'] state=COMPLETED  /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_pool4_dtw_amp_4n/ca3_pyramidal_pool4/pool4_80k/out
