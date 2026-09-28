| cell | params | probes | ncomp | stim | stim scale | max I (nA) | rest mV | spikes | peak mV | width ms | AHP mV | block | Vmin/Vmax | verdict | sim s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_comp | 3 | 1 | 1 | 5k0step_200 | 0.05 | 0.01 | -65.0 | 0 | nan | nan | nan | no | -66/-64 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k0step_500 | 0.05 | 0.03 | -65.0 | 0 | nan | nan | nan | no | -66/-60 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | BBP_Exp_Step1000 | 0.05 | 0.05 | -65.0 | 1 | 36 | 1.00 | -76 | no | -76/36 | FAIL: 1 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k0ramp | 0.05 | 0.02 | -65.0 | 0 | nan | nan | nan | no | -66/-63 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k50kInterChaoticB | 0.05 | 0.34 | -65.0 | 8 | 40 | 1.00 | -76 | no | -77/42 | PASS | 1 |
| single_comp | 3 | 1 | 1 | 5kChaoticRamp | 0.05 | 0.30 | -65.0 | 19 | 30 | 0.88 | -75 | no | -76/36 | PASS | 1 |

| dataset | cell | N | T | probes×stims | box ±log10 | stim scale | ncomp | norm used | rest mV | silent % | spikes med / p95 / max | peak mV | width ms | AHP mV | block % | non-finite % | out-of-range % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ball_and_stick_5k50kInterChaoticB/ball_and_stick_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5k50kInterChaoticB | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.6 | 4 | 8 / 34 / 37 | 40 | 1.07 | -76 | 0 | 0.0 | 0.0 | PASS |
|  |  | 40,000 | 5001 | dend/5k50kInterChaoticB | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.6 | 4 | 8 / 34 / 37 | 39 | 1.09 | -76 | 0 | 0.0 | 0.0 | PASS |
| ball_and_stick_5kChaoticRamp/ball_and_stick_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5kChaoticRamp | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.6 | 9 | 16 / 39 / 41 | 34 | 1.19 | -75 | 0 | 0.0 | 0.0 | PASS |
|  |  | 40,000 | 5001 | dend/5kChaoticRamp | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.6 | 9 | 16 / 39 / 41 | 33 | 1.19 | -75 | 0 | 0.0 | 0.0 | PASS |
| ball_and_stick_bbp_5k50kInterChaoticB/ball_and_stick_bbp_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5k50kInterChaoticB | 0.5 | 0.07 |  | fixed -60.1/19.0 | -76.7 | 0 | 2 / 5 / 6 | 44 | 0.35 | -87 | 9 | 0.0 | 0.0 | PASS |
| ball_and_stick_bbp_5kChaoticRamp/ball_and_stick_bbp_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5kChaoticRamp | 0.5 | 0.07 |  | fixed -60.1/19.0 | -76.7 | 0 | 4 / 14 / 19 | 43 | 0.35 | -83 | 10 | 0.0 | 0.0 | PASS |
| l5ttpc_nc2_5k50kInterChaoticB/l5ttpc_nc2_synth.mlPack1.h5 | model ladder (2026-09) | 20,000 | 5001 | soma/5k50kInterChaoticB | 0.5 | 1.5 | 2 | fixed -60.1/19.0 | -74.5 | 0 | 7 / 11 / 18 | 36 | 0.49 | -80 | 0 | 0.0 | 0.0 | PASS |
| l5ttpc_nc2_5kChaoticRamp/l5ttpc_nc2_synth.mlPack1.h5 | model ladder (2026-09) | 20,000 | 5001 | soma/5kChaoticRamp | 0.5 | 1.5 | 2 | fixed -60.1/19.0 | -74.5 | 0 | 14 / 23 / 32 | 24 | 1.00 | -60 | 0 | 0.0 | 0.0 | PASS |
| l5ttpc_nc4_5k50kInterChaoticB/l5ttpc_nc4_synth.mlPack1.h5 | model ladder (2026-09) | 20,000 | 5001 | soma/5k50kInterChaoticB | 0.5 | 1.5 | 4 | fixed -60.1/19.0 | -74.5 | 0 | 7 / 10 / 19 | 37 | 0.50 | -81 | 0 | 0.0 | 0.0 | PASS |
| l5ttpc_nc4_5kChaoticRamp/l5ttpc_nc4_synth.mlPack1.h5 | model ladder (2026-09) | 20,000 | 5001 | soma/5kChaoticRamp | 0.5 | 1.5 | 4 | fixed -60.1/19.0 | -74.5 | 0 | 15 / 20 / 29 | 24 | 0.99 | -61 | 0 | 0.0 | 0.0 | PASS |
| single_comp_5k50kInterChaoticB/single_comp_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5k50kInterChaoticB | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.9 | 4 | 8 / 35 / 41 | 39 | 1.04 | -76 | 0 | 0.0 | 0.0 | PASS |
| single_comp_5kChaoticRamp/single_comp_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 5001 | soma/5kChaoticRamp | 0.5 | 0.05 |  | fixed -60.1/19.0 | -65.9 | 10 | 15 / 40 / 44 | 32 | 1.19 | -75 | 0 | 0.0 | 0.0 | PASS |
