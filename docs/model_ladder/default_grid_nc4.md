| cell | params | probes | ncomp | stim | stim scale | max I (nA) | rest mV | spikes | peak mV | width ms | AHP mV | block | Vmin/Vmax | verdict | sim s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_comp | 3 | 1 | 1 | 5k0step_200 | 0.05 | 0.01 | -65.0 | 0 | nan | nan | nan | no | -66/-64 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k0step_500 | 0.05 | 0.03 | -65.0 | 0 | nan | nan | nan | no | -66/-60 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | BBP_Exp_Step1000 | 0.05 | 0.05 | -65.0 | 1 | 36 | 1.00 | -76 | no | -76/36 | FAIL: 1 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k0ramp | 0.05 | 0.02 | -65.0 | 0 | nan | nan | nan | no | -66/-63 | FAIL: 0 spikes | 1 |
| single_comp | 3 | 1 | 1 | 5k50kInterChaoticB | 0.05 | 0.34 | -65.0 | 8 | 40 | 1.00 | -76 | no | -77/42 | PASS | 1 |
| single_comp | 3 | 1 | 1 | 5kChaoticRamp | 0.05 | 0.30 | -65.0 | 19 | 30 | 0.88 | -75 | no | -76/36 | PASS | 1 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | 5k0step_200 | 0.05 | 0.01 | -65.0 | 0 | nan | nan | nan | no | -66/-64 | FAIL: 0 spikes | 1 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | 5k0step_500 | 0.05 | 0.03 | -65.0 | 0 | nan | nan | nan | no | -66/-61 | FAIL: 0 spikes | 2 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | BBP_Exp_Step1000 | 0.05 | 0.05 | -65.0 | 1 | 35 | 1.00 | -76 | no | -76/35 | FAIL: 1 spikes | 1 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | 5k0ramp | 0.05 | 0.02 | -65.0 | 0 | nan | nan | nan | no | -66/-63 | FAIL: 0 spikes | 1 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | 5k50kInterChaoticB | 0.05 | 0.34 | -65.0 | 8 | 39 | 0.99 | -76 | no | -77/42 | PASS | 2 |
| ball_and_stick | 4 | 2 | 2 (soma+stick) | 5kChaoticRamp | 0.05 | 0.30 | -65.0 | 14 | 29 | 0.88 | -74 | no | -75/35 | PASS | 1 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | 5k0step_200 | 0.07 | 0.01 | -76.9 | 0 | nan | nan | nan | no | -77/-68 | FAIL: 0 spikes | 5 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | 5k0step_500 | 0.07 | 0.04 | -76.9 | 3 | 44 | 0.37 | -84 | no | -84/45 | PASS | 5 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | BBP_Exp_Step1000 | 0.07 | 0.07 | -76.9 | 6 | 44 | 0.38 | -84 | no | -84/45 | PASS | 5 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | 5k0ramp | 0.07 | 0.03 | -76.9 | 2 | 45 | 0.35 | -84 | no | -84/45 | FAIL: 2 spikes | 4 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | 5k50kInterChaoticB | 0.07 | 0.48 | -76.9 | 4 | 45 | 0.35 | -88 | no | -100/46 | PASS | 5 |
| ball_and_stick_bbp | 13 | 1 | 2 (soma+stick) | 5kChaoticRamp | 0.07 | 0.42 | -76.9 | 8 | 44 | 0.36 | -83 | no | -86/46 | PASS | 4 |
| ca3_pyramidal | 6 | 1 | 1 | 5k0step_200 | 1 | 0.20 | -65.0 | 0 | nan | nan | nan | no | -65/-56 | FAIL: 0 spikes | 2 |
| ca3_pyramidal | 6 | 1 | 1 | 5k0step_500 | 1 | 0.50 | -65.0 | 0 | nan | nan | nan | no | -67/-40 | FAIL: 0 spikes | 2 |
| ca3_pyramidal | 6 | 1 | 1 | BBP_Exp_Step1000 | 1 | 1.00 | -65.0 | 29 | 42 | 0.91 | -39 | no | -65/43 | FAIL: AHP -39 | 2 |
| ca3_pyramidal | 6 | 1 | 1 | 5k0ramp | 1 | 0.50 | -65.0 | 0 | nan | nan | nan | no | -66/-42 | FAIL: 0 spikes | 3 |
| ca3_pyramidal | 6 | 1 | 1 | 5k50kInterChaoticB | 1 | 6.82 | -65.0 | 6 | 40 | 0.78 | -80 | no | -98/42 | PASS | 2 |
| ca3_pyramidal | 6 | 1 | 1 | 5kChaoticRamp | 1 | 6.00 | -65.0 | 34 | 23 | 4.54 | -28 | YES | -77/44 | FAIL: width 4.5ms, AHP -28, block | 2 |
| l5ttpc | 19 | 1 | 4 | 5k0step_200 | 1.5 | 0.30 | -74.9 | 0 | nan | nan | nan | no | -75/-60 | FAIL: 0 spikes | 27 |
| l5ttpc | 19 | 1 | 4 | 5k0step_500 | 1.5 | 0.75 | -74.9 | 2 | 25 | 0.40 | -61 | no | -78/30 | FAIL: 2 spikes | 27 |
| l5ttpc | 19 | 1 | 4 | BBP_Exp_Step1000 | 1.5 | 1.50 | -74.9 | 5 | 8 | 55.28 | -53 | no | -75/34 | FAIL: width 55.3ms | 28 |
| l5ttpc | 19 | 1 | 4 | 5k0ramp | 1.5 | 0.75 | -74.9 | 1 | 30 | 0.40 | -64 | no | -75/30 | FAIL: 1 spikes | 25 |
| l5ttpc | 19 | 1 | 4 | 5k50kInterChaoticB | 1.5 | 10.23 | -74.9 | 7 | 40 | 0.47 | -82 | no | -91/42 | PASS | 27 |
| l5ttpc | 19 | 1 | 4 | 5kChaoticRamp | 1.5 | 9.00 | -74.9 | 16 | 27 | 0.45 | -62 | no | -83/38 | PASS | 27 |

| dataset | cell | N | T | probes×stims | box ±log10 | stim scale | ncomp | norm used | rest mV | silent % | spikes med / p95 / max | peak mV | width ms | AHP mV | block % | non-finite % | out-of-range % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
