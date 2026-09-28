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
| l5ttpc_nc2_icb4k_bbp_dt02/l5ttpc_nc2_bbp_synth.mlPack1.h5 | model ladder (2026-09) | 40,000 | 2001 | soma/4k50kInterChaoticB | 1.0 | 1.5 | 2 | fixed -60.1/19.0 | -74.2 | 6 | 6 / 15 / 23 | 33 | 0.63 | -77 | 0 | 0.0 | 0.0 | PASS |
