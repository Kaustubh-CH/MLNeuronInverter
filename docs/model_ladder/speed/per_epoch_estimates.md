# Per-epoch wall time on 16 A100 GPUs (4 nodes), from the measured forward+backward cost

minutes per epoch = samples x GPU-s/sample / 16 GPUs; add ~5 % for the forward-only validation pass.
'(est.)' rows are scaled from measured neighbours (nc3 = geometric mean of nc2/nc4; fp32 = x0.70 at equal batch; dt 0.2 = x0.5; 400 ms stim = x0.8).

| technique (best per-GPU batch) | model | GPU-s/sample | 40k: min/epoch | 80k | 200k | 100 epochs @40k (h) |
|---|---|---|---|---|---|---|
| fp64 dt0.1, B=64 (old config, 40 GB) | nc1 B=64 | 0.406 | 16.9 | 33.8 | 84.6 | 28.2 |
| fp64 dt0.1, B=64 (old config, 40 GB) | nc2 B=64 | 0.575 | 24.0 | 47.9 | 119.8 | 39.9 |
| fp64 dt0.1, B=64 (old config, 40 GB) | nc3 B=64 | 0.733 (est.) | 30.5 | 61.1 | 152.7 | 50.9 |
| fp64 dt0.1, B=64 (old config, 40 GB) | nc4 B=64 | 0.934 | 38.9 | 77.8 | 194.6 | 64.9 |
| fp64 dt0.1, best batch on 80 GB | nc1 B=512 | 0.097 | 4.0 | 8.0 | 20.1 | 6.7 |
| fp64 dt0.1, best batch on 80 GB | nc2 B=256 | 0.236 | 9.8 | 19.7 | 49.2 | 16.4 |
| fp64 dt0.1, best batch on 80 GB | nc3 B=256 | 0.384 (est.) | 16.0 | 32.0 | 80.0 | 26.7 |
| fp64 dt0.1, best batch on 80 GB | nc4 B=128 | 0.625 | 26.0 | 52.1 | 130.2 | 43.4 |
| fp64 dt0.2 | nc1 B=512 | 0.048 | 2.0 | 4.0 | 10.0 | 3.3 |
| fp64 dt0.2 | nc2 B=256 | 0.118 | 4.9 | 9.8 | 24.6 | 8.2 |
| fp64 dt0.2 | nc3 B=256 | 0.192 (est.) | 8.0 | 16.0 | 40.0 | 13.3 |
| fp64 dt0.2 | nc4 B=128 | 0.313 | 13.0 | 26.1 | 65.2 | 21.7 |
| fp64 dt0.25 | nc1 B=512 | 0.039 | 1.6 | 3.2 | 8.1 | 2.7 |
| fp64 dt0.25 | nc2 B=256 | 0.095 | 3.9 | 7.9 | 19.7 | 6.6 |
| fp64 dt0.25 | nc3 B=256 | 0.154 (est.) | 6.4 | 12.8 | 32.1 | 10.7 |
| fp64 dt0.25 | nc4 B=128 | 0.251 (est.) | 10.4 | 20.9 | 52.2 | 17.4 |
| fp32 dt0.1 | nc1 B=1024 | 0.068 (est.) | 2.8 | 5.6 | 14.1 | 4.7 |
| fp32 dt0.1 | nc2 B=512 | 0.113 | 4.7 | 9.4 | 23.5 | 7.8 |
| fp32 dt0.1 | nc3 B=512 | 0.184 (est.) | 7.7 | 15.3 | 38.3 | 12.8 |
| fp32 dt0.1 | nc4 B=256 | 0.299 | 12.5 | 24.9 | 62.3 | 20.8 |
| fp32 dt0.2 | nc1 B=1024 | 0.034 (est.) | 1.4 | 2.8 | 7.0 | 2.3 |
| fp32 dt0.2 | nc2 B=512 | 0.083 (est.) | 3.4 | 6.9 | 17.2 | 5.7 |
| fp32 dt0.2 | nc3 B=512 | 0.135 (est.) | 5.6 | 11.2 | 28.0 | 9.3 |
| fp32 dt0.2 | nc4 B=256 | 0.219 (est.) | 9.1 | 18.3 | 45.6 | 15.2 |
| fp32 dt0.2 + 400 ms stim | nc1 B=1024 | 0.027 (est.) | 1.1 | 2.2 | 5.6 | 1.9 |
| fp32 dt0.2 + 400 ms stim | nc2 B=256 | 0.066 (measured) | 2.7 | 5.5 | 13.7 | 4.6 |
| fp32 dt0.2 + 400 ms stim | nc3 B=512 | 0.108 (est.) | 4.5 | 9.0 | 22.4 | 7.5 |
| fp32 dt0.2 + 400 ms stim | nc4 B=256 | 0.175 (est.) | 7.3 | 14.6 | 36.5 | 12.2 |
