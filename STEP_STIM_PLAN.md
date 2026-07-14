# Session 5 prep — voltage-only chaoticRamp + step (crack kdr)

**Why:** kdr's correlation is stuck at ~0.44 on chaoticRamp alone (non-specific rate
signal). A sustained current step gives kdr a clean, specific firing-rate readout.
Add it as a 2nd stim; loss stays strictly voltage-only. User approved the helper stim.

## Data — generate `ca3_chaoramp_step_v1` (no ready pack has 5kChaoticRamp+step)
Stims on the PROBE axis (num_probs=2, num_stims=1). Step = `5k0step_500` (500 ms, 0.5 nA,
matches 5kChaoticRamp length so no t_max mismatch). Generator is `scripts/gen_ca3_sharded.py`
(two phases; `--stims` is COMMA-separated). Run in the conda env, 4 nodes/16 GPU (debug queue):
```
OUT=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoramp_step_v1
SHD=$SCRATCH/tmp_neuInv/shards/ca3_chaoramp_step_v1
srun -n16 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none python -u scripts/gen_ca3_sharded.py \
  --source-cell ca3_pyramidal --cell-name ca3_pyramidal_chaoramp_step \
  --out $OUT --shard-dir $SHD --stims 5kChaoticRamp,5k0step_500 --n 50000 --batch 256 --seed 0
srun -n1 python -u scripts/gen_ca3_sharded.py --merge \
  --source-cell ca3_pyramidal --cell-name ca3_pyramidal_chaoramp_step \
  --out $OUT --shard-dir $SHD --stims 5kChaoticRamp,5k0step_500 --n 50000 --world 16
# -> $OUT/ca3_pyramidal_chaoramp_step.mlPack1.h5  (50k: 40k/5k/5k, 5001 bins, num_probs=2)
```
Alt (skip gen): `ca3_multistim_v1/ca3_pyramidal_multistim.mlPack1.h5` probes 0,1 =
`[5k50kInterChaoticB, 5k0step_500]` — but chaotic stim differs from 5kChaoticRamp.

## Config (pure soft-DTW, 2-stim — NO blur; blur regressed in R1)
Clone `ca3_vo_chaoticramp_dtw.hpar.yaml`, set `data_path` → the new pack, and:
```
data_conf: { serialize_stims: True, append_stim: False, parallel_stim: False, num_data_workers: 4 }
voltage_loss:
    ...pure DTW as baseline (dtw_weight 1, blur/mse/efel 0, band 8, fp64, tanh)...
    stim_name:        5kChaoticRamp          # drives t_max_override:auto -> 500 ms
    stim_names_multi: [5kChaoticRamp, 5k0step_500]   # MUST match probe/channel order
    # probe_loss_indices: unset (soma-only per stim)
```

## Launch gotcha — probsSelect
Launcher hardcodes `probsSelect="0"`. For 2 stims need `probsSelect="0 1"`. Use a launcher
variant (copy with probsSelect="0 1") OR run train_dist.py directly `--probsSelect 0 1
--stimsSelect 0 --validStimsSelect 0`. Data channel ci pairs with stim_names_multi[ci].
CNN adapts automatically (inputShape [5001,2], outputSize 6). Cost ≈ 2× (2 solves/step)
→ ~2.7h/100ep; gen(20m)+train fits one 4h salloc.
```
