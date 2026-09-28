#!/bin/bash -l
# Pack for the nc2 fp32/dt0.2 voltage-only pilot: L5 ncomp=2, solver dt 0.2 ms, 4k50kInterChaoticB
# (400 ms), BBP run.py sampling (+-1 decade conductances, linear e_pas/cm), 50k samples, stim x1.5.
#   salloc -N1 -C gpu -q interactive -t 0:45:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_pilot_gen_salloc.sh
set -u
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur || exit 2
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
export L5TTPC_NCOMP=2 NEUINV_LOG_HALFSPAN=1.0 SOURCE_CELL=l5ttpc GEN_BATCH=64 GEN_DT=0.2
export SHARD_ROOT=/pscratch/sd/k/ktub1999/ca3_gen_shards
OUT=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02
echo "PILOT-GEN start $(date) job=${SLURM_JOB_ID:-?}"
bash scripts/run_ca3_gen.sh l5ttpc_nc2_bbp_synth "$OUT" 4k50kInterChaoticB 50000 400
echo "PILOT-GEN exit=$? $(date)"; ls -la "$OUT"
