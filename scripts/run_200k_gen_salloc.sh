#!/bin/bash -l
# 250k pack (200k train / 25k valid / 25k test) for the L5 nc2 200k voltage-only run: identical
# recipe to the 50k pilot pack (ncomp 2, solver dt 0.2, 400 ms 4k50kInterChaoticB, run.py sampling,
# stim x1.5, fp64), just N=250000.  ~37 min on one node (50k took 7.4 min on 4 GPUs).
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_200k_gen_salloc.sh
set -u
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp || exit 2
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
export L5TTPC_NCOMP=2 NEUINV_LOG_HALFSPAN=1.0 SOURCE_CELL=l5ttpc GEN_BATCH=64 GEN_DT=0.2
export SHARD_ROOT=/pscratch/sd/k/ktub1999/ca3_gen_shards
OUT=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02_250k
echo "200K-GEN start $(date) job=${SLURM_JOB_ID:-?}"
bash scripts/run_ca3_gen.sh l5ttpc_nc2_bbp_synth "$OUT" 4k50kInterChaoticB 250000 400
echo "200K-GEN exit=$? $(date)"; ls -la "$OUT"
[ -f "$OUT/l5ttpc_nc2_bbp_synth.mlPack1.h5" ] || exit 1     # afterok dependency of the training job
