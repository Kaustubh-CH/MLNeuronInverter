#!/bin/bash -l
# Model-ladder stimulus-scale sweep on 4 GPUs (one cell family per rank).
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g \
#          --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_model_ladder_scan_salloc.sh
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
export LADDER_OUT=${LADDER_OUT:-docs/model_ladder} LADDER_NBOX=${LADDER_NBOX:-128}
echo "LADDER-SCAN start $(date) job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"
srun -n4 --gpus-per-node=4 --gpu-bind=none bash scripts/model_ladder_scan_worker.sh
echo "LADDER-SCAN done $(date)"
