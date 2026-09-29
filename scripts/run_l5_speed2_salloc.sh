#!/bin/bash -l
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_l5_speed_salloc.sh
set -u
module load python; source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp || exit 2
echo "L5-SPEED2 start $(date) job=${SLURM_JOB_ID:-?}"
srun -n4 --gpus-per-node=4 --gpu-bind=none bash scripts/l5_speed_worker2.sh
echo "L5-SPEED2 done $(date)"
