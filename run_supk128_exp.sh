#!/bin/bash -l
#SBATCH -N1 --time=30:00 -J supk128-exp -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Roy/Paula exp predictions for the Jul-8 SUPERVISED + kernel-128 control model
# (1 GPU, debug queue — ~5 min of work). Unclamped outputs (clamp_unit_tanh
# False) may produce extreme/NaN sims on OOD real traces — that IS the result.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_supk128_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

SUP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_supervised_interchaoticB_k128/ca3_pyramidal_synth/super/out

echo "=== supervised_k128 @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$SUP" --outDir "$SUP/exp_roy" \
       --simScale rig || echo "SUPK128: rig FAILED"
echo "=== supervised_k128 @ training amplitude (protocol-matched InterChaoticB)"
python -u plot_exp_overlay_roy.py -m "$SUP" --outDir "$SUP/exp_roy_train" \
       --simScale train || echo "SUPK128: train FAILED"
echo "=== SUPK128 ALL DONE $(date)"
