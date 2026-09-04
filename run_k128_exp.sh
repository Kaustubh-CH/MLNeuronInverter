#!/bin/bash -l
#SBATCH -N1 --time=30:00 -J k128-exp -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Roy/Paula exp predictions for the DTW k128 model (re-run of the two steps
# that failed in job 57628704 with "python: command not found" — the driver
# lacked its own conda activation).
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_k128exp_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

RUN=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k

echo "=== k128 @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
       --simScale rig || echo "K128EXP: rig FAILED"
echo "=== k128 @ training amplitude (stim-mismatched 5kChaoticRamp, firing bracket only)"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy_train" \
       --simScale train || echo "K128EXP: train FAILED"
echo "=== K128EXP ALL DONE $(date)"
