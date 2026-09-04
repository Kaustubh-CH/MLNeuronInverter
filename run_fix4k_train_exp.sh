#!/bin/bash -l
#SBATCH -N1 --time=30:00 -J fix4k-exp -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Re-run of the two @train rows that used the 4000-bin 4k50kInterChaoticB stim
# under the forced 500 ms window (sim spikes plotted ~100 ms early). The
# overlay script now sets T_MAX from the simulated stim's length (400 ms).
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_fix4k_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

VO=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_voltonly_efel_interchaoticB_k128/ca3_pyramidal_synth/super/out
SUP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_supervised_interchaoticB_k128/ca3_pyramidal_synth/super/out

echo "=== voefel_k128 @ train, FIXED 400 ms window"
python -u plot_exp_overlay_roy.py -m "$VO" --outDir "$VO/exp_roy_train" \
       --simScale train || echo "FIX4K: voefel FAILED"
echo "=== supervised_k128 @ train, FIXED 400 ms window"
python -u plot_exp_overlay_roy.py -m "$SUP" --outDir "$SUP/exp_roy_train" \
       --simScale train || echo "FIX4K: supervised FAILED"
echo "=== FIX4K ALL DONE $(date)"
