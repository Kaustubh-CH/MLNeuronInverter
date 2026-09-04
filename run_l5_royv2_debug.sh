#!/bin/bash -l
#SBATCH -N1 --time=30:00 -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Cross-cell-model probe: L5TTPC_jaxley_nc2 supervised model scored on the Roy
# v2 recordings (held-out ca3ft pack), faithful (unclamped) + tanh-clamped
# passes.  Submit: sbatch -J l5-royv2 run_l5_royv2_debug.sh
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
MODEL=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/l5ttpc_supervised/L5TTPC_jaxley_nc2/super/out
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_l5royv2_$SLURM_JOBID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "=== L5TTPC royv2 FAITHFUL (unclamped, as trained) $(date)"
python -u plot_exp_overlay_royv2_l5.py -m "$MODEL" \
       --outDir "$MODEL/exp_royv2_l5" || echo "L5-ROYV2: faithful FAILED"
echo "=== L5TTPC royv2 TANH-CLAMPED $(date)"
python -u plot_exp_overlay_royv2_l5.py -m "$MODEL" --clampTanh \
       --outDir "$MODEL/exp_royv2_l5_tanh" || echo "L5-ROYV2: tanh FAILED"
echo "L5-ROYV2: ALL DONE $(date)"
