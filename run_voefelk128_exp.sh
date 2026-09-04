#!/bin/bash -l
#SBATCH -N1 --time=30:00 -J voefelk128-exp -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Standard Roy exp predictions for the Jul-8 VOLTAGE-ONLY + soft-eFEL k128
# model (the original "big kernel -> best experimental transfer" model; its
# July verdict was on the OLD exp datasets). Training stim 4k50kInterChaoticB
# is protocol-matched to the rig, so --simScale train is meaningful.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_voefel_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

VO=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_voltonly_efel_interchaoticB_k128/ca3_pyramidal_synth/super/out

echo "=== voefel_k128 @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$VO" --outDir "$VO/exp_roy" \
       --simScale rig || echo "VOEFELK128: rig FAILED"
echo "=== voefel_k128 @ training amplitude (protocol-matched InterChaoticB)"
python -u plot_exp_overlay_roy.py -m "$VO" --outDir "$VO/exp_roy_train" \
       --simScale train || echo "VOEFELK128: train FAILED"
echo "=== VOEFELK128 ALL DONE $(date)"
