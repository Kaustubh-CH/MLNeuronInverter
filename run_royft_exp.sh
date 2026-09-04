#!/bin/bash -l
#SBATCH -N1 --time=30:00 -J royft-exp -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Standard Roy exp-prediction pipeline (plot_exp_overlay_roy.py -> table rows)
# for the experimental FINE-TUNE royexp_ft_57631428 (k128 base, fine-tuned on
# Roy recordings with Roy*_icav2_5k stims by the user's session).
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_royft_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428/out

echo "=== royexp_ft @ rig amplitude (per-amp Roy<amp>_ica5k)"
python -u plot_exp_overlay_roy.py -m "$FT" --outDir "$FT/exp_roy" \
       --simScale rig || echo "ROYFT: rig FAILED"
echo "=== royexp_ft @ training stim (Roy2000_icav2_5k, matched at 2000 only)"
python -u plot_exp_overlay_roy.py -m "$FT" --outDir "$FT/exp_roy_train" \
       --simScale train || echo "ROYFT: train FAILED"
echo "=== ROYFT ALL DONE $(date)"
