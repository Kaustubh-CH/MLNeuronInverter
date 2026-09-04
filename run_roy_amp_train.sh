#!/bin/bash
# dtw_amp_200k on the Roy/Paula recordings, re-simulated at TRAINING amplitude (5kChaoticRamp).
set -e
module load conda
conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_roy_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
AMP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_8n/ca3_pyramidal_synth/dtw_amp_200k/out
python -u plot_exp_overlay_roy.py -m $AMP --outDir $AMP/exp_roy_train --simScale train
echo "=== ALL DONE"
