#!/bin/bash
# Roy/Paula chaotic recordings vs the DTW-era CA3 models (1 GPU, interactive).
set -e
module load conda
conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_roy_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

ICA=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_interchaoticB_dtw_8n/ca3_pyramidal_synth/dtw_ica_200k/out
AMP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_amp_8n/ca3_pyramidal_synth/dtw_amp_200k/out
D80=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_80k/out

echo "=== dtw_ica_200k (protocol-matched) @ rig amplitude"
python -u plot_exp_overlay_roy.py -m $ICA --outDir $ICA/exp_roy       --simScale rig
echo "=== dtw_ica_200k @ training amplitude (scaling decision probe)"
python -u plot_exp_overlay_roy.py -m $ICA --outDir $ICA/exp_roy_train --simScale train
echo "=== dtw_amp_200k (chaoticRamp champion, OOD protocol) @ rig amplitude"
python -u plot_exp_overlay_roy.py -m $AMP --outDir $AMP/exp_roy       --simScale rig
echo "=== dtw_80k @ rig amplitude"
python -u plot_exp_overlay_roy.py -m $D80 --outDir $D80/exp_roy       --simScale rig
echo "=== ALL DONE"
