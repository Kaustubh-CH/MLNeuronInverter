#!/bin/bash -l
# Session 3 (one 4h/16-GPU salloc): broad-win refinement + dataset-size HPO.
#   C  ca3_vo_dtw_precond2_chaoticramp  (refined kdr-neutral grad-precond, 40k)
#   D1 ca3_vo_chaoticramp_dtw @ 20k     (baseline recipe, half data)
#   D2 ca3_vo_chaoticramp_dtw @ 10k     (baseline recipe, quarter data)
# The 40k baseline (0.593) already exists -> D1/D2 give the data-scaling curve.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/
J=$SCRATCH/tmp_neuInv/jaxley_ca3

run_arm () {  # $1=design  $2=suffix  $3=numGlobSamp
  echo "S3: ===== arm $1 suffix=$2 numGlob=$3 start $(date) job=$SLURM_JOBID ====="
  NEUINV_WRK_SUFIX=$2 NEUINV_NUMGLOBSAMP=$3 bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$1"
  echo "S3: ===== arm $1 launcher exit=$? $(date) ====="
}

run_arm ca3_vo_dtw_precond2_chaoticramp precond2 40000
run_arm ca3_vo_chaoticramp_dtw          dtw_20k  20000
run_arm ca3_vo_chaoticramp_dtw          dtw_10k  10000

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
echo "S3: ===== ledger $(date) ====="
python scripts/collect_vo_ledger.py \
  "$J/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2" \
  "$J/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_20k" \
  "$J/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_10k" \
  "$J/ca3_vo_dtw_precond_chaoticramp/ca3_pyramidal_synth/precond" \
  "$J/ca3_vo_dtw_band4_chaoticramp/ca3_pyramidal_synth/band4" \
  /pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/ca3_vo_chaoticramp_dtw \
  -o "$J/vo_ledger/results.csv"
echo "S3: session 3 done $(date)"
