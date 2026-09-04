#!/bin/bash -l
# Two orthogonal voltage-only arms on the BASELINE soft-DTW (no blur), 100 ep each,
# in one 4-node/16-GPU salloc. Each drives the committed launcher (train + auto
# predict + evaluate_voltage), then all runs are scored into the vo ledger.
#   A: ca3_vo_dtw_precond_chaoticramp  (feature-sensitivity grad-precond, kdr x1.68)
#   B: ca3_vo_dtw_band4_chaoticramp    (DTW warp band 8 -> 4 ms)
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/
J=$SCRATCH/tmp_neuInv/jaxley_ca3

run_arm () {  # $1=design  $2=wrk_suffix
  echo "A: ===== arm $1 (suffix=$2) start $(date) job=$SLURM_JOBID ====="
  NEUINV_WRK_SUFIX=$2 bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$1"
  echo "A: ===== arm $1 launcher exit=$? $(date) ====="
}

run_arm ca3_vo_dtw_precond_chaoticramp precond
run_arm ca3_vo_dtw_band4_chaoticramp   band4

# Score everything into the ledger.
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
echo "A: ===== ledger $(date) ====="
python scripts/collect_vo_ledger.py \
  "$J/ca3_vo_dtw_precond_chaoticramp/ca3_pyramidal_synth/precond" \
  "$J/ca3_vo_dtw_band4_chaoticramp/ca3_pyramidal_synth/band4" \
  "$J/ca3_vo_dtwblur_chaoticramp/ca3_pyramidal_synth/r1_dtwblur" \
  /pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/ca3_vo_chaoticramp_dtw \
  -o "$J/vo_ledger/results.csv"
echo "A: arms A+B done $(date)"
