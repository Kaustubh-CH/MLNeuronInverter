#!/bin/bash -l
# PARALLEL session (own 4-node/16-GPU salloc): kdr step-stim.
# 1) generate ca3_chaoramp_step_v1 (5kChaoticRamp + 5k0step_500 on the probe axis) if
#    missing, via the working sharded generator; 2) train the 2-stim voltage-only
#    soft-DTW recipe (multiprobe launcher, probsSelect="0 1"); 3) score into the ledger.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
OUT=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoramp_step_v1
PACKH5=$OUT/ca3_pyramidal_chaoramp_step.mlPack1.h5

echo "STEP: session start $(date) job=$SLURM_JOBID nodes=$SLURM_JOB_NODELIST"

if [ ! -f "$PACKH5" ]; then
  echo "STEP: generating 2-stim pack (5kChaoticRamp,5k0step_500) ..."
  bash scripts/run_ca3_gen.sh ca3_pyramidal_chaoramp_step "$OUT" 5kChaoticRamp,5k0step_500 50000
  echo "STEP: gen returned $? $(date)"
else
  echo "STEP: pack exists, skipping gen"
fi
ls -la "$PACKH5" || { echo "STEP: GEN FAILED - no pack, aborting"; exit 3; }

export NEUINV_CELL=ca3_pyramidal_chaoramp_step   # data cellName = pack filename base
export NEUINV_WRK_SUFIX=chaoramp_step
echo "STEP: training 2-stim voltage-only $(date)"
bash batchShifterJaxleyCA3_multiprobe.slr "$OUT/" ca3_vo_dtw_chaoramp_step
echo "STEP: train launcher exit=$? $(date)"

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
RUN=$SCRATCH/tmp_neuInv/jaxley_ca3/ca3_vo_dtw_chaoramp_step/ca3_pyramidal_chaoramp_step/chaoramp_step
if [ -f "$RUN/out/eval/summary.yaml" ]; then
  echo "STEP: ===== ledger ====="
  python scripts/collect_vo_ledger.py "$RUN" \
    /pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/ca3_vo_chaoticramp_dtw \
    -o $SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv
else
  echo "STEP: no eval summary at $RUN/out/eval/ - check $RUN/log.train / log.evalvolt"
fi
echo "STEP: session done $(date)"
