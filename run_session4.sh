#!/bin/bash -l
# Session 4: dataset up-sweep. Session 3 showed mean R2 climbs with data
# (10k 0.440 -> 20k 0.489 -> 40k 0.593), so test 80k (v2 100k pack, 80k train)
# on the baseline DTW recipe -> clean extension of the data-scaling curve.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK_V2=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v2/
J=$SCRATCH/tmp_neuInv/jaxley_ca3

echo "S4: dtw_80k start $(date) job=$SLURM_JOBID nodes=$SLURM_JOB_NODELIST"
NEUINV_WRK_SUFIX=dtw_80k NEUINV_NUMGLOBSAMP=80000 \
  bash batchShifterJaxleyCA3_100ep.slr "$PACK_V2" ca3_vo_chaoticramp_dtw
echo "S4: dtw_80k launcher exit=$? $(date)"

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
RUN=$J/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_80k
if [ -f "$RUN/out/eval/summary.yaml" ]; then
  echo "S4: ===== ledger ====="
  python scripts/collect_vo_ledger.py "$RUN" -o "$J/vo_ledger/results.csv"
else
  echo "S4: no eval summary at $RUN/out/eval/ - check $RUN/log.*"
fi
echo "S4: session 4 done $(date)"
