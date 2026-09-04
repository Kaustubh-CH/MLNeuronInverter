#!/bin/bash -l
# Best broad-spectrum recipe (precond2) at 80k data (v2 pack): combine the two levers
# that work (kdr-neutral grad-precond + more data). Parallel salloc.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK_V2=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v2/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
echo "P2_80k: start $(date) job=$SLURM_JOBID nodes=$SLURM_JOB_NODELIST"
NEUINV_WRK_SUFIX=precond2_80k NEUINV_NUMGLOBSAMP=80000 \
  bash batchShifterJaxleyCA3_100ep.slr "$PACK_V2" ca3_vo_dtw_precond2_chaoticramp
echo "P2_80k: launcher exit=$? $(date)"
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
RUN=$J/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2_80k
if [ -f "$RUN/out/eval/summary.yaml" ]; then
  python scripts/collect_vo_ledger.py "$RUN" -o "$J/vo_ledger/results.csv"
else
  echo "P2_80k: no eval summary at $RUN/out/eval/ - check $RUN/log.*"
fi
echo "P2_80k: done $(date)"
