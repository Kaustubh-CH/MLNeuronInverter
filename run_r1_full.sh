#!/bin/bash -l
# Full R1 (DTW + coarse->fine van-Rossum blur) voltage-only training + auto-eval.
# Runs INSIDE a 4-node (16-GPU) salloc. Drives the committed launcher, which trains
# 100 epochs then auto-chains predict.py + evaluate_voltage.py into out/eval/, then
# scores the result (and the DTW baseline) into the vo ledger.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2

# Fixed run-dir suffix so the output path is predictable (not the salloc job id).
export NEUINV_WRK_SUFIX=r1_dtwblur
PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/
RUN=$SCRATCH/tmp_neuInv/jaxley_ca3/ca3_vo_dtwblur_chaoticramp/ca3_pyramidal_synth/r1_dtwblur
DTW_BASE=/pscratch/sd/k/ktub1999/tmp_neuInv/ca3_ablation/ca3_vo_chaoticramp_dtw
LEDGER=$SCRATCH/tmp_neuInv/jaxley_ca3/vo_ledger/results.csv

echo "F: full R1 start $(date)  job=$SLURM_JOBID  nodes=$SLURM_JOB_NODELIST"
echo "F: run dir will be $RUN"

# The launcher does its own module load / conda activate / env exports / srun.
bash batchShifterJaxleyCA3_100ep.slr "$PACK" ca3_vo_dtwblur_chaoticramp
echo "F: launcher exit=$?  $(date)"

# Score into the ledger (re-activate env; launcher ran in its own subshell).
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
if [[ -f "$RUN/out/eval/summary.yaml" ]]; then
  echo "F: === R1 eval summary ==="
  python scripts/collect_vo_ledger.py "$RUN" "$DTW_BASE" -o "$LEDGER"
else
  echo "F: NO eval/summary.yaml at $RUN/out/eval/ -- check log.train / log.evalvolt in $RUN"
fi
echo "F: full R1 done $(date)"
