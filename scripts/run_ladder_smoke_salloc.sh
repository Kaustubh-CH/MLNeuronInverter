#!/bin/bash -l
# 2-epoch smoke of the ladder training path (supervised + voltage-only) on the
# point-neuron InterChaoticB pack, inside one interactive node (4 GPUs).
#   salloc -N1 -C gpu -q interactive -t 0:40:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_ladder_smoke_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/global/homes/k/ktub1999/model_ladder_data/single_comp_5k50kInterChaoticB/
export NEUINV_ROOT=model_ladder NEUINV_TMP_ROOT=/global/homes/k/ktub1999/tmp_neuInv NEUINV_JAXCC_ROOT=/tmp/jaxcc NEUINV_CELL=single_comp_synth NEUINV_PROBS="0" NEUINV_EPOCHS=2 NEUINV_NUMGLOBSAMP=8192
export NCCL_P2P_DISABLE=1
for mode in sup vo; do
  design=ladder_single_comp_icb_$mode
  if [[ $mode == sup ]]; then export NEUINV_EVAL_ARGS="--cellSim soma_only --stimNames 5k50kInterChaoticB --clampTanh"; else export NEUINV_EVAL_ARGS=""; fi
  export NEUINV_WRK_SUFIX=smoke_${mode} NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_smoke_${mode}_$SLURM_JOB_ID
  echo "SMOKE $design start $(date)"
  bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$design" > docs/model_ladder/smoke_${mode}.log 2>&1
  echo "SMOKE $design exit=$? $(date)"
  R=$NEUINV_TMP_ROOT/model_ladder/$design/single_comp_synth/smoke_${mode}
  grep -E "Average loss|#res|mean R|MSE_z|spike count|Error|error|Traceback" $R/log.train $R/log.predict $R/log.evalvolt 2>/dev/null | tail -12
done
echo "SMOKE all done $(date)"
