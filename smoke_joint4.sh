#!/bin/bash -l
# Short GPU smoke for the joint4 / pool4 arms — run INSIDE an interactive allocation.
#
# Purpose is NOT accuracy, it is two facts neither arm has ever produced on a GPU:
#   1. does variant A fit in memory? It holds all FOUR stims' sim graphs at once for
#      backward, ~4x the champion's peak activation memory, at batch_size 64/rank.
#   2. what is the REAL seconds/epoch? The 24 h wall on the queued jobs came from
#      extrapolating the campaign's 247 s/epoch at 160k solves; this measures it.
#
# Usage (inside salloc):
#   bash smoke_joint4.sh A     # joint, needs ca3_joint4_v1
#   bash smoke_joint4.sh B     # pooled, needs ca3_joint4_v1_pooled
set -u
ARM=${1:?pass A or B}
EP=${NEUINV_EPOCHS:-2}

if [ "$ARM" = "A" ]; then
  PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_joint4_v1/
  DESIGN=ca3_joint4_dtw_amp_4n
  CELL=ca3_pyramidal_joint4
  export NEUINV_PROBS="0 1 2 3" NEUINV_STIMS="0" NEUINV_VALIDSTIMS="0"
else
  PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_joint4_v1_pooled/
  DESIGN=ca3_pool4_dtw_amp_4n
  CELL=ca3_pyramidal_pool4
  export NEUINV_PROBS="0" NEUINV_STIMS="0 1 2 3" NEUINV_VALIDSTIMS="2"
fi

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/smoke_${SLURM_JOBID}_${ARM}
mkdir -p "$(dirname "$NEUINV_RDZV_FILE")"; rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1

echo "=== SMOKE $ARM: $EP epochs, $SLURM_NNODES nodes, design=$DESIGN ==="
date
NEUINV_CELL=$CELL NEUINV_WRK_SUFIX=smoke_${ARM}_${SLURM_JOBID} \
NEUINV_NUMGLOBSAMP=80000 NEUINV_EPOCHS=$EP \
  bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$DESIGN"
echo "=== SMOKE $ARM exit=$? ==="
date

RUN=$SCRATCH/tmp_neuInv/jaxley_ca3/$DESIGN/$CELL/smoke_${ARM}_${SLURM_JOBID}
echo "--- per-epoch timing + any OOM ---"
grep -iE "timePerEpoch|sec/epoch|out of memory|CUDA|RESOURCE_EXHAUSTED|Error" "$RUN/log.train" 2>/dev/null | tail -20
echo "--- last loss lines ---"
grep -iE "save_checkpoint|val-loss|loss=" "$RUN/log.train" 2>/dev/null | tail -6
echo "RUN=$RUN"
