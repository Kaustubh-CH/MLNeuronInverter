#!/bin/bash -l
# 2-epoch smoke of the nc2 fp32/dt0.2/400ms voltage-only pilot on one interactive node (4 GPUs).
#   salloc -N1 -C gpu -q interactive -t 0:40:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_pilot_smoke_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/
design=ladder_l5ttpc_nc2_icb4k_vo_fp32dt02
export L5TTPC_NCOMP=2 NEUINV_JAX_X64=false NEUINV_ROOT=model_ladder NEUINV_CELL=l5ttpc_nc2_bbp_synth NEUINV_PROBS="0"
export NEUINV_EPOCHS=2 NEUINV_NUMGLOBSAMP=4096 NEUINV_WRK_SUFIX=smoke2_fp32dt02 NEUINV_TRIM_LAST=1
export NEUINV_EVAL_ARGS="--noGrad --simBatch 64" NCCL_P2P_DISABLE=1
unset NEUINV_RDZV_FILE
echo "PILOT-SMOKE start $(date) job=${SLURM_JOB_ID:-?}"
bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$design" > docs/model_ladder/pilot_smoke.log 2>&1
echo "PILOT-SMOKE exit=$? $(date)"
R=$SCRATCH/tmp_neuInv/model_ladder/$design/l5ttpc_nc2_bbp_synth/smoke2_fp32dt02
grep -E "HybridLoss\]|Epoch [0-9]+ took|mean R|MSE_z|spike count|Traceback|Error|nan" $R/log.train $R/log.evalvolt 2>/dev/null | tail -14 | cut -c1-180
