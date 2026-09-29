#!/bin/bash -l
# Supervised twin of the L5 nc2 fp32/dt0.2 pilot, on ONE interactive node (training ~2 min,
# eval re-simulates 200 test traces at dt 0.2 / 400 ms).  Run inside:
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_nc2_icb4k_sup_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp
cd "$WT" || exit 2
pack=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/; design=ladder_l5ttpc_nc2_icb4k_sup
export L5TTPC_NCOMP=2 NEUINV_ROOT=model_ladder NEUINV_TMP_ROOT=$SCRATCH/tmp_neuInv NEUINV_JAXCC_ROOT=/tmp/jaxcc
export NEUINV_CELL=l5ttpc_nc2_bbp_synth NEUINV_PROBS="0" NEUINV_NUMGLOBSAMP=40000 NEUINV_EPOCHS=${LADDER_EPOCHS:-100}
export NEUINV_WRK_SUFIX=sup_icb4k NEUINV_TRIM_LAST=1 NCCL_P2P_DISABLE=1
export NEUINV_EVAL_ARGS="--cellSim l5ttpc --stimNames 4k50kInterChaoticB --clampTanh --noGrad --simBatch 32"
unset NEUINV_RDZV_FILE            # 1 node -> TCP rendezvous
echo "===== $design start $(date) on $(hostname)"
bash batchShifterJaxleyCA3_100ep.slr "$pack" "$design" > docs/model_ladder/train_${design}.log 2>&1
R=$NEUINV_TMP_ROOT/model_ladder/$design/l5ttpc_nc2_bbp_synth/sup_icb4k
echo "===== $design exit=$? $(date)"; grep -E "mean R²|spike count:|val-loss=|MSE_z" $R/log.evalvolt $R/log.train 2>/dev/null | tail -4 | cut -c1-170
