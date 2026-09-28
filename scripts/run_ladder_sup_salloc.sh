#!/bin/bash -l
# All 10 supervised (param-MSE) model-ladder runs, sequentially, on ONE interactive node
# (each is minutes: the CNN never calls jaxley during training; eval re-simulates 200 test traces).
#   salloc -N1 -C gpu -q interactive -t 2:30:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_ladder_sup_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
ROOT=${LADDER_DATA_ROOT:-/global/homes/k/ktub1999/model_ladder_data}
export NEUINV_ROOT=model_ladder NEUINV_TMP_ROOT=/global/homes/k/ktub1999/tmp_neuInv NEUINV_JAXCC_ROOT=/tmp/jaxcc
export NEUINV_EPOCHS=${LADDER_EPOCHS:-100} NEUINV_TRIM_LAST=1 NCCL_P2P_DISABLE=1
RUNGS=${LADDER_ONLY:-"single_comp ball_and_stick ball_and_stick_bbp l5ttpc_nc2 l5ttpc_nc4"}
for rung in $RUNGS; do
  case $rung in
    single_comp)        cell=soma_only;          token=single_comp_synth;        probes="0";   nsamp=40000; ncomp="" ;;
    ball_and_stick)     cell=ball_and_stick;     token=ball_and_stick_synth;     probes="0 1"; nsamp=40000; ncomp="" ;;
    ball_and_stick_bbp) cell=ball_and_stick_bbp; token=ball_and_stick_bbp_synth; probes="0";   nsamp=40000; ncomp="" ;;
    l5ttpc_nc2)         cell=l5ttpc;             token=l5ttpc_nc2_synth;         probes="0";   nsamp=20000; ncomp=2 ;;
    l5ttpc_nc4)         cell=l5ttpc;             token=l5ttpc_nc4_synth;         probes="0";   nsamp=20000; ncomp=4 ;;
    *) echo "unknown rung '$rung'"; exit 2 ;;
  esac
  for stag in icb cr; do
    [[ $stag == icb ]] && stim=5k50kInterChaoticB || stim=5kChaoticRamp
    pack=$ROOT/${rung}_${stim}/; design=ladder_${rung}_${stag}_sup
    for i in $(seq 1 60); do [ -f "$pack/$token.mlPack1.h5" ] && break; echo "waiting for $pack ($i)"; sleep 60; done
    [ -f "$pack/$token.mlPack1.h5" ] || { echo "SKIP $design: pack missing"; continue; }
    if [ -f "$NEUINV_TMP_ROOT/model_ladder/$design/$token/sup_${stag}/out/eval/summary.yaml" ]; then echo "SKIP $design: done"; continue; fi
    if [ -n "$ncomp" ]; then export L5TTPC_NCOMP=$ncomp; evx="--noGrad --simBatch 32"; else unset L5TTPC_NCOMP; evx=""; fi
    export NEUINV_CELL=$token NEUINV_PROBS="$probes" NEUINV_NUMGLOBSAMP=$nsamp NEUINV_WRK_SUFIX=sup_${stag}
    export NEUINV_EVAL_ARGS="--cellSim $cell --stimNames $stim --clampTanh $evx"
    # single node -> TCP rendezvous (MASTER_ADDR).  A FileStore on HOME fails with
    # DistStoreError 524 (no file locking on the HOME filesystem).
    unset NEUINV_RDZV_FILE
    echo "===== $design start $(date)"
    bash batchShifterJaxleyCA3_100ep.slr "$pack" "$design" > docs/model_ladder/train_${design}.log 2>&1
    R=$NEUINV_TMP_ROOT/model_ladder/$design/$token/sup_${stag}
    echo "===== $design exit=$? $(date)"; grep -E "mean R²|spike count:|val-loss=" $R/log.evalvolt $R/log.train 2>/dev/null | tail -3 | cut -c1-170
  done
done
echo "LADDER-SUP all done $(date)"
