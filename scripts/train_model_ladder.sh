#!/bin/bash
# Submit model-ladder training jobs (one sbatch per rung x stim x mode).
#   bash scripts/train_model_ladder.sh sup  [rungs...]   # supervised param-MSE, 1 node, ~0.5 h
#   bash scripts/train_model_ladder.sh vo   [rungs...]   # voltage-only DTW+eFEL, 4 nodes (HH/BBP-ball only)
# Runs land in $SCRATCH/tmp_neuInv/model_ladder/<design>/<cellname>/<mode>_<stim>/out
# and auto-evaluate (predict.py + evaluate_voltage.py -> out/eval/summary.yaml).
set -u
MODE=${1:?sup|vo}; shift
RUNGS=${*:-"single_comp ball_and_stick ball_and_stick_bbp l5ttpc_nc2 l5ttpc_nc4"}
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
ROOT=${LADDER_DATA_ROOT:-/global/homes/k/ktub1999/model_ladder_data}
RUNROOT=${NEUINV_TMP_ROOT:-/global/homes/k/ktub1999/tmp_neuInv}     # pscratch over quota -> HOME
SLOG=/global/homes/k/ktub1999/tmp_neuInv/slurm_logs; mkdir -p "$SLOG"
for rung in $RUNGS; do
  case $rung in
    single_comp)        cell=soma_only;          token=single_comp_synth;        probes="0";   nodes=4; ncomp="" ;;
    ball_and_stick)     cell=ball_and_stick;     token=ball_and_stick_synth;     probes="0 1"; nodes=4; ncomp="" ;;
    ball_and_stick_bbp) cell=ball_and_stick_bbp; token=ball_and_stick_bbp_synth; probes="0";   nodes=4; ncomp="" ;;
    l5ttpc_nc2)         cell=l5ttpc;             token=l5ttpc_nc2_synth;         probes="0";   nodes=4; ncomp=2 ;;
    l5ttpc_nc4)         cell=l5ttpc;             token=l5ttpc_nc4_synth;         probes="0";   nodes=4; ncomp=4 ;;
    *) echo "unknown rung $rung"; continue ;;
  esac
  for stag in icb cr; do
    [[ $stag == icb ]] && stim=5k50kInterChaoticB || stim=5kChaoticRamp
    pack=$ROOT/${rung}_${stim}/
    design=ladder_${rung}_${stag}_${MODE}
    if [ ! -f "$pack/$token.mlPack1.h5" ]; then echo "SKIP $design: pack missing ($pack)"; continue; fi
    if [[ $MODE == vo && $rung == l5ttpc_* ]]; then echo "SKIP $design: in-loop L5 is 28-110x CA3 cost (see RESULTS_vo.md); run supervised"; continue; fi
    # evaluate_voltage flags: supervised runs carry no voltage-loss block -> simulator from CLI.
    if [[ $MODE == sup ]]; then evargs="--cellSim $cell --stimNames $stim --clampTanh"; else evargs=""; fi
    [[ -n "$ncomp" ]] && evargs="$evargs --noGrad --simBatch 32"
    if [[ $MODE == sup ]]; then N=1; T=1:30:00; Q=regular; else N=$nodes; T=4:00:00; Q=regular; fi
    echo "SUBMIT $design  pack=$pack  nodes=$N  eval='$evargs'"
    L5TTPC_NCOMP=$ncomp NEUINV_ROOT=model_ladder NEUINV_TMP_ROOT=$RUNROOT NEUINV_JAXCC_ROOT=/tmp/jaxcc \
    NEUINV_CELL=$token NEUINV_PROBS="$probes" NEUINV_TRIM_LAST=1 \
    NEUINV_WRK_SUFIX=${MODE}_${stag} NEUINV_NUMGLOBSAMP=40000 NEUINV_EVAL_ARGS="$evargs" \
    NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_ladder_${design}_$$ NCCL_P2P_DISABLE=1 \
      sbatch -N $N -t $T -q $Q -J $design -o $SLOG/slurm-%j.out --export=ALL batchShifterJaxleyCA3_100ep.slr "$pack" "$design"
  done
done
