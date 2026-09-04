#!/bin/bash
# Body of `salloc -N1 --ntasks-per-node=4 --gpus-per-node=4 ... bash run_stimch_salloc.sh`.
# STIM-AS-CHANNEL A/B (user 2026-08-28): the efel5-ema champion recipe, from
# scratch, on the 2-channel RoyExpStimCh pack.
#   arm 2ch: voltage + recorded-stim channels (NEUINV_PROBS="0 1")
#   arm 1ch: voltage only, SAME pack/loss/LR (the control)
# Each trained 80 ep then scored under the exact recorded stims.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_stimch/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
DESIGN=ca3_royexp_stimch

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
export NCCL_P2P_DISABLE=1
export NEUINV_STIMS="0" NEUINV_VALIDSTIMS="0"

for ARM in 2ch 1ch; do
  if [ "$ARM" = 2ch ]; then export NEUINV_PROBS="0 1"; CH="0 1"; else export NEUINV_PROBS="0"; CH="0"; fi
  SUF=stimch_$ARM
  RUN=$J/$DESIGN/RoyExpStimCh/$SUF
  cd "$WT"; rm -f "$NEUINV_RDZV_FILE"
  if [ -f "$RUN/out/checkpoints/ckpt.pth" ] && [ -f "$RUN/out/sum_train.yaml" ]; then
    echo "STIMCH-$ARM: completed model present -- SKIP training"
  else
    echo "===== STIMCH-$ARM: train $(date) probs='$NEUINV_PROBS' ====="
    NEUINV_CELL=RoyExpStimCh NEUINV_EPOCHS=80 NEUINV_WRK_SUFIX=$SUF \
      bash batchShifterJaxleyCA3_100ep.slr "$DATA" $DESIGN
    echo "STIMCH-$ARM: launcher exit=$?"
  fi
  if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
    echo "STIMCH-$ARM: NO checkpoint"; tail -25 "$RUN/log.train" 2>/dev/null; continue
  fi
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_stimch_$SLURM_JOB_ID
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  echo "=== STIMCH-$ARM eval (icaRec, held-out neurons)"
  python -u plot_exp_overlay_stimch.py -m "$RUN/out" --channels "$CH" \
         --outDir "$RUN/out/exp_stimch" || echo "STIMCH-$ARM: eval FAILED"
done
echo "STIMCH: ALL DONE $(date)"
