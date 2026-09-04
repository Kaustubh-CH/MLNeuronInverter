#!/bin/bash
# Body of `salloc -N1 --ntasks-per-node=4 --gpus-per-node=4 ... bash run_1stim_dtw_salloc.sh`.
# Trains the TWO salloc-side PURE-soft-DTW one-stim exp-only k128 models
# (Roy500 + Roy2000; Roy1000/1500 go through run_1stim_dtw_debug.sh on the debug
# queue), sequentially, then scores each: royv2 held-out + rig rows (no Roy100).
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_1stim/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
export NCCL_P2P_DISABLE=1

for AMP in 500 2000; do
  echo "===== 1DTW-$AMP: train $(date) ====="
  cd "$WT"
  rm -f "$NEUINV_RDZV_FILE"
  NEUINV_CELL=RoyExp$AMP NEUINV_EPOCHS=40 NEUINV_WRK_SUFIX=onestim_dtw_$AMP \
    bash batchShifterJaxleyCA3_100ep.slr "$DATA" ca3_royexp_1stim_${AMP}_dtw
  echo "1DTW-$AMP: launcher exit=$?"
  RUN=$J/ca3_royexp_1stim_${AMP}_dtw/RoyExp$AMP/onestim_dtw_$AMP
  if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
    echo "1DTW-$AMP: NO checkpoint"; tail -20 "$RUN/log.train" 2>/dev/null; continue
  fi
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_1dtw_$SLURM_JOB_ID
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  cd "$WT"
  echo "=== 1dtw_$AMP @ rig (no Roy100)"
  python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
         --simScale rig || echo "1DTW-$AMP: rig FAILED"
  echo "=== 1dtw_$AMP royv2 held-out (own family)"
  cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" \
         --packFile "$DATA/RoyExp$AMP.mlPack1.h5" \
         --outDir "$RUN/out/exp_royv2" || echo "1DTW-$AMP: royv2 FAILED"
done
echo "1DTW salloc pair: ALL DONE $(date)"
