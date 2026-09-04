#!/bin/bash
# Body of `salloc -N1 -n4 --gpus-per-node=4 ... bash run_1stim_salloc.sh` (2h).
# Trains FOUR one-stim exp-only k128 models (Roy500/1000/1500/2000, each on its
# own family's sweeps + its own icav2 stim; Roy100 dropped), sequentially, then
# scores each: royv2 held-out on its family pack + standard rig rows (no Roy100).
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

for AMP in 500 1000 1500 2000; do
  echo "===== 1STIM-$AMP: train $(date) ====="
  cd "$WT"
  rm -f "$NEUINV_RDZV_FILE"
  NEUINV_CELL=RoyExp$AMP NEUINV_EPOCHS=40 NEUINV_WRK_SUFIX=onestim_$AMP \
    bash batchShifterJaxleyCA3_100ep.slr "$DATA" ca3_royexp_1stim_$AMP
  echo "1STIM-$AMP: launcher exit=$?"
  RUN=$J/ca3_royexp_1stim_$AMP/RoyExp$AMP/onestim_$AMP
  if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
    echo "1STIM-$AMP: NO checkpoint"; tail -20 "$RUN/log.train" 2>/dev/null; continue
  fi
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_1stim_$SLURM_JOB_ID
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  cd "$WT"
  echo "=== 1stim_$AMP @ rig (all amps, no Roy100)"
  python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
         --simScale rig || echo "1STIM-$AMP: rig FAILED"
  echo "=== 1stim_$AMP royv2 held-out (own family)"
  cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" \
         --packFile "$DATA/RoyExp$AMP.mlPack1.h5" \
         --outDir "$RUN/out/exp_royv2" || echo "1STIM-$AMP: royv2 FAILED"
done
echo "1STIM: ALL DONE $(date)"
