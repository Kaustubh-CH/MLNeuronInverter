#!/bin/bash
# Body of `salloc -N1 --ntasks-per-node=4 --gpus-per-node=4 ... bash run_efel5_stab_salloc.sh`.
# STABILIZED spike-champion run (user: watch the loss curves): the w=5 rate-
# dominated recipe (ca3_royexp_ft_icarec_wide_dtwefel5_ema) with LR 5e-5 -> 2e-5
# and 40 -> 80 epochs, to converge IN the firing basin instead of scoring a
# diverging run's early checkpoint.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428
SUP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_supervised_interchaoticB_k128/ca3_pyramidal_synth/super/out

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1
export NEUINV_FT_MODEL=$SUP/blank_model.pth
export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
export NEUINV_INITLR=2e-5

DESIGN=ca3_royexp_ft_icarec_wide_dtwefel5_ema
SUF=ft_icarec_wide_efel5_ema
echo "===== EFEL5-EMA: train $(date) LR=2e-5 epochs=80 ====="
NEUINV_CELL=RoyExpChaotic NEUINV_EPOCHS=80 NEUINV_WRK_SUFIX=$SUF \
  bash batchShifterJaxleyCA3_100ep.slr "$DATA" $DESIGN
echo "EFEL5-EMA: launcher exit=$?"
RUN=$J/$DESIGN/RoyExpChaotic/$SUF
if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
  echo "EFEL5-EMA: NO checkpoint"; tail -20 "$RUN/log.train" 2>/dev/null; exit 5
fi
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_e5ema_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "=== EFEL5-EMA held-out @ icaRec"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" --physFromModel \
     --packFile "$DATA/RoyExpChaotic.mlPack1.h5" --stimSuffix icaRec_5k \
     --outDir "$RUN/out/exp_royv2_icarec") || echo "EFEL5-EMA: icaRec score FAILED"
echo "=== EFEL5-EMA held-out @ icav2"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" --physFromModel \
     --packFile "$DATA/RoyExpChaotic.mlPack1.h5" \
     --outDir "$RUN/out/exp_royv2") || echo "EFEL5-EMA: icav2 score FAILED"
echo "EFEL5-EMA: ALL DONE $(date)"
