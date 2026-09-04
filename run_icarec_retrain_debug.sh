#!/bin/bash -l
#SBATCH -N1 --time=30:00 -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 -C gpu -A m2043_g
# icaRec retrain (finding 11): pooled exp-only model with the EXACT RECORDED rig
# stimulus in the training loop.  VARIANT=scratch -> from-scratch twin of
# ca3_royexp_scratch_k128; VARIANT=ft -> fine-tune twin of royexp_ft (warm-start
# from supervised_interchaoticB_k128).  Scored held-out under BOTH icav2 (old
# rows comparable) and icaRec stims.
# Submit: sbatch --export=ALL,VARIANT=<scratch|ft> -J icarec-<v> run_icarec_retrain_debug.sh
set -u
: "${VARIANT:?set VARIANT=scratch|ft}"
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428
SUP=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_supervised_interchaoticB_k128/ca3_pyramidal_synth/super/out

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOBID}
rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1

SCORE_FLAGS=""
if [ "$VARIANT" = scratch ]; then
  DESIGN=ca3_royexp_scratch_icarec; SUF=scratch_icarec
elif [ "$VARIANT" = ftwide ]; then
  DESIGN=ca3_royexp_ft_icarec_wide; SUF=ft_icarec_wide
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = ftwidedtw ]; then
  DESIGN=ca3_royexp_ft_icarec_wide_dtw; SUF=ft_icarec_wide_dtw
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = ftwidedtwefel ]; then
  DESIGN=ca3_royexp_ft_icarec_wide_dtwefel; SUF=ft_icarec_wide_dtwefel
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = efel5 ]; then
  DESIGN=ca3_royexp_ft_icarec_wide_dtwefel5; SUF=ft_icarec_wide_efel5
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = efel3 ]; then
  DESIGN=ca3_royexp_ft_icarec_wide_dtwefel3; SUF=ft_icarec_wide_efel3
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = efel8 ]; then
  DESIGN=ca3_royexp_ft_icarec_wide_dtwefel8; SUF=ft_icarec_wide_efel8
  SCORE_FLAGS="--physFromModel"
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
elif [ "$VARIANT" = polish ]; then
  # warm-start from the FIRING ftwide checkpoint (4.8 spikes @Roy2000) and
  # polish with the stable DTW+eFEL objective -- start inside the firing
  # region instead of the supervised basin
  DESIGN=ca3_royexp_ft_icarec_wide_dtwefel; SUF=ft_icarec_wide_polish
  SCORE_FLAGS="--physFromModel"
  FTWIDE=$J/ca3_royexp_ft_icarec_wide/RoyExpChaotic/ft_icarec_wide/out
  export NEUINV_FT_MODEL=$FTWIDE/blank_model.pth
  export NEUINV_FT_CKPT=$FTWIDE/checkpoints/ckpt.pth
else
  DESIGN=ca3_royexp_ft_icarec; SUF=ft_icarec
  export NEUINV_FT_MODEL=$SUP/blank_model.pth
  export NEUINV_FT_CKPT=$SUP/checkpoints/ckpt.pth
fi

echo "===== ICAREC-$VARIANT: train $(date) job=$SLURM_JOBID design=$DESIGN ====="
NEUINV_CELL=RoyExpChaotic NEUINV_EPOCHS=40 NEUINV_WRK_SUFIX=$SUF \
  bash batchShifterJaxleyCA3_100ep.slr "$DATA" $DESIGN
echo "ICAREC-$VARIANT: launcher exit=$?"
RUN=$J/$DESIGN/RoyExpChaotic/$SUF
if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
  echo "ICAREC-$VARIANT: NO checkpoint"; tail -20 "$RUN/log.train" 2>/dev/null; exit 5
fi
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_icarec_$SLURM_JOBID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "=== ICAREC-$VARIANT held-out @ icaRec (self-consistent, TRUE drive)"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" $SCORE_FLAGS \
     --packFile "$DATA/RoyExpChaotic.mlPack1.h5" --stimSuffix icaRec_5k \
     --outDir "$RUN/out/exp_royv2_icarec") || echo "ICAREC-$VARIANT: icaRec score FAILED"
echo "=== ICAREC-$VARIANT held-out @ icav2 (comparable to previous rows)"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" $SCORE_FLAGS \
     --packFile "$DATA/RoyExpChaotic.mlPack1.h5" \
     --outDir "$RUN/out/exp_royv2") || echo "ICAREC-$VARIANT: icav2 score FAILED"
echo "ICAREC-$VARIANT: ALL DONE $(date)"
