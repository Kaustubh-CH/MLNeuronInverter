#!/bin/bash
# Submit the L5 nc2 voltage-only pilot: fp32, dt 0.2, 400 ms InterChaoticB, 100 epochs,
# 2 x 80 GB A100 nodes (B=256/GPU -> global batch 2048).  ~9.5 h estimated; 12 h wall.
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/
design=ladder_l5ttpc_nc2_icb4k_vo_fp32dt02
SLOG=$SCRATCH/tmp_neuInv/slurm_logs; mkdir -p "$SLOG" $SCRATCH/tmp_neuInv/rdzv
L5TTPC_NCOMP=2 NEUINV_JAX_X64=false NEUINV_ROOT=model_ladder NEUINV_CELL=l5ttpc_nc2_bbp_synth NEUINV_PROBS="0" \
NEUINV_EPOCHS=${PILOT_EPOCHS:-100} NEUINV_NUMGLOBSAMP=40000 NEUINV_WRK_SUFIX=${PILOT_SUFIX:-vo_fp32dt02_100ep} NEUINV_TRIM_LAST=1 \
NEUINV_EVAL_ARGS="--noGrad --simBatch 64" NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_pilot_$$ NCCL_P2P_DISABLE=1 \
  sbatch -N 2 -t 12:00:00 -q regular -C "gpu&hbm80g" -J l5nc2_pilot -o $SLOG/slurm-%j.out --export=ALL \
  batchShifterJaxleyCA3_100ep.slr "$PACK" "$design"
