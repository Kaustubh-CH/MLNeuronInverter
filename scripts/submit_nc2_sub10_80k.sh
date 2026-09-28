#!/bin/bash
# L5 nc2 voltage-only on the SENSITIVE SUBSET (10 of 19 channels; the other 9 pinned at the BBP
# default in both the pack and the in-loop sim).  Recipe = the 200k run (job 58131394): soft-DTW +
# ramped eFEL aux, fp32, dt 0.2, 400 ms 4k50kInterChaoticB x1.5, kernel 128, 50 ep, patience 25,
# B=128/GPU.  80k train on 4 x 80 GB nodes (16 GPUs -> global 2048).  Est 80k x 0.108 GPU-s x 50 /16
# = ~7.5 h (+eval); 11 h wall.
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
PACK=/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02_sub10_100k/
design=ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_sub10_80k
SLOG=$SCRATCH/tmp_neuInv/slurm_logs; mkdir -p "$SLOG" $SCRATCH/tmp_neuInv/rdzv
L5TTPC_NCOMP=2 NEUINV_JAX_X64=false NEUINV_ROOT=model_ladder NEUINV_CELL=l5ttpc_nc2_bbp_synth NEUINV_PROBS="0" \
NEUINV_EPOCHS=${RUN_EPOCHS:-50} NEUINV_NUMGLOBSAMP=80000 NEUINV_WRK_SUFIX=${RUN_SUFIX:-vo_sub10_80k_50ep} NEUINV_TRIM_LAST=1 \
NEUINV_EVAL_ARGS="--noGrad --simBatch 64" NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_sub10_$$ NCCL_P2P_DISABLE=1 \
  sbatch -N 4 -t 11:00:00 -q regular -C "gpu&hbm80g" -J l5nc2_sub10 -o $SLOG/slurm-%j.out --export=ALL \
  batchShifterJaxleyCA3_100ep.slr "$PACK" "$design"
