#!/bin/bash
# sbatch twin of scripts/run_l5_royexp_ft_salloc.sh step (b): the full champion budget
# (40 epochs, LR 5e-5) on one regular-queue node, 8 h.  Default design = the +voltage_base
# variant (the plain 16-ep run drifted the resting level to -40 mV); FT_DESIGN overrides.  Overlay it afterwards with
#   python plot_exp_overlay_royv2_l5dt02.py -m <run>/out --tag "exp fine-tune 40ep"
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp
cd "$WT" || exit 2
PILOT=$SCRATCH/tmp_neuInv/model_ladder/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02/l5ttpc_nc2_bbp_synth/vo_fp32dt02_100ep/out
PACK=/pscratch/sd/k/ktub1999/RoyExpPack_l5dt02/; design=${FT_DESIGN:-l5nc2_royexp_ft_dt02_efel5_vb}
SLOG=$SCRATCH/tmp_neuInv/slurm_logs; mkdir -p "$SLOG" $SCRATCH/tmp_neuInv/rdzv
L5TTPC_NCOMP=2 NEUINV_JAX_X64=false NEUINV_ROOT=model_ladder NEUINV_CELL=RoyExpChaotic NEUINV_PROBS="0" \
NEUINV_EPOCHS=${FT_EPOCHS:-40} NEUINV_NUMGLOBSAMP=40000 NEUINV_WRK_SUFIX=${FT_SUFIX:-ft_efel5vb_40ep} NEUINV_TRIM_LAST=1 \
NEUINV_FT_MODEL=$PILOT/blank_model.pth NEUINV_FT_CKPT=$PILOT/checkpoints/ckpt.pth NEUINV_INITLR=${FT_LR:-0.00005} \
NEUINV_EVAL_ARGS="--noGrad --simBatch 16 --numSamples 8 --numOverlay 2" \
NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_royft_$$ NCCL_P2P_DISABLE=1 \
  sbatch -N 1 -t 8:00:00 -q regular -C gpu -J l5_royexp_ft -o $SLOG/slurm-%j.out --export=ALL \
  batchShifterJaxleyCA3_100ep.slr "$PACK" "$design"
