#!/bin/bash
# 2026-09-24 copy of scripts/submit_l5_royexp_ft.sh warm-starting from the FINISHED 200k L5 nc2 model
# (vo_fp32dt02_200k_50ep, job 58131394) instead of the 100-ep pilot.  Same budget: 40 ep, LR 5e-5,
# 1 regular node, 8 h.  FT_DESIGN picks the recipe:
#   l5nc2_royexp_ft_dt02_efel5_vb_c3  (default; stim_scale 3.0)
#   l5nc2_royexp_ft_dt02_efel5_vb     (x1 control -- set FT_SUFIX so it does not land in ft_efel5vb_40ep)
# Overlay afterwards with plot_exp_overlay_royv2_l5dt02.py -m <run>/out --stimScale <c> --tag ...
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp
cd "$WT" || exit 2
PILOT=${FT_PILOT:-$SCRATCH/tmp_neuInv/model_ladder/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_200k/l5ttpc_nc2_bbp_synth/vo_fp32dt02_200k_50ep/out}
PACK=/pscratch/sd/k/ktub1999/RoyExpPack_l5dt02/; design=${FT_DESIGN:-l5nc2_royexp_ft_dt02_efel5_vb_c3}
# 2026-09-26: pscratch went 102.5% over quota and killed the first launch (58842880/58842882) in torch.save;
# everything the job WRITES goes to TMPR (default HOME).  The DDP FileStore must NOT live in HOME (no flock on
# compute nodes -> DistStoreError 524, relaunch 58895462/3); single node -> node-local /tmp.  Reads (pack, warm start) stay on pscratch.
TMPR=${FT_TMP_ROOT:-/global/homes/k/ktub1999/tmp_neuInv}
SLOG=$TMPR/slurm_logs; mkdir -p "$SLOG"
[[ -f $PILOT/blank_model.pth && -f $PILOT/checkpoints/ckpt.pth ]] || { echo "missing warm-start in $PILOT"; exit 3; }
NEUINV_TMP_ROOT=$TMPR NEUINV_JAXCC_ROOT=/tmp/jaxcc \
L5TTPC_NCOMP=2 NEUINV_JAX_X64=false NEUINV_ROOT=model_ladder NEUINV_CELL=RoyExpChaotic NEUINV_PROBS="0" \
NEUINV_EPOCHS=${FT_EPOCHS:-40} NEUINV_NUMGLOBSAMP=40000 NEUINV_WRK_SUFIX=${FT_SUFIX:-ft200k_40ep} NEUINV_TRIM_LAST=1 \
NEUINV_FT_MODEL=$PILOT/blank_model.pth NEUINV_FT_CKPT=$PILOT/checkpoints/ckpt.pth NEUINV_INITLR=${FT_LR:-0.00005} \
NEUINV_EVAL_ARGS="--noGrad --simBatch 16 --numSamples 8 --numOverlay 2" \
NEUINV_RDZV_FILE=/tmp/rdzv_royft_$$_${FT_SUFIX:-c3} NCCL_P2P_DISABLE=1 \
  sbatch -N 1 -t 8:00:00 -q regular -C gpu -J l5_royexp_ft -o $SLOG/slurm-%j.out --export=ALL \
  batchShifterJaxleyCA3_100ep.slr "$PACK" "$design"
