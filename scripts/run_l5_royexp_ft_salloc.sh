#!/bin/bash -l
# L5 nc2 fp32/dt0.2 pilot on Paula's Roy v2 recordings: (a) zero-shot overlay, (b) voltage-only
# fine-tune on the exp pack (efel5 recipe, 16 ep fp32 LR 1e-4 -- ~10 min/epoch: 5 family sims per step), (c) overlay of the fine-tuned model.
# ONE interactive node:
#   salloc -N1 -C gpu -q interactive -t 4:00:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_l5_royexp_ft_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp
cd "$WT" || exit 2
PILOT=${PILOT_OUT:-$SCRATCH/tmp_neuInv/model_ladder/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02/l5ttpc_nc2_bbp_synth/vo_fp32dt02_100ep/out}
design=l5nc2_royexp_ft_dt02_efel5_ema; suffix=${FT_SUFIX:-ft_efel5ema}
PACK=/pscratch/sd/k/ktub1999/RoyExpPack_l5dt02/
LOGD=docs/model_ladder/exp; mkdir -p $LOGD
PYENV=/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
module load python; source activate $PYENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true L5TTPC_NCOMP=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

echo "===== (a) zero-shot overlay of the pilot  $(date)"
srun -n1 --gpus=1 python plot_exp_overlay_royv2_l5dt02.py -m "$PILOT" --outDir "$PILOT/exp_royv2_l5dt02" \
     --dom test --numOverlay 4 --tag "pilot zero-shot" > $LOGD/zeroshot_pilot.log 2>&1
grep "royv2-l5dt02\] Roy" $LOGD/zeroshot_pilot.log | cut -c1-150

echo "===== (b) fine-tune on the exp pack  $(date)"
export NEUINV_ROOT=model_ladder NEUINV_TMP_ROOT=$SCRATCH/tmp_neuInv NEUINV_JAXCC_ROOT=/tmp/jaxcc
export NEUINV_CELL=RoyExpChaotic NEUINV_PROBS="0" NEUINV_NUMGLOBSAMP=40000 NEUINV_EPOCHS=${FT_EPOCHS:-16}
export NEUINV_WRK_SUFIX=$suffix NEUINV_TRIM_LAST=1 NCCL_P2P_DISABLE=1
export NEUINV_FT_MODEL=$PILOT/blank_model.pth NEUINV_FT_CKPT=$PILOT/checkpoints/ckpt.pth
export NEUINV_JAX_X64=false NEUINV_INITLR=${FT_LR:-0.0001}   # fp32 solve; 2x the champion LR for the short interactive budget
export NEUINV_EVAL_ARGS="--noGrad --simBatch 16 --numSamples 8 --numOverlay 2"   # exp pack: labels are dummies; real scoring is (c)
unset NEUINV_RDZV_FILE            # 1 node -> TCP rendezvous
bash batchShifterJaxleyCA3_100ep.slr "$PACK" "$design" > $LOGD/train_${design}.log 2>&1
FT=$NEUINV_TMP_ROOT/model_ladder/$design/RoyExpChaotic/$suffix/out
echo "===== fine-tune exit=$? $(date)  run=$FT"
grep -h "took .* sec" $(dirname $FT)/log.train 2>/dev/null | tail -2 | cut -c1-150

echo "===== (c) overlay of the fine-tuned model  $(date)"
if [ -f "$FT/checkpoints/ckpt.pth" ]; then
  srun -n1 --gpus=1 python plot_exp_overlay_royv2_l5dt02.py -m "$FT" --outDir "$FT/exp_royv2_l5dt02" \
       --dom test --numOverlay 4 --tag "exp fine-tune" > $LOGD/zeroshot_ft.log 2>&1
  grep "royv2-l5dt02\] Roy" $LOGD/zeroshot_ft.log | cut -c1-150
else
  echo "no fine-tuned checkpoint at $FT -- skipped (c)"
fi
echo "===== done $(date)"
