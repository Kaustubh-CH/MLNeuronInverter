#!/bin/bash -l
# 2026-09-24: Roy-exp overlays (with roy_traces.npz) + scripts/compare_royexp_ft.py for the c = 3 vs x1
# fine-tunes of the 200k L5 nc2 model.  STAGE=ep0: the epoch-0 ckpt of the quota-killed first c3 launch.  STAGE=ref (default): references that exist now (old x1 pilot
# fine-tune; 200k model zero-shot at x1 and x3).  STAGE=ft: the two 200k fine-tunes (58842880 c3,
# 58842882 x1) + the full comparison table.  Every compute line under srun (salloc <script> runs on
# the LOGIN node).
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g --gpus-per-node=1 bash scripts/run_royexp_ft_compare_salloc.sh
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp
cd "$WT" || exit 2
module load conda; conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export L5TTPC_NCOMP=2 JAX_ENABLE_X64=true
M=$SCRATCH/tmp_neuInv/model_ladder
B200=$M/ladder_l5ttpc_nc2_icb4k_vo_fp32dt02_200k/l5ttpc_nc2_bbp_synth/vo_fp32dt02_200k_50ep/out
PILOT_FT=$M/l5nc2_royexp_ft_dt02_efel5_vb/RoyExpChaotic/ft_efel5vb_40ep/out
# relaunch 2026-09-26 (58895462 c3 / 58895463 x1) writes to HOME: pscratch is over quota
MH=/global/homes/k/ktub1999/tmp_neuInv/model_ladder
FT_C3=$MH/l5nc2_royexp_ft_dt02_efel5_vb_c3/RoyExpChaotic/ft200k_40ep/out
FT_X1=$MH/l5nc2_royexp_ft_dt02_efel5_vb/RoyExpChaotic/ft200k_40ep/out
# first c3 launch 58842880: died in torch.save at epoch 11 (quota); its ckpt.pth = epoch 0 (best val 4.61)
C3_EP0=/global/homes/k/ktub1999/tmp_neuInv/royexp_ft_c3_ep0_model   # symlinks to its weights + reconstructed sum_train.yaml
EP0_OUT=/global/homes/k/ktub1999/tmp_neuInv/royexp_ft_c3_ep0
DOC=docs/model_ladder/sensitivity/royexp_ft_c3; mkdir -p $DOC
ov() { # model outDir scale tag
  srun -n1 --gpus=1 python -u plot_exp_overlay_royv2_l5dt02.py -m "$1" --outDir "$2" --stimScale "$3" --tag "$4"; }
if [[ "${STAGE:-ref}" == ep0 ]]; then
  ov $C3_EP0 $EP0_OUT 3.0 "200k exp-ft c3 epoch0 (58842880)"
  srun -n1 python -u scripts/compare_royexp_ft.py ft200k_c3_ep0=$EP0_OUT zs200k_c3=$B200/exp_royv2_l5dt02_c3 \
     zs200k_x1=$B200/exp_royv2_l5dt02_x1 pilotft_x1=$PILOT_FT/exp_royv2_l5dt02_traces --outCsv $DOC/compare_ep0.csv
  cp $EP0_OUT/roy_overlay_grid.png $DOC/overlay_ft200k_c3_ep0.png
elif [[ "${STAGE:-ref}" == ref ]]; then
  ov $PILOT_FT $PILOT_FT/exp_royv2_l5dt02_traces 1.0 "pilot exp-ft x1"
  ov $B200 $B200/exp_royv2_l5dt02_x1 1.0 "200k zero-shot x1"
  ov $B200 $B200/exp_royv2_l5dt02_c3 3.0 "200k zero-shot x3"
  srun -n1 python -u scripts/compare_royexp_ft.py pilotft_x1=$PILOT_FT/exp_royv2_l5dt02_traces \
     zs200k_x1=$B200/exp_royv2_l5dt02_x1 zs200k_c3=$B200/exp_royv2_l5dt02_c3 --outCsv $DOC/compare_ref.csv
else
  ov $FT_C3 $FT_C3/exp_royv2_l5dt02 3.0 "200k exp-ft c3"
  ov $FT_X1 $FT_X1/exp_royv2_l5dt02 1.0 "200k exp-ft x1"
  srun -n1 python -u scripts/compare_royexp_ft.py ft200k_c3=$FT_C3/exp_royv2_l5dt02 ft200k_x1=$FT_X1/exp_royv2_l5dt02 \
     pilotft_x1=$PILOT_FT/exp_royv2_l5dt02_traces zs200k_x1=$B200/exp_royv2_l5dt02_x1 \
     zs200k_c3=$B200/exp_royv2_l5dt02_c3 ft200k_c3_ep0=$EP0_OUT --outCsv $DOC/compare.csv
  cp $FT_C3/exp_royv2_l5dt02/roy_overlay_grid.png $DOC/overlay_ft200k_c3.png
  cp $FT_X1/exp_royv2_l5dt02/roy_overlay_grid.png $DOC/overlay_ft200k_x1.png
  cp $FT_C3/exp_royv2_l5dt02/roy_unit_params.png $DOC/unit_params_ft200k_c3.png
fi
