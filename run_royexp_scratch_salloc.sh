#!/bin/bash
# Body of `salloc -N1 ... bash run_royexp_scratch_salloc.sh` (interactive, 2h).
# FROM-SCRATCH k128 training on the Roy exp pack (ablation twin of
# royexp_ft_57631428: same data/loss/epochs, NO warm start, LR 1e-4), then the
# standard Roy exp predictions + the per-family royv2 eval.
# NOTE: the launcher's auto predict/evaluate_voltage run against DUMMY labels
# and a single stim — ignore their channel numbers; the royv2 eval is the
# meaningful held-out score for this pack.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
mkdir -p "$(dirname "$NEUINV_RDZV_FILE")"; rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1
echo "SCRATCHK128: start $(date) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"

NEUINV_CELL=RoyExpChaotic NEUINV_EPOCHS=40 NEUINV_WRK_SUFIX=scratch_k128 \
  bash batchShifterJaxleyCA3_100ep.slr "$DATA" ca3_royexp_scratch_k128
echo "SCRATCHK128: launcher exit=$? $(date)"

RUN=$J/ca3_royexp_scratch_k128/RoyExpChaotic/scratch_k128
if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
  echo "SCRATCHK128: NO checkpoint — dumping log tail"; tail -30 "$RUN/log.train" 2>/dev/null; exit 5
fi

export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_scr_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$WT"
echo "=== scratch_k128 @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
       --simScale rig || echo "SCRATCHK128: rig FAILED"
echo "=== scratch_k128 @ training stim (Roy2000_icav2_5k)"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy_train" \
       --simScale train || echo "SCRATCHK128: train FAILED"
echo "=== scratch_k128 royv2 per-family held-out eval"
cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" \
       --outDir "$RUN/out/exp_royv2" || echo "SCRATCHK128: royv2 FAILED"
echo "SCRATCHK128: ALL DONE $(date)"
