#!/bin/bash -l
#SBATCH -N1 --time=30:00 -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 -C gpu -A m2043_g
# One one-stim exp-only k128 model (family $ONESTIM_AMP), trained + scored in a
# single debug job. Submit with: sbatch --export=ALL,ONESTIM_AMP=<amp> -J 1stim-<amp> run_1stim_debug.sh
set -u
: "${ONESTIM_AMP:?set ONESTIM_AMP}"
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_1stim/
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOBID}
rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1

echo "===== 1STIM-$ONESTIM_AMP: train $(date) job=$SLURM_JOBID ====="
NEUINV_CELL=RoyExp$ONESTIM_AMP NEUINV_EPOCHS=40 NEUINV_WRK_SUFIX=onestim_$ONESTIM_AMP \
  bash batchShifterJaxleyCA3_100ep.slr "$DATA" ca3_royexp_1stim_$ONESTIM_AMP
echo "1STIM-$ONESTIM_AMP: launcher exit=$?"
RUN=$J/ca3_royexp_1stim_$ONESTIM_AMP/RoyExp$ONESTIM_AMP/onestim_$ONESTIM_AMP
if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
  echo "1STIM-$ONESTIM_AMP: NO checkpoint"; tail -20 "$RUN/log.train" 2>/dev/null; exit 5
fi
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_1stim_$SLURM_JOBID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$WT"
echo "=== 1stim_$ONESTIM_AMP @ rig (no Roy100)"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
       --simScale rig || echo "1STIM-$ONESTIM_AMP: rig FAILED"
echo "=== 1stim_$ONESTIM_AMP royv2 held-out (own family)"
cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$RUN/out" \
       --packFile "$DATA/RoyExp$ONESTIM_AMP.mlPack1.h5" \
       --outDir "$RUN/out/exp_royv2" || echo "1STIM-$ONESTIM_AMP: royv2 FAILED"
echo "1STIM-$ONESTIM_AMP: ALL DONE $(date)"
