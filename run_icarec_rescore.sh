#!/bin/bash -l
#SBATCH -N1 --time=30:00 -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# Re-score every model of record under the EXACT RECORDED rig stimulus
# (Roy<amp>_icaRec_5k, holding ~0 nA; finding 11) instead of the icav2
# reconstruction (constant -0.05 nA low).  Outputs land in
# <model>/exp_royv2_icarec/.  Missing models (still training) are skipped
# with a message -- re-run for stragglers.
# Submit: sbatch -J icarec-rescore run_icarec_rescore.sh
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
J=$SCRATCH/tmp_neuInv/jaxley_ca3
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_1stim/
CA3FT=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_icarec_$SLURM_JOBID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

score () {  # $1 model-out dir, $2 pack file, $3 tag
  if [ ! -f "$1/checkpoints/ckpt.pth" ] && [ ! -f "$1/blank_model.pth" ]; then
    echo "SKIP $3: no model at $1"; return
  fi
  echo "=== icaRec rescore: $3  $(date)"
  (cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$1" \
       --packFile "$2" --stimSuffix icaRec_5k \
       --outDir "$1/exp_royv2_icarec") || echo "icaRec $3 FAILED"
}

# pooled models on the full ca3ft pack (all families)
score "$FT/out"                                                        "$CA3FT" ft_champion
score "$J/ca3_royexp_scratch_k128/RoyExpChaotic/scratch_k128/out"      "$CA3FT" scratch_k128
score "$J/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k/out" "$CA3FT" dtw_k128_80k
# DTW specialists on their own family packs
for A in 500 1000 1500 2000; do
  score "$J/ca3_royexp_1stim_${A}_dtw/RoyExp$A/onestim_dtw_$A/out" \
        "$DATA/RoyExp$A.mlPack1.h5" 1dtw_$A
done
# MSE+eFEL specialists (previous round) for symmetry
for A in 500 1000 1500 2000; do
  score "$J/ca3_royexp_1stim_$A/RoyExp$A/onestim_$A/out" \
        "$DATA/RoyExp$A.mlPack1.h5" 1stim_$A
done
echo "ICAREC-RESCORE: ALL DONE $(date)"
