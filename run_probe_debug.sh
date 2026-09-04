#!/bin/bash -l
#SBATCH -N1 --time=30:00 -q debug
#SBATCH -o /pscratch/sd/k/ktub1999/tmp_neuInv/slurm_logs/slurm-%j.out
#SBATCH --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 -C gpu -A m2043_g
# (a) Excitability probe: can ANY CA3 params (in/out of box) fire at data rates
#     under the TRUE recorded stimulus?  (b) dtw_k128 icaRec+icav2 rescore, now
#     that the royv2 scorer pads 4001->5001-bin inputs.
# Submit: sbatch -J probe run_probe_debug.sh
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
FT=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/royexp_ft_57631428
DTW=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k/out
CA3FT=/pscratch/sd/k/ktub1999/RoyExpPack_ca3ft/RoyExpChaotic.mlPack1.h5
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_probe_$SLURM_JOBID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "=== excitability probe $(date)"
python -u excitability_probe_icarec.py \
       --outDir /pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/excitability_probe_icarec \
       || echo "PROBE FAILED"
echo "=== dtw_k128 @ icaRec (padded)"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$DTW" \
     --packFile "$CA3FT" --stimSuffix icaRec_5k \
     --outDir "$DTW/exp_royv2_icarec") || echo "dtw_k128 icaRec FAILED"
echo "=== dtw_k128 @ icav2 (padded)"
(cd "$FT" && python -u plot_exp_overlay_royv2.py -m "$DTW" \
     --packFile "$CA3FT" \
     --outDir "$DTW/exp_royv2") || echo "dtw_k128 icav2 FAILED"
echo "PROBE JOB: ALL DONE $(date)"
