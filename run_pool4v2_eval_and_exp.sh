#!/bin/bash
# Runs on the compute node of an interactive salloc (1 GPU).
# Part A: per-stim voltage eval of pool4v2 (reconstruct traces for ALL 4 stims,
#         not just 5kChaoticRamp) via evaluate_voltage.py --stimIndex 0..3.
# Part B: Roy/Paula experimental predictions — the 3 staged champion models
#         (run_roy_exp.sh) plus the pool4v2 pooled model.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_evalexp_${SLURM_JOB_ID}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
RUN=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/ca3_pool4_dtw_amp_v2_4n/ca3_pyramidal_pool4/pool4v2_80k

echo "===== PART A: pool4v2 per-stim eval $(date) ====="
cd "$RUN" || exit 2
for k in 0 1 2 3; do
  echo "--- stimIndex $k $(date) ---"
  python -u evaluate_voltage.py --modelPath ./out --numSamples 200 \
         --numOverlay 12 --stimIndex "$k" >& "log.evalstim$k" \
    || echo "PART A: stimIndex $k FAILED"
  grep -E "overall MSE|MSE_z: mean|spike count" "log.evalstim$k" | head -5
done
echo "===== PART A done $(date) ====="

echo "===== PART B: Roy experimental predictions $(date) ====="
cd "$WT" || exit 2
bash run_roy_exp.sh || echo "PART B: run_roy_exp.sh FAILED"
echo "=== pool4v2 (pooled model) @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
       --simScale rig || echo "PART B: pool4v2 exp overlay FAILED"
echo "===== ALL DONE $(date) ====="
