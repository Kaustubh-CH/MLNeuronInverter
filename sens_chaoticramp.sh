#!/bin/bash -l
# Per-channel voltage sensitivity ||dV_z/dtheta|| on 5kChaoticRamp, for the
# grad-preconditioner. Emits grad_precond_sensitivity.voltage_mse (length-6, CA3
# PARAM_KEYS order) -> the weights that un-starve low-sensitivity channels.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/sens/rank_${SLURM_PROCID:-0}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

OUT=$SCRATCH/tmp_neuInv/ca3_ablation/sens_chaoticramp
PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/ca3_pyramidal_synth.mlPack1.h5
echo "SENS start $(date) host=$(hostname)"
srun -N1 --ntasks=1 --gpus-per-task=1 python -u feature_channel_sensitivity.py \
  --cell ca3_pyramidal --stims 5kChaoticRamp --h5 "$PACK" \
  --numOp 32 --features STRONG -o "$OUT"
echo "SENS srun exit=$?"
echo "=== grad_precond_sensitivity from summary.yaml ==="
grep -A10 "grad_precond_sensitivity" "$OUT/feature_sensitivity_summary.yaml" 2>/dev/null || echo "NO summary.yaml written"
echo "SENS end $(date)"
