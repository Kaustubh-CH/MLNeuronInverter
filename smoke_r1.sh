#!/bin/bash -l
# 1-epoch, 1-GPU smoke test for the R1 (DTW+blur) voltage-only recipe.
# Validates: HybridLoss imports (trace_metrics blur path), CA3 cell builds, fp64
# jaxley solve runs, the new coarse-sigma blur forward+backward works, per-step
# time is sane, and sum_train.yaml is written. Runs INSIDE an salloc.
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export NCCL_P2P_DISABLE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/r1_smoke/rank_${SLURM_PROCID:-0}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# DDP rendezvous vars: train_dist.py reads MASTER_ADDR/MASTER_PORT even for
# world_size=1 (a logging line). Derive from the allocation's first node.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
export MASTER_PORT=8881

OUT=$SCRATCH/tmp_neuInv/ca3_ablation/smoke_r1_dtwblur/out
rm -rf "$OUT"; mkdir -p "$OUT"
PACK=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v1/

echo "S: smoke start $(date)  host=$(hostname)  nodelist=${SLURM_JOB_NODELIST}"
time srun -N1 --ntasks=1 --gpus-per-task=1 python -u train_dist.py \
  --design ca3_vo_dtwblur_chaoticramp --cellName ca3_pyramidal_synth --facility perlmutter \
  --outPath "$OUT" --jobId smoke --probsSelect 0 --stimsSelect 0 --validStimsSelect 0 \
  --epochs 1 --numGlobSamp 2048 --data_path_temp "$PACK"
rc=$?
echo "S: === SMOKE srun exit code = $rc ==="
if [[ -f "$OUT/sum_train.yaml" ]]; then
  echo "S: SMOKE_OK  sum_train.yaml written"
else
  echo "S: SMOKE_FAIL  no sum_train.yaml (import/config/solve error above)"
fi
echo "S: smoke end $(date)"
