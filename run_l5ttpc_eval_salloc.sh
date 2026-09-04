#!/bin/bash
# Test-split evaluation (predicted-vs-true voltage overlays + channel recovery)
# for every L5TTPC ncomp=2 run that was trained 2026-06-27..07-08 but never scored.
# Outputs go to HOME (pscratch is at quota).  Uses the no-grad jaxley path so the
# 19-parameter L5 cell does not OOM (the July attempt OOM'd in the VJP).
#
# Usage from a login node:
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g \
#          --ntasks-per-node=1 --gpus-per-node=1 bash run_l5ttpc_eval_salloc.sh
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export L5TTPC_NCOMP=2                       # the runs were trained/generated at ncomp=2
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache_l5eval_${SLURM_JOB_ID:-nojob}
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
OUT=$WT/l5ttpc_eval
mkdir -p "$OUT"
V=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_voltage_only
S=/pscratch/sd/k/ktub1999/tmp_neuInv/jaxley_ca3/l5ttpc_supervised/L5TTPC_jaxley_nc2/super

RUNS=(
  "supervised_nc2|$S/out"
  "voltage_only_nc2|$V/l5ttpc_jaxley_ncomp2/L5TTPC_jaxley_nc2/55135593/out"
  "hybrid_nc2|$V/l5ttpc_jaxley_hybrid/L5TTPC_jaxley_nc2/salloc_55144257/out"
  "multiprobe_nc2|$V/l5ttpc_multiprobe/L5TTPC_multiprobe/salloc_55163753/out"
  "multistim_nc2|$V/l5ttpc_multistim/L5TTPC_multistim/salloc_55166043/out"
  "multistim_efel|$V/l5ttpc_multistim_efel/L5TTPC_multistim/salloc_55370141/out"
  "hybrid_efel|$V/l5ttpc_jaxley_hybrid_efel/L5TTPC_multistim/reg15/out"
  "paramonly_3stim|$V/l5ttpc_multistim_paramonly/L5TTPC_multistim/debug_55374316/out"
  "paramonly_ft80k|$V/l5ttpc_multistim_paramonly/L5TTPC_multistim/finetune_55375352/out"
)

echo "L5TTPC eval driver start $(date)  job=${SLURM_JOB_ID:-?}  node=$(srun -n1 hostname 2>/dev/null | head -1)"
for r in "${RUNS[@]}"; do
  name=${r%%|*}; mp=${r#*|}
  echo "===== $name  ($mp)  $(date)"
  if [ -f "$OUT/$name/summary.yaml" ]; then echo "  already scored -- skip"; continue; fi
  srun -n1 --gpus-per-task=1 python -u evaluate_voltage.py -m "$mp" --outDir "$OUT/$name" \
       -n 200 --numOverlay 6 --noGrad --simBatch 64 --savePng > "$OUT/$name.log" 2>&1 \
    || echo "  $name: FAILED (see $OUT/$name.log)"
  grep -E 'mean R²|MSE_z mean|spike count|DONE|Error|error' "$OUT/$name.log" | tail -8
done
echo "L5TTPC eval driver ALL DONE $(date)"
