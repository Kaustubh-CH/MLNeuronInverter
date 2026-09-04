#!/bin/bash -l
# Score the two 80k models whose salloc timed out before the auto-eval chain ran.
# Run on 1 GPU via srun on an existing interactive allocation.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/eval80k
mkdir -p $JAX_COMPILATION_CACHE_DIR

WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
J=$SCRATCH/tmp_neuInv/jaxley_ca3
D=$J/ca3_vo_chaoticramp_dtw/ca3_pyramidal_synth/dtw_80k
P=$J/ca3_vo_dtw_precond2_chaoticramp/ca3_pyramidal_synth/precond2_80k

for RUN in "$D" "$P"; do
  echo "==================== EVAL $RUN $(date) ===================="
  if [ ! -f "$RUN/out/sum_train.yaml" ]; then echo "  MISSING sum_train.yaml, skip"; continue; fi
  if [ -f "$RUN/out/eval/summary.yaml" ]; then echo "  already scored, skip"; continue; fi
  cd "$RUN" || { echo "  cd fail"; continue; }
  # copy the eval scripts into the frozen run dir if absent
  for f in predict.py evaluate_voltage.py; do
    [ -f "$RUN/$f" ] || cp "$WT/$f" "$RUN/$f"
  done
  python -u predict.py --modelPath ./out --dom test -X >& log.predict.rescore || echo "  predict FAILED"
  python -u evaluate_voltage.py --modelPath ./out --numSamples 200 --numOverlay 20 >& log.evalvolt.rescore || echo "  evalvolt FAILED"
  echo "  --- summary ---"
  grep -E "overall MSE|mean R|voltage MSE_z|spike count" log.evalvolt.rescore 2>/dev/null | tail -6
done

echo "==================== LEDGER $(date) ===================="
cd "$WT" || exit 2
python scripts/collect_vo_ledger.py "$D" -o "$J/vo_ledger/results.csv"
python scripts/collect_vo_ledger.py "$P" -o "$J/vo_ledger/results.csv"
echo "eval_80k: DONE $(date)"
