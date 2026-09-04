#!/bin/bash -l
# Score the 200k epoch-95 (converged) model whose training timed out at 95/100 before
# the auto-eval + sum_train.yaml write. sum_train.yaml was reconstructed (v3 test pack).
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/eval200k
mkdir -p $JAX_COMPILATION_CACHE_DIR

WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
J=$SCRATCH/tmp_neuInv/jaxley_ca3
RUN=$J/ca3_vo_chaoticramp_dtw_8n/ca3_pyramidal_synth/dtw_200k
echo "==================== EVAL 200k $RUN $(date) ===================="
if [ ! -f "$RUN/out/sum_train.yaml" ]; then echo "  MISSING sum_train.yaml"; exit 2; fi
cd "$RUN" || exit 2
for f in predict.py evaluate_voltage.py; do [ -f "$RUN/$f" ] || cp "$WT/$f" "$RUN/$f"; done
[ -d "$RUN/toolbox" ] || cp -rp "$WT/toolbox" "$RUN/toolbox"
python -u predict.py --modelPath ./out --dom test -X >& log.predict.rescore || echo "  predict FAILED"
python -u evaluate_voltage.py --modelPath ./out --numSamples 200 --numOverlay 20 >& log.evalvolt.rescore || echo "  evalvolt FAILED"
echo "  --- summary ---"
grep -E "overall MSE|mean R|voltage MSE_z|spike count" log.evalvolt.rescore 2>/dev/null | tail -6
echo "==================== LEDGER $(date) ===================="
cd "$WT" || exit 2
python scripts/collect_vo_ledger.py "$RUN" -o "$J/vo_ledger/results.csv"
echo "eval_200k: DONE $(date)"
