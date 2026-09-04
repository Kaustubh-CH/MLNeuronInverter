#!/bin/bash
# Body of `salloc -N4 ... bash run_k128_salloc.sh` (interactive, 4h).
# Trains ca3_vo_chaoticramp_dtw_k128 (dtw_80k champion recipe, conv kernel
# [4,4,4] -> [128,128,128]) on the same 80k chaoticRamp pack, then runs the
# Roy/Paula experimental overlays. Judge by exp transfer, not synthetic R².
# Budget: dtw_80k measured 98.83 s/ep on 4n/16GPU -> ~2.9h train + eval + exp.
set -u
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_chaoticramp_v2/
J=$SCRATCH/tmp_neuInv/jaxley_ca3

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
mkdir -p "$(dirname "$NEUINV_RDZV_FILE")"; rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1
echo "K128: start $(date) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"

NEUINV_CELL=ca3_pyramidal_synth \
NEUINV_NUMGLOBSAMP=80000 NEUINV_EPOCHS=100 NEUINV_WRK_SUFIX=dtw_k128_80k \
  bash batchShifterJaxleyCA3_100ep.slr "$DATA" ca3_vo_chaoticramp_dtw_k128
echo "K128: launcher exit=$? $(date)"

RUN=$J/ca3_vo_chaoticramp_dtw_k128/ca3_pyramidal_synth/dtw_k128_80k
if [ ! -f "$RUN/out/eval/summary.yaml" ]; then
  echo "K128: NO eval summary at $RUN/out/eval — dumping log tails"
  tail -30 "$RUN/log.train" 2>/dev/null
  exit 5
fi
python scripts/collect_vo_ledger.py "$RUN" -o "$J/vo_ledger/results.csv" \
  || echo "K128: ledger append FAILED"

echo "K128: === Roy experimental overlays $(date) ==="
cd "$WT"
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_k128_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "=== k128 @ rig amplitude"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy" \
       --simScale rig || echo "K128: exp rig FAILED"
echo "=== k128 @ training amplitude (stim-mismatched, firing bracket only)"
python -u plot_exp_overlay_roy.py -m "$RUN/out" --outDir "$RUN/out/exp_roy_train" \
       --simScale train || echo "K128: exp train FAILED"
echo "K128: ALL DONE $(date)"
