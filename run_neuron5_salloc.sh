#!/bin/bash
# Body of `salloc -N1 --ntasks-per-node=4 --gpus-per-node=4 ... bash run_neuron5_salloc.sh`.
# Trains the NEURON-level joint 5-sweep battery model (option 2: one theta per
# neuron from all its sweeps; ca3_royexp_neuron5_joint) then scores held-out
# neurons with plot_exp_overlay_neuron5.py.
set -u
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1
WT=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
cd "$WT" || exit 2
DATA=/pscratch/sd/k/ktub1999/RoyExpPack_neuron5/
J=$SCRATCH/tmp_neuInv/jaxley_ca3

export NEUINV_RDZV_FILE=$SCRATCH/tmp_neuInv/rdzv/rdzv_${SLURM_JOB_ID}
rm -f "$NEUINV_RDZV_FILE"
export NCCL_P2P_DISABLE=1
export NEUINV_PROBS="0 1 2 3 4" NEUINV_STIMS="0" NEUINV_VALIDSTIMS="0"

DESIGN=ca3_royexp_neuron5_joint
SUF=neuron5_joint
RUN=$J/$DESIGN/RoyExpNeuron5/$SUF
if [ -f "$RUN/out/checkpoints/ckpt.pth" ] && [ -f "$RUN/out/sum_train.yaml" ]; then
  echo "NEURON5: completed model already present -- SKIP training, eval only"
else
  echo "===== NEURON5: train $(date) ====="
  NEUINV_CELL=RoyExpNeuron5 NEUINV_EPOCHS=60 NEUINV_WRK_SUFIX=$SUF \
    bash batchShifterJaxleyCA3_100ep.slr "$DATA" $DESIGN
  echo "NEURON5: launcher exit=$?"
fi
if [ ! -f "$RUN/out/checkpoints/ckpt.pth" ]; then
  echo "NEURON5: NO checkpoint"; tail -30 "$RUN/log.train" 2>/dev/null; exit 5
fi
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_n5_$SLURM_JOB_ID
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "=== NEURON5 held-out eval (test neurons)"
python -u plot_exp_overlay_neuron5.py -m "$RUN/out" \
       --outDir "$RUN/out/exp_neuron5" || echo "NEURON5: eval FAILED"
echo "NEURON5: ALL DONE $(date)"
