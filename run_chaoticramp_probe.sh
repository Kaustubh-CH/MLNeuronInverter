#!/bin/bash -l
# Run the ChaoticRamp variance probe inside a 1-node interactive salloc.
#   salloc -N1 -C gpu -q interactive -t 0:30:00 -A m2043_g \
#          --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none --cpus-per-task=32 \
#          bash run_chaoticramp_probe.sh
set -e
module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
srun -n1 bash -c '
  export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/probe_gpu0; mkdir -p $JAX_COMPILATION_CACHE_DIR
  python -u chaoticramp_variance_probe.py --N 64 --stim 5kChaoticRamp \
     --outDir $SCRATCH/tmp_neuInv/chaoticramp_probe'
echo "PROBE_DONE rc=$?"
