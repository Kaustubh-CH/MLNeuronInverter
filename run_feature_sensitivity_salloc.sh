#!/bin/bash -l
# Feature x Channel sensitivity, run INSIDE an interactive salloc allocation.
# Launch with:
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g \
#          --gpus-per-node=1 --cpus-per-task=32 bash run_feature_sensitivity_salloc.sh
set -e
REPO=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
CELL=${CELL:-ca3_pyramidal}
STIMS=${STIMS:-4k50kInterChaoticB}          # space-separated stim stems
H5=${H5:-/pscratch/sd/k/ktub1999/synthetic_ca3_data/ca3_4kinterchaoticB_v1/ca3_pyramidal.mlPack1.h5}
PHYS=${PHYS:-$REPO/ca3_efel_precond_interchaoticB.hpar.yaml}
NUMOP=${NUMOP:-32}
FEATS=${FEATS:-ALL}                         # STRONG | ALL | comma-list
OUT=${OUT:-$SCRATCH/tmp_neuInv/feature_sensitivity/${CELL}_$SLURM_JOB_ID}

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/featsens_$SLURM_LOCALID
mkdir -p $JAX_COMPILATION_CACHE_DIR
cd $REPO

python -u feature_channel_sensitivity.py \
   --cell $CELL --stims $STIMS \
   --physRange $PHYS --h5 $H5 \
   --numOp $NUMOP --features $FEATS -o $OUT
echo "S: DONE -> $OUT"
ls -la $OUT
