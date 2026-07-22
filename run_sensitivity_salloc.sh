#!/bin/bash -l
# Driver run INSIDE an interactive salloc allocation.  Runs the OAT
# voltage+eFEL sensitivity sweep on BOTH the native and interpolated stim
# dirs (4 GPUs, one stim-chunk per GPU), merges each, then compares them.
#
# Launch with:
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g \
#          --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
#          --cpus-per-task=32 bash run_sensitivity_salloc.sh
set -e

cell=${CELL:-ca3_pyramidal}
nSamples=${NSAMPLES:-500}
NATIVE=/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims
INTERP=$NATIVE/stim_interpolated
REPO=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
base=$SCRATCH/tmp_neuInv/sensitivity_variation/${cell}/salloc_${SLURM_JOB_ID}

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
cd $REPO

NTASK=${SLURM_NTASKS:-4}
echo "S: base=$base  ntasks=$NTASK  cell=$cell  nSamples=$nSamples"

run_one() {
  local tag=$1 stimDir=$2
  local partsDir=$base/$tag/parts combined=$base/$tag/combined
  mkdir -p $partsDir $combined
  local NSTIM=$(ls $stimDir/*.csv 2>/dev/null | wc -l)
  local CHUNK=$(( (NSTIM + NTASK - 1) / NTASK ))
  echo "S: [$tag] stims=$NSTIM chunk=$CHUNK  dir=$stimDir"
  date
  srun -n $NTASK bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID JAX_PLATFORMS=cuda
    export JAX_COMPILATION_CACHE_DIR='"$SCRATCH"'/jax_cc/sensvar_gpu_$SLURM_LOCALID
    mkdir -p $JAX_COMPILATION_CACHE_DIR
    START=$(( SLURM_LOCALID * '"$CHUNK"' ))
    python -u sensitivity_variation.py --cell '"$cell"' --stimDir '"$stimDir"' \
       --nSamples '"$nSamples"' --stimStart $START --stimCount '"$CHUNK"' --efel \
       --outDir '"$partsDir"'/part_$SLURM_LOCALID > '"$partsDir"'/log_part_$SLURM_LOCALID.txt 2>&1'
  echo "S: [$tag] sweeps done; merging"; date
  JAX_PLATFORMS=cpu python -u sensitivity_variation_merge.py \
     --partsDir $partsDir --outDir $combined \
     --cell $cell --stimDir $stimDir --nSamples $nSamples >& $base/$tag/log_merge.txt
  echo "S: [$tag] combined -> $combined"
}

run_one interp4000 $INTERP
run_one native      $NATIVE

echo "S: comparing native vs interp4000"
JAX_PLATFORMS=cpu python -u sensitivity_variation_compare.py \
   --a native=$base/native/combined \
   --b interp4000=$base/interp4000/combined \
   --outDir $base/compare | tee $base/compare_stdout.txt

echo "S: ALL DONE"
echo "S: base=$base"
ls -la $base/compare $base/native/combined $base/interp4000/combined
