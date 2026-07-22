#!/bin/bash -l
# 4-NODE (16-GPU) interactive sensitivity sweep with the FULL metric set:
#   raw-voltage std + eFEL(11 = 6 spike + 5 sub-threshold) + blurred-MSE + soft-DTW.
# Each of the 16 GPUs takes a global stim slice (chunked by SLURM_PROCID, NOT
# localid — localid repeats 0-3 per node), writes part_<procid>/, then one merge
# combines them into interp4000/combined/.
#
# Launch (interactive, 4 nodes):
#   salloc -N 4 -C gpu -q interactive -t 1:30:00 -A m2043_g \
#          --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=32 \
#          bash run_sensitivity_metrics_salloc.sh
set -e

cell=${CELL:-ca3_pyramidal}
nSamples=${NSAMPLES:-500}
stimDir=${STIMDIR:-/global/homes/k/ktub1999/mainDL4/DL4neurons2/stims/stim_interpolated}
REPO=/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter
base=$SCRATCH/tmp_neuInv/sensitivity_variation/${cell}/metrics4node_${SLURM_JOB_ID}/interp4000
partsDir=$base/parts; combined=$base/combined
mkdir -p $partsDir $combined

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export NCCL_P2P_DISABLE=1
cd $REPO

NSTIM=$(ls $stimDir/*.csv 2>/dev/null | wc -l)
NTASK=${SLURM_NTASKS:-16}
CHUNK=$(( (NSTIM + NTASK - 1) / NTASK ))
echo "S: base=$base  nstim=$NSTIM ntask=$NTASK chunk=$CHUNK nSamples=$nSamples cell=$cell"

# ── preflight: 1 GPU, 1 stim, tiny N, all metrics — abort if the path is broken.
echo "S: preflight (1 stim, N=8, +blur +dtw) ..."; date
srun -N1 -n1 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR='"$SCRATCH"'/jax_cc/sensvar_metrics_preflight
  mkdir -p $JAX_COMPILATION_CACHE_DIR
  python -u sensitivity_variation.py --cell '"$cell"' --stimDir '"$stimDir"' \
     --nSamples 16 --metricSamples 12 --stimStart 0 --stimCount 1 \
     --efel --blur --dtw --sysvar --smoothvar --saveTraces \
     --outDir '"$base"'/preflight' 2>&1 | tail -25
if [ ! -f $base/preflight/sysvar_variation_aggregate.csv ] || [ ! -f $base/preflight/saved_traces.npz ]; then
  echo "S: PREFLIGHT FAILED — sysvar/saved_traces output missing; aborting before the full run."; exit 1
fi
echo "S: preflight OK"; date

# ── full sweep: 16 tasks, global-PROCID chunking ─────────────────────────────
echo "S: full sweep ..."; date
time srun -n $NTASK bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID JAX_PLATFORMS=cuda
  export JAX_COMPILATION_CACHE_DIR='"$SCRATCH"'/jax_cc/sensvar_metrics_gpu_$SLURM_PROCID
  mkdir -p $JAX_COMPILATION_CACHE_DIR
  START=$(( SLURM_PROCID * '"$CHUNK"' ))
  if [ $START -ge '"$NSTIM"' ]; then echo "rank $SLURM_PROCID: empty slice, skip"; exit 0; fi
  python -u sensitivity_variation.py --cell '"$cell"' --stimDir '"$stimDir"' \
     --nSamples '"$nSamples"' --stimStart $START --stimCount '"$CHUNK"' \
     --efel --blur --dtw --sysvar --smoothvar --saveTraces \
     --outDir '"$partsDir"'/part_$SLURM_PROCID \
     > '"$partsDir"'/log_part_$SLURM_PROCID.txt 2>&1'
echo "S: sweeps done; merging"; date

JAX_PLATFORMS=cpu python -u sensitivity_variation_merge.py \
   --partsDir $partsDir --outDir $combined \
   --cell $cell --stimDir $stimDir --nSamples $nSamples >& $base/log_merge.txt || echo "S: merge FAILED (see log_merge.txt)"

echo "S: combined outputs:"; ls -la $combined
echo "S: DONE base=$base"
