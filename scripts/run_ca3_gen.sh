#!/bin/bash -l
# Sharded CA3 data generation driver. Run INSIDE a 4-node GPU allocation.
#   Args: CELLNAME OUTDIR STIMS N
# e.g.  bash scripts/run_ca3_gen.sh ca3_pyramidal_synth <out> 5k50kInterChaoticB 50000
set -e
CELL=$1; OUT=$2; STIMS=$3; N=$4; TMAX=$5
: "${CELL:?}"; : "${OUT:?}"; : "${STIMS:?}"; : "${N:?}"
# Optional 5th arg: t_max (ms) override — pass 400 for the 4000-sample interpolated
# stims (BBP_Exp_Step600_i4k etc.), else the cell default (500 ms) is used.
TMAX_ARG=""; [[ -n "$TMAX" ]] && TMAX_ARG="--t-max $TMAX"

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SHARD=$SCRATCH/ca3_gen_shards/${CELL}
rm -rf "$SHARD"; mkdir -p "$SHARD" "$OUT"
G=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-1} ))
echo "=== worker phase: G=$G ranks, stims=$STIMS, N=$N -> $OUT/$CELL.mlPack1.h5 ==="
date
srun -n "$G" python scripts/gen_ca3_sharded.py \
     --cell-name "$CELL" --out "$OUT" --shard-dir "$SHARD" \
     --stims "$STIMS" --n "$N" --batch 256 --seed 0 $TMAX_ARG
echo "=== merge phase ==="
srun -n 1 python scripts/gen_ca3_sharded.py --merge --world "$G" \
     --cell-name "$CELL" --out "$OUT" --shard-dir "$SHARD" \
     --stims "$STIMS" --n "$N" --batch 256 --seed 0 $TMAX_ARG
date
echo "=== done: $OUT/$CELL.mlPack1.h5 ==="
