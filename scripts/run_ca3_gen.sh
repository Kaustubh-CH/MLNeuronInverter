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
# Optional env: SOURCE_CELL (jaxley_cells module name, default ca3_pyramidal),
# GEN_BATCH (jaxley batch, default 256; use 32-64 for the L5 cells),
# L5TTPC_NCOMP is passed through the environment to the workers.
SOURCE_CELL=${SOURCE_CELL:-ca3_pyramidal}; GEN_BATCH=${GEN_BATCH:-256}
# GEN_DT: coarser solver step (ms) recorded in the pack (pair with voltage_loss.sim_dt_override).
DT_ARG=""; [[ -n "${GEN_DT:-}" ]] && DT_ARG="--dt $GEN_DT"
# GEN_VARY: comma-separated PARAM_KEYS to vary; all others pinned at the cell default.
VARY_ARG=""; [[ -n "${GEN_VARY:-}" ]] && VARY_ARG="--vary $GEN_VARY"

module load python
source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
# GEN_X64 (default true): generate in fp64.  Set false to produce a pack whose traces come
# from the SAME fp32 solver an `fp64: False` training loss uses (no precision floor);
# recorded truthfully in meta.simu_info.fp64.
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=${GEN_X64:-true}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# SHARD_ROOT: where the per-rank raw shards go (default $SCRATCH/ca3_gen_shards;
# the model ladder uses HOME while pscratch is over quota).  Removed after the merge.
SHARD=${SHARD_ROOT:-$SCRATCH/ca3_gen_shards}/${CELL}
rm -rf "$SHARD"; mkdir -p "$SHARD" "$OUT"
G=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-1} ))
echo "=== worker phase: G=$G ranks, source=$SOURCE_CELL ncomp=${L5TTPC_NCOMP:-n/a} stims=$STIMS, N=$N batch=$GEN_BATCH -> $OUT/$CELL.mlPack1.h5 ==="
date
srun -n "$G" python scripts/gen_ca3_sharded.py --source-cell "$SOURCE_CELL" \
     --cell-name "$CELL" --out "$OUT" --shard-dir "$SHARD" \
     --stims "$STIMS" --n "$N" --batch "$GEN_BATCH" --seed 0 $TMAX_ARG $DT_ARG $VARY_ARG
echo "=== merge phase ==="
srun -n 1 python scripts/gen_ca3_sharded.py --merge --world "$G" --source-cell "$SOURCE_CELL" \
     --cell-name "$CELL" --out "$OUT" --shard-dir "$SHARD" \
     --stims "$STIMS" --n "$N" --batch "$GEN_BATCH" --seed 0 $TMAX_ARG $DT_ARG $VARY_ARG
date
[ -f "$OUT/$CELL.mlPack1.h5" ] && rm -rf "$SHARD" && echo "=== shards removed: $SHARD ==="
echo "=== done: $OUT/$CELL.mlPack1.h5 ==="
