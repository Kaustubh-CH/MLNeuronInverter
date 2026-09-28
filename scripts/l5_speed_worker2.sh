#!/bin/bash
# Big-batch follow-up: the first pass showed fwd+bwd wall time nearly independent of B
# (per-step latency bound), so throughput should scale ~linearly with batch until memory.
set -u
R=${SLURM_PROCID:-0}; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_speed2_$R; mkdir -p $JAX_COMPILATION_CACHE_DIR
OUT=docs/model_ladder/speed; mkdir -p $OUT
log() { echo "[rank $R] $(date +%T) $*"; }
case $R in
  0) log "nc4 big batch";  L5TTPC_NCOMP=4 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 512,1024 --dts 0.1 --iters 2 --out $OUT/timing_big.csv > $OUT/timing_big_nc4.log 2>&1 ;;
  1) log "nc2 big batch";  L5TTPC_NCOMP=2 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 512,1024 --dts 0.1 --iters 2 --out $OUT/timing_big.csv > $OUT/timing_big_nc2.log 2>&1 ;;
  2) log "nc1 big batch";  L5TTPC_NCOMP=1 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 512,1024,2048 --dts 0.1 --iters 2 --out $OUT/timing_big.csv > $OUT/timing_big_nc1.log 2>&1 ;;
  3) log "nc2 fp32 big + dt0.2 big"; L5TTPC_NCOMP=2 JAX_ENABLE_X64=false python scripts/bench_l5_speed.py --batches 512,1024 --dts 0.1 --iters 2 --out $OUT/timing_big.csv > $OUT/timing_big_fp32.log 2>&1
     L5TTPC_NCOMP=2 JAX_ENABLE_X64=true python scripts/bench_l5_speed.py --batches 512 --dts 0.2 --iters 2 --out $OUT/timing_big.csv >> $OUT/timing_big_fp32.log 2>&1 ;;
esac
log "done"
