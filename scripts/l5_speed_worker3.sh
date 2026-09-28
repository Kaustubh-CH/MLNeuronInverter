#!/bin/bash
# Pass 3 on an 80 GB A100 node: the batch sizes that OOM'd on 40 GB, plus dt 0.25 and 400 ms.
set -u
R=${SLURM_PROCID:-0}; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_speed3_$R; mkdir -p $JAX_COMPILATION_CACHE_DIR
OUT=docs/model_ladder/speed; mkdir -p $OUT
log() { echo "[rank $R] $(date +%T) $*"; }
case $R in
  0) log "nc4 80g"; L5TTPC_NCOMP=4 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 128,256 --dts 0.1 --iters 2 --out $OUT/timing_80g.csv > $OUT/timing_80g_nc4.log 2>&1
                    L5TTPC_NCOMP=4 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 128 --dts 0.2 --iters 2 --out $OUT/timing_80g.csv >> $OUT/timing_80g_nc4.log 2>&1 ;;
  1) log "nc2 80g"; L5TTPC_NCOMP=2 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 256,512 --dts 0.1 --iters 2 --out $OUT/timing_80g.csv > $OUT/timing_80g_nc2.log 2>&1
                    L5TTPC_NCOMP=2 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 256 --dts 0.2,0.25 --iters 2 --out $OUT/timing_80g.csv >> $OUT/timing_80g_nc2.log 2>&1 ;;
  2) log "nc1 80g"; L5TTPC_NCOMP=1 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 512,1024 --dts 0.1 --iters 2 --out $OUT/timing_80g.csv > $OUT/timing_80g_nc1.log 2>&1
                    L5TTPC_NCOMP=1 JAX_ENABLE_X64=true  python scripts/bench_l5_speed.py --batches 512 --dts 0.2,0.25 --iters 2 --out $OUT/timing_80g.csv >> $OUT/timing_80g_nc1.log 2>&1 ;;
  3) log "fp32 80g"; L5TTPC_NCOMP=2 JAX_ENABLE_X64=false python scripts/bench_l5_speed.py --batches 256,512 --dts 0.1 --iters 2 --out $OUT/timing_80g.csv > $OUT/timing_80g_fp32.log 2>&1
                     L5TTPC_NCOMP=4 JAX_ENABLE_X64=false python scripts/bench_l5_speed.py --batches 128,256 --dts 0.1 --iters 2 --out $OUT/timing_80g.csv >> $OUT/timing_80g_fp32.log 2>&1
                     L5TTPC_NCOMP=2 JAX_ENABLE_X64=false python scripts/bench_l5_speed.py --batches 256 --dts 0.2 --tmax 400 --iters 2 --out $OUT/timing_80g.csv >> $OUT/timing_80g_fp32.log 2>&1 ;;
esac
log "done"
