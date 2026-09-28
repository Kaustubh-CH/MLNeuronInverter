#!/bin/bash
# One rank of the L5 speed study (4 GPUs): timing matrix + accuracy traces + nc1 physiology.
set -u
R=${SLURM_PROCID:-0}; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_speed_$R; mkdir -p $JAX_COMPILATION_CACHE_DIR
OUT=docs/model_ladder/speed; mkdir -p $OUT
P=python
log() { echo "[rank $R] $(date +%T) $*"; }
case $R in
  0) log "nc4 timing"; L5TTPC_NCOMP=4 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 64,128,256 --dts 0.1 --out $OUT/timing_nc4.csv > $OUT/timing_nc4.log 2>&1
     L5TTPC_NCOMP=4 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 128 --dts 0.2,0.25 --out $OUT/timing_nc4.csv >> $OUT/timing_nc4.log 2>&1
     L5TTPC_NCOMP=4 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 128 --dts 0.1 --tmax 400 --out $OUT/timing_nc4.csv >> $OUT/timing_nc4.log 2>&1
     log "nc4 accuracy"; L5TTPC_NCOMP=4 $P scripts/l5_accuracy_dt_ncomp.py --simulate --out $OUT/acc_nc4.npz > $OUT/acc_nc4.log 2>&1 ;;
  1) log "nc2 timing"; L5TTPC_NCOMP=2 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 64,128,256 --dts 0.1 --out $OUT/timing_nc2.csv > $OUT/timing_nc2.log 2>&1
     L5TTPC_NCOMP=2 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 128 --dts 0.2,0.25 --out $OUT/timing_nc2.csv >> $OUT/timing_nc2.log 2>&1
     L5TTPC_NCOMP=2 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 128 --dts 0.1 --tmax 400 --out $OUT/timing_nc2.csv >> $OUT/timing_nc2.log 2>&1
     log "nc2 accuracy"; L5TTPC_NCOMP=2 $P scripts/l5_accuracy_dt_ncomp.py --simulate --out $OUT/acc_nc2.npz > $OUT/acc_nc2.log 2>&1 ;;
  2) log "nc1 timing"; L5TTPC_NCOMP=1 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 64,128,256 --dts 0.1 --out $OUT/timing_nc1.csv > $OUT/timing_nc1.log 2>&1
     L5TTPC_NCOMP=1 JAX_ENABLE_X64=true $P scripts/bench_l5_speed.py --batches 128 --dts 0.2,0.25 --out $OUT/timing_nc1.csv >> $OUT/timing_nc1.log 2>&1
     log "nc1 accuracy"; L5TTPC_NCOMP=1 $P scripts/l5_accuracy_dt_ncomp.py --simulate --out $OUT/acc_nc1.npz > $OUT/acc_nc1.log 2>&1
     log "nc1 physiology sweep"; L5TTPC_NCOMP=1 JAX_ENABLE_X64=true $P scripts/stim_scale_scan.py --cell l5ttpc --scales 1,1.5 --nbox 128 --batch 64 --out docs/model_ladder/scan_l5ttpc_nc1 > docs/model_ladder/scan_l5ttpc_nc1.log 2>&1 ;;
  3) log "fp32 timing"; for nc in 1 2 4; do L5TTPC_NCOMP=$nc JAX_ENABLE_X64=false $P scripts/bench_l5_speed.py --batches 128 --dts 0.1 --out $OUT/timing_fp32.csv >> $OUT/timing_fp32.log 2>&1; done
     log "fp32 timing no-ckpt"; L5TTPC_NCOMP=2 JAX_ENABLE_X64=false $P scripts/bench_l5_speed.py --batches 128 --dts 0.1 --ckpt none --out $OUT/timing_fp32.csv >> $OUT/timing_fp32.log 2>&1 ;;
esac
log "done"
