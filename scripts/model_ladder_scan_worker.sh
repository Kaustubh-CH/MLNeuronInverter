#!/bin/bash
# One rank of the model-ladder stimulus-scale sweep (launched by
# scripts/run_model_ladder_scan_salloc.sh via srun -n4).  Rank -> task list
# below; each task = one scripts/stim_scale_scan.py call on this rank's GPU.
set -u
R=${SLURM_PROCID:-0}
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cc/ladder_${SLURM_JOB_ID:-x}_$R
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
OUT=${LADDER_OUT:-docs/model_ladder}
NBOX=${LADDER_NBOX:-128}
mkdir -p "$OUT"
run() {  # name cell scales [env...]
  local name=$1 cell=$2 scales=$3; shift 3
  echo "[rank $R] $(date +%T) START $name"
  env "$@" python -u scripts/stim_scale_scan.py --cell "$cell" --scales "$scales" \
      --nbox "$NBOX" --batch "${LADDER_BATCH:-64}" --out "$OUT/scan_$name" \
      > "$OUT/scan_$name.log" 2>&1 && echo "[rank $R] $(date +%T) DONE $name" || echo "[rank $R] FAILED $name (see $OUT/scan_$name.log)"
}
if [[ "${LADDER_TASKSET:-all}" == "l5rerun" ]]; then
  case $R in
    0) run l5ttpc_nc4 l5ttpc 1,1.5,2 L5TTPC_NCOMP=4 LADDER_BATCH=32 ;;
    1) run l5ttpc_nc2 l5ttpc 1,1.5,2 L5TTPC_NCOMP=2 ;;
    2) run ball_and_stick_bbp ball_and_stick_bbp 0.05,0.07,0.1 ;;
    3) echo "[rank $R] idle" ;;
  esac
  echo "[rank $R] $(date +%T) all tasks finished"; exit 0
fi
case $R in
  0) run l5ttpc_nc4 l5ttpc 0.5,0.75,1,1.5,2 L5TTPC_NCOMP=4 LADDER_BATCH=32 ;;
  1) run l5ttpc_nc2 l5ttpc 0.5,0.75,1,1.5,2 L5TTPC_NCOMP=2 ;;
  2) run ball_and_stick_bbp ball_and_stick_bbp 0.03,0.05,0.07,0.1,0.15,0.2
     run ca3_pyramidal ca3_pyramidal 0.25,0.5,0.75,1 ;;
  3) run single_comp single_comp 0.02,0.03,0.05,0.07,0.1,0.15
     run ball_and_stick ball_and_stick 0.02,0.03,0.05,0.07,0.1,0.15 ;;
esac
echo "[rank $R] $(date +%T) all tasks finished"
