#!/bin/bash -l
# Score the POOLED model separately on every stimulus in the battery.
# A pooled model sees one untagged trace, so recovery differs per protocol and
# the stock eval (stims_select[0] = 5kChaoticRamp) reports only the first one.
set -u
cd "$(dirname "$0")"
module load conda >/dev/null 2>&1
conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true PYTHONNOUSERSITE=1
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/tmp_neuInv/jaxcache_evalpool_${SLURM_JOBID:-0}
for k in 0 1 2 3; do
  echo "############ stimIndex $k  $(date +%T)"
  python -u evaluate_voltage.py --modelPath ./out --numSamples 200 \
         --numOverlay 20 --stimIndex $k >& log.evalvolt_stim$k
  tail -3 log.evalvolt_stim$k
done
echo "############ collect"
python - <<'PY'
import glob, yaml, os
rows=[]
for d in sorted(glob.glob("out/eval_stim*")):
    s=yaml.safe_load(open(os.path.join(d,"summary.yaml")))
    rows.append((s.get("eval_stim_index"), s.get("eval_stim_name"),
                 s["channel_r2_overall"], s["voltage_mse_z_mean"],
                 s["spike_count_diff_mean_abs"],
                 {p["name"]:round(p["r2"],3) for p in s["channel_per_param"]}))
print(f"{'k':>2} {'stim':<24} {'meanR2':>7} {'mse_z':>7} {'spkdiff':>8}  per-channel R2")
for k,n,r2,mz,sd,per in rows:
    print(f"{k:>2} {n:<24} {r2:>7.3f} {mz:>7.3f} {sd:>8.2f}  {per}")
PY
