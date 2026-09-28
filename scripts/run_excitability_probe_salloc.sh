#!/bin/bash -l
# L5 nc2 excitability probe (2026-09-23): 10 variants of the fixed dendritic passive properties /
# stim scale, 4 GPU-parallel groups, each `srun -n1 --gpus=1` (salloc runs THIS script on the login node).
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_excitability_probe_salloc.sh
set -u
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur || exit 2
module load python; source activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true L5TTPC_NCOMP=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
OUT=${OUT:-/pscratch/sd/k/ktub1999/tmp_neuInv/excitability_probe/l5nc2_${SLURM_JOB_ID}}
mkdir -p $OUT
G=("--variant base: --variant c2:scale=2 --variant c3:scale=3 --variant c4:scale=4"
   "--variant g1e5:gpas=1e-5 --variant g3e6:gpas=3e-6"
   "--variant cm1:cm=1 --variant g1e5cm1:gpas=1e-5,cm=1"
   "--variant g3e6cm1:gpas=3e-6,cm=1 --variant g1e6cm1:gpas=1e-6,cm=1")
for k in 0 1 2 3; do
  JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_probe_${SLURM_JOB_ID}_$k \
  srun -n1 --gpus=1 --cpus-per-task=32 --exact python -u scripts/l5_excitability_probe.py ${G[$k]} \
       -o $OUT/g$k > $OUT/g$k.log 2>&1 &
done
wait
echo "PROBE DONE $(date) -> $OUT"; tail -n 25 $OUT/g*.log
