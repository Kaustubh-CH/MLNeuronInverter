#!/bin/bash -l
# Fisher / Cramér-Rao + feature x channel sensitivity of a jaxley cell under a stim battery,
# run INSIDE an interactive salloc allocation:
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g --gpus-per-node=1 --cpus-per-task=32 \
#          bash scripts/run_sensitivity_salloc.sh
# Defaults = L5TTPC nc2 under the five Roy exp stimuli at the exp fine-tune physics
# (stim_scale 1.0, solver dt 0.2 ms, loss window skip 99.8 ms, fp64 solve for clean FDs).
set -e -o pipefail
# NB: salloc runs this script on the LOGIN node; only the srun lines land on the compute node.
REPO=${REPO:-/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp}
PYENV=/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
CELL=${CELL:-l5ttpc}
export L5TTPC_NCOMP=${L5TTPC_NCOMP:-2}
STIMS=${STIMS:-"Roy100_icaRec_5k Roy500_icaRec_5k Roy1000_icaRec_5k Roy1500_icaRec_5k Roy2000_icaRec_5k"}
H5=${H5:-/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/l5ttpc_nc2_bbp_synth.mlPack1.h5}
OUT=${OUT:-/pscratch/sd/k/ktub1999/tmp_neuInv/sensitivity/${CELL}_nc${L5TTPC_NCOMP}_roy_${SLURM_JOB_ID}}
NUMOP=${NUMOP:-32}; EPS=${EPS:-0.02}; SIMBATCH=${SIMBATCH:-128}
STIM_SCALE=${STIM_SCALE:-1.0}; SIM_DT=${SIM_DT:-0.2}; TSKIP=${TSKIP:-99.8}

module load python; source activate $PYENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_sens_${SLURM_JOB_ID}
mkdir -p $JAX_COMPILATION_CACHE_DIR $OUT
cd $REPO
echo "S: job $SLURM_JOB_ID -> $OUT"; srun -n1 --gpus=1 bash -c 'echo "S: running on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
srun -n1 --gpus=1 --cpus-per-task=32 python -u sensitivity_analysis.py --cell $CELL --stims $STIMS --h5 $H5 \
   --stimScale $STIM_SCALE --simDt $SIM_DT --tSkipMs $TSKIP \
   --numOp $NUMOP --eps $EPS --simBatch $SIMBATCH -o $OUT "$@" 2>&1 | tee $OUT/run.log
echo "S: DONE -> $OUT"; ls -la $OUT
