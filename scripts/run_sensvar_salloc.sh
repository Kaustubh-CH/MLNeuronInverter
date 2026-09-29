#!/bin/bash -l
# One-at-a-time (OAT) voltage-variation sweep (sensitivity_variation.py) for a jaxley cell under a
# stim battery, run INSIDE an interactive salloc allocation (1 GPU):
#   salloc -N1 -C gpu -q interactive -t 1:30:00 -A m2043_g --gpus-per-node=1 --cpus-per-task=32 \
#          bash scripts/run_sensvar_salloc.sh
# Defaults = L5TTPC nc2 under the five Roy exp stimuli at the real current (x1.0) plus the training
# stimulus 5k50kInterChaoticB at the pack's x1.5, solver dt 0.2 ms, fp64, N=500 draws per channel.
# NB: salloc runs this script on the LOGIN node; only the srun lines land on the compute node.
set -e -o pipefail
REPO=${REPO:-/global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/roy-exp}
PYENV=/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
CELL=${CELL:-l5ttpc}
export L5TTPC_NCOMP=${L5TTPC_NCOMP:-2}
STIMDIR=${STIMDIR:-/pscratch/sd/k/ktub1999/main/DL4neurons2/stims}
STIMS=${STIMS:-"Roy100_icaRec_5k Roy500_icaRec_5k Roy1000_icaRec_5k Roy1500_icaRec_5k Roy2000_icaRec_5k 5k50kInterChaoticB@1.5"}
H5=${H5:-/pscratch/sd/k/ktub1999/model_ladder_data/l5ttpc_nc2_icb4k_bbp_dt02/l5ttpc_nc2_bbp_synth.mlPack1.h5}
TAG=${TAG:-roy}
RUN=${RUN:-/pscratch/sd/k/ktub1999/tmp_neuInv/sensitivity_variation/${CELL}_nc${L5TTPC_NCOMP}/${TAG}_${SLURM_JOB_ID}}
OUT=$RUN/combined
NSAMPLES=${NSAMPLES:-500}; SIMBATCH=${SIMBATCH:-128}
STIM_SCALE=${STIM_SCALE:-1.0}; SIM_DT=${SIM_DT:-0.2}
FISHER_SRC=${FISHER_SRC:-/pscratch/sd/k/ktub1999/tmp_neuInv/sensitivity/l5ttpc_nc2_roy_58383215}

module load python; source activate $PYENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_sensvar_${SLURM_JOB_ID}
mkdir -p $OUT
cd $REPO
echo "S: job $SLURM_JOB_ID -> $RUN"
srun -n1 --gpus=1 bash -c 'echo "S: running on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
srun -n1 --gpus=1 --cpus-per-task=32 python -u sensitivity_variation.py --cell $CELL \
   --stimDir $STIMDIR --stims $STIMS --h5 $H5 --stimScale $STIM_SCALE --simDt $SIM_DT \
   --nSamples $NSAMPLES --simBatch $SIMBATCH --saveTraces --topK 6 -o $OUT "$@" 2>&1 | tee $RUN/run.log
# Sibling copy of the Fisher / CRB analysis (same cell, stims, physics) for the roy2000_decomp layout.
if [ -d "$FISHER_SRC" ]; then mkdir -p $RUN/fisher; cp $FISHER_SRC/*.csv $FISHER_SRC/*.yaml $FISHER_SRC/*.png $RUN/fisher/; fi
echo "S: DONE -> $RUN"; ls -la $OUT
