#!/bin/bash -l
# Same as scripts/gen_model_ladder.slr but for an INTERACTIVE allocation:
#   LADDER_ONLY="single_comp ball_and_stick ball_and_stick_bbp" \
#   salloc -N1 -C gpu -q interactive -t 1:00:00 -A m2043_g --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 \
#          bash scripts/run_gen_ladder_salloc.sh
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
bash /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur/scripts/gen_model_ladder.slr
