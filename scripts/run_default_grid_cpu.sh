#!/bin/bash
# Cross-cell DEFAULT-parameter physiology grid (CPU, fp64) for the model ladder:
# one run at ncomp=4 (all cells) + one at ncomp=2 (l5ttpc only). ~10-15 min on a login node.
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
P=/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python
SKIP_DATASETS=1 L5TTPC_NCOMP=4 $P scripts/scan_cells_default_physiology.py docs/model_ladder/default_grid_nc4 \
   single_comp,ball_and_stick,ball_and_stick_bbp,ca3_pyramidal,l5ttpc > docs/model_ladder/default_grid_nc4.log 2>&1
SKIP_DATASETS=1 L5TTPC_NCOMP=2 $P scripts/scan_cells_default_physiology.py docs/model_ladder/default_grid_nc2 \
   l5ttpc > docs/model_ladder/default_grid_nc2.log 2>&1
echo "DEFAULT-GRID done $(date)"
