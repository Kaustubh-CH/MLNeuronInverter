#!/bin/bash
# Physiology scan of the model-ladder packs (CPU; HOME).  Part 1 restricted to single_comp to keep it fast.
cd /global/u1/k/ktub1999/Neuron/neuron4/neuroninverter/.claude/worktrees/ca3-vo-dtwblur
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true LADDER_ONLY=1
P=/pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley/bin/python
sed -i 's#glob.glob("/pscratch/sd/k/ktub1999/model_ladder_data/\*/\*.mlPack1.h5")#glob.glob(os.environ.get("LADDER_DATA_ROOT", "/global/homes/k/ktub1999/model_ladder_data") + "/*/*.mlPack1.h5")#' scripts/scan_cells_default_physiology.py
$P scripts/scan_cells_default_physiology.py docs/model_ladder/packs single_comp > docs/model_ladder/packs.log 2>&1
echo "PACK-SCAN done $(date)"
