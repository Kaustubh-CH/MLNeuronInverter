#!/bin/bash
# Halve each cell's contribution in all ONTRA Inhibitory datasets.
# Data is GLOBALLY SHUFFLED (verified: aggregate_All65.py np.random.shuffle, and
# on-disk KS tests), so keeping the first 50% of every train/valid/test split
# halves every cell uniformly and preserves the 8/1/1 ratio. No per-cell labels
# exist in the packed file, so contiguous per-cell slicing is not possible.
#
#   module load conda
#   conda activate /pscratch/sd/k/ktub1999/conda_envs/neuroninverter_jaxley
#   ./halveInhAll.sh
#
# Output: a sibling dir <dataset>_half/ next to each input dataset dir.
set -u

PARENT="/pscratch/sd/k/ktub1999"
DIRGLOB="BBP_Ontra_Inhibitory_Exclude_*"

python3 halve_inh_dataset.py \
    --parent   "$PARENT" \
    --dirGlob  "$DIRGLOB" \
    --fileGlob '*.mlPack1.h5' \
    --fraction 0.5
    # add --fileGlob '*.h5'   to also halve the .simRaw.h5 files
    # add --random            to keep a random (vs first-rows) half
    # add --dryRun            to preview without writing
