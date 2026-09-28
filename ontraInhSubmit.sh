#!/bin/bash

# Base directory pattern
base_dir="/pscratch/sd/k/ktub1999/BBP_Ontra_Inhibitory_Exclude_*"

# Iterate over all matching directories
for dir in $base_dir; do
    if [ -d "$dir" ]; then
        echo "Submitting job for directory: $dir"
        sbatch batchShifterOntraInh.slr "$dir"
    else
        echo "Skipping: $dir is not a directory"
    fi
done