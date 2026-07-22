#!/bin/bash
set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

k=0

# Directory containing the files
dataPath="/pscratch/sd/k/ktub1999/Ontra_Single_Inh_dataMay18th/"

# Iterate through all files ending with simRaw.h5
for file in "$dataPath"/*simRaw.h5; do
    # Extract the cell name (remove the directory path and .h5 extension)
    # cell=$(basename "$file" .h5)
    cell=$(basename "$file" | sed 's/.simRaw\.h5$//')
    echo "Processing cell=$cell"
    time python3 format_bbp3_for_ML_paralelly.py --cellName "$cell" --dataPath "$dataPath"
    
    k=$((k + 1))
done

echo
echo "SCAN: packed-dom $k jobs"