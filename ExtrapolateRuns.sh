#!/bin/bash

# Directory containing the .h5 files
directory="/pscratch/sd/k/ktub1999/Ontra_Single_Inh_dataMay18th"
mkdir -p outExtrapolate
# Loop through all files ending with mlpack1.h5
for file in "$directory"/*mlPack1.h5; do
    # Extract the base name of the file (without extension)
    cell_name=$(basename "$file" | sed 's/.mlPack1\.h5$//')
    # cell_name=$(basename "$file" .mlpack1.h5)
    echo $cell_name
    echo $file
    # Construct and execute the command
    command="srun -n1 shifter python -u ./predict.py --modelPath out -X --venue poster --segmentColors -o outExtrapolate --sortBy axonal somatic dend apical all --cellName $cell_name"
    echo "Executing: $command"
    eval "$command"
done