#!/bin/bash -l
#SBATCH -t 2:30:00
#SBATCH -q regular
#SBATCH -J ontraInhAllPacker
#SBATCH --error=logs/%A_%a.err
#SBATCH -L SCRATCH,cfs
#SBATCH -N 15
#SBATCH -C cpu
#SBATCH --output logs/%A_%a
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

grouped_csv="/global/homes/k/ktub1999/mainDL4/DL4neurons2/Grouped_PMJobs.csv" 
Path="/pscratch/sd/k/ktub1999/BBP_Inh_Feb5thAll150CellsNoNoise/runs2"
outPath_base="/pscratch/sd/k/ktub1999/BBP_Ontra_Inhibitory_Exclude"

 srun -n15 shifter python3 ontraInhAllSubmit.py \
    --csv_file $grouped_csv \
    --path $Path \
    --outPath_base $outPath_base \
