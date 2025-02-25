#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -J bigDom
#SBATCH -L SCRATCH,cfs
#SBATCH -C cpu
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=balewski/ubu20-neuron8:v5
#SBATCH --array 1-1 #a

shifter ./bigDom.sh