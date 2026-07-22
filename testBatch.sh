#!/bin/bash -l

#SBATCH  -N1  
#SBATCH --time=30:00
#SBATCH  -J ni_h5
#SBATCH   -q debug
#SBATCH  --ntasks-per-node=1 
#SBATCH  --gpus-per-task=1 
#SBATCH  --cpus-per-task=128 -C gpu -A  m2043_g 
echo "H1"
