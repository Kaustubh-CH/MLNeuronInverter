#!/bin/bash

for i in {0..12}
do
    sbatch batchShifter.slr //pscratch/sd/k/ktub1999/BBP_Ontra_Exc_Mar19_NoNoise_Exclude${i}
done