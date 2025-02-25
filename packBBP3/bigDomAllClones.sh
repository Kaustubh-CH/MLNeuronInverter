#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#   module load pytorch
#  salloc  -C cpu -q interactive  -t4:00:00  -A  m2043  -N 1
# shifter python3 format_bbp3_for_ML_paralelly_cell_wise.py --dataPath /pscratch/sd/k/ktub1999/ExC_Ontra_Excluded2 --cellName ALL_CELLS
cellL="
AllCellsTestOnly
"
cellL="
ALL_CELLS_interpolated
"

jidL="
10
"

for jid in $jidL ; do
    dataPath="/pscratch/sd/k/ktub1999/Oct29OntraExcExclude"$jid

    for cell in $cellL ; do
        echo cell=$cell
        time  python3 format_bbp3_for_ML_paralelly.py --cellName ${cell}   --dataPath "$dataPath"
        k=$[ ${k} + 1 ]
    done
done
echo
echo SCAN: packed-dom $k jobs
