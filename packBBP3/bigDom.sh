#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#  salloc  -C cpu -q interactive  -t4:00:00  -A  m2043  -N 1
#   module load pytorch

cellL="
L5_TTPC1cADpyr
"
jidL="
8944917
8944944
"

for jid in $jidL ; do
    dataPath=/pscratch/sd/k/ktub1999/bbp_May_18

    for cell in $cellL ; do
        echo cell=$cell
        time  ./format_bbp3_for_ML.py --cellName ${cell}0   --dataPath "$dataPath"_"$jid"
        k=$[ ${k} + 1 ]
    done
done
echo
echo SCAN: packed-dom $k jobs
