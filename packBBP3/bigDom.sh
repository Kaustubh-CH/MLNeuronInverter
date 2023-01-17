#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#  salloc  -C cpu -q interactive  -t4:00:00  -A  m2043  -N 1
#   module load pytorch

cellL="
L4_SScADpyr
L4_SPcADpyr
L5_TTPC1cADpyr
L5_TTPC2cADpyr
L4_PCcADpyr
L6_BPCcADpyr
L6_TPC_L4cADpyr
L6_TPC_L1cADpyr
L6_UTPCcADpyr
L23_PCcADpyr
L5_UTPCcADpyr
L5_STPCcADpyr
L6_IPCcADpyr
"



dataPath=/global/cfs/cdirs/m2043/balewski/neuronBBP3-10kHz_3pr_6stim/dec26_simRaw

for cell in $cellL ; do
    echo cell=$cell
    time  ./format_bbp3_for_ML.py --cellName ${cell}4   --dataPath $dataPath
    k=$[ ${k} + 1 ]
done

echo
echo SCAN: packed-dom $k jobs
