#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#

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
cellL="
L5_STPCcADpyr
L6_IPCcADpyr
"
for cell in $cellL ; do
    echo cell=$cell
    time  ./format_bbp3_for_ML.py --cellName ${cell}3
    k=$[ ${k} + 1 ]
done

echo
echo SCAN: packed-dom $k jobs
