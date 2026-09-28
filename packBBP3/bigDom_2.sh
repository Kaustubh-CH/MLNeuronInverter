#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#   module load pytorch
#  salloc  -C cpu -q interactive  -t4:00:00  -A  m2043  -N 1

cellL="
L5_STPCcADpyr0
L5_STPCcADpyr1
L5_STPCcADpyr2
L5_STPCcADpyr3
L5_STPCcADpyr4
L6_IPCcADpyr0
L6_IPCcADpyr1
L6_IPCcADpyr2
L6_IPCcADpyr3
L6_IPCcADpyr4
L6_TPC_L4cADpyr0
L6_TPC_L4cADpyr1
L6_TPC_L4cADpyr2
L6_TPC_L4cADpyr3
L6_TPC_L4cADpyr4
L5_TTPC1cADpyr0
L5_TTPC1cADpyr1
L5_TTPC1cADpyr2
L5_TTPC1cADpyr3
L5_TTPC1cADpyr4
L5_TTPC2cADpyr0
L5_TTPC2cADpyr1
L5_TTPC2cADpyr2
L5_TTPC2cADpyr3
L5_TTPC2cADpyr4
L4_PCcADpyr0
L4_PCcADpyr1
L4_PCcADpyr2
L4_PCcADpyr3
L4_PCcADpyr4
L6_BPCcADpyr0
L6_BPCcADpyr1
L6_BPCcADpyr2
L6_BPCcADpyr3
L6_BPCcADpyr4

"
jidL="
0

"

for jid in $jidL ; do
    dataPath="/pscratch/sd/k/ktub1999/BB_Exc_test_all_cells_processed/"

    for cell in $cellL ; do
        echo cell=$cell
        time  python3 format_bbp3_for_ML_paralelly.py --cellName ${cell}  --dataPath "$dataPath"
        k=$[ ${k} + 1 ]
    done
done
echo
echo SCAN: packed-dom $k jobs
