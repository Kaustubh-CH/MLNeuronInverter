#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

#   module load pytorch
#  salloc  -C cpu -q interactive  -t4:00:00  -A  m2043  -N 1

cellL="
L6_TPC_L1cADpyr0
L6_TPC_L1cADpyr1
L6_TPC_L1cADpyr2
L6_TPC_L1cADpyr3
L6_TPC_L1cADpyr4
L4_SScADpyr0
L4_SScADpyr1
L4_SScADpyr2
L4_SScADpyr3
L4_SScADpyr4
L4_SPcADpyr0
L4_SPcADpyr1
L4_SPcADpyr2
L4_SPcADpyr3
L4_SPcADpyr4
L6_UTPCcADpyr0
L6_UTPCcADpyr1
L6_UTPCcADpyr2
L6_UTPCcADpyr3
L6_UTPCcADpyr4
L23_PCcADpyr0
L23_PCcADpyr1
L23_PCcADpyr2
L23_PCcADpyr3
L23_PCcADpyr4
L5_UTPCcADpyr0
L5_UTPCcADpyr1
L5_UTPCcADpyr2
L5_UTPCcADpyr3
L5_UTPCcADpyr4
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
cellL="
L4_BPdSTUT0
L4_BTCbIR0
L4_BTCbSTUT0
L4_BTCdNAC0
L5_DBCbAC0
L5_DBCcNAC0
L6_MCcIR0
L23_DBCcAC0
L23_LBCbNAC0
L23_LBCcSTUT0
"

cellL="
L4_BTCbAC0
L6_BTCbAC0
L6_MCbAC0
L6_LBCbIR0
L6_MCbIR0
L4_BPbIR0
L4_NGCbNAC0
L6_LBCbNAC0
L5_MCbSTUT0
L6_DBCbSTUT0
L6_NBCbSTUT0
L23_DBCcAC0
L4_BTCcAC0
L5_MCcAC0
L6_NBCcAC0
L1_HACcIR0
L6_NBCcIR0
L1_NGC-SAcNAC0
L23_BPcNAC0
L4_LBCcNAC0
L6_LBCcNAC0
L1_NGC-DAcSTUT0
L6_NGCcSTUT0
L23_NGCcSTUT0
L6_NBCcSTUT0
L23_LBCdNAC0
L23_MCdNAC0
L4_ChCdNAC0
L4_SBCdNAC0
"
cellL="
L5_NBCcIR0
L5_DBCcIR0	
L5_BPdSTUT0
L23_BPdSTUT0
L6_NBCdSTUT0
L6_BPdSTUT0
"
#ALL 50 INH
cellL="
L5_DBCbAC0
L23_BTCbAC0
L4_BTCbAC0
L6_BTCbAC0
L6_MCbAC0
L4_BTCbIR0
L23_DBCbIR0
L6_LBCbIR0
L6_MCbIR0
L4_BPbIR0
L23_LBCbNAC0
L23_NGCbNAC0
L23_NGCbNAC0
L4_NGCbNAC0
L6_LBCbNAC0
L4_BTCbSTUT0
L5_DBCbSTUT0
L5_MCbSTUT0
L6_DBCbSTUT0
L6_NBCbSTUT0
L23_DBCcAC0
L23_BPcAC0
L4_BTCcAC0
L5_MCcAC0
L6_NBCcAC0
L6_MCcIR0
L1_HACcIR0
L6_NBCcIR0
L5_NBCcIR0
L5_DBCcIR0
L5_DBCcNAC0
L1_NGC-SAcNAC0
L23_BPcNAC0
L4_LBCcNAC0
L6_LBCcNAC0
L23_LBCcSTUT0
L1_NGC-DAcSTUT0
L6_NGCcSTUT0
L23_NGCcSTUT0
L6_NBCcSTUT0
L4_BTCdNAC0
L23_LBCdNAC0
L23_MCdNAC0
L4_ChCdNAC0
L4_SBCdNAC0
L4_BPdSTUT0
L5_BPdSTUT0
L23_BPdSTUT0
L6_NBCdSTUT0
L6_BPdSTUT0
"

# for jid in $jidL ; do
dataPath="/pscratch/sd/k/ktub1999/BBP_Sep_19_Inh_50Cells_OntraHandPicked/"

    for cell in $cellL ; do
        echo cell=$cell
        time  python3 format_bbp3_for_ML_paralelly.py --cellName ${cell}   --dataPath "$dataPath"
        k=$[ ${k} + 1 ]
done
# done
echo
echo SCAN: packed-dom $k jobs
