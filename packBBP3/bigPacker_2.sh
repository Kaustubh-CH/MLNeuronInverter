#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="
46690059
46690060
46690061
46690062
46690064
46690065
46690067
46690171
46690172
46690173
46690179
46690181
46690182
46690183
46690184
46690185
46690186
46690188
46690189
"

Path="/pscratch/sd/k/ktub1999/BB_Exc_Test_data_All_Cells/runs2" 
outPath="/pscratch/sd/k/ktub1999/BB_Exc_test_all_cells_processed/"


for jid in $jidL ; do
    echo jid=$jid
    mkdir -p "$outPath"
    time  python3  aggregate_Kaustubh.py --jid ${jid}_1 --simPath $Path --outPath "$outPath"
    k=$[ ${k} + 1 ]
done
# --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# --idx 0 1 2 3 4 5 6 7 8 9 10 13 14 15 18 --numExclude 4 
echo
echo SCAN: packed1 $k jobs

#python3 -m pdb aggregate_All65.py --jid 17710172 17710128 --simPath /pscratch/sd/k/ktub1999/Feb24Nrow/runs2/ --outPath /pscratch/sd/k/ktub1999/M1_ALL
