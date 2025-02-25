#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="
32208934
32208935
32208936
32208937

32208939
32208940
32208941
32208942

32208944
32208945
32208947
32208948

32208950
32208951
32208952
32208953

32208956
32208957
32208958
32208959

32208961
32208962
32208963
32208964

32208966
32208967
32208968
32208969

32208972
32208973
32208974
32208979

32208983
32208985
32208987
32208989

32263650
32263651
32263652
32263653

32208999
32209000
32209001
32209002

32209004
32209005
32209006
32209007
"

Path="/pscratch/sd/k/ktub1999/BBP_Exp_Exc/runs2/" 
outPath="/pscratch/sd/k/ktub1999/BBP_Inh_Rest_aug27"


for jid in $jidL ; do
    echo jid=$jid
    mkdir -p "$outPath"_"$jid"
    time  python3  aggregate_Kaustubh.py --jid ${jid}_1 --simPath $Path --outPath "$outPath"_"$jid" 
    k=$[ ${k} + 1 ]
done
# --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
# --idx 0 1 2 3 4 5 6 7 8 9 10 13 14 15 18 --numExclude 4 
echo
echo SCAN: packed1 $k jobs

#python3 -m pdb aggregate_All65.py --jid 17710172 17710128 --simPath /pscratch/sd/k/ktub1999/Feb24Nrow/runs2/ --outPath /pscratch/sd/k/ktub1999/M1_ALL
