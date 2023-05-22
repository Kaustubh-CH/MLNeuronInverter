#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="8944917
8944944
"
Path="/pscratch/sd/k/ktub1999/Feb24Nrow/runs2/" 
outPath="/pscratch/sd/k/ktub1999/bbp_May_18"


for jid in $jidL ; do
    echo jid=$jid
    mkdir -p "$outPath"_"$jid"
    time  ./aggregate_Kaustubh.py --jid ${jid}_1 --simPath $Path --outPath "$outPath"_"$jid"
    k=$[ ${k} + 1 ]
done
# --idx 0 1 2 3 4 5 6 7 8 9 10 13 14 15 18 --numExclude 4 
echo
echo SCAN: packed1 $k jobs
