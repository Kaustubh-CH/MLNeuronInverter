#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="6712138
"
Path="/pscratch/sd/k/ktub1999/Feb24Nrow/runs2/" 
outPath="/pscratch/sd/k/ktub1999/bbp_Mar30"

for jid in $jidL ; do
    echo jid=$jid
    time  ./aggregate_Kaustubh.py --jid ${jid}_1 --simPath $Path --outPath $outPath
    k=$[ ${k} + 1 ]
done

echo
echo SCAN: packed1 $k jobs
