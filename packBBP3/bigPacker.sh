#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

jidL="4077401
4077404
4077407
4077410
4077414
4077418
4077422
4077425
4077428
4077431
4077434
4077438
4077441
"

for jid in $jidL ; do
    echo jid=$jid
    time  ./aggregate_Kaustubh.py --jid ${jid}_1
    k=$[ ${k} + 1 ]
done

echo
echo SCAN: packed1 $k jobs
