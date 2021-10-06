#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0


for cell in bbp176 ; do # one-off
    echo 
    echo start cell=$cell

    ./formatSimV8kHz.py -n 500  --cellName $cell

    ./findSpikesSimB.py  --dataPath /global/homes/b/balewski/prjn/2021-roys-simulation/vyassa8kHz/ --formatName  simV.8kHz --dataName $cell
    
    ./plotSpikesSimB.py  --dataName  $cell
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: submitted $k jobs
