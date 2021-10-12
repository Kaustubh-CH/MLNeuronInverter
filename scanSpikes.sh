#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
#set -e ;  #  bash exits if any statement returns a non-true return value
#set -o errexit ;  # exit if any statement returns a non-true return value
k=0

# FIX: minimal sensible height form the base : 30 mV
# FIX  minimal sensible width  measured at half amplitude: 0.5 ms

#Roy:
spikePath=out-inhib-roy/
#Vyassa:
spikePath=out-inhib-vyassa/

for cell in bbp124 bbp125 bbp126 bbp127 bbp128 bbp129 bbp139 bbp140 bbp141 bbp142 bbp143 bbp144 ; do # one-off
    echo 
    echo start cell=$cell

    # Roy:
    #./formatRawSimB.py --cellName  $cell
    #./findSpikesSimB.py   --outPath $spikePath --dataName $cell 

    # Vyassa:
    ./formatSimV8kHz.py -n 1000 --cellName  $cell 
    ./findSpikesSimB.py  --dataPath /global/homes/b/balewski/prjn/2021-roys-simulation/vyassa8kHz/ --formatName  simV.8kHz --outPath $spikePath --dataName $cell
    
    ./plotSpikesSimB.py  --dataName  $cell  -X  --dataPath $spikePath --outPath $spikePath
    k=$[ ${k} + 1 ]
    #exit
done


echo
echo SCAN: processed $k cells

