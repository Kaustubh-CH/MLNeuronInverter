#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;  #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value
k=0

inpPath=/global/cfs/cdirs/m2043/balewski/neuronBBP2-data_67pr
hsiPath=neuronBBP2-data_67pr
# excitatory cells
for cell in  bbp054 bbp098 bbp102 bbp152 bbp153 bbp154 bbp155 bbp156 bbp176 bbp205 bbp206 bbp207 bbp208 ; do
    echo start cell=$cell  `date`
    for clone in  `seq 5 5` ; do
	core=${cell}${clone}
	echo core=$core
	tgt=${hsiPath}/${core}.tar
	time  htar -cf $tgt ${inpPath}/$core  >& out/log-htar.$core 
	#hsi ls -B $tgt
	k=$[ ${k} + 1 ]	
	#exit
    done
done

echo
echo HTAR: processed $k cells

exit
screen: cori1

To restore a  dir from HPSS
cd abc
time htar -xf neuronBBP2-data_67pr/bbp0541.tar
