#!/bin/bash
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value

# use case: ./postproc_oneSimu.sh  bbp205 L6_TPC_L1_cADpyr231 48781230,48788634_[3-6]  2 0K 
# note: the 1st jobId can't be expandable-list

# to prevent: H5-write error: unable to lock file, errno = 524
export HDF5_USE_FILE_LOCKING=FALSE

files_out=131 #  null   or a number : how many h5 will be produced before packing quits

simuSourcePath=/global/cscratch1/sd/adisaran/DL4neurons/runs/
#targetPath0=/global/cscratch1/sd/balewski/
targetPath0=/global/cfs/cdirs/m2043/balewski/
targetPath=$targetPath0/neuronBBP2-data_67pr/

bbpNNN=$1
bbpBase=$2  # e.g.  L5_STPC_cADpyr232 w/o cloneId
simuJobIdS=$3  # list:  123,567,324  (commas, no spaces, can have [a-b])
cloneId=$4  # only integer w/o 'c'
packExt=${5-4K}

bbpId=${bbpNNN}${cloneId} # bbbpNNNC
bbpName=${bbpBase}_${cloneId}

# get 1st jobid for the list for meta-data
simuJobId=`echo $simuJobIdS | cut -f1 -d,`

# init other variables
if [[ $packExt == "1K" ]]; then
    frames_out=1024 
elif [[ $packExt == "6K" ]]; then
    frames_out=6144 
elif [[ $packExt == "0K" ]]; then
    frames_out=30  # testing for small data size
else 
    echo wrong packExt=$packExt=
    exit 99
fi

echo P: packTExt  $packExt $frames_out

inpPath1=$simuSourcePath/$simuJobId/${bbpBase}_1/c${cloneId}/
outPath=$targetPath/$bbpId/
echo neuronSim prep simuJobId=$simuJobId, bbpName=$bbpName bbpId=$bbpId  inp1=$inpPath1  out=$outPath frames_out=$frames_out
date

echo 1b-count hd5 files in $inpPath1 ...
nHD5=` find $inpPath1/ -type f -name  *h5 |wc -l`
echo see $nHD5 files in $simuJobId , there may be more...

echo 2-create output dir $outPath
rm -rf $outPath
mkdir -p $outPath

echo 3-produce rawMeta stub
metaF=$outPath/stubMeta.yaml
echo "# automatic metadata generated on `date` " > $metaF
echo "bbpId: $bbpId " >> $metaF
echo "cloneId: $cloneId " >> $metaF
echo "simuJobId: $simuJobId " >> $metaF
echo "maxOutFiles: $files_out"  >> $metaF
echo "numFramesPerOutput: $frames_out "  >> $metaF  
echo "shuffle: True # use False only for debugging"  >> $metaF
echo "digiNoise: None"  >> $metaF
echo "useQA: False "  >> $metaF
echo "rawPath0: $simuSourcePath "  >> $metaF
echo "rawJobIdS: '$simuJobIdS'  "  >> $metaF

echo "# "  >> $metaF
echo "# add meta data from Vyassa below: "  >> $metaF

echo 'metaStub dump - - - - - -'
cat $metaF

metaF2=$outPath/rawMeta.cellSpike.yaml
metaF0=`ls $inpPath1/../../${bbpBase}_1*-meta-${cloneId}.yaml`
ls -l $metaF0
cat  $metaF   > $metaF2

# remove 'rawPath' from Vyassa's yaml because now we have muti-job scheme and this path points to just 1 job
cat  $metaF0 |grep -v rawPath >> $metaF2
echo created complete metafile
ls -l $metaF2

echo 4-execute formating of Vyassas $nHD5 HD5 files ...
time python3 -u ./format_Vyassa.py  --dataPath $outPath 

echo 5-open read access
chmod a+rx -R $targetPath/$bbpId
chmod a+rx -R $outPath

echo DONE $bbpId
date
