#!/usr/bin/env python3
""" 
 ONTRA-3,4 transformation for ONE cell belonging to any inhib etype: see Readme.ontra-3,-4
 INPUT:  67probes,  40 kHz, ~40 cells
 OUTPUT:  3(4) probes, 8 kHz, train+val+test subsest stored in the common h5 file + maching meta-file
  input conductances are reduced to 19 which are common to all 3 etaypes and U-par are lineary transformed to preserve phys-par mapping across all calls and e-types

 it reads *all* data in to RAM and then writes one output file at the end
 data are normalized to N(0,1) and saved as float32
 Output:
 - bbp019_8inhib.cellSpike_3prB8kHz.data.h5
 - bbp019_8inhib.cellSpike_3prB8kHz.meta.yaml

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from toolbox.Util_IOfunc import write_yaml, read_yaml,write_data_hdf5,read_data_hdf5
from pprint import pprint
import numpy as np

import random,time

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellName", type=str, default='bbp153', help="cell shortName list, blanks separated")
    parser.add_argument( "--transformConfig", default='ontra4_excite2', help="defines all needed switches and choices")

    args = parser.parse_args()
    args.verb=1
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

# only to rebin stimulus I forgot earlier

#...!...!..................
def rebin_data1D(X,nReb):
        tBin=X.shape[0]
        #print('X1D',X.shape,nReb)
        assert tBin==8000 # tested only for original data
        assert tBin%nReb==0
        a=X.reshape(tBin//nReb,nReb)
        b=np.sum(a,axis=1)/nReb
        #print('X2',a.shape,b.shape,b.dtype)
        return b

def rebin_data3D(X,nReb):
        nS,tBin,nF=X.shape
        #print('X3D',X.shape,'nReb=',nReb)
        assert tBin==8000 # tested only for original data
        assert tBin%nReb==0
        a=X.reshape(nS,tBin//nReb,nReb,nF)
        b=np.sum(a,axis=2)/nReb
        #print('X2',a.shape,b.shape,b.dtype)
        return b


#...!...!..................
def repack_one_cell(confD,goalD,nReb,verb=1):
    numShardsPerCell=goalD['dataInfo'].pop('numDataFiles')
    numFramesPerShard=6144 # excatly 6k
    #numShardsPerCell=3;# numFramesPerShard=1024  #tmp , for quck test

    # out dims
    totalFrameCount=numFramesPerShard*numShardsPerCell
    num_tbin=goalD['dataInfo']['numTimeBin']
    out_numFeature=goalD['dataInfo']['numFeature']
    num_par=goalD['dataInfo']['numPar']
    cellName = goalD['rawInfo']['bbpId']
    probSel=[ x[0] for x in goalD['packInfo']['probeName'] ]
    
    # efficient u--> ustar transformation:
    #  ustar= (upar + log10P_jk - centP_k ) / delP

    parSel=[]
    utrans_bias=[]
    utrans_scale=[]
    parNameL=[]
    for [idx, name, centP,delP] in goalD['packInfo']['conductName']:
        [nameOrg,blo,bhi]=goalD['rawInfo']['physRange'][idx]
        # one more check conductances re-mapping is consisten
        assert name==nameOrg 
        assert blo>0
        base=np.sqrt(blo*bhi)
        ln10p=np.log10(base)
        parSel.append(idx)
        utrans_bias.append(float(-ln10p+centP))
        utrans_scale.append(delP)
        parNameL.append(name)
    goalD['packInfo']['utrans_bias']=utrans_bias
    goalD['packInfo']['utrans_scale']=utrans_scale
    goalD['dataInfo']['parName']=parNameL
    
    print('cell=',cellName,',probSel:',probSel,'shards:',numShardsPerCell,',parSel:',parSel)
    print('goal conductances:',parNameL)
    
    # convert to np-array for speed 
    utrans_bias=np.array(utrans_bias)
    utrans_scale=np.array(utrans_scale)
    
    bigD={}
    bigD['Fall']=np.zeros((totalFrameCount,num_tbin,out_numFeature),dtype='float32')
    bigD['Uall']=np.zeros((totalFrameCount,num_par),dtype='float32')
    bigD['Pall']=np.zeros((totalFrameCount,num_par),dtype='float32')
    print(' created output storage for a cell Fall:',bigD['Fall'].shape,'Uall:',bigD['Uall'].shape)

    assert out_numFeature==len(probSel)

    #pprint(goalD['rawInfo']['physRange'])

    frmOff=0
    startT0 = time.time()
    for ii in range(numShardsPerCell): 
        shardId=ii
        print(cellName,'read shard %d of %d, elaT=%.1f (min) ...'%(ii,numShardsPerCell,(time.time()-startT0)/60.))
        inpF=confD['inpPath']+cellName+"/%s.cellSpike.data_%d.h5"%(cellName,shardId)
        inpD=read_data_hdf5(inpF,verb=shardId<1)
        assert max(probSel)<=inpD['frames'].shape[2]
        assert numFramesPerShard <=inpD['frames'].shape[0]
        #print('rr',inpD['unit_par'][:numFramesPerShard][:,parSel].shape)
        upar=inpD['unit_par'][:numFramesPerShard][:,parSel]

        #linear transfomation of U-->U* to preserve meaning of conductances in Si
        ustar= (upar- utrans_bias)/ utrans_scale

        bigD['Uall'][frmOff:frmOff+numFramesPerShard]=ustar
        bigD['Pall'][frmOff:frmOff+numFramesPerShard]=inpD['phys_par'][:numFramesPerShard][:,parSel]

        # clip frame count to exactly 6k
        X=inpD['frames'][:numFramesPerShard]
        
        # select probes of interest
        X=X[...,probSel].astype('float32')
        
        # rebin time axis
        X=rebin_data3D(X,nReb)
        
        if ii==0: # compute avr+std once per cell from the 1st shard
            #  double precision for std computation but results can be back in float32
            X=X.astype('float64')
            fnorm_mean=np.mean(X,axis=(0,1)).astype('float32')
            fnorm_std=np.std(X,axis=(0,1)).astype('float32')
            X=X.astype('float32')

            goalD['packInfo']['fnorm_mean']=[ float(x) for x in fnorm_mean]
            goalD['packInfo']['fnorm_std']=[ float(x) for x in fnorm_std]
            
        # scale data to ~N(0,1), vectorized
        X=(X - fnorm_mean) / fnorm_std
        
        bigD['Fall'][frmOff:frmOff+numFramesPerShard]=X[:]
        frmOff+=numFramesPerShard
    packTime=time.time() - startT0

    
    print('repacking done for',cellName,frmOff,' elaT=%.1f (min)\n'%(packTime/60.))

    # moments for Action Potential
    if 0: # must use double precision for checking computaton of std:
        print('\ncross-check AP avr+std , compute avr again...',bigD['Fall'].shape)
        F64=bigD['Fall'].astype('float64')
        all_avr=np.mean(F64,axis=(0,1))
        print('F avr64 again:',all_avr,F64.dtype)
        all_std=np.std(F64,axis=(0,1))
        print('F std64 again:',all_std)
        kill_64

    # moments for U*
    if 0: # must use double precision
        print('\ncross-check U* avr+std , compute avr again...',bigD['Uall'].shape)
        F64=bigD['Uall'].astype('float64')
        all_avr=np.mean(F64,axis=(0))
        print('U* avr64 again:',all_avr,F64.dtype)
        all_std=np.std(F64,axis=(0))
        print('U*std64 again:',all_std)
        kill_65

    return bigD
    

#...!...!..................
def repack_stimulus(stimName):
    # ================= open stimulus to compress it and save
    
    stimD=read_yaml(confD['inpPath']+args.cellName+"/stim.%s.yaml"%stimName)
    print('stimD:', sorted(stimD))
    stimD['timeAxis']=metaD['rawInfo']['timeAxis']
    stimA=stimD.pop('stimFunc')
    print('inp stim',stimA.shape, stimA.dtype)
    stimA=rebin_data1D(stimA,freqRebin)
    stimName+='_8kHz'
    print('out stim',stimA.shape, stimA.dtype)
    stimD['stimFunc']=stimA
    stimD['stimName']=stimName

    metaD['rawInfo']['stimName']=stimName

    stimF=confD['outPath']+"stim.%s.yaml"%(stimName)
    write_yaml(stimD,stimF)


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    confF=args.transformConfig+'.conf.yaml'
    confD=read_yaml(confF)
    #print(confD)
    #print(confD['probeNamePerCell'].keys())
    out_numFeature=len(confD['probeNamePerCell'][args.cellName])
    freqRebin=5  # rebin from 40kHz to 8kHz
    skimStr='%dprB8kHz'%out_numFeature  # 3 or 4 probes , has 'B' !
    #skimStr='%dpr8kHz'%out_numFeature  # 67 probes 
    out_name=args.cellName+'_excite.cellSpike_%s'%(skimStr)
    out_h5name=out_name+'.data.h5'

    metaF=confD['inpPath']+args.cellName+"/meta.cellSpike.yaml"
    metaD=read_yaml(metaF)
    #pprint(metaD)

    metaRaw=metaD['rawInfo']
    metaInp=metaD['dataInfo']
    metaPack={} # repacking info
    metaD['packInfo']=metaPack

    # - - - - not needed  - - -
    dropL=['seenBadFrames','totalGoodFrames','usedRawInpFiles','splitIdx','h5nameTemplate']
    for x in dropL:
        metaInp.pop(x)

    dropL2=['maxOutFiles','numFramesPerOutput','rawPath0','simuJobId','rawJobIdL']
    for x in dropL2:
        metaRaw.pop(x)


    # ---- new/changed  info added
    stimNameInp=metaRaw.pop('stimName')
    metaRaw['stimName']=stimNameInp+'_8kHz'

    metaRaw['timeAxis']['step']*=freqRebin
    # move some records
    for x in ['physRange','parName','numIcedPar']:
        metaRaw[x]=metaInp.pop(x)

    metaInp['numTimeBin']//=freqRebin
    metaInp['numFeature']=out_numFeature
    metaInp['featureName']=confD['goalFeatureName']
    metaInp['h5name']=out_h5name
    metaInp['featureType']='probes_'+skimStr
    metaInp['dataNormalized']=True
    metaInp['ustarTransform']=True

    metaPack['probeName']=confD['probeNamePerCell'][args.cellName]
    metaPack['conductName']=confD['conductName']
    metaInp['numPar']=len(metaPack['conductName'])

    if 0: #-----stim should be DONE once
        repack_stimulus(stimNameInp)
        exit(0)
    print('\n\nM:Start repacking frames for cell:',args.cellName)

    # ******* this takes the most of time *****
    bigD=repack_one_cell(confD,metaD,freqRebin,verb=1)

    totalFrameCount=bigD['Fall'].shape[0]
    frac=0.10
    nFrac=max(1,int(totalFrameCount*frac))
    splitSize={}
    splitSize['train']=totalFrameCount-2*nFrac
    splitSize['test']=nFrac
    splitSize['val']=nFrac
    print('splitFrames:',splitSize)

    metaInp['numDataFiles']=len(splitSize)
    metaInp['splitIdx']=splitSize  # this name is kept for backward compatibility 

    print('M:write big file...')
    outD={}

    for dom in [ 'test','val','train']:
        if dom=='test': # 1st block
            outD.update({dom+'_unitStar_par':bigD['Uall'][:nFrac], dom+'_phys_par':bigD['Pall'][:nFrac], dom+'_frames':bigD['Fall'][:nFrac]})
        if dom=='val': #2nd block
            outD.update({dom+'_unitStar_par':bigD['Uall'][nFrac:2*nFrac], dom+'_phys_par':bigD['Pall'][nFrac:2*nFrac], dom+'_frames':bigD['Fall'][nFrac:2*nFrac]})
        if dom=='train': # take the rest 8+ blocks
            outD.update({dom+'_unitStar_par':bigD['Uall'][2*nFrac:], dom+'_phys_par':bigD['Pall'][2*nFrac:], dom+'_frames':bigD['Fall'][2*nFrac:]})

    bigF=confD['outPath']+out_h5name
    write_data_hdf5(outD,bigF)

    metaF=confD['outPath']+out_name+".meta.yaml"
    write_yaml(metaD,metaF)

    print('DONE')
