__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# those functions ar needed only for data repacking in the format stage

import numpy as np
import random
import time,  os
import h5py
from pprint import pprint

from toolbox.Util_IOfunc import write_data_hdf5

#...!...!..................
def save_packed_data(framesL,unitXL, physXL,k,doShuffle,args,varParL):
    frames=np.concatenate(framesL,axis=0)
    unitX=np.concatenate(unitXL,axis=0)
    physX=np.concatenate(physXL,axis=0)
    #print('qqq',frames.shape,unitX.shape)
    assert frames.shape[0]==unitX.shape[0]
        
    nPar=len(varParL)
    assert unitX.shape[1]==nPar

    for X in [frames, unitX,physX ]:
        numNaN= np.sum(np.isnan(X))
        #print('see NaN=%d  : '%numNaN,X.shape)
        if numNaN>0:
            #print(frames[17,3000:3100,0])
            print('NaN=%d detected, aborting for : '%numNaN,X.shape)
            if len(X.shape)==2: print(X[1])
        assert numNaN ==0

    # synchronized shuffle of frames within each output file
    if doShuffle:
        idxL=[i for i in range(frames.shape[0]) ]
        random.shuffle(idxL)
        frames=frames[idxL]
        unitX=unitX[idxL]
        physX=physX[idxL]
        
    dataD={'frames':frames,'unit_par':unitX,'phys_par':physX}
    outF=args.dataPath+args.targetName+'_%d.h5'%k
    write_data_hdf5(dataD,outF, k%10==0 )
    return k+1,frames.shape[0]

#...!...!..................
def add_digital_noise(fV,digiNoise):
    #print('addDGN:',digiNoise)
    fVn = ((digiNoise[0] * fV) + np.random.normal(0, digiNoise[1])).astype(int) / digiNoise[0]
    return fVn

        
#...!...!..................
def read_one_raw_data(fname,conf):
    h5f = h5py.File(fname, 'r')
    frames=np.array(h5f['voltages'])    
    unitX=np.array(h5f['norm_par'])
    physX=np.array(h5f['phys_par'])
    qaA=np.array(h5f['binQA'])
    h5f.close()

    if 'num_probes' in conf:
        #print('qq1',frames.shape)
        frames=frames[...,:conf['num_probes']]
        #print('qq2',frames.shape)

    if conf['useQA']:
        # apply trace QA on 
        nBad=np.sum(qaA==0)
        frames=frames[qaA==1]
        unitX=unitX[qaA==1]
        physX=physX[qaA==1]
    else:
        nBad=0

    # apply data pre-processing
    # Note, the order of operations does matter!
    # add digital noise procured by Harry, expects mV scale

    if conf['digiNoise']!='None':
        frames=add_digital_noise(frames,conf['digiNoise'])

    # catch NaN or Inf
    for xx in [frames, unitX, physX]:
        numFin= np.sum(np.isfinite(xx))
        assert numFin== np.prod(xx.shape)
    
    return frames,unitX,physX,nBad


#...!...!..................
def read_one_raw_metadata(fname,varParL):        
    h5f = h5py.File(fname, 'r')
    stimul=np.array(h5f['stim'])
    physRangeAll=np.array(h5f['phys_par_range'])
    #print('physRangeAll shape=',physRangeAll.shape)
    h5f.close()
    nPar=len(varParL)
    assert physRangeAll.shape[0]==nPar
    outL=[]
    # repack param ranges and names
    nNeg=0
    nDyn=0
    for i in range(nPar):
        parRange=physRangeAll[i].tolist()
        name=varParL[i]
        outL.append([name]+parRange)
        if 'const' in name: continue
        if 'fixed' in name: continue
        nDyn+=1 # params which we actually varied
        if  min(parRange) <0 :
            print('FATAL, see negative ', name,i,parRange)
            nNeg+=1

    if nNeg >0 :
        print('see %d negative ranges for used phys params, ABORT'%nNeg)
        exit(99)
    return outL,stimul,nPar-nDyn
   

#...!...!..................
def agregate_raw_data(args,rawMeta):
        # unpack rawMeta
        print(rawMeta)
        
        rawPath0=rawMeta['rawPath0']+'/'
        rawJobIdL=rawMeta['rawJobIdL']
        bbpName=rawMeta['bbpName']
        doShuffle=rawMeta['shuffle']
        rawDataName=rawMeta['rawDataName']
        maxOutFiles=rawMeta['maxOutFiles']
        varParL=rawMeta['varParL']
        
        # use a subset of  probes
        numProbes=rawMeta['num_probes']
        if numProbes < len(rawMeta['probeName']):
            print('\nfrm: clip probes from %d to %d , '%(len(rawMeta['probeName']),numProbes),rawMeta['probeName'])
            rawMeta['probeName']=rawMeta['probeName'][:numProbes]
        
        print("\nRoy's hd5 path0=%s rawDataName=%s, pathJobs="%(rawPath0,rawDataName),rawJobIdL,'bbpName=',bbpName)
        print('varPar :',varParL,'\n used numProbes=',numProbes)
        
        numFilesOut=0
        totBad=0; totGood=0
        assert '*' in rawDataName
        fnameTmpl=rawDataName.split('*')[0]

        # adjust for Adytia jobs
        # edit template: move cloneID to 
        xL=fnameTmpl.split('-')
        xL[0]=xL[0][:-1]+'1'
        fnameTmpl='-'.join(xL)
        mark2='%d.h5'%rawMeta['cloneId'] 
        
        shardL=[]
        for jid in rawJobIdL:
            name2=bbpName[:-1]+'1/c%d'%rawMeta['cloneId'] # adjust for Adytia jobs
            path1='%s/%s/'%(jid,name2)
            path2=rawPath0+path1
            if not os.path.exists(path2):
                print('Warn, skip missing data path:',path2)
                continue
            
            allL=os.listdir(path2)
            print('get_glob_info len',len(allL),jid)
            #print('XX',allL)
            allL=sorted(allL) # to easier find bugs while reading
            for x in allL:
                if mark2 not in x: continue
                #print('ww',fnameTmpl , x)
                assert fnameTmpl in x
                shardL.append(path1+x)
                    
        print('end-shardL  len=',len(shardL))#,shardL[0],shardL[-1])

        if doShuffle:
            random.shuffle(shardL) # works in place 

        nInpF=0
        accFrames=0
        nGood=0
        # .......... main loop over raw input files 
        for shard in shardL:
            if accFrames>= rawMeta['numFramesPerOutput'] or accFrames==0:
                 if  accFrames>0 :
                     numFilesOut,nGood=save_packed_data(framesOut,unitXOut, physXOut,numFilesOut,doShuffle,args,varParL)
                     totGood+=nGood
                 framesOut=[] # tmp storage for writing 1 output
                 unitXOut=[]
                 physXOut=[]
                 accFrames=0
                 if maxOutFiles!=None:
                     if numFilesOut >=maxOutFiles: break
                
            fname=rawPath0+shard
            if nInpF==0 :
                print('try 1st shard:',fname)
            if not os.path.exists(fname): 
                if accFrames==0 : print('not found:',fname)
                continue
            if nInpF%200==0 or nInpF<6: print(nInpF,'=shards --------START\n',fname)

            try:
            #if 1:
                frames,unitX,physX,nBad=read_one_raw_data(fname,rawMeta)
            except:
                print('SKIP corrupted :',fname)
                continue
            
            nInpF+=1
            assert numProbes==frames.shape[2]
            framesOut.append(frames)
            unitXOut.append(unitX)
            physXOut.append(physX)
            if accFrames==0:  # it is 1st valid input
                physRangeL,stimul,nIcedParam=read_one_raw_metadata(fname,varParL)
            accFrames+=frames.shape[0]
            totBad+=nBad

        # . . . . . .   end of main loop . . . . . . .

        # skip last if too short
        if accFrames>= rawMeta['numFramesPerOutput']:
            numFilesOut,nGood=save_packed_data(framesOut,unitXOut, physXOut,numFilesOut,doShuffle,args,varParL) 
            totGood+=nGood
        print('P:total saved %d files'%numFilesOut)
        assert nGood>0
        print('\nsaved %d files, frames: totGood=%d, totBad=%d, badFract=%.3g'%(numFilesOut,totGood,totBad, totBad/totGood))

        # randomize order of output files order        
        allL=[i for i in range(numFilesOut)]
        if doShuffle:
            random.shuffle(allL) # works in place 
        frac=0.10
        nFrac=max(1,int(numFilesOut*frac))

        splitIdxD={'test': allL[:nFrac]}
        splitIdxD['val']=allL[nFrac:2*nFrac]
        splitIdxD['train']=allL[2*nFrac:]
                            
        print('  achieved split for  %d files to: '%numFilesOut, [ (sg,len(splitIdxD[sg])) for sg in splitIdxD ])

        # construct metaD
        outD={}
        outD['physRange']= physRangeL
        outD['parName']=varParL
        outD['numPar']=len(varParL)
        outD['numIcedPar']=nIcedParam
        
        outD['h5nameTemplate']=args.targetName+'_*.h5'
        outD['numDataFiles']=numFilesOut
        outD['splitIdx']=splitIdxD
        outD['totalGoodFrames']=totGood
        outD['seenBadFrames']=totBad
        outD['usedRawInpFiles']=nInpF
        outD['numTimeBin']=frames.shape[1]
        metaD={'rawInfo':rawMeta,'dataInfo':outD}

        #pprint(outD)
        return metaD,stimul
