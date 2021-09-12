#!/usr/bin/env python3
'''
identify spikes in experimental wave forms
score wavforms by counting valid spikes

'''
from pprint import pprint

#import sys,os
#sys.path.append(os.path.abspath("../"))
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from toolbox.Util_Experiment import SpikeFinder

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/june/data8kHz/',help="formated data location")

    parser.add_argument("--dataName",  default='210611_3_NI', help="shortName for a set of routines ")
    parser.add_argument("--formatName",  default='expB.8kHz', help="data name extesion maps to sampling rate ")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!..................
def pack_scores(spikeTraitL2,mxSpk, inp_shape,bigD,traitUnits):
    #...... pack scores & spikeTraits  as np-arrays 
    #bigD['score']=np.array(scoreL,dtype=np.float32).reshape(inp_shape)
    nsw=len(spikeTraitL2)
    totSpikes=0
    nSweep=0
    ntr=len(traitUnits)
    spCnt=np.zeros(nsw,dtype=np.int32)
    spTrt=np.zeros((nsw,mxSpk,ntr),dtype=np.float32)
    for isw in range(nsw): # loop over sweeps
        spikeL=spikeTraitL2[isw]
        m=len(spikeL)
        if m==0: continue
        nSweep+=1
        totSpikes+=m
        spCnt[isw]=m
        spTrt[isw,:m]=spikeL
    
    bigD['spikeCount']=spCnt.reshape(inp_shape)
    bigD['spikeTrait']=spTrt.reshape(inp_shape+(mxSpk,ntr))
    
    return totSpikes,nSweep,mxSpk
    
#...!...!..................
def score_me(waves,spiker):
    N= waves.shape[0]
    assert waves.ndim==2
    spikeTraitL2=[]
    mxSpk=0
    for isw in range(N):
        nSpike=spiker.make(waves[isw])
        spikeA=np.array(spiker.spikes)
        print('waveform=%d of %d nSpike=%d '%(isw,N,nSpike))
        spikeTraitL2.append(spikeA)
        if mxSpk<nSpike: mxSpk=nSpike 
    return spikeTraitL2,mxSpk
        

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,inpMD=read3_data_hdf5(args.dataPath+inpF, verb=1)
    
    pprint(inpMD)

    timeV=bigD['time']
    ampls=inpMD['stimAmpl']
    
    spiker=SpikeFinder(timeV,verb=args.verb)    
    waves2D=bigD['waveform'] # score also empty waveforms
    #waves2D=waves2D[4:5] ; print('aa',ampls[4:5])
    #waves2D=waves2D[6:7,:1]  # pick 1 good waveform, for debug
    inp_shape=tuple(waves2D.shape[:-1])
    numTimeBin=inpMD['numTimeBin']
    
    print('inp measurement shape:',inp_shape)
    spikeTraitL2,mxSpk=score_me(waves2D.reshape(-1,numTimeBin),spiker)
    traitUnits=[['tPeak','ms'],['yPeak','mV'],['twidth','ms'],['y_twidth','mV'],['twidth_at_base','ms']]
    
    totSpikes,totSweep,maxSpikes=pack_scores(spikeTraitL2,mxSpk, inp_shape,bigD,traitUnits)

    # save results
    inpMD['spikes']={'traits':traitUnits,'totSpikes':totSpikes,'maxSpikes':maxSpikes,'totWaves':len(spikeTraitL2)}
    inpMD['spikerConf']=spiker.conf
    
    outF=inpF.replace(args.formatName,'spikerSum')
    write3_data_hdf5(bigD,args.outPath+outF,metaD=inpMD)
    print('M:done %s totSpikes=%d totSpikySweep=%d'%(args.dataName,totSpikes,totSweep))
    #1pprint(inpMD)
