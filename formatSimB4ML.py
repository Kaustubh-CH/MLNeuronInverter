#!/usr/bin/env python3
'''
select subset of simulated waveforms and re-pack them for ML-predictions 

'''
from pprint import pprint

import sys,os
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from toolbox.Util_Experiment import SpikeFinder

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-simulation/sim8kHz/',help="formated data location")

    parser.add_argument("--dataName",  default='bbp153', help="shortName for a set of routines ")
    
    parser.add_argument("--amplIdx",  default=19,type=int, help="amplitude index")
    parser.add_argument("--numPredConduct",  default=15,type=int, help="number of predicted conductances")
    parser.add_argument("--holdCurrIdx",  default=None,type=int, help="(optional) holdingcurrent index")
    parser.add_argument("--formatName",  default='simB.8kHz', help="data name extesion maps to sampling rate ")
    parser.add_argument("--comment3",  default=None, help="additional info stored in meta-data")

    parser.add_argument("-o","--outPath", default='sim4ml/',help="output path for plots and tables")
    
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


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
    currs=inpMD['holdCurr']
    waves2D=bigD['waveform'] # score also empty waveforms
    stim2D=bigD['stim'] 
    #1trait2D=bigD['sweepTrait']

    print('waves2D:',waves2D.shape)
    # select amplitude row
    ampl=ampls[args.amplIdx]    
    waves=waves2D[:,args.amplIdx]
    print('waves1D:',waves.shape)
    nSweep=waves.shape[0]
    
    if args.holdCurrIdx!=None:
        hcIdx=args.holdCurrIdx
        assert hcIdx>=0 and hcIdx < nSweep
        waves=waves[hcIdx:hcIdx+1]
        print('waves0D:',waves.shape)
        currs=currs[hcIdx:hcIdx+1]
        nSweep=1    

    waves=np.expand_dims(waves,2) # add 1-channel index for ML compatibility
    #1sweepTrait=trait2D[args.amplIdx][:nSweep]
    stims=stim2D[:,args.amplIdx] # for future plotting 
    
    # do NOT normalize waveforms - Dataloader uses per-wavform normalization

    if 0:  # degrade resolution to match how Vyassa waveorms were processed
        X=waves*150
        X=X.astype(np.int16)
        X=X/150.
        waves=X.astype(np.float32)
        
    #add fake Y
    unitStar=np.zeros((nSweep,args.numPredConduct))
    print('use ampl=',ampl,nSweep,waves.shape,unitStar.shape)

    #... assemble output meta-data
    keyL=['formatName','numTimeBin','sampling','shortName','stimName']
    outMD={ k:inpMD[k] for k in keyL}
    outMD['stimAmpl']=float(ampl)
    outMD['normWaveform']=True
    outMD['holdCurr']=currs
    outMD['numSweep']=int(nSweep)
    outMD['comment1']='added fake unitStar_par for consistency with simulations'
    outMD['comment2']='simulated data produced by Roy in June 2021'
    if args.comment3!=None:
        outMD['comment3']=args.comment3
    
    # ... wrap it up
    outF='%s-a%.2f.h5'%(args.dataName,ampl)
    if  args.holdCurrIdx!=None:
        outF=outF.replace('.h5','-c%.2f.h5'%(currs[0]))
    bigD={'exper_frames':waves,'exper_unitStar_par':unitStar,'time':timeV,'stims':stims}
    #1bigD['sweep_trait']=sweepTrait
    bigD['stims']=stims
    write3_data_hdf5(bigD,args.outPath+outF,metaD=outMD)
    print('M:done,\n outMD:')
    pprint(outMD)
