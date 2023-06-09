#!/usr/bin/env python3
'''
select subset of experimental waveforms and re-pack them for ML-predictions 

'''
from pprint import pprint

from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from fftRawExpB import Plotter

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/june/data8kHz/',help="formated data location")

    parser.add_argument("--dataName",  default='210611_3_NI', help="shortName for a set of routines ")
    parser.add_argument("--amplIdx",  default=6,type=int, help="amplitude index")
    parser.add_argument("--numPredConduct",  default=15,type=int, help="number of predicted conductances")
    parser.add_argument("--formatName",  default='expB.8kHz', help="data name extesion maps to sampling rate ")

    parser.add_argument("-o","--outPath", default='exp4ml/',help="output path for plots and tables")
    
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
    waves2D=bigD['waveform'] # score also empty waveforms
    stim2D=bigD['stim'] 
    trait2D=bigD['sweepTrait']
     
    # select amplitude row
    ampl=ampls[args.amplIdx]
    nSweep=bigD['sweepCnt'][args.amplIdx]
    waves=waves2D[args.amplIdx][:nSweep]
    waves=np.expand_dims(waves,2) # add 1-channel index for ML compatibility
    sweepTrait=trait2D[args.amplIdx][:nSweep]
    stims=stim2D[args.amplIdx][:nSweep]  # for future plotting 
    shortName='%s-a%.2f'%(args.dataName,ampl)
    # do NOT normalize waveforms - Dataloader uses per-wavform normalization

    if 0: # skip some waveforms : cherry picking for proposal 2021-10
        #dropL=[2,5,6]  #  'a' = no drug
        #useL=[ i for i in range(waves.shape[0])]
        #[useL.remove(x) for x in dropL ]
        useL= [0, 1, 3, 4, 7, 8, 9]  #  'a' = no drug
        #useL= [1,4] # [3,5] #  'c' = 0.5umol drug
        #useL=[ 0,1, 2,3 , 4,5 ,6,7, 8,9,12,13] #  'd' = 1.0umol drug
        print('ww-in',waves.shape)        
        print('useL=',useL)
        waves=waves[useL]
        shortName+='s'
        if 1:
            nSweep=1
            sweepTrait=sweepTrait[:1]
            stims=stims[:1]
            waves=np.mean(waves,axis=0)
            waves=np.expand_dims(waves,0) 
        print('ww',waves.shape)
        
    
    #add fake Y
    unitStar=np.zeros((nSweep,args.numPredConduct))
    print('use ampl=',ampl,nSweep,waves.shape,unitStar.shape)

    #... assemble output meta-data
    keyL=['formatName','numTimeBin','rawDataPath','sampling','stimName']
    outMD={ k:inpMD[k] for k in keyL}
    outMD['stimAmpl']=float(ampl)
    outMD['normWaveform']=True
    outMD['numSweep']=int(nSweep)
    outMD['comment']='added fake unitStar_par for consistency with simulations'
    outMD['shortName']=shortName
    
    # ... wrap it up
    outF=shortName+'.h5'
    bigD={'exper_frames':waves,'exper_unitStar_par':unitStar,'time':timeV,'stims':stims}
    bigD['sweep_trait']=sweepTrait
    bigD['stims']=stims
    write3_data_hdf5(bigD,args.outPath+outF,metaD=outMD)
    print('M:done,\n outMD:')
    pprint(outMD)

    
    # - - - - - PLOTTER - - - - -
    args.noXterm=0
    args.formatVenue='prod'
    args.prjName='repackB'
    plot=Plotter(args)
    plDD={}
    plDD['shortName']=shortName
    plDD['timeV']=timeV
    plDD['timeLR']=[0.,200]  # (ms)  time range 
    plot.waveform(waves,plDD)
    plot.display_all()
    
