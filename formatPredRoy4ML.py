#!/usr/bin/env python3
'''
 2nd iteration of predictions
  Takes Roy's waveforms geneerated for my predictions
  format them to allow ML predictions again
Output has fixed name:  pred2/roy-pred-Nov12a.h5

To make predictions do:
./predict_exp.py  --modelPath /global/homes/b/balewski/prjn/2021-ml-opt_4roys-simu/practice50c_1pr_expF2us/out  --dataPath  pred2 --outPath pred2 --dataName roy-pred-Nov12a 

see:
https://docs.google.com/document/d/1Bq30vIQWP831yR2ChSbhLqOaLaeAfQxYWJZXQV4TVzg/edit?usp=sharing 


'''
from pprint import pprint

import sys,os
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from toolbox.Util_Experiment import SpikeFinder
from toolbox.Util_Experiment import rebin_data1D

import numpy as np
import argparse

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    dataPath='pred2'
    outPath=dataPath
    inpF='210611_3_NI-a0.17_j329867.mlPred._neurPred_L5TTPC2_1.h5'
    bigD,inpMD=read3_data_hdf5( os.path.join(dataPath,inpF))
    #pprint(inpMD)
    
    timeV=bigD['time']
    conds=bigD['pred_cond']
    #ampls=inpMD['stimAmpl']
    #currs=inpMD['holdCurr']
    waves40kHz=bigD['pred_vs']
    stims=bigD['stims'] 

    print('waves 40kHz:',waves40kHz.shape)
    numTimeRaw=waves40kHz.shape[-1]
    nRebin=5
    numTime=numTimeRaw//nRebin
    numSamp=waves40kHz.shape[1]
    waves=np.zeros((numSamp,numTime)).astype('float32')
    for i in range(numSamp):
        waves[i]=rebin_data1D(waves40kHz[0,i],nRebin)
     
    print('waves 8 kHz:',waves.shape)
    nSweep=waves.shape[0]
    
    waves=np.expand_dims(waves,2) # add 1-channel index for ML compatibility
    
    # do NOT normalize waveforms - Dataloader uses per-wavform normalization
        
    #add fake Y
    unitStar=np.zeros_like(conds)
    print('M:aa ',nSweep,waves.shape,unitStar.shape)

    #... assemble output meta-data
    #keyL=['formatName','numTimeBin','sampling','shortName','stimName']
    outMD={'numTimeBin':numTime,'nRebin':nRebin,'sampling': '8kHz','formatName': 'simB.8kHz'}
    #outMD['stimAmpl']=float(ampl)
    outMD['normWaveform']=True
    #outMD['holdCurr']=currs
    #outMD['numSweep']=int(nSweep)
    outMD['comment1']='Roys waveforms geneerated for my predictions'

    
    # ... wrap it up
    outF='roy-pred-Nov12a.h5'
    bigD={'exper_frames':waves,'exper_unitStar_par':unitStar,'time':timeV,'stims':stims}
    
    bigD['stims']=stims
    write3_data_hdf5(bigD, os.path.join(outPath,outF),metaD=outMD)
    print('M:done,\n outMD:')
    pprint(outMD)
