#!/usr/bin/env python3
'''
 formats small hd5-inputs from individual NEURON simulations for bbp153 with some choice of stim amplitude and hodlingCurrent
Uses advanced hd5 storage, includes meta-data in hd5

'''
#import sys,os
from pprint import pprint
import numpy as np

from toolbox.Util_H5io3  import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_Experiment import rebin_data1D 

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--rawPath", default='/global/homes/b/balewski/prjn/2021-roys-simulation/simRaw40kHz/',help="input  raw data path for experiments")
    parser.add_argument("--dataPath", default='/global/homes/b/balewski/prjn/2021-roys-simulation/sim8kHz/',help="output path  rebinned Waveforms  ")

    parser.add_argument("--cellName",  default='bbp153', help="cell shortName ")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................

inpConf={'bbpName':'L5_TTPC1_cADpyr','shortName':'bbp153'}
# hopefully generic
inpConf['holdCurr']=[0.00, 0.05, 0.20, 0.35 ]
inpConf['stimAmpl']=[float('%.2f'%(0.05+i*0.05)) for i in range(0,40)]
#inpConf['stimAmpl']=[float('%.2f'%(0.05+i*0.05)) for i in range(0,24)]  # amp<=1.2

inpConf['units']={'stimAmpl':'FS','holdCurr':'nA','waveform':'mV','time':'ms'}
inpConf['probe']=['soma']
inpConf['stimName']='chaotic_2'
                 
#...!...!..................
def read_one(ic,ia):
    inpF='%s/%s_factor_%.2f_add%.2f.h5'%(args.cellName,inpConf['bbpName'],inpConf['stimAmpl'][ia],inpConf['holdCurr'][ic])
    print('RO:',ic,ia,inpF)
    blob,_=read3_data_hdf5(args.rawPath+inpF)
    stimA=blob['stim']
    timeA=blob['time']
    stimName='chaotic_2'
    soma=blob['soma_Vs'][:-1]
    #axon=blob['axon_Vs'][:-1]
    #dend=blob['dend_Vs'][:-1]
    numTimeBin=stimA.shape[0]
    print(stimName,'numTimeBin=',numTimeBin,'soma:',soma.shape)
    return soma,stimA,timeA
    
   
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    pprint(inpConf)
    assert args.cellName==inpConf['shortName']
    numHold=len(inpConf['holdCurr'])
    numAmpl=len(inpConf['stimAmpl'])
    
    numTimeRaw=8000
    nRebin=5
    numTime=numTimeRaw//nRebin
    #numProb=len(inpConf['probes'])
  
    #create common stim container 
    waveforms=np.zeros((numHold,numAmpl,numTime)).astype('float32')
    stims=np.zeros((numHold,numAmpl,numTime)).astype('float32')
    print('M:wafeforms',waveforms.shape,', stims:',stims.shape)

    bigD={'waveform':waveforms,'stim':stims}
    nok=0
    for ic in range(numHold):
        for ia in range(numAmpl):
            wave40kHz,stim,time=read_one(ic,ia)
            if ic==0 and ia==0:
                timeB=rebin_data1D(time,nRebin).astype('float32')
                timeB-=timeB[0] # make time start at 0
                print('time (ms) rebin:',time.shape, timeB.shape,timeB[:4],timeB[-4:])
                #print('timeA (ms) ',time.shape, time[:2],time[-2:])
                bigD['time']=timeB
            # populate bigData
            stims[ic,ia]=rebin_data1D(stim,nRebin)
            # store  waveforms ...
            nok+=1
            waveforms[ic,ia]=rebin_data1D(wave40kHz,nRebin)
            #break
    print('M:nok=',nok)
    metaD={}
    for x in inpConf:
        metaD[x]=inpConf[x]
    metaD.update({'numTimeBin':numTime,'numHoldCurr':numHold,'numStimAmpl':numAmpl,'nRebin':nRebin,'sampling': '8kHz','formatName': 'simB.8kHz'})

    print('M:MD');pprint(metaD)
    
    outF='%s.%s.h5'%(args.cellName,metaD['formatName'])

    write3_data_hdf5(bigD,args.dataPath+outF,metaD=metaD)
    print('M:done   ')
