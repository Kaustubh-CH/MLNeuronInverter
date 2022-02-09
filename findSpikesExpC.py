#!/usr/bin/env python3
'''
identify spikes in experimental wave forms
score wavforms by counting valid spikes

'''
from pprint import pprint

from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml, read_yaml
from toolbox.Util_Experiment import SpikeFinder
import os
import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/december/data8kHz/',help="formated data location")
    parser.add_argument("--dataName",  default='211219_5a', help="shortName for a set of routines ")

    parser.add_argument("--ampl",  default='0.14',type=str, help="(FS) amplitude ")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument( "--stimSpikes", action='store_true', 
        default=False,help="overwrites first waveform with scaled stim")

       
    args = parser.parse_args()
    args.formatName='expC.8kHz'
    args.save_tspike=True
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def score_me(waves,spiker): # waves must be flattened to 2D (if it has channels)
    N= waves.shape[0]
    #print('SCMin:',waves.shape)
    assert waves.ndim==2
    spikeTraitL2=[]
    mxSpk=0
    # self.spikes.append([tPeak,ampPeak,twidthFwhm,tBase,ampBase,twidthBase])
    for isw in range(N):
        nSpike=spiker.make(waves[isw])
        spikeA=np.array(spiker.spikes)
        print('SCM: waveform=%d of %d nSpike=%d '%(isw,N,nSpike),spikeA.shape)
        spikeTraitL2.append(spikeA)
        if mxSpk<nSpike: mxSpk=nSpike 
    return spikeTraitL2,mxSpk
        

#...!...!..................
def pack_scores(spikeTraitL2,mxSpk,bigD,ntr=6):
    #...... pack scores & spikeTraits  as np-arrays 
    nsw=len(spikeTraitL2)
    totSpikes=0
    nSweep=0
    spCnt=np.zeros(nsw,dtype=np.int32)
    spTrt=np.zeros((nsw,mxSpk,ntr),dtype=np.float32)
    #print('PSC:',spCnt.shape,spTrt.shape)
    for isw in range(nsw): # loop over sweeps
        spikeL=spikeTraitL2[isw]
        m=len(spikeL)
        if m==0: continue
        nSweep+=1
        totSpikes+=m
        spCnt[isw]=m
        spTrt[isw,:m]=spikeL
    
    bigD['spikeCount']=spCnt
    bigD['spikeTrait']=spTrt
    
    return totSpikes,nSweep,mxSpk
    
#...!...!..................
def M_build_metaData():
    spikeUnits=[['tPeak','ms'],['ampPeak','mV'],['twidthFwhm','ms'],['tBase','ms'],['ampBase','mV'],['twidthBase','ms']]
    swtrait=exp_info['sweep_trait'][i0:i1]
        
    ac={'spikeUnits':spikeUnits,'totSpikes':totSpikes,'maxSpikes':maxSpikes,'totWaves':len(spikeTraitL2)}
    ac['exp_log']=expMD['expInfo']['exp_log']
    for k in ['stimName','numTimeBin']: ac[k]=expMD[k]
    if  args.stimSpikes:
        ac['shortName']='%s-stim'%(args.dataName)
    else:
        ac['shortName']='%s-A%s'%(args.dataName,args.ampl)
    ac['spikerConf']=spiker.conf
    ac['stimAmpl']=float(args.ampl)
    ac['sweepTrait']=swtrait
    ac['expUnits']=expMD['units'] # sweepId,timeLive,stimAmpl
    ac['has_stim_spikes']=args.stimSpikes
        
    return ac

#...!...!..................
def M_save_tspike(): # for George, binned time of spikes
    print('M_save_tspike:')
    timeV=bigD['time']
    tstep=timeV[2]-timeV[1]
    spikeC=bigD['spikeCount']
    spikeT=bigD['spikeTrait']
    print('spikeC',spikeC,'tstep=',tstep)
    tBase=np.rint(spikeT[...,3]/tstep).astype(np.uint16)
    twidthBase=np.rint(spikeT[...,5]/tstep).astype(np.uint16)
    print('tBase\n',tBase.shape,tBase)
    print('twidthBase\n',twidthBase.shape,twidthBase)
    tHigh=tBase
    tLow=tBase+twidthBase
    wave_spikesUD=np.stack( ( tHigh,tLow),axis=-1)
    print('wave_spikesUD',wave_spikesUD.shape)
    pprint(wave_spikesUD[0,:,0])
    pprint(wave_spikesUD[0,:,1])
    name='soma_spikesUD'
    if args.stimSpikes: name='stim_spikesUD'
    bigD[name]=wave_spikesUD
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    inpF=os.path.join(args.dataPath,'%s.%s.h5'%(args.dataName,args.formatName))
    bigD,expMD=read3_data_hdf5(inpF, verb=1)
    pprint(expMD)

    #k=list(exp_info['map_idx'])
    #print('k',k,args.ampl,type(args.ampl))
    exp_info=expMD['expInfo']
    i0,i1=exp_info['map_idx'][args.ampl]
    if args.stimSpikes:  i1=i0+1 # for stim-spikes
    
    nSweep=i1-i0

    timeV=bigD['time']
    waves=bigD['soma_wave'][i0:i1]  # score also empty waveforms
    stims=bigD['stim_wave'][i0:i1]

    if args.stimSpikes:
        print('M:WARN - 1st wavefor overwritten by stim')
        waves[0]=stims[0]*70-40.
        waves[0][-1]=waves[0][-2]  # hack for NaN - check next time better:  numpy.isnan(a).any()
    
    spiker=SpikeFinder(timeV,verb=args.verb)    
    spikeTraitL2,mxSpk=score_me(waves,spiker)
    
    totSpikes,totSweep,maxSpikes=pack_scores(spikeTraitL2,mxSpk,bigD)

    # save results
    bigD['soma_wave']=waves
    bigD['stim_wave']=stims

    spkMD=M_build_metaData()
    if args.save_tspike: M_save_tspike()
    
    outF=os.path.join(args.outPath,'%s.spiker.h5'%(spkMD['shortName']))
    write3_data_hdf5(bigD,outF,metaD=spkMD)
    print('M:done %s totSpikes=%d totSpikySweep=%d'%(args.dataName,totSpikes,totSweep))
    pprint(spkMD)
