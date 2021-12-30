#!/usr/bin/env python3
'''
agregate and format Roy's experimental data for one cell
store waveforms and meta-data in hd5

sort file name by routin-id first

'''
import sys,os
from pprint import pprint
import numpy as np

from toolbox.Util_H5io3  import  write3_data_hdf5, read3_data_hdf5
from toolbox.Util_Experiment import rebin_data1D ,rebin_data2D 

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--rawPath",
                        default='/global/homes/r/roybens/fromMac/neuron_wrk/cori_mount/',
                        help="input  raw data path for experiments")
    parser.add_argument("--dataPath", default='/global/homes/b/balewski/prjn/2021-roys-experiment/december/data8kHz/',help="output path  rebinned Waveforms  ")
    parser.add_argument('-c',"--cellName", type=str, default='211219_5', help=" [_analyzed.h5] raw measurement file")

    parser.add_argument("--saveNameExt",  default='x', help="optional output h5 output extension ")
    parser.add_argument("--routineRange", default=None,nargs="+",type=int,help="bracket used data, inclusive [a,b]",required=True)
    #1parser.add_argument("--initWallTime", default=0,type=int,help="optional offset in seconds")
    
    args = parser.parse_args()
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if args.routineRange!=None: # skip some data
        assert  len(args.routineRange)==2
        assert args.routineRange[0]<=args.routineRange[1]
    return args

#...!...!..................
def find_exp_log(expLog,ir):
    rec0=None
    for i,rec in enumerate(expLog):
        if rec['rtn_id_start'] >ir: return rec0,i-1
        rec0=rec
    return rec0,i
#...!...!..................
def survey_routines(bigRaw,rawMD, rr):  
    tmpD={}
    cnt=0
    for ir in range(rr[0],rr[1]+1):
        stimAmpl=float(bigRaw['stim_ampl'][ir])
        print('ir:',ir,stimAmpl)
        if stimAmpl not in tmpD: tmpD[stimAmpl]=[]
        tmpD[stimAmpl].append(ir)
        cnt+=1

    amps=sorted(tmpD)
    #print('SRV:amps=',amps)
    routs=[]
    for a in amps:
        routs.append( sorted(tmpD[a]))
    #print('SRV:asweeps=',sweeps)
    exp_info={'amps':amps,'routines':routs}

    # find exp setup for routines
    expLog=rawMD['exp_log']
    rec,ir=find_exp_log(expLog,rr[0])
    rec1,ir1=find_exp_log(expLog,rr[1])
    assert ir==ir1
    print('rec=',rec)
    exp_info['exp_log']=rec
    
    return exp_info

#...!...!..................
def extract_routines(exp_info,bulk):
    
    # 1-dim
    timeA=bulk['stim_time']*1000 # in ms
    #3-dim:
    waveA=bulk['Vs']*1000   # in mV
    waveFA=bulk['60HzFilteredVs']*1000   # in mV
    stimA=bulk['stim_waveform']*1e9  # nA

    # rebin + interpolate
    # A: time axis
    nReb=6
    timeB=rebin_data1D(timeA,nReb,clip=True)
    #print('reb timeA,B',timeA.shape, timeB.shape, timeA[-1])
    numTimeBin=1600 # 200ms @ 8 kHz
    stepC=timeA[-1] /numTimeBin
    # 'C' is new target time axis
    timeC=np.linspace(0,numTimeBin, num=numTimeBin,endpoint=False)*stepC
    #print('reb timeC',timeC.shape,timeC[-1],stepC)

    lastTime=0
    mapD={}
    widx0=0
    waveCL=[];stimCL=[]
    sweepTrait=[]  # [sweepId, sweepTime, stimAmplFS]
    for a,rs in zip(exp_info['amps'],exp_info['routines']):
        print('ER:',a,rs,type(a))
        for ir in rs:
            #  waveforms 
            outB=rebin_data2D(waveA[ir],nReb,clip=True)
            outC=[np.interp(timeC,timeB,outX) for outX in outB ]
            waveCL+=outC
            #print('waveCL[0]',waveCL[0].shape)
            
            #  stim-waveforms (2x the same)
            outB=rebin_data1D(stimA[ir],nReb,clip=True)
            outC=np.interp(timeC,timeB,outB)
            stimCL+=[outC,outC]
            # auxil info for each sweep
            swt=float(bulk['time_from_start'][ir])
            sweepTrait.append([ir+0.1,swt,a])
            swt+=sweepDurationInRoutine
            sweepTrait.append([ir+0.2,swt,a])
            if lastTime<swt: lastTime=swt

        widx1= len(waveCL)   
        #print(a,'nwave=',widx1)
        mapD[a]=[widx0,widx1]
        widx0=widx1
    print(mapD)
    exp_info['map_idx']=mapD
    exp_info['sweep_trait']=sweepTrait
    
    # pack np-arrays
    bigD={}
    bigD['soma_wave']=np.array(waveCL,dtype=np.float32)
    bigD['stim_wave']=np.array(stimCL,dtype=np.float32)
    bigD['time']=np.array(timeC,dtype=np.float32)
    
    return bigD,lastTime

#...!...!..................
def QA_one(wave1):
    avrPreMemBias=np.mean(wave1[:50])
    if avrPreMemBias > -30:
        print('JSPM corruped waveform, skip it')
        print('wave1',avrPreMemBias,wave1[:50])
        bad55
    return 0



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    args.prjName='formExpC'
    numSweepPerRoutine=2  # each routine has 2 sweeps 
    sweepDurationInRoutine=3 # Roy: 3 sec separation between sweeps

    inpF=os.path.join(args.rawPath,args.cellName+'_analyzed.h5')
    print('M:rawPath=',args.rawPath)
    bigRaw,rawMD=read3_data_hdf5(inpF)

    exp_info=survey_routines(bigRaw,rawMD,args.routineRange)
    print('M:exp_info'); pprint(exp_info)
    
    bigData,lastTime=extract_routines(exp_info,bigRaw) # final HD5-stoarge
    #surveyCnt['lastRoutineTime']=int(lastTime)   
    
    if 0:
        print('M:sweepCnt',bigData['sweepCnt'])
        #print(bigData['waveform'][:,:,:5])

    
    # - - - - -  assemble meta data - - - - - -
    saveF=args.cellName+args.saveNameExt
    metaD={}
    metaD['rawDataPath']=args.rawPath
    metaD['shortName']=saveF
    metaD['stimName']='chaotic_2'
    metaD['numStimAmpl']=len(exp_info['amps'])
    metaD['numTimeBin']=bigData['soma_wave'].shape[1]
    metaD['totNumSweep']=bigData['soma_wave'].shape[0]
    metaD['units']={'stimAmpl':'nA or FS','somaWaveform':'mV','stimTime':'ms','stimWaveform':'nA','sweepId':'routine+0.1*idx','timeLive':'s'}
    metaD['sampling']='8 kHz'
    metaD['formatName']='expC.8kHz'
    metaD['expInfo']=exp_info
    metaD['lastRoutineTime']=int(lastTime) 
    print('M:metaD');pprint(metaD)
         
    outF='%s.%s.h5'%(saveF,metaD['formatName'])
    write3_data_hdf5(bigData,os.path.join(args.dataPath,outF),metaD=metaD)
    print('M:done %s numSweep=%d'%(saveF,metaD['totNumSweep']))

 
