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
                        #default='/global/homes/b/balewski/prjn/2021-roys-experiment/october/raw/',
                        default='/global/homes/r/roybens/fromMac/',
                        help="input  raw data path for experiments")
    parser.add_argument("--dataPath", default='/global/homes/b/balewski/prjn/2021-roys-experiment/october/Xdata8kHz/',help="output path  rebinned Waveforms  ")

    parser.add_argument("--templeteName",  default='211002_1_NI', help="all files matching  *template* will be agregated ")
    parser.add_argument("--saveNameExt",  default='', help="optional output h5 output extension, if same cell data needs split ")
    parser.add_argument("--routineRange", default=None,nargs="+",type=int,help="bracket used data")
    parser.add_argument("--initWallTime", default=0,type=int,help="optional offset in seconds")
    
    args = parser.parse_args()
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if args.routineRange!=None: # skip some data
        assert  len(args.routineRange)==2
        assert args.routineRange[0]<=args.routineRange[1]
    return args



#...!...!..................
def survey_routines(allName, routRange):  # sort by routine-ID,
    expDB={}
    cnt={'nok':0,'minId':9999,'maxId':-1,'seen':0}
    minId=9999; maxId=0
    for x in allN:
        if 'h5' not in x : continue
        if 'sweep_all' in x : continue
        if args.templeteName not in x : continue
        
        # D: parse name  , expected: 210607_1_NI_04_47.h5  
        xL=x[:-3].split('_')
        routNo=int(xL[4])
        cnt['seen']+=1
        if routRange!=None: # skip some data
            if routNo <routRange[0]: continue 
            if routNo >routRange[1]: continue
            
        campl=xL[3][1:]
        stimAmpl=float('.'+campl)
        cnt['nok']+=1
        if cnt['minId']>routNo :  cnt['minId']=routNo
        if cnt['maxId']<routNo :  cnt['maxId']=routNo 
        #print('aa',nOK,x,routNo,stimAmpl)
        if stimAmpl not in expDB: expDB[stimAmpl]={}
        expDB[stimAmpl][routNo]=x


    # sort routones for each amplitude
    for x in sorted(expDB):
        D=expDB[x]
        yL=sorted(D)
        print('ampl:',x,'numRout:',len(yL))
        expDB[x]=[ [y,D[y]] for y in yL ]
        
    #pprint(expDB)
    print('survey: numAmpl=%d '%(len(expDB)),cnt)
    return expDB,cnt

#...!...!..................
def init_bigData(expDB): # final HD5-stoarge

    mxStim=len(expDB)
    mxRo=0
    for x in expDB:
        nr=len(expDB[x])
        if mxRo<nr: mxRo=nr
    mxSweep=numSweepPerRoutine*mxRo  
    mxTB=1600

    bigD={}
    bigD['waveform']=np.zeros((mxStim,mxSweep,mxTB)).astype('float32')
    bigD['sweepCnt']=np.zeros(mxStim).astype('int32')
    bigD['stim']=np.zeros((mxStim,mxSweep,mxTB)).astype('float32')
    bigD['sweepTrait']=np.zeros((mxStim,mxSweep,numSweepTrait)).astype('float32')
    print('waveform shape',bigD['waveform'].shape)
    return bigD

#...!...!..................
def read_all_data(expDB,bigD):
    amplL=sorted(expDB)
    totSw=0
    maxTime=-1
    for ia,ampl in enumerate(amplL):
        iSw=0
        for routId,fname in expDB[ampl]:
            print('r',ia,ampl,routId,fname)
            stim,wave,trait,routTime=M_read_one(fname,ampl,routId)
            bigD['waveform'][ia,iSw:iSw+numSweepPerRoutine]=wave
            bigD['stim'][ia,iSw:iSw+numSweepPerRoutine]=stim
            bigD['sweepTrait'][ia,iSw:iSw+numSweepPerRoutine]=trait
            iSw+=numSweepPerRoutine
            if maxTime < routTime: maxTime = routTime
        print('done read ampl=%.2f numSweep=%d'%(ampl,iSw))
        bigD['sweepCnt'][ia]=iSw
        totSw+=iSw
    return totSw,amplL,maxTime

#...!...!..................
def QA_one(wave1):
    avrPreMemBias=np.mean(wave1[:50])
    if avrPreMemBias > -30:
        print('JSPM corruped waveform, skip it')
        print('wave1',avrPreMemBias,wave1[:50])
        bad55
    return 0


#...!...!..................
def M_read_one(inpF,stimAmpl,routId):
    print('\nRO:',inpF,stimAmpl,routId)
    bulk,_=read3_data_hdf5(args.rawPath+inpF)

    timeA=bulk['time']*1000 # in ms
    waveA=bulk['Vs']*1000   # in mV
    stimA=bulk['stim']*1e9  # nA
    numMeas=waveA.shape[0]
    try:
        Rserial=bulk['Rs'][0] # MOhm
        routTime=bulk['time_from_start'][0] # sec
        
    except:
        Rserial=19.  # for 210607_1_NI
        routTime=18*routId

    routTime+=args.initWallTime
    # rebin + interpolate
    # A: time axis
    nReb=6
    timeB=rebin_data1D(timeA,nReb,clip=True)
    #print('reb timeA,B',timeA.shape, timeB.shape, timeA[-1])
    numTimeBin=1600 # 200ms @ 8 kHz
    stepC=timeA[-1] /numTimeBin
    timeC=np.linspace(0,numTimeBin, num=numTimeBin,endpoint=False)*stepC
    #print('reb timeC',timeC.shape,timeC[-1],stepC)

    # B: waveforms and stim
    outB=rebin_data2D(waveA,nReb,clip=True)
    waveCL=[np.interp(timeC,timeB,outX) for outX in outB ]
    #print('waveCL[0]',waveCL[0].shape)
    
    outB=rebin_data2D(stimA,nReb,clip=True)
    stimCL=[np.interp(timeC,timeB,outX) for outX in outB ]
    #print('stimCL[0]',stimCL[0].shape)

    sweepQA=[QA_one(waveCL[i])  for i in range(numMeas)]
    
    sweepIdL=[routId+i/10. for i in range(1,numMeas+1)]
    sweepTimeL=[routTime+i*sweepDurationInRoutine for i in range(numMeas)]  
    traitLT=[sweepIdL,sweepTimeL, [Rserial]*numMeas]
    trait=np.array(traitLT).T
    print('ttt',trait.shape,trait)
    if 'time' not in bigData: bigData['time']=timeC
    return  stimCL, waveCL,trait,routTime
   

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    args.prjName='formExpB'
    numSweepPerRoutine=2  # each routine has 2 sweeps 
    numSweepTrait=3 # [sweepId, sweepTime, serialRes]
    sweepDurationInRoutine=3 # Roy: 3 sec separation between sweeps
    
    # collect valid h5-file names
    xL=args.templeteName.split('_')
    args.rawPath+='/'+'_'.join(xL[:2])+'/'
    print('M:rawPath=',args.rawPath)
    allN=os.listdir(args.rawPath)
    expDB,surveyCnt=survey_routines(allN,args.routineRange)

    bigData=init_bigData(expDB) # final HD5-stoarge

    totNumSweep,amplL,maxTime=read_all_data(expDB,bigData)
    surveyCnt['lastRoutineTime']=int(maxTime)
    
    if 1:
        print('M:sweepCnt',bigData['sweepCnt'])
        #print(bigData['waveform'][:,:,:5])

    
    # - - - - -  assemble meta data - - - - - -
    [nSt,nSw,nTB]= bigData['waveform'].shape
    saveF=args.templeteName+args.saveNameExt
    metaD={}
    metaD['rawDataPath']=args.rawPath
    metaD['shortName']=saveF
    metaD['stimName']='chaotic_2'
    metaD['maxSweepPerStimAmpl']=nSw
    metaD['numStimAmpl']=nSt
    metaD['numTimeBin']=nTB
    metaD['totNumSweep']=totNumSweep
    #pprint(metaD)
    metaD['stimAmpl']=amplL
    metaD['units']={'stimAmpl':'nA','waveform':'mV','time':'ms','sweepId':'routine+idx','serialRes':'MOhm','timeLive':'s'}
    metaD['sampling']='8 kHz'
    metaD['formatName']='expB.8kHz'
    metaD['surveyCounter']=surveyCnt
    
    print('M:metaD');pprint(metaD)
         
    outF='%s.%s.h5'%(saveF,metaD['formatName'])
    write3_data_hdf5(bigData,os.path.join(args.dataPath,outF),metaD=metaD)
    print('M:done %s numSweep=%d'%(saveF,totNumSweep))

 
