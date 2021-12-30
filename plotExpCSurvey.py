#!/usr/bin/env python3
'''
inspect formated  experiment

'''
import sys,os
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from pprint import pprint
import json
import copy

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/december/data8kHz/',help="formated data location")

    parser.add_argument("--dataName",  default='211219_5x', help="shortName for a set of routines ")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

 
    args = parser.parse_args()
    args.formatName='expC.8kHz'  # file name extension
    args.formatVenue='prod'
    args.prjName='expC'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def print_exp_summary(bigD,expMD):
    exp_info=expMD['expInfo']
    print('Exp-summary:',expMD['shortName'])
            
    for a,rs in zip(exp_info['amps'],exp_info['routines']):
        sa=str(a)
        print('stimAmpl/FS=%.2f routies=%s, idxs=%s'%(a,str(rs),exp_info['map_idx'][sa]))


    if 0:
        ia=8
        print('details for ia=%d ampl=%.2f'%(ia,ampls[ia]))
        for iw in range(sweepCnt[ia]):
            sweepId, sweepTime, serialRes=bigD['sweepTrait'][ia,iw]
            print(iw,sweepId, sweepTime, serialRes)
        

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.cL10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 10 distinct colors
        
        
#...!...!..................
    def stims(self,stims,plDD,presc=1,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)
        #pprint(plDD)
        timeV=plDD['timeV']
        ampls=plDD['stimAmpl']
        N=stims.shape[0]
        for n in range(0,N,presc):
            hexc=self.cL10[ n%10]
            ax.plot(timeV,stims[n], color=hexc, label='%.2f (FS)'%ampls[n],linewidth=0.7)

        ax.legend(loc='best',title='stim ampl')
        tit=plDD['shortName']
        xLab='stim time '+plDD['units']['stimTime']
        yLab='stim ampl '+plDD['units']['stimAmpl']
        ax.set(title=tit,xlabel=xLab,ylabel='stim (nA)')
        ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
 
#...!...!..................
    def waveform(self,waves,plDD,presc=1,tit2='',figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)

        timeV=plDD['timeV']
        ampls=plDD['stimAmpl']
        N=waves.shape[0]

        for n in range(0,N,presc):
            hexc=self.cL10[ n%10] 
            ax.plot(timeV,waves[n], color=hexc, label='ampl=%.2f'%ampls[n],linewidth=0.7)

        ax.legend(loc='best')
        tit=plDD['shortName']+tit2
        xLab='stim time (%s)'%plDD['units']['stimTime']
        yLab='soma waveform (%s)'%plDD['units']['somaWaveform']

        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
        ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
        if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

        
#...!...!..................
    def survey_exp(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,4))

        exp_info=expMD['expInfo']
        mapD=exp_info['map_idx']
        traitA=exp_info['sweep_trait']

        print('mm',mapD)
        # repack data for plotting
        stimA=[]; swTimeA=[]; swIdA=[]

        for a in sorted(mapD):
            i0,i1=mapD[a]
            for i in range(i0,i1): # loop over stmAmpl
                [sweepId,sweepTime,ampl]=traitA[i]
                swTimeA.append(sweepTime)
                stimA.append(ampl)
                swIdA.append(sweepId)
                print(i,sweepId,sweepTime,a,ampl)
                
        wallTA=np.array(swTimeA)/60.
        stimA=np.array(stimA)
        wtIdx = np.argsort(wallTA) # needed to sort data by wall time
        #print('ii',wtIdx)
        ax = self.plt.subplot(nrow,ncol,1)

        ax.plot(wallTA[wtIdx],stimA[wtIdx],'*-', alpha=0.6)
        ax.set(xlabel='wall time (min)',ylabel='stim ampl (FS)')
        ax.text(0.3,0.90,plDD['shortName'],transform=ax.transAxes,color='m')
        ax.grid()


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    outF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,expMD=read3_data_hdf5(args.dataPath+outF)
    print_exp_summary(bigD,expMD)
    pprint(expMD)

    exp_info=expMD['expInfo']
    ampls=exp_info['amps']
    waves=bigD['soma_wave']
    stims=bigD['stim_wave']
    timeV=bigD['time']
    

    # extract 1 stim per ampl
    idxL=[]
    mapD=exp_info['map_idx']
    isw0=1
    for a in sorted(mapD):
        idxL.append(mapD[a][0]+isw0)
        #if len(idxL)>0: break
    print('M:sdxL',idxL)
    waves=waves[idxL]
    stims=stims[idxL]
    print('M:ws',waves.shape)
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    for x in [ 'units','shortName']: plDD[x]=expMD[x]
    plDD.update({'timeV':timeV,'stimAmpl':ampls})

    #- - - -  display
    plDD['timeLR']=[10.,160.]  # (ms)  time range
    plDD['timeLR']=[0.,200.]  # (ms)  time range
    presc=1
    if 1:
        plot.stims(stims,plDD,presc=presc)

    if 1:
        #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        plot.waveform(waves,plDD,presc=presc,tit2=' sweep=%d'%(isw0+1))

    if 1:
        plot.survey_exp(bigD,plDD)
    
    plot.display_all('baseC')
    
