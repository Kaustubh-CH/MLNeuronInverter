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

    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-experiment/october/out-inhib-roy-ml2/',help="formated data location")
 
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")
    args = parser.parse_args()

    args.outPath+='/'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

        

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
         
#...!...!..................
    def signalRoy(self,data,plDD,figId=5):
        figId=self.smart_append(figId)
        xyIn=(6,5)
        mkS=[30,45]
        
        if args.formatVenue=='poster':
            xyIn=(7.5,4.5)
            mkS=[50,70]
            self.plt.rcParams['axes.spines.left'] = False
               
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        ax = self.plt.subplot(1,1,1)
        
        parName=plDD['parName']
        nd,nc=data.shape
        mkL=['D','o']
        fL=['none','k']
        dL=plDD['dose'][1:]
        
        yL=[j+0.4 for j in range(nc)]
        for i in range(nd):
            mk=mkL[i]
            mc='k'            
            ax.scatter( data[i],yL, color=mc, marker=mk, s=mkS[i], facecolors=fL[i],label=dL[i])
        #ax.axvline(0, color='k',linewidth=0.7,linestyle='--')
        ax.axvline(0, color='k',linewidth=1.,linestyle='-')
        ax.get_yaxis().set_visible(False)
        
        for j in range(nc):
            xv=0.6
            ax.text(xv,yL[j]-0.1,parName[j])

        ax.set_xlim(-1.5,1.5)
        tit=plDD['shortName']
        ax.set(xlabel='log10( conductance / no drug )')
        ax.legend(loc='upper left',title='dose (umol)')
        if args.formatVenue!='poster':
            ax.set(title=tit)

 
#...!...!..................
    def signalKris(self,data,plDD,figId=5):
        figId=self.smart_append(figId)
        xyIn=(8,4)
        xyIn=(5,4.5)
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        ax = self.plt.subplot(1,1,1)
        
        parName=plDD['parName']
        nd,nc=data.shape
        assert nc==len(parName)
        dL=plDD['dose'][1:]
        colL=['silver','dimgrey']
        X=np.arange(nc)
        
        xLab=[]; n1=1;n2=1
        for name in parName:
            if 'soma' in name: xLab.append('S%d'%n1); n1+=1
            if 'axon' in name: xLab.append('A%d'%n2); n2+=1

        for a,b in zip(xLab,parName): print(a,b)    
        width = 0.35  # the width of the bars
        for i in range(nd):
            xoff=-width*(2*i-1)/2.
            ax.bar(X - xoff,data[i], width,label=dL[i], color=colL[i])
        
        ax.set_xticks(X)
        ax.set_xticklabels(xLab)
     
        ax.axhline(0, color='k',linewidth=0.7,linestyle='-')
        tit=plDD['shortName']
        ax.set(ylabel='log10( conductance / no drug )',xlabel='conductance type')
        ax.legend(loc='lower left',title='dose (umol)')
        ax.set_ylim(-1.3,1.3)
        if args.formatVenue!='poster':
            ax.set(title=tit)
        return
 
#...!...!..................
    def waveforms(self,waves,plDD,presc=1,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,8))
        ax = self.plt.subplot(1,1,1)

        timeV=plDD['timeV']
        dose=plDD['dose']
        N=waves.shape[0]

        colL=['r','b','lime']
        for n in range(0,N,presc):
            hexc=colL[n]
            ax.plot(timeV,waves[n], label=dose[n],color=hexc,linewidth=0.7)

        ax.legend(loc='best',title='dose (umol)')
        tit=plDD['shortName']
        xLab='stim time (ms) '
        yLab='soma signal (mV)'
        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
        if args.formatVenue!='poster':
            ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
        if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    dDose=[1,2,3]
    #       dose, file,
    dDose[0]=[0,'a']
    dDose[1]=[0.5,'c']
    dDose[2]=[1.0,'d']
    predCond=[]
    waves=[]
    for rec in dDose:
        x=rec[1]
        outF='211002_1_NI%s-a0.18s_ML-ontra3-1pr.mlPred.h5'%x
        bigD,expMD=read3_data_hdf5(args.dataPath+outF)
        pred=bigD['pred_cond'][0]
        parNameAll=expMD['parNameOrg']
        assert expMD['numSamples']==1  # tested for averaged data only
        predCond.append(pred)
        waves.append(bigD['exper_frames'][0])
        
    waves=np.array(waves)
    predCond=np.array(predCond)
    pprint(expMD)
    #for a,b in zip(parName,pred):   print(a,b)
    print('M:predCond=',predCond.shape)

    # select 10 conductances to which ML was senitive
    usePar=[0,1,4,5,8,9,10,11,12,14]
    parName=[parNameAll[i] for i in usePar]
    predCond=predCond[:,usePar]
    print('aa',parName,predCond.shape)

    # compute  singnal
    nPar=len(usePar)
    signal=np.zeros((2,nPar))
    for i in range(2):
        signal[i]=predCond[i+1]/predCond[0]

    signal=np.log10(signal)
    print('signal:',signal)
   
    
    #stims=bigD['stim']
    # - - - - - PLOTTER - - - - -
    args.prjName=expMD['exper_info']['shortName']
    plot=Plotter(args)
    plDD={}
    plDD['parName']=parName
    plDD['timeV']=bigD['time']
    plDD['shortName']='experiment '+expMD['exper_info']['shortName']
    plDD['dose']=[0.,0.5,1.0]
    #for x in [ 'units','shortName']: plDD[x]=expMD[x]
    
    #- - - -  display
    if 1:
        plDD['amplLR']=[-90,-30]  #  (mV) amplitude range
        plDD['timeLR']=[10.,160]  # (ms)  time range 
        #plDD['timeLR']=[0.,200]  # (ms)  time range 
        plot.waveforms(waves,plDD)

    if 1:
        plot.signalKris(signal,plDD)
        
    if 1:  # must be last
        plot.signalRoy(signal,plDD)


     
    plot.display_all('money',png=0)
    
