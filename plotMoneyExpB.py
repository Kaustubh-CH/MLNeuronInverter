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
 
    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='money1'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

        

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.cL10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 10 distinct colors
        
        
#...!...!..................
    def signalRoy(self,data,plDD,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
        ax = self.plt.subplot(1,1,1)
        

        parName=plDD['parName']
        nd,nc=data.shape
        mkL=['D','o']
        fL=['none','k']
        dL=['0.5','1.0']
        mkS=[30,40]
        yL=[j for j in range(nc)]
        for i in range(nd):
            mk=mkL[i]
            mc='k'            
            ax.scatter( data[i],yL, color=mc, marker=mk, s=mkS[i], facecolors=fL[i],label=dL[i])
        ax.axvline(0, color='k',linewidth=0.7,linestyle='--')

        for j in range(nc):
            xv=-0.88
            ax.text(xv,j,parName[j])

        ax.set_xlim(-1.4,.4)
        ax.set_yticklabels([])
        ax.set(xlabel='log10( conductance/no drug)')
        ax.legend(loc='best',title='dose (umol)')
        
        return
        ax.legend(loc='best',title='stim ampl')
        tit=plDD['shortName']
        xLab='stim time '+plDD['units']['time']
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
        xLab='time '+plDD['units']['time']
        yLab='AP '+plDD['units']['waveform']
        ax.set(title=tit,xlabel=xLab,ylabel=yLab)
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
    for rec in dDose:
        x=rec[1]
        outF='211002_1_NI%s-a0.18s_ML-ontra3-1pr.mlPred.h5'%x
        bigD,expMD=read3_data_hdf5(args.dataPath+outF)
        pred=bigD['pred_cond'][0]
        parNameAll=expMD['parNameOrg']
        assert expMD['numSamples']==1  # tested for averaged data only
        predCond.append(pred)

    predCond=np.array(predCond)
    #pprint(expMD)
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

    
    #timeV=bigD['time']
    #ampls=expMD['stimAmpl']
    #waves=bigD['waveform']
    #stims=bigD['stim']
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    plDD['parName']=parName
    #for x in [ 'units','shortName']: plDD[x]=expMD[x]
    
    #- - - -  display
    if 1:
        plot.signalRoy(signal,plDD)

    if 0:
        plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        plot.waveform(waves,plDD,presc=presc,tit2=' sweep=%d'%isw)
    
    plot.display_all('money')
    
