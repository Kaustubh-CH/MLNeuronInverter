#!/usr/bin/env python3
'''
plot scores and waveforms w/ spikes

'''
from pprint import pprint
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from toolbox.Plotter import get_arm_color

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='out/',help="scored data location")

    parser.add_argument("--dataName",  default='bbp153', help="shortName ")
    parser.add_argument("--formatName",  default='mlPred', help="data name extesion ")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='plotPred'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#............................
#............................
#............................

class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)        
        self.mL7=['*','^','x','h','o','x','D']        
        
#...!...!..................
    def waveArray(self,bigD,plDD,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=4,2; yIn=9
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,yIn))
        
        timeV=bigD['time']
        waveA=bigD[ plDD['wave_data'] ]
        stimA=bigD['stims']
            
        idxL,idxR,idxS=plDD['idxLR']
        M=(idxR-idxL)//idxS
        #print('wdif1:',M,idxL,idxR)
        assert M<=nrow*ncol

        ssum=0
        for n in range(idxL,idxR,idxS): # want to dispaly column first
            j= (n-idxL)//idxS; ja=j*ncol; jb=j//nrow; jc=(ja+jb)%(nrow*ncol)
            #print('n',n,j,ja,jb,ja+jb,jc)
            if n>=waveA.shape[0]: break
            wave=waveA[n]
            ax = self.plt.subplot(nrow,ncol,1+jc)
            if 'raw' in plDD['yLab']:
                ax.plot(timeV,stimA[n], 'C2',linewidth=0.5,label='stim')
            ax.plot(timeV,wave, 'b',linewidth=0.7,label='soma AP')
            xm=np.mean(wave,axis=0)  # average over 1600 time bins
            xs=np.std(wave,axis=0)
            print('wave n=%d'%n, plDD['wave_data'],'avr:',xm,'std:',xs)

            yLab=plDD['yLab']
            xLab='time (ms), n=%d'%n
            ax.set(xlabel=xLab,ylabel=yLab)
            ax.grid()
            
            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))
            
            if jc==0:
                ax.legend(loc='best')
                ax.text(0.01,0.9,plDD['short_name'],transform=ax.transAxes,color='m')
            if jc==2:  ax.text(0.01,0.9,plDD['text1'],transform=ax.transAxes,color='m')


#...!...!..................
    def physParams1D(self,bigD,plDD,figId=4):

        predC=bigD['pred_cond']
        logC=np.log10(predC)
        logB=plDD['log10_phys_cent']
        parName=plDD['parName']
        crossTile=plDD['crossTile']
        
        nPar=len(parName)
        assert nPar==logC.shape[1]
        print('PP1Da:',logC.shape)
        nrow,ncol=4,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(3.1*ncol,2.2*nrow))
        
        for i in range(0,nPar):
            ax=self.plt.subplot(nrow,ncol,i+1)
            v=logC[:,i]
            #v=v[8]  # pick only one sample to plot
            lbase=logB[i]
            # compute avr on phys values of conductance
            pv=predC[:,i]
            avrC=np.mean(pv); stdC=-1
            if pv.shape[0]>1: stdC=np.std(pv)/np.sqrt( len(pv)-1)
            #print('iii',i,v,pv)
            avrTxt='avr(S)=%.2g\n   +/- %.1g'%(avrC, stdC)
            #print(i,'avrTxt',avrTxt)
            #ok11
            hcol=get_arm_color(parName[i])
            lfac=1.5
            binsX=np.linspace(lbase-lfac,lbase+lfac,30)
            (binCnt,_,_)=ax.hist(v,binsX,color=hcol)

            ax.text(0.6,0.65,avrTxt,transform=ax.transAxes, color='b')
            ax.set(title=parName[i], xlabel='log10(cond/(S)) p=%d'%(i),ylabel='samples')
            ary=0.7
            #arrow(x, y, dx, dy, **kwargs)
            ax.arrow(lbase, ary, 0,-ary+0.1, head_width=0.2, head_length=0.15, fc='k', ec='k')
            if i in crossTile:
                ax.text(0.8,0.1,"BAD\nML",transform=ax.transAxes, color='b')
            
            for x in [-1,1]:
                ax.axvline(lbase+x, color='black', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(-0.05,0.85,plDD['short_name'][:25],transform=ax.transAxes, color='r')
            if i==1:
                ax.text(0.05,0.85,'n=%d'%logC.shape[0],transform=ax.transAxes, color='r')

#...!...!..................
            
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,inpMD=read3_data_hdf5(args.dataPath+inpF, verb=args.verb)

    #pprint(inpMD)

    #for a,b in zip(inpMD['parName'],inpMD['log10_phys_cent']):   print(a,b)
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    for x in ['short_name', 'parName', 'log10_phys_cent']: plDD[x]=inpMD[x]
   
    if 0:  # wavforms as array, decimated
        plDD['idxLR']=[0,8,1] # 1st range(n0,n1,step) ampl-index
        plDD['text1']='raw waveform'
        plDD['yLab']='AP (mV)'
        plDD['wave_data']='exper_frames'
        #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        plot.waveArray(bigD,plDD)

    if 0:  # wavforms as array, decimated
        plDD['idxLR']=[0,4,1] # 1st range(n0,n1,step) ampl-index
        plDD['text1']='ML input waveform'
        plDD['yLab']='normalized (a.u.)'
        plDD['wave_data']='waves_ml'
        # the second pop-argument prevents the conditional exception.
        plDD.pop('amplLR', None) 
        plot.waveArray(bigD,plDD)

    if 1:  # phys conductances
        plDD['crossTile']=[6,7,11,13,15,16,17,18,19]
        plot.physParams1D(bigD,plDD)
        
    plot.display_all('scoreSim')
