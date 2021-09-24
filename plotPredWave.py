#!/usr/bin/env python3
'''
overlays  wavformes predicted by Roy to ML pred coductances

'''
from pprint import pprint
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from toolbox.Plotter import get_arm_color
from toolbox.Util_Experiment import rebin_data1D


import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='/global/homes/b/balewski/prjn/2021-roys-simulation/2021-neuron-pred-from-ml/exp_pred_NEURON_2019_1/',help=" data location")

    
    parser.add_argument("--formatName",  default='neurPred', help="data name extesion ")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='neurPred'
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
    def waveArrayOne(self,h5D,plDD,iSamp,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=1,1; 
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,8))
        ax = self.plt.subplot(nrow,ncol,1)

        # experiment
        bigD=h5D['15-pred']
        
        timeV=bigD['time']
        waveA=bigD['waves_ml']
        stimA=bigD['stims']
            
        wave=waveA[iSamp,:,0]
        stim=stimA[iSamp]
            
        ax.plot(timeV,stim+5, 'C2',linewidth=0.5,label='stim+5(nA)')
        ax.plot(timeV,wave, 'b',linewidth=0.7,label='exp soma')
        if 1:
            xm=np.mean(wave,axis=0)  # average over 1600 time bins
            xs=np.std(wave,axis=0)
            print('wave i=%d'%iSamp, 'avr:',xm,'std:',xs)

        yLab='norm. waveform (a.u.)'
        xLab='time (ms)'
        tit='%s  iSamp=%d'%(plDD['short_name'],iSamp)
        ax.set(xlabel=xLab,ylabel=yLab,title=tit)
        ax.axhline(0,color='C8')
        ax.grid()

        dC=['C3','C0','C4']
        for j,pred in enumerate(h5D):
            wave8kA=h5D[pred]['pred_vs']
            nRebin=5
            print('pred:', pred,wave8kA.shape,nRebin)
            wave8k=wave8kA[0,iSamp]
            wave=rebin_data1D(wave8k,nRebin)
            xm=np.mean(wave,axis=0)  # average over 1600 time bins
            xs=np.std(wave,axis=0)
            print('pred:', pred, 'avr:',xm,'std:',xs)
            ax.plot(timeV,wave, color=dC[j],ls='--',linewidth=0.8,label='ML:'+pred)
        
            #if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            #if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))
            

        ax.legend(loc='best')
        #ax.text(0.01,0.9,xx,transform=ax.transAxes,color='m')
            

#...!...!..................
    def waveArray(self,h5D,plDD,figId=5):
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
    def XphysParams1D(self,bigD,plDD,figId=4):

        logC=np.log10(bigD['pred_cond'])
        logB=np.log10(plDD['base_cond'])
        parName=plDD['parName']
        
        nPar=len(parName)
        assert nPar==logC.shape[1]
        print('PP1Da:',logC.shape)
        nrow,ncol=3,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.75*ncol,2.2*nrow))

        binsX=30
        for i in range(0,nPar):
            ax=self.plt.subplot(nrow,ncol,i+1)
            v=logC[:,i]
            v=v[8]  # pick only one sample to plot
            lbase=logB[i]
            #print('iii',i,v,base)
            hcol=get_arm_color(parName[i])
            lfac=1.5
            binsX=np.linspace(lbase-lfac,lbase+lfac,50)
            (binCnt,_,_)=ax.hist(v,binsX,color=hcol)
                                  
            ax.set(title=parName[i], xlabel='log10(cond/(S)) %d'%(i),ylabel='samples')
            ary=0.6
            #arrow(x, y, dx, dy, **kwargs)
            ax.arrow(lbase, ary, 0,-ary+0.1, head_width=0.2, head_length=0.15, fc='k', ec='k')
            for x in [-1,1]:
                ax.axvline(lbase+x, color='black', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(-0.05,0.85,plDD['short_name'],transform=ax.transAxes, color='r')
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

    inpD={'15-pred':'all_params','13-pred':'2params_fixed','12-pred':'3params_fixed'}

    h5D={}
    md={}
    for x in inpD:        
        inpF='210611_3_exp_predictions_%s.%s.h5'%(inpD[x],args.formatName)
        bigD,inpMD=read3_data_hdf5(args.dataPath+inpF, verb=args.verb)
        h5D[x]=bigD
        md[x]=inpMD
   
    #pprint(inpMD)

    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    #for x in ['short_name', 'parName', 'base_cond']: plDD[x]=inpMD[x]
    plDD['short_name']='210611_3_NI'
    
    if 1:  # wavforms as array, decimated
        plDD['idxLR']=[0,8,1] # 1st range(n0,n1,step) ampl-index
        #plDD['wave_data']='exper_frames'
        #plDD['amplLR']=[-90,70]  #  (mV) amplitude range
        plot.waveArrayOne(h5D,plDD,iSamp=3)
        #plot.waveArray(h5D,plDD)
        
    if 0:  # phys conductances
        plot.physParams1D(bigD,plDD)
        
    plot.display_all('scoreSim')
