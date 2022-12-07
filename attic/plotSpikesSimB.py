#!/usr/bin/env python3
'''
plot scores and waveforms w/ spikes

'''

from pprint import pprint
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='out2019/',help="scored data location")

    parser.add_argument("--dataName",  default='bbp153', help="shortName ")
    parser.add_argument("--formatName",  default='spikerSum', help="data name extesion ")

    parser.add_argument("-o","--outPath", default='out2019/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='scoreB'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#............................
#............................
#............................

class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.cL10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 10 distinct colors
        self.mL7=['*','^','x','h','o','x','D']
                
#...!...!..................
    def waveArray(self,bigD,plDD,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=4,2; yIn=8
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,yIn))
        
        timeV=bigD['time']
        ihc=plDD['iHoldCurr']
        
        stimA=bigD['stim'][ihc]*5 # better visibility
        spikeC=bigD['spikeCount'][ihc]
        spikeT=bigD['spikeTrait'][ihc]
        waveA=bigD['waveform'][ihc]
        
        #print('wdif0',waveA.shape,stimT.shape,len(spikeA[0]))
      
        idxL,idxR,idxS=plDD['idxLR']
        M=(idxR-idxL)//idxS
        print('wdif1:',M,idxL,idxR)
        assert M<=nrow*ncol

        ssum=0
        for n in range(idxL,idxR,idxS): # want to dispaly column first
            j= (n-idxL)//idxS; ja=j*ncol; jb=j//nrow; jc=(ja+jb)%(nrow*ncol)
            #print('n',n,j,ja,jb,ja+jb,jc)
            ax = self.plt.subplot(nrow,ncol,1+jc)
            ks=spikeC[n]
            ssum+=ks
            ax.plot(timeV,stimA[n], 'k',linewidth=0.5,label='stim')
            ax.plot(timeV,waveA[n], 'b',linewidth=0.7,label='soma AP')
            if plDD['simAuth']=='simRoy':
                txt=' numSpike=%d  stimAmpl=%.2f'%(ks,plDD['stimAmpl'][n])
                ax.text(0.25,0.9,txt,transform=ax.transAxes,color='g')
            ax.axhline(plDD['peak_thr'],color='m', linestyle='--',linewidth=0.5)
            if 1: # show valid spikes
                spikes=spikeT[n][:ks]
                tPeak=spikes[:,0]
                yPeak=spikes[:,1]
                ax.plot(tPeak,yPeak,"*",color='m')
  
            '''
            if 1: # show valid spikes
                spikes=spikeA[n][:]
                #pprint(spikes)
                for rec in spikes:
                   tPeak,yPeak,twidth,ywidth,_=rec 
                   x0=tPeak-twidth/2
                   y0=ywidth
                   ax.add_patch(Rectangle((x0, y0), twidth, yPeak-ywidth,alpha=0.7,fc='r'))
            '''
            
            yLab='AP (mV)'
            xLab='time (ms), '+plDD['shortName']+', n=%d'%n
            ax.set(xlabel=xLab,ylabel=yLab)
            ax.grid()
            if jc==0: ax.legend(loc='best')
            #print('P: %s avrScore=%.1f '%(plDD['text1'],ssum/(idxR-idxL)))
            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

            
            if jc==0:   ax.text(0.01,0.9,plDD['shortName'],transform=ax.transAxes,color='m')
            if jc==1:  ax.text(0.01,0.9,plDD['text1'],transform=ax.transAxes,color='m')
            if jc==2:   ax.text(0.01,0.9,plDD['bbpName'],transform=ax.transAxes,color='m')

#...!...!..................
    def score_stimAmpl(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,5))
        ax = self.plt.subplot(nrow,ncol,1)

        stimA=plDD['stimAmpl']
        scoreA=bigD['spikeCount'] ; yTit='spike count'
        nhc=scoreA.shape[0]
        assert scoreA.shape[1]==stimA.shape[0]

        avrScore=np.mean(scoreA,axis=0)
        for ihc in range(nhc):
            dLab='holdCurr=%.2f nA'%(inpMD['holdCurr'][ihc])
            dmk=self.mL7[ihc%7]
            ax.plot(stimA,scoreA[ihc], dmk+'-',linewidth=0.8,label=dLab)

        #ax.set_xlim(0.6,1.5)
        
        tit1='%s simulation:   %s,   %s'%(plDD['simVer'],plDD['shortName'],plDD['bbpName'])
        ax.set(ylabel=yTit,xlabel='stim ampl (FS)',title=tit1)
        ax.legend(loc='best')
        ax.grid()
        
#...!...!..................
    def stims_fixedAmpl(self,bigD,plDD,figId=51):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))
        ax = self.plt.subplot(nrow,ncol,1)
         
        timeV=bigD['time']
        iAmp=plDD['iAmp']
        
        stimA=bigD['stim'][:,iAmp]
        M=stimA.shape[0]
        for j in range(M):
            dLab='holdCurr=%.2f nA'%(inpMD['holdCurr'][j])
            ax.plot(timeV,stimA[j],color='C%d'%j,label=dLab)

        yLab='stimulus (nA)'
        xLab='time (ms) '
        ax.set(xlabel=xLab,ylabel=yLab)
        ax.legend(loc='best')
        ax.grid()
        
#...!...!..................
    def stims_fixedHoldCurr(self,bigD,plDD,figId=52):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))
        ax = self.plt.subplot(nrow,ncol,1)
         
        timeV=bigD['time']
        ihc=plDD['iHoldCurr']
        stimA=bigD['stim'][ihc]

        for j in range(15,24,2):
            dLab='stimAmpl=%.2f FS'%(inpMD['stimAmpl'][j])
            ax.plot(timeV,stimA[j],color='C%d'%j,label=dLab,linewidth=0.8)

        yLab='stimulus (nA)'
        xLab='time (ms) '
        ax.set(xlabel=xLab,ylabel=yLab)
        ax.legend(loc='best')
        ax.grid()
        
#...!...!..................
    def spikes_survey2D(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,6))
        #fig.suptitle("Title for whole figure", fontsize=16)

        spikeC=bigD['spikeCount'][ihc]
        spikeA=bigD['spikeTrait'][ihc]
        N=spikeC.shape[0]
        
        # 
        # repack data for plotting
        tposA=[]; widthA=[]; amplA=[]; stimA=[]
        for n in range(N): # loop over stmAmpl
            ks=spikeC[n]            
            spikes=spikeA[n][:ks] # use valid spikes
            if simAuth=='simRoy':
                stimAmpl=plDD['stimAmpl'][n]
            for rec in spikes:
                tPeak,yPeak,twidth,ywidth,_=rec
                tposA.append(tPeak)
                widthA.append(twidth)
                amplA.append(yPeak)
                if simAuth=='simRoy':
                    stimA.append(stimAmpl)
                    
                #if tPeak<10: print(n,'rec:',rec)
                    
        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(tposA,amplA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike ampl (mV)')
        ax.text(0.01,0.9,plDD['shortName'],transform=ax.transAxes,color='m')

        if simAuth=='simRoy':
            ax = self.plt.subplot(nrow,ncol,2)
            ax.scatter(stimA,amplA, alpha=0.6)
            ax.set(xlabel='stim ampl (FS)',ylabel='spike ampl (mV)')
            ax.text(0.01,0.9,plDD['text1'],transform=ax.transAxes,color='m')
        
        ax = self.plt.subplot(nrow,ncol,3)
        ax.scatter(tposA,widthA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike width (ms)')
        if 'fwhmLR' in plDD: ax.set_ylim(tuple(plDD['fwhmLR']))
        ax.text(0.01,0.9,plDD['bbpName'],transform=ax.transAxes,color='m')

        if simAuth=='simRoy':
            ax = self.plt.subplot(nrow,ncol,4)
            ax.scatter(stimA,widthA, alpha=0.6)
            ax.set(xlabel='stim ampl (FS)',ylabel='spike width (ms)')

#...!...!..................
    def spikes_survey1D(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,2.5))

        if plDD['simAuth']=='simRoy':
            ia=plDD['iStimAmpl']
            stimAmpl=plDD['stimAmpl'][ia]
            spikeC=bigD['spikeCount'][:,ia]
            spikeA=bigD['spikeTrait'][:,ia]
        else: # Vyassa, has no stim-ampl variation
            stimAmpl=1.0
            spikeC=bigD['spikeCount'][0]
            spikeA=bigD['spikeTrait'][0]
            
        M=spikeC.shape[0] # num of waveform/ampl
        print('pl1D: ampl=%.2f, M=%d'%(stimAmpl,M))
        # repack data for plotting
        widthA=[]; amplA=[]; tbaseA=[]
        for j in range(M): #loop over waveforms
            ks=spikeC[j]
            if ks==0: continue                
            spikes=spikeA[j][:ks] # use valid spikes
            for rec in spikes:
                tPeak,yPeak,twidth,ref_amp,twidth_base=rec                  
                widthA.append(twidth)
                amplA.append(yPeak)
                tbaseA.append(twidth_base)

        #print('pl1D: widthA:',widthA)
        tit='%s stim ampl=%.2f'%(plDD['shortName'],stimAmpl)
        ax = self.plt.subplot(nrow,ncol,1)
        binsX= np.linspace(-0.5,10.5,12)  # good choice
        ax.hist(spikeC[:M],binsX,facecolor='g')        
        ax.set(xlabel='num spikes per sweep',ylabel='num sweeps',title=tit)
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,2)
        binsX= np.linspace(0.5,3.5,20)
        ax.hist(widthA,bins=binsX,facecolor='b')        
        ax.set(xlabel='spike half-width (ms), aka FWHM',ylabel='num spikes',title=plDD['bbpName'])
        if 'fwhmLR' in plDD: ax.set_xlim(tuple(plDD['fwhmLR']))
        ax.grid()
                
        ax = self.plt.subplot(nrow,ncol,3)
        binsX= np.linspace(20,60,40)
        ax.hist(amplA,bins=binsX,facecolor='C1')        
        ax.set(xlabel='spike peak ampl (mV)',ylabel='num spikes',title='gen:'+plDD['simAuth'])
        ax.grid()
                
        ax = self.plt.subplot(nrow,ncol,4)
        binsX= np.linspace(0,10,40)
        ax.hist(tbaseA,bins=binsX,facecolor='C3')        
        ax.set(xlabel='spike base width (ms)',ylabel='num spikes')
        ax.grid()
            
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF='%s.%s.h5'%(args.dataName,args.formatName)
    bigD,inpMD=read3_data_hdf5(args.dataPath+inpF, verb=args.verb)

    pprint(inpMD)

    simAuth='simRoy'
    if 'simV' in inpMD['formatName']: simAuth='simVyassa'
        
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    for x in [ 'units','shortName','bbpName']: plDD[x]=inpMD[x]
    plDD['stimAmpl']=np.array(inpMD['stimAmpl'])
    plDD['simAuth']=simAuth
    plDD['simVer']='NeuronSim_2019_1'
    
    if 1:  # wavforms as array, decimated
        plDD['idxLR']=[8,24,2] # 1st range(n0,n1,step) ampl-index
        #plDD['idxLR']=[0,8,1] # low ampl
        
        if simAuth=='simRoy':
            ihc=0  # select holding current
            plDD['iHoldCurr']=ihc # holding current index
            plDD['text1']='holdCurr=%.2f nA'%(inpMD['holdCurr'][ihc])
        else: # Vyassa's simu,  fake holdCurr
            plDD['iHoldCurr']=0
            plDD['text1']='Vyassa-sim'
        #plDD['timeLR']=[10.,160.]  # (ms)  time range
        #plDD['timeLR']=[15.,40.]  # (ms)  time range 
        plDD['amplLR']=[-100,70]  #  (mV) amplitude range
        plDD['peak_thr']=inpMD['spikerConf']['min_peak_ampl_mV']
        plot.waveArray(bigD,plDD)

    if 1 and  simAuth=='simRoy':   # score vs. stimAmpl , all data
        plot.score_stimAmpl(bigD,plDD)
        
    if 1 and  simAuth=='simRoy':
        plDD['iAmp']=19
        plot.stims_fixedAmpl(bigD,plDD)
        plDD['iHoldCurr']=0
        plot.stims_fixedHoldCurr(bigD,plDD)
        
    if 1:   # spike analysis, all data, many 2D plots
        ihc=0  # select holding current
        #plDD['iHoldCurr']=ihc # holding current index
        plDD['fwhmLR']=[0.5,3.5] # clip plotting range
        plDD['text1']='holdCurr=%.2f nA'%(inpMD['holdCurr'][ihc])
        plot.spikes_survey2D(bigD,plDD)
        
    if 1:   # spike analysis, one stim-ampl, many 1D plots
        if simAuth=='simRoy':
            plDD['iStimAmpl']=19 # stim-ampl index --> ampl=1.0 FS
        plDD['fwhmLR']=[0.5,3.5] # clip plotting range
        plot.spikes_survey1D(bigD,plDD)        
        
    plot.display_all(args.dataName)
