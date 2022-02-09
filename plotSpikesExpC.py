#!/usr/bin/env python3
'''
plot scores and waveforms w/ spikes

'''
import sys,os
from pprint import pprint
from matplotlib.patches import Rectangle

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

    parser.add_argument("-d", "--dataPath",  default='out/',help="scored data location")

    parser.add_argument("--dataName",  default='211219_5a-A0.14', help="shortName ")
    
    #parser.add_argument("--amplIdx",  default=6,type=int, help="amplitude index")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.formatName='spiker'
    args.prjName='scoreC'

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
        nrow,ncol=4,2; yIn=9
        #nrow,ncol=2,2; yIn=5
        if plDD['has_stim_spikes'] : nrow,ncol=2,1; yIn=8
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,yIn))
        
        timeV=bigD['time']
        spikeC=bigD['spikeCount']
        spikeT=bigD['spikeTrait']
        sweepT=plDD['sweepTrait']
        N=spikeC.shape[0] 
        
        stimAmpl=plDD['stimAmpl']

        stimA=bigD['stim_wave']*30 # better visibility
        waveA=bigD['soma_wave']
        
        #print('wdif0',waveA.shape,stimA.shape,len(spikeA[0]))
        idxL,idxR,idxS=plDD['idxLR']
        j=0
        for n in range(idxL,idxR,idxS):
            if n>=N : continue
            j+=1
            if j>=nrow*ncol: continue
            
            ax = self.plt.subplot(nrow,ncol,j)
            ks=spikeC[n]
            sweepId,timeLive,xstimAmpl=sweepT[n]

            ax.plot(timeV,waveA[n], 'b',linewidth=0.7,label='soma wave')
            if not ['has_stim_spikes']:
                ax.plot(timeV,stimA[n], 'C1',linewidth=0.5,label='stim (a.u.)')
                txt='numSpike=%d  sweep=%d  wallTime=%.1f min'%(ks,n,timeLive/60.)
                ax.text(0.35,0.9,txt,transform=ax.transAxes,color='g')
            if 1: # show valid spikes
                spikes=spikeT[n][:ks]
                #print(n,'spikes=',spikes)
                tPeak=spikes[:,0]
                ampPeak=spikes[:,1]
                tBase=spikes[:,3]
                ampBase=spikes[:,4]
                twidthBase=spikes[:,5]
                ax.plot(tPeak,ampPeak,"*",color='m')
                ax.plot(tBase,ampBase,"x",color='r')
                ax.plot(tBase+twidthBase,ampBase,".",color='r')
    
            if j==1:   ax.text(0.01,0.9,'cell='+plDD['shortName'],transform=ax.transAxes,color='m')
            #if j==2:  ax.text(0.01,0.9,plDD['text1'],transform=ax.transAxes,color='m')

            ax.axhline(plDD['peak_thr'],color='m', linestyle='--',linewidth=0.5)
            ax.grid()
            if plDD['has_stim_spikes']: return
            yLab='waveform (mV)'
            xLab='time (ms), '+plDD['shortName']+', n=%d'%n
            ax.set(xlabel=xLab,ylabel=yLab)
            
            if j==1 : ax.legend(loc='center right')

            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

 
#...!...!..................
    def score_wallTime(self,bigD,plDD,figId=6): # NEVER USED
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
        ax = self.plt.subplot(nrow,ncol,1)

        sweepC=bigD['sweepCnt']
        traitA=bigD['sweepTrait']
        N=sweepC.shape[0] # num stim ampl
        stimA=plDD['stimAmpl']
        scoreA=bigD['spikeCount'] 
        stimAmplA=plDD['stimAmpl']
        #print('P:spikeCount',bigD['spikeCount'] )

        avrScore=[]; jj=0
        for n in range(N):  # loop over stim-ampl
           k=sweepC[n]
           traits=traitA[n][:k]  #[sweepId, sweepTime, serialRes]*nSweeps
           #print('q3',traits.shape,traits)
           wallTA=np.array(traits[:,1])/60.
           scoreV=scoreA[n][:k]
           if np.max(scoreV)<=0: continue
           dLab='ampl=%.2f'%(stimAmplA[n])
           dmk=self.mL7[jj%7]; jj+=1
           ax.plot(wallTA,scoreV,dmk+"-",label=dLab)
        ax.legend(bbox_to_anchor=(0., 1.06, 1., .106), loc=2,
                   ncol=4, mode="expand", borderaxespad=0.,
                  title=plDD['shortName']+', stim ampl (FS)' )
        
        yTit='spike count'
        ax.set(ylabel=yTit,xlabel='wall time (min)')
        ax.grid()
        ax.set_ylim(0,10.5)
        
#...!...!..................
    def spikes_survey2D(self,bigD,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=3,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,9))

        spikeC=bigD['spikeCount']
        spikeA=bigD['spikeTrait']
        sweepA=plDD['sweepTrait']

        N=spikeC.shape[0] 
        
        stimAmpl=plDD['stimAmpl']
        
        # repack data for plotting
        tposA=[]; twidthA=[]; amplA=[]; swTimeA=[]
        for j in range(N): #loop over waveforms
            ks=spikeC[j]
            #print('qwq',n,j,ks)
            if ks==0: continue
            sweepId,timeLive,xstimAmpl=sweepA[j]
            assert xstimAmpl==stimAmpl
            print('j',j,spikeA[j].shape)
            spikes=spikeA[j][:ks] # use valid spikes
            for rec in spikes:
                tPeak,ampPeak,twidthFwhm,tBase,ampBase,twidthBase=rec
                tposA.append(tPeak)
                twidthA.append(twidthFwhm)
                amplA.append(ampPeak)
                swTimeA.append(timeLive)
                
        wallTA=np.array(swTimeA)/60.

        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(tposA,amplA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike ampl (mV)')
        ax.text(0.01,0.9,plDD['shortName'],transform=ax.transAxes,color='m')
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

        ax = self.plt.subplot(nrow,ncol,2)
        # free
                
        ax = self.plt.subplot(nrow,ncol,3)
        ax.scatter(tposA,twidthA, alpha=0.6)
        ax.set(xlabel='stim time (ms)',ylabel='spike width (ms)')
        if 'fwhmLR' in plDD: ax.set_ylim(tuple(plDD['fwhmLR']))
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))

        ax = self.plt.subplot(nrow,ncol,4)
        # free
        
        ax = self.plt.subplot(nrow,ncol,5)
        ax.scatter(wallTA,amplA, alpha=0.6,color='g')
        ax.set(xlabel='wall time (min)',ylabel='spike ampl (mV)')
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,6)
        ax.scatter(wallTA,twidthA, alpha=0.6,color='g')
        ax.set(xlabel='wall time (min)',ylabel='spike width (ms)')
        ax.grid()

        return twidthA, amplA
        
#...!...!..................
    def spikes_survey1D(self, twidthA, amplA,plDD,figId=6):
        figId=self.smart_append(figId)
        nrow,ncol=1,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(9,2.5))

        ax = self.plt.subplot(nrow,ncol,1)
        binsX= np.linspace(0.5,3.5,20)
        ax.hist(twidthA,bins=binsX,facecolor='b')        
        ax.set(xlabel='spike half-width (ms), aka FWHM',ylabel='num spikes')
        if 'fwhmLR' in plDD: ax.set_xlim(tuple(plDD['fwhmLR']))
        ax.grid()
                
        ax = self.plt.subplot(nrow,ncol,2)
        binsX= np.linspace(20,60,40)
        ax.hist(amplA,bins=binsX,facecolor='C1')        
        ax.set(xlabel='spike peak ampl (mV)',ylabel='num spikes')
        ax.grid()
                
        
#...!...!..................
def M_save_summary(sumL):
    sum2D=np.array(sumL)
    outF=args.dataName+'.sum.h5'
    print('sum2D:',sum2D.shape)
    outD={'spikeCount':sum2D}
    write3_data_hdf5(outD,args.outPath+outF)
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpF=os.path.join(args.dataPath,'%s.%s.h5'%(args.dataName,args.formatName))
    bigD,spkMD=read3_data_hdf5(inpF, verb=1)
    pprint(spkMD)

    print('M:numWaves=%d  stimAmpl=%.2f'%(bigD['spikeCount'].shape[0],spkMD['stimAmpl']))

    print('M:spikeTrait',bigD['spikeTrait'].shape )
    
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    plDD={}
    for x in [ 'expUnits','shortName','stimAmpl','sweepTrait','has_stim_spikes']: plDD[x]=spkMD[x]
    #plDD['stimAmpl']=np.array(inpMD['stimAmpl'])
    
    #plDD['timeLR']=[0.,200.]  # (ms)  time range
        
    if 1:  # wavforms as array, decimated
        plDD['idxLR']=[0,8,1] # 1st,last,step

        #plDD['text1']='stim ampl=%.2f FS'%(inpMD['stimAmpl'][args.amplIdx])
        #plDD['timeLR']=[10.,160.]  # (ms)  time range 
        
        plDD['peak_thr']=spkMD['spikerConf']['min_peak_ampl_mV']
        plot.waveArray(bigD,plDD)
                
    if 0:   # score vs. wall time survey
        plot.score_wallTime(bigD,plDD)
        
    if 0:   # spike analysis, all data, many 2D plots
        #plDD['fwhmLR']=[0.5,3.5] # clip plotting range
        twidthA, amplA=plot.spikes_survey2D(bigD,plDD)        
        plot.spikes_survey1D( twidthA, amplA,plDD)        
        
    plot.display_all('scoreExp')
