#!/usr/bin/env python3
'''
plot BBP3 simulation data

./plotBaseVolts.py --dataPath  /pscratch/sd/k/ktub1999/BBP_TEST2/runs2/3800565_1/L4_SScADpyr4/ --dataName L4_SS_cADpyr_4-v3-304-c4.h5  -i 161

'''

from pprint import pprint
from toolbox.Util_H5io3 import   read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
import sys,os,time
from aggregate_Kaustubh import normalize_volts

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='/pscratch/sd/k/ktub1999/BBP_TEST2/runs2/3800564_1/L6_TPC_L1cADpyr4/',help="scored data location")

    parser.add_argument("--dataName",  default='L6_TPC_L1_cADpyr_4-v3-206-c4.h5', help="BBP3 simu file")
   
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-i","--sampleIdx",  default=4,type=int, help="sample index")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='baseBBP3'
    args.showPlots=''.join(args.showPlots)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):  os.mkdir(args.outPath) 
    return args


#............................
#............................
#............................

class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.mL7=['*','^','x','h','o','x','D']

#...!...!..................
    def overlay_50Kstims(self,simD,simMD,plDD,jSamp,figId=7):
        figId=self.smart_append(figId)
        nrow,ncol=5,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,13))

        stimHRN=simMD['stimName50k']
        numTbinHR=25000
        timeHR =np.arange(numTbinHR,dtype=np.float32)*simMD['timeAxis']['step'] /5. 
        
        stimLRN=simMD['stimName']
        stimLR=simD['stims'][stimLRN[0]] # to generate x-axis
        numTbinLR=stimLR.shape[0]
        timeLR =np.arange(numTbinLR,dtype=np.float32)*simMD['timeAxis']['step']
        Nst=len(stimLRN)
        
        #print('zzzm',timeHR.shape,stimLR.shape)

        def export_stim(name,data,fact):
            outF='out/'+name+'_kevin.csv'
            fd = open(outF, 'w')
            for line in data:
                fd.write('%.4e\n'%(line*fact))
            fd.close()
            print('saved', outF)
        
        for j in range(Nst):
            ax = self.plt.subplot(nrow,ncol,1+j)
            nameHR=stimHRN[j]
            stimHR=np.zeros(numTbinHR)
            stimHR[5000:]=simD['stims50k'][nameHR]
            if 'step' in nameHR or 'ramp' in nameHR: stimHR/=1000.
            ax.plot(timeHR,stimHR, linewidth=0.7,label=nameHR)

            stimDHR=np.interp(timeLR,timeHR,stimHR)
            ax.plot(timeLR,stimDHR, linewidth=0.7,label='down-sampl',ls='--')
                    
            nameLR=stimLRN[j]
            stimLR=simD['stims'][nameLR]
            ax.plot(timeLR,stimLR, linewidth=0.7,label=nameLR)    
            export_stim(nameLR,stimDHR,0.3)
            ax.set_xlim(-5.,505.)
                
            ax.legend(loc='best')
            ax.set_xlabel('time (ms) ')
            ax.set_ylabel('stim (nA) ')

            if i==0:
                ax.text(0.2,0.02,simMD['bbpName'],transform=ax.transAxes,color='g')
            if i==1:
                txt='sample=%d'%jSamp
                ax.text(0.2,0.02,txt,transform=ax.transAxes,color='g')
            if i==2:
                ax.text(0.2,0.02,args.dataName,transform=ax.transAxes,color='g')
            

#...!...!..................
    def overlay_volts(self,simD,simMD,plDD,jSamp,figId=7):
        figId=self.smart_append(figId)
        nrow,ncol=3,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,13))

        waveA=simD['volts'][jSamp]
        stimNL=simMD['stimName']
               
        # prep time axis
        numTbin,Nbo,Nst=waveA.shape
        timeV =np.arange(numTbin,dtype=np.float32)*simMD['timeAxis']['step']
        jj=Nst-1  # index of the last stim
        
        bodyL=simMD['probeName']
        for i in range(Nbo):
            ax = self.plt.subplot(nrow,ncol,1+i)
            for j in range(Nst-1):
                dwave=waveA[:,i,j]-waveA[:,i,jj]
                ax.plot(timeV,dwave, linewidth=0.7,label=stimNL[j])
                #ax.set_xlim(100.,120.)
                
            ax.legend(loc='best', title='DIFF vs. '+stimNL[jj])
            ax.set_xlabel('time (ms) ')
            ax.set_ylabel( bodyL[i])

            if i==0:
                ax.text(0.2,0.02,simMD['bbpName'],transform=ax.transAxes,color='g')
            if i==1:
                txt='sample=%d'%jSamp
                ax.text(0.2,0.02,txt,transform=ax.transAxes,color='g')
            

#...!...!..................
    def waveArray(self,simD,simMD,plDD,jSamp,figId=5):
        figId=self.smart_append(figId)
        nrow,ncol=3+1,6
        fig=self.plt.figure(figId,facecolor='white', figsize=(20,10))
                
        waveA=simD['volts'][jSamp]#- simD['volts'][1]
        stimNL=simMD['stimName']
               
        # prep time axis
        numTbin,Nbo,Nst=waveA.shape
        timeV =np.arange(numTbin,dtype=np.float32)*simMD['timeAxis']['step']
        
        bodyL=simMD['probeName']
        print('plWA:',waveA.shape)        
        assert Nst*Nbo<=nrow*ncol

        yMin,yMax=np.min(waveA),np.max(waveA)
        for j in range(Nst):
            ax = self.plt.subplot(nrow,ncol,1+j)
            stimN=simMD['stimName'][j]
            print('bbb',stimN)
            stim=simD['stims'][stimN]
            ax.plot(timeV,stim)
            
            ax.set_ylabel(stimN)
            #.. print stim adj params
            parU=simD['unit_stim_adjust'][jSamp,:,j]
            txt='  '.join(['u:%.3f'%u for u in parU ])
            ax.text(0.25,0.6,txt,transform=ax.transAxes,color='g')
            
            parP=simD['phys_stim_adjust'][jSamp,:,j]
            txt='  '.join(['p:%.3f'%u for u in parP ])
            ax.text(0.25,0.3,txt,transform=ax.transAxes,color='b')
            
            for i in range(Nbo):
                ii=j +Nst*(i+1)
                ax = self.plt.subplot(nrow,ncol,1+ii)
            
                #ax.plot(timeV,stimA[n]*spFac[n], 'r',linewidth=0.5,label='stim')
                ax.plot(timeV,waveA[:,i,j], 'b',linewidth=0.7)#,label=yLab[i])
                ax.set_ylim(yMin,yMax)
                #break
                #if i<2:  ax.set_xticklabels([])
                if j==0:
                    tt=bodyL[i]+' (mV)'
                    ax.set_ylabel(tt)
                if i==0:
                    if j==0:
                        ax.text(0.2,0.02,simMD['bbpName'],transform=ax.transAxes,color='g')
                    if j==1:
                        txt='sample=%d'%jSamp
                        ax.text(0.2,0.02,txt,transform=ax.transAxes,color='g')
                    if j==2:
                        ax.text(0.2,0.02,args.dataName,transform=ax.transAxes,color='g')
 
            
            ax.set_xlabel('time (ms) ')

            continue
            
            if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            if 'amplLR' in plDD: ax.set_ylim(tuple(plDD['amplLR']))

            
#...!...!..................
    def paramCondCorr(self,simD,simMD,plDD,figId=2):
        figId=self.smart_append(figId)
        
        parName=simMD['parName']
        parU=simD['unit_par']
        parP=simD['phys_par']
        nrow,ncol=4,5
        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,10))
        
        npar=parP.shape[1]
        #binsX=np.linspace(mm1,mm2,30)
        binsX=30
        
        colMap='GnBu'
        for iPar in range(npar):

            ax = self.plt.subplot(nrow,ncol,1+iPar)
            # plot stim(j)
            
            u=parU[:,iPar]
            p=parP[:,iPar]
            pLog= iPar not in simMD['linearParIdx']  # move it up?
            if pLog:  p=np.log10(p)

            uL=min(u);uR=max(u)
            pL=min(p);pR=max(p)

            print(iPar,'U:[%.2f,%.2f] d=%.2f'%(uL,uR,uR-uL),'P:[%.2f,%.2f] d=%.2f'%(pL,pR,pR-pL), pLog,parName[iPar])

            zsum,xbins,ybins,img = ax.hist2d(u,p,bins=binsX, cmin=0., cmap = colMap) #norm=mpl.colors.LogNorm(),

            ax.set_title('%d:%s'%(iPar,parName[iPar]), size=10)
            

            # more details  will make plot more crowded
            self.plt.colorbar(img, ax=ax)
            txt1='y=log10(p)' if pLog else 'y=p'
            ax.text(0.2,0.8,txt1,transform=ax.transAxes)
            if iPar==0:
                ax.text(0.05,0.02,simMD['bbpName'],transform=ax.transAxes,color='b')
            yy=0.90; xx=0.04
            #if j==0: ax1.text(xx,yy,tit1,transform=ax1.transAxes)
            #if j==1: ax1.text(xx,yy,'nSampl=%d'%(u.shape[0]),transform=ax1.transAxes)
 
            
#...!...!..................
    def paramStimCorr(self,simD,simMD,plDD,figId=3):
        figId=self.smart_append(figId)
       
        parName=simMD['stimParName']
        parU=simD['unit_stim_adjust']
        parP=simD['phys_stim_adjust']
        nrow,ncol=1,2
        
        fig=self.plt.figure(figId,facecolor='white', figsize=(7.,3))
        
        npar=parP.shape[1]
        binsX=30
        
        colMap='GnBu'
        for iPar in range(npar):
            ax = self.plt.subplot(nrow,ncol,1+iPar)
            # plot stim(j)
            print('\nstimpar:',iPar,parName[iPar],parU.shape)
            print('parU',parU[:30,iPar])
            u=parU[:,iPar].reshape(-1)  # needed only for stim-adj
            p=parP[:,iPar].reshape(-1)
            zsum,xbins,ybins,img = ax.hist2d(u,p,bins=binsX, cmin=0., cmap = colMap) #norm=mpl.colors.LogNorm(),
           
            ax.set_title('%d:%s'%(iPar,parName[iPar]), size=10)            

            # more details  will make plot more crowded
            self.plt.colorbar(img, ax=ax)
            
            if iPar==0:
                ax.text(0.05,0.02,simMD['bbpName'],transform=ax.transAxes,color='b')
            yy=0.90; xx=0.04
            #if j==0: ax1.text(xx,yy,tit1,transform=ax1.transAxes)
            #if j==1: ax1.text(xx,yy,'nSampl=%d'%(u.shape[0]),transform=ax1.transAxes)
 


#...!...!..................
def import_stims_from_CVS(do50k=False):
    name50k={'5k0chaotic4.csv':'chaotic_50khz.csv',
             '5k0step_200.csv':'step_200_50khz.csv',
             '5k0ramp.csv':'ramp_50khz.csv',
             '5k0chirp.csv':'chirp_50khz.csv',
             '5k0step_500.csv':'step_500_50khz.csv'
    }
    #name50k['5k0chaotic4.csv']='chaotic_50khz.csv'
    name50k['5k0chaotic4.csv']='cahotic_50khz.csv'
    if do50k: stimPath='Kevinl5pyr'
    #else:    stimPath='/global/cscratch1/sd/ktub1999/stims/'
    else:    stimPath='/global/homes/b/balewski/neuronInverter/packBBP3/stims_dec26'
    nameL2=[]
    outD={}
    for name in  simMD['stimName']:
        name2=name[:-4]
        if do50k:
            name=name50k[name]
            name2=name2.replace('5k','50k')
        inpF=os.path.join(stimPath,name)
        print('import  stim', inpF)
        fd = open(inpF, 'r')
        lines = fd.readlines()
        print('  got',name, len(lines), type(lines[0]))
        vals=[float(x) for x in lines ]
        data=np.array( vals, dtype=np.float32)
        #print('ddd',data.shape, data[::100],data.dtype)
        
        nameL2.append(name2)
        if do50k:
            outD[name2]=data*1.5 # no pre 0s
        else:
            outD[name2]=data[1000:]
        #break
    # ... store results in containers
    
    if do50k:
        simD['stims50k']=outD
        simMD['stimName50k']=nameL2
    else:
        simD['stims']=outD
        simMD['stimName']=nameL2
        
    print('rrr',sorted(outD),simMD['stimName'])
    #print('zzz',sorted(simD['stims']),type(data), type(simD['stims']['5k0chaotic4']))
    


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    inpF=os.path.join(args.dataPath,args.dataName)
    simD,simMD=read3_data_hdf5(inpF)
    print('M:sim meta-data');   pprint(simMD)
    if 0:
        print('all unit_stim_adjust[:,0]')
        print(simD['unit_stim_adjust'][-30:,0])
        ok99_jan
    
    if 0:
        for i,x in enumerate(simMD[ 'parName']):
            print(i,x)
        ok0
    
    if 0: # also import 50k stims for comparison
        import_stims_from_CVS(do50k=True)
    import_stims_from_CVS()
    
    if 1:  # patch 2
        simMD['stimParName']= ['Mult','Offset']

    if 0:
         normalize_volts(simD['volts'],args.dataName,verb=1)
         
    # print selected data
    j=args.sampleIdx
    print('jSamp=%d'%j)
    assert j < simD['volts'].shape[0] , 'not enouh samples in the input, reduce idx'
    
    nStim=len(simMD['stimName'])
    print('stim adjust:',      simMD['stimParName'])
    for i in range(nStim):
        stN=simMD['stimName'][i]
        print(i,stN,'phys:',simD['phys_stim_adjust'][j,:,i],'unit:',simD['unit_stim_adjust'][j,:,i])
        

    print('phys_par:',simD['phys_par'][j])
    print('unit_par:',simD['unit_par'][j])

    
    #test_fp16(simD)
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    plDD={}
    #for x in [ 'units','shortName','bbpName']: plDD[x]=inpMD[x]
    #plDD['stimAmpl']=np.array(inpMD['stimAmpl'])

    plDD['shortName']=args.dataName
    #plDD['simVer']='NeuronSim_2019_1'
        

    if 'a' in args.showPlots:
        plot.waveArray(simD,simMD,plDD,j)
        
    if 'b' in args.showPlots:
        plot.paramCondCorr(simD,simMD,plDD)
        
    if 'c' in args.showPlots:
        plot.paramStimCorr(simD,simMD,plDD)
        
    if 'd' in args.showPlots:
        plot.overlay_volts(simD,simMD,plDD,j)
        
    if 'e' in args.showPlots:
        plot.overlay_50Kstims(simD,simMD,plDD,j)
        

    plot.display_all(args.dataName)
