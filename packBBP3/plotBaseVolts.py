#!/usr/bin/env python3
'''
plot BBP3 simulation: soma volts

'''

from pprint import pprint
from toolbox.Util_H5io3 import   read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
import sys,os

import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='data5/',help="scored data location")

    parser.add_argument("--dataName",  default='L4_SS_cADpyr230_1-v3-1-1-c1.h5', help="BBP3 simu file")
    #parser.add_argument("--metaData",  default='data1/bbp3_simu_feb9.meta.h5', help="meta-data for BBP3 simu")
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")

    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-i","--sampleIdx",  default=6,type=int, help="sample index")

    args = parser.parse_args()
    args.formatVenue='prod'
    args.prjName='baseBBP3'

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
    def overaly_volts(self,simD,simMD,plDD,jSamp,figId=7):
        figId=self.smart_append(figId)
        nrow,ncol=3,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,13))

        waveA=simD['volts'][jSamp]
        stimNL=simMD['stimName']
               
        # prep time axis
        numTbin,Nbo,Nst=waveA.shape
        timeV =np.arange(numTbin,dtype=np.float32)*simMD['timeAxis']['step']
        
        bodyL=simMD['probeName']
        for i in range(Nbo):
            ax = self.plt.subplot(nrow,ncol,1+i)
            for j in range(Nst-1):
                dwave=waveA[:,i,j]-waveA[:,i,4]
                ax.plot(timeV,dwave, linewidth=0.7,label=stimNL[j])
                #ax.set_xlim(100.,120.)
                
            ax.legend(loc='best', title='DIFF vs. '+stimNL[4])
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
        nrow,ncol=3+1,5
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,10))
                
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

            #ax.plot([0, 1], [0,1], color='magenta', linestyle='--',linewidth=0.5,transform=ax.transAxes) #diagonal
            # 
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
        nrow,ncol=1,5
        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,3))
        
        npar=parP.shape[1]
        #binsX=np.linspace(mm1,mm2,30)
        binsX=30
        
        colMap='GnBu'
        for iPar in range(npar):

            ax = self.plt.subplot(nrow,ncol,1+iPar)
            # plot stim(j)
            print(iPar,npar)
            u=parU[:,iPar].reshape(-1)  # needed only for stim-adj
            p=parP[:,iPar].reshape(-1)
            zsum,xbins,ybins,img = ax.hist2d(u,p,bins=binsX, cmin=0., cmap = colMap) #norm=mpl.colors.LogNorm(),

            #ax.plot([0, 1], [0,1], color='magenta', linestyle='--',linewidth=0.5,transform=ax1.transAxes) #diagonal
            # 
            ax.set_title('%d:%s'%(iPar,parName[iPar]), size=10)
            

            # more details  will make plot more crowded
            self.plt.colorbar(img, ax=ax)
            
            if iPar==0:
                ax.text(0.05,0.02,simMD['bbpName'],transform=ax.transAxes,color='b')
            yy=0.90; xx=0.04
            #if j==0: ax1.text(xx,yy,tit1,transform=ax1.transAxes)
            #if j==1: ax1.text(xx,yy,'nSampl=%d'%(u.shape[0]),transform=ax1.transAxes)
 


#...!...!..................
def import_stims_from_CVS():
    nameL2=[]
    outD={}
    for name in  simMD['stimName']:
        inpF=os.path.join('data5',name)
        print('import ', inpF)
        fd = open(inpF, 'r')
        lines = fd.readlines()
        #print('aaa', len(lines), type(lines[0]))
        vals=[float(x) for x in lines ]
        data=np.array( vals, dtype=np.float32)
        #print('ddd',data.shape, data[::100],data.dtype)
        name2=name[:-4]
        nameL2.append(name2)
        outD[name2]=data[1000:]
        #ok22

    # ... store results in containers
    simMD['stimName']=nameL2
    simD['stims']=outD
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

    import_stims_from_CVS()
    
    if 1:  # patch 2
        simMD['stimParName']= ['Mult','Offset']

        
    # print selected data
    j=args.sampleIdx
    print('jSamp=%d'%j)
    
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
        plot.overaly_volts(simD,simMD,plDD,j)
        

    plot.display_all(args.dataName)
