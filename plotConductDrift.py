#!/usr/bin/env python3
'''
plot scores and waveforms w/ spikes

'''
from pprint import pprint
from toolbox.Util_H5io3 import  write3_data_hdf5, read3_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from toolbox.Plotter import get_arm_color
import os 
import numpy as np
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument("-d", "--dataPath",  default='out-exp-2012dec/',help="scored data location")

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
    def physParamsDrift(self,bigD,plDD,figId=4):
        logB=plDD['log10_phys_cent']
        parName=plDD['parName']
        crossTile=plDD['crossTile']
        
        nPar=len(parName)
        #assert nPar==logB.shape[1]
        #xprint('PP1Da:',logB.shape)
        nrow,ncol=4,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(3.3*ncol,2.2*nrow))
        
        for i in range(0,nPar):
            #if i>0: break
            if i in crossTile: continue  # show only conduct w/ good ML pred
            
            ax=self.plt.subplot(nrow,ncol,i+1)

            xV=[];eV=[];yLab=[]
            for rec in plDD['exp']:
                txt,avr,std,M=rec
                yLab.append(txt)
                xV.append(avr[i])
                eV.append(std[i])

            ax.errorbar(xV,yLab,xerr=eV,marker='o')
            lbase=logB[i]
            hcol=get_arm_color(parName[i])
            lfac=1.5
            binsX=np.linspace(lbase-lfac,lbase+lfac,30)      

            # compute num std from 'a'
            for j in range(1,len(xV)):
                dx=xV[j]-xV[0]
                re2=dx*dx/(eV[j]**2 + eV[0]**2)
                nSig=np.sqrt(re2)
                #print(i,parName[i],j,dx,re2,nSig)
                if nSig>3.0:
                    if dx<0: nSig*=-1
                    txt='%.1f sig'%nSig
                    ax.text(xV[j]+0.1,j-0.2,txt, color='m')
                    txt2=yLab[j].replace('\n','')
                    print('%2d=%s  drug:%s,  change: %s'%(i,parName[i],txt2,txt))
                    
            ax.set(title=parName[i], xlabel='log10(cond/(S)) p=%d'%(i),ylabel='experiment')
            ary=0.5
            #arrow(x, y, dx, dy, **kwargs)
            ax.arrow(lbase, ary, 0,-ary+0.06, head_width=0.10, head_length=0.2*ary, fc='k', ec='k')
            if i in crossTile:
                ax.text(0.8,0.1,"BAD\nML",transform=ax.transAxes, color='g')
            
            for x in [-1,1]:
                ax.axvline(lbase+x, color='black', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(-0.05,0.78,plDD['short_name'][:25],transform=ax.transAxes, color='r')
                ax.text(-0.05,0.65,plDD['cell_type'],transform=ax.transAxes, color='r')
            if i==-1:
                ax.text(0.05,0.85,'n=%d'%logC.shape[0],transform=ax.transAxes, color='r')

#...!...!..................
def process_one( bigD):
    predC=bigD['pred_cond']
    logC=np.log10(predC)
    numMeas=predC.shape[0]
    #print('qq0',predC.shape)
    assert numMeas>1 
    avrC=np.mean(logC,axis=0)
    #print('qq',avrC.shape)
    stdC=np.std(logC,axis=0)/np.sqrt(numMeas)
    return avrC,stdC,numMeas
#...!...!..................
#...!...!..................
            
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()

    inpD={'coreName':'211215_0*-A0.14_MLontra3',
          'cell_type': 'inhibitory',
          'exp':[ ['a','no drug'],['b','0.5 nM'],['c','0.5 nM\n+Dclamp'],
                  ['d','1.0 nM'],['e','1.0 nM\n+Dclamp']] }

    if 0:
        inpD={'coreName':'211219_5*-A0.14_MLontra4cl',
              'cell_type': 'excitatory',
              'exp':[ ['a','no drug'],['b','0.25 nM'],['c','0.5 nM']] }
          
    
    plDD={'exp':[],'short_name':inpD['coreName'], 'cell_type':inpD['cell_type']}
    if inpD['cell_type']== 'excitatory':
        plDD['crossTile']=[12]
    else:  # inhibitory
        plDD['crossTile']=[2,3,6,7,13,15,16,17,18]
        
    for rec in inpD['exp']:
        x,txt=rec
        name=inpD['coreName'].replace('*',x)
        inpF='%s.%s.h5'%(name,args.formatName)
        print('M:read',inpF)
        bigD,expMD=read3_data_hdf5(os.path.join(args.dataPath,inpF), verb=args.verb-1)
        avr,std,M=process_one( bigD)
        plDD['exp'].append([txt,avr,std,M])
    
    # - - - - - PLOTTER - - - - -

    plot=Plotter(args)
    for x in [ 'parName', 'log10_phys_cent']: plDD[x]=expMD[x]
   
    if 1:  # phys conductances

        plot.physParamsDrift(bigD,plDD)
        
    plot.display_all('scoreSim')
