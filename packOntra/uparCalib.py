#!/usr/bin/env python3
import sys,os
'''
re-calibrates upar for a set of cells to conver common physical range
Note1, the name of params must all match across cells
Note2, frozen params are ignored: if 'fixed' or 'const'


module load pytorch
export HDF5_USE_FILE_LOCKING=FALSE 

'''

import numpy as np
import time
from pprint import pprint

from toolbox.Util_IOfunc import write_yaml, read_yaml, write_data_hdf5, read_data_hdf5
from toolbox.Plotter_Backbone import Plotter_Backbone
from matplotlib import cm as cmap

import argparse 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False,help="disable X-term for batch mode")
        
    parser.add_argument("-o","--outPath",  default='out/',help="output path for plots and tables")

    parser.add_argument("--cellName", type=str, nargs='+', default=['bbp019'], help="cell shortName list, blanks separated")

    args = parser.parse_args()
    #args.dataPath='/global/cfs/cdirs/m2043/balewski/neuronBBP-pack40kHzDisc/probe_orig/'
    args.dataPath='/global/cfs/cdirs/m2043/balewski/neuronBBP2-data_67pr/'
    args.prjName='uparCal'
    args.formatVenue='prod'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter_UparCalib(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)
        
#...!...!..................
    def paramRange(self,log10Base,parNL,idx0=0,figId=10,tit=''):
        
        nrow,ncol=5,2
        figId=self.smart_append(figId+idx0)
        # Remove horizontal space between axes
        fig,axA = self.plt.subplots(nrow, ncol, sharex=True, sharey=True,num=figId,facecolor='white', figsize=(14,10))
        axA=axA.flatten()
        
        nPar=nrow*ncol
        data2D=log10Base[:,idx0:idx0+nPar]
        parNV=parNL[idx0:idx0+nPar]
        nPar=data2D.shape[1] # for the last slice over parameters
                
        for ip in range(nPar):
            logP=data2D[:,ip]
            parN=parNV[ip]
            #print('ip:',ip,parN,logP.shape,logP)
            ax=axA[ip]
            nSamp=20
            xLo=-7; xHi=2
            binsX=np.linspace(xLo,xHi,int((xHi-xLo)*nSamp/2.)+1)
            cumHit=[]
            for logp in logP:
                valA=np.linspace(logp-1,logp+1,nSamp+1)
                #print('aa',valA.shape)
                cumHit.append(valA)
                #break
            cumHit=np.array(cumHit).flatten()    
            ax.hist(cumHit,bins=binsX)
            ax.set_title(parN+' '+tit)
            if ip>=nPar-ncol:
                ax.set(xlabel='log10(conductance)')
            if ip%ncol==0: ax.set(ylabel='cells')
            ax.grid(linestyle=':')
            #break

        
#...!...!..................
    def paramCorrel(self,Ua,Ub,parNL,figId=1,tit=''):
        
        nrow,ncol=4,5
        figId=self.smart_append(figId)
        # Remove horizontal space between axes
        fig,axA = self.plt.subplots(nrow, ncol, sharex=True, sharey=True,num=figId,facecolor='white', figsize=(14,8))
        axA=axA.flatten()
        
        nPar=Ua.shape[1]
        assert nPar<=nrow*ncol
                
        for ip in range(nPar):
            Xa=Ua[:,ip]
            Xb=Ub[:,ip]
            parN=parNL[ip]
            ax=axA[ip]
            ax.set_aspect(1.0)
            ax.plot(Xa,Xb)
            ax.set_title(parN+' '+tit)
            if ip>=nPar-ncol:
                ax.set(xlabel='U_A')
            if ip%ncol==0: ax.set(ylabel='U_B')
            ax.grid(linestyle=':')
            #break

#...!...!..................
def extractCellParams(metaD,parN2D):
    parBase=[]
    physRange=metaD['dataInfo']['physRange']

    for [idx,name0] in parN2D:
        [name,a,b] =physRange[idx]
        #print (name0,name)
        assert name0==name
        assert a>0
        base=a*10.                    
        ln10p=np.log10(base)
        #print(name,base,lgb)
        #print (name,name0)
        parBase.append(ln10p)
        
    '''
    print('LL:',len(parBase),len(parNL))
    aS=set(parNL)
    nameL2=[ name for name,a,b in physRange ]
    bS=set(nameL2)
    print('A-B:',aS-bS)
    print('B-A:',bS-aS)
    '''
    return parBase
    
            
   
#...!...!..................
def find_u2ustar_trans(parN2D,data2D):
    print('FEC: log10Base:',data2D.shape)

    nCell,nPar=data2D.shape
    for j in range( nPar):
        valV=data2D[:,j]
        a=np.min(valV)
        b=np.max(valV)
        
        centP=(a+b)/2.
        delP=(b-a)/2.+1
        
        parN2D[j]+=[centP,delP]
        #print(j,a,b,abh,dh1)

#...!...!..................
def buil_var_parL(metaD):

    #blackList={'gkbar.StochKv.den','gK.Tstbar.K.Tst.axn','gNap.Et2bar.Nap.Et2.den','gkbar.KdShu2007.den'}  # for all inhibitory cells 
    blackList={} # nothing for excitatory
    # build list of varied params
    parNLall=[ a for a,b,c in metaD['dataInfo']['physRange'] ]
    parN2D=[]
    for i,name in enumerate(parNLall):
        if 'fixed'  in name: continue
        if 'const'  in name: continue
        if name in blackList: continue
        parN2D.append([i,name])
    print('var-par list :%d'%len(parN2D))
    #pprint( parN2D)
    return parN2D

#...!...!..................
def test_u2ustar(jCell=1):
    nSamp=100
    nPar=4
    u=np.random.uniform(-1,1,size=(nSamp,nPar))
    ustar=u
    return u,ustar


#=================================
#=================================
#  M A I N 
#=================================
#=================================
args=get_parser()
plot=Plotter_UparCalib(args)

parRng2D=None
cellNL0=args.cellName
# expand cells to clones
cellNL=[]
for shortN in cellNL0:
    for x in [1,2,3,4,5]:
        if 'bbp208'==shortN and x<3 : continue
    #for x in [3]:
        cellNL.append('%s%d'%(shortN,x))
print('cc',cellNL,len(cellNL))
lgp2D=[]
for shortN in cellNL:
    metaF=args.dataPath+shortN+"/meta.cellSpike.yaml"
    metaD=read_yaml(metaF)
    if parRng2D==None: parRng2D=buil_var_parL(metaD) # do it once, use 1st cell
    lgp2D.append(extractCellParams(metaD,parRng2D))

lgp2D=np.array(lgp2D)
#parNL=np.array(parNL)
#print('M: dump:',log10Base[:3])

print('M: cellN:',len(cellNL),cellNL)
parNL=[x[1] for x in parRng2D]
print('M: parNL len :',len(parNL))#,parNL)
print('\nparNames: [',', '.join(parNL),' ]')


print('M: lgp2D:',lgp2D.shape)
find_u2ustar_trans(parRng2D,lgp2D)
pprint(parRng2D)

u,ustar=test_u2ustar(jCell=2)
#plot.paramCorrel(u,ustar,parNL)

if 1:  # log10(base) overlaps
    #tit=',  all 8 inhib e-types'
    tit=',  all 12*5+3=63 excitatory clones'
    plot.paramRange(lgp2D,parNL,idx0=0,tit=tit)
    plot.paramRange(lgp2D,parNL,idx0=10,tit=tit)

# select random 5 cells
totCell=len(cellNL)
if 0: # pick random
    #totCell=37
    testCellIdx=np.random.choice(totCell,size=3,replace=False)
    print('M: testCellIdx=',testCellIdx)
    print('M: exclude:',np.array(cellNL)[testCellIdx])

#process(log10Base,parNL)
#find_edge_cells(log10Base)

plot.display_all('parRange')
