__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np

from matplotlib import cm as cmap
from toolbox.Plotter_Backbone import Plotter_Backbone
from pprint import pprint

#...!...!..................
def get_arm_color(parName):
    armCol={'apical':'C2', 'axonal':'C3','somatic':'C4','dend':'C5','all':'C6'}
    arm=parName.split('_')[-1]
    #print('ccc',parName, arm)
    if(arm not in armCol.keys()):
        return 'C2'
    hcol=armCol[arm]
    return hcol

#............................
#............................
#............................
class Plotter_NeuronInverter(Plotter_Backbone):
    def __init__(self, args,inpMD,sumRec=None):
        Plotter_Backbone.__init__(self,args)
        self.maxU=1.1
        self.inpMD=inpMD
        self.sumRec=sumRec
        self.formatVenue=args.formatVenue
        # self.idx=range(len(inpMD['parName']))
        # if(args.idx is not None):
        #     self.idx=args.idx
        #     self.idx=[int(i) for i in self.idx]
        
#...!...!..................
    def frames_vsTime(self,X,Y,nFr,figId=7,metaD=None, stim=[]):
        
        if metaD==None:  metaD=self.inpMD
        probeNameL=metaD['featureName']
        
        nBin=X.shape[1]
        maxX=nBin ; xtit='time bins'
        binsX=np.linspace(0,maxX,nBin)
        numProbe=X.shape[-1]

        #print('zz',numProbe,len(probeNameL))
        assert numProbe<=len(probeNameL)
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,8))

        nFr=min(X.shape[0],nFr)
        nrow,ncol=3,3
        if nFr==1:  nrow,ncol=1,1
        print('plot input traces, numProbe',numProbe)

        yLab='ampl (a.u.)'
        
        j=0
        for it in range(nFr):
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax = self.plt.subplot(nrow, ncol, 1+j)
            j+=1
                        
            for ip in range(0,numProbe):
                amplV=X[it,:,ip]#/metaD2['voltsScale']                
                ax.plot(binsX,amplV,label='%d:%s'%(ip,probeNameL[ip]))
                
            if len(stim)>0:
                ax.plot(stim*10, label='stim',color='black', linestyle='--')

            tit='id%d'%(it)
            ptxtL=[ '%.2f'%x for x in Y[it]]
            tit+=' U:'+','.join(ptxtL)
            ax.set(title=tit[:45]+'..',ylabel=yLab, xlabel=xtit)
                        
            # to zoom-in activate:
            if nFr==11:
                ax.set_xlim(4000,5000)
                ax.set_ylim(-1.6,-1.4)
            ax.grid()
            if it==0:
                ax.legend(loc='best', title='input channels')

#...!...!..................
    def param_residua2D(self,U,Z, do2DRes=False, figId=9):
        #colMap=cmap.rainbow
        assert U.shape[1]==Z.shape[1]
        colMap=cmap.GnBu
        sumRec=self.sumRec
        parName=self.inpMD['parName']
        # idx=[0,1,2,3,4,5,6,7,8,9,10,13,14,15]
        if('include' in self.inpMD.keys()):
            idx=self.inpMD['include']
        else:
            idx = range(len(parName))
        # parName=[parName[id] for id in idx]
        # nPar=self.inpMD['num_phys_par']
        nPar=len(idx)
        residualL=sumRec['residual_mean_std']
        
        nrow,ncol=-(-(nPar+1) // 5),5 # BBP3
        #nrow,ncol=4,4  # for proposal update, 2022-01

        if  self.formatVenue=='poster':
            # grant August-2020
            colMap=cmap.Greys
            figId+=100

        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(14,int(3*nrow)))

        #1fig, axs = self.plt.subplots(nrow,ncol, sharex='col', sharey='row', gridspec_kw={'hspace': 0.3, 'wspace': 0.1},num=figId)

        fig, axs = self.plt.subplots(nrow,ncol,num=figId)
        axs=axs.flatten()
        j=0

        for iPar in range(0,nPar):
            ax1=axs[j]; j+=1
            ax1.set_aspect(1.0)
                        
            u=U[:,iPar]
            z=Z[:,iPar]

            mm2=self.maxU
            mm1=-mm2
            mm3=self.maxU/3.  # adjust here of you want narrow range for 1D residua
            binsX=np.linspace(mm1,mm2,30)

            zsum,xbins,ybins,img = ax1.hist2d(z,u,bins=binsX,#norm=mpl.colors.LogNorm(),
                                               cmin=1, cmap = colMap)

            
            ax1.plot([0, 1], [0,1], color='magenta', linestyle='--',linewidth=0.5,transform=ax1.transAxes) #diagonal
            # 
            ax1.set_title('%d:%s'%(iPar,parName[idx[iPar]]), size=10)
            tit4='job:'+str(self.sumRec['jobId'])
            
            # ax1.text(0.0,-0.01,'%.1f'%(self.inpMD['phys_par_range'][idx[iPar]][0]),transform=ax1.transAxes, color = 'blue')
            # ax1.text(1.0,-0.01,'%.1f'%(self.inpMD['phys_par_range'][idx[iPar]][1]),transform=ax1.transAxes, color = 'blue')
            lower_limit = self.inpMD['phys_par_range'][idx[iPar]][0]
            upper_limit = self.inpMD['phys_par_range'][idx[iPar]][1]
            original_ticks=[-1,0,1]
            new_ticks = [lower_limit,'%.1f'%((lower_limit+upper_limit)/2),upper_limit]
            new_ticks_dict={-1:lower_limit,
                            0:(lower_limit+upper_limit)/2,
                            1:upper_limit}
            ax1.set_xticks(original_ticks,new_ticks)
            ax1.set_yticks(original_ticks,new_ticks)
            

            # ax1.figtext(0.5, 0.01, self.inpMD[iPar][0], wrap=True, horizontalalignment='center', fontsize=12)
            if  self.formatVenue=='poster': continue

            # more details  will make plot more crowded
            self.plt.colorbar(img, ax=ax1)
            
            [ _,resM, resS]=residualL[iPar]

            # additional info Roy+Kris did not wanted to see for nicer look
            if j>(nrow-1)*ncol: ax1.set_xlabel('pred (a.u.)')
            if j%ncol==1: ax1.set_ylabel('truth (a.u.)')

            ax1.text(0.4,0.03,'avr=%.3f\nstd=%.3f'%(resM,resS),transform=ax1.transAxes)
            #print('aa z, u, umz, umz-z',parName[iPar],z.mean(),u.mean(),umz.mean(),z.mean()-umz.mean())

            if resS > self.sumRec['lossThrHi']:
                ax1.text(0.1,0.7,'BAD',color='red',transform=ax1.transAxes)
            if resS < 0.01:
                ax1.text(0.1,0.6,'CONST',color='sienna',transform=ax1.transAxes)

            #print('sss');pprint(self.sumRec)
            dom=self.sumRec['domain']
            tit1='dom='+dom
            tit2='MSEloss=%.3g'%self.sumRec[dom+'LossMSE']
            tit3='inp:'+str(self.sumRec['inpShape'])
            
            
            yy=0.90; xx=0.04
            if j==0: ax1.text(xx,yy,tit1,transform=ax1.transAxes)
            if j==1: ax1.text(xx,yy,'nSampl=%d'%(u.shape[0]),transform=ax1.transAxes)
            if j==2: ax1.text(xx,yy,tit2,transform=ax1.transAxes)
            if j==3: ax1.text(xx,yy,'short:'+self.sumRec['short_name'][:20],transform=ax1.transAxes)
            if j==4: ax1.text(xx,yy,'dom:'+self.sumRec['domain'][:20],transform=ax1.transAxes)
            
            if j==6: ax1.text(0.2,yy,tit3,transform=ax1.transAxes)
            if j==7: ax1.text(0.2,yy,tit4,transform=ax1.transAxes)
            
        #.....  more info in not used pannel last pane;
        dataTxt='data:'+sumRec['short_name'][:20]
        txt3='\ndesign:%s\n'%(sumRec['modelDesign'])+dataTxt
        txt3+='\n'+tit4
        txt3+='\n train time/min=%.1f '%(sumRec['trainTime']/60.)
        txt3+='\ntrain.stims %s \n samples: %d'%(sumRec['train_stims_select'],sumRec['train_glob_sampl'])
        txt3+='\ntrain.loss valid %.3g'%(sumRec['loss_valid'])
        txt3+='\npred.loss %s %.3g'%(sumRec['domain'],sumRec[sumRec['domain']+'LossMSE'])
        txt3+='\ninp:'+str(sumRec['inpShape'])
        txt3+='\npred.stims %s\n samples: %d'%(sumRec['pred_stims_select'],u.shape[0])
        ax1=axs[j]
        ax1.axis('off')
        ax1.text(-0.15,-0.10,txt3,transform=ax1.transAxes)



#...!...!..................
    def params1D(self,P,tit1,figId=4,doRange=True):
        
        metaD=self.inpMD
        parName=metaD['parName']
        # idx=[0,1,2,3,4,5,6,7,8,9,10,13,14,15]
        if('include' in self.inpMD.keys()):
            idx=self.inpMD['include']
        else:
            idx = range(len(parName))
        parName=[parName[id] for id in idx]
        nPar=len(idx)
        nrow,ncol=-(-(nPar+1) // 5),5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.75*ncol,2.2*nrow))

        tit0=self.sumRec['short_name'][:27]
        j=1
        binsX=30
        if doRange:
            mm2=1.2; mm1=-mm2
            binsX= np.linspace(mm1,mm2,30)            
        for i in range(0,nPar):
            ax=self.plt.subplot(nrow,ncol,j)
            p=P[:,i]
            hcol=get_arm_color(parName[i])
            if 'true' in tit1: hcol='C0'

            j+=1            
            (binCnt,_,_)=ax.hist(p,binsX,color=hcol)
            cntIn=sum(binCnt)
                        
            ax.set(title=parName[i], xlabel='Upar %d, inRange=%d'%(i,cntIn),ylabel='samples')
                        
            for x in [-1.,1.]:
                ax.axvline(x, color='m', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(-0.05,0.85,tit0,transform=ax.transAxes, color='r')
            if i==2:
                ax.text(0.05,0.85,tit1,transform=ax.transAxes, color='r')
            if i==1:
                ax.text(0.05,0.85,'n=%d'%p.shape[0],transform=ax.transAxes, color='r')

#............................
    def params1D_vyassa(self,P,tit,figId=4,tit2='',isPhys=False):
        #fix_phys_range_handling_iced_params
        metaD=self.inpMD
        parName=metaD['parName']
        nPar=metaD['numPar']
        rangeLD=metaD['physRange']
        nrow,ncol=4,8

        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.7*ncol,2.*nrow))

        j=1
        for i in range(0,nPar):

            if isPhys:
                range12=rangeLD[i]
                #print('\n',i,rangeLD[i][0],parName[i],range12,j)
                base=np.sqrt(range12[1]*range12[2])
                lgBase=np.log10(base)
                mm2=lgBase+1.1
                mm1=lgBase-1.1
                #print('www4',base,lgBase,mm1,mm2)
                p=np.log10(P[:,i])
                unitStr='log10(phys)'
                hcol='g'
            else:
                mm2=1.2; mm1=-mm2
                p=P[:,i]
                unitStr='unit'
                hcol='b'
            binsX= np.linspace(mm1,mm2,30)

            ax=self.plt.subplot(nrow,ncol,j)
            j+=1
            binsY=np.linspace(mm1,mm2,50)
            ax.hist(p,binsX,color=hcol)
            ax.set(title='par %d'%(i),ylabel='frames',xlabel=unitStr)
            ax.text(0.02,0.5,parName[i],rotation=20, fontsize=10,transform=ax.transAxes)

            for x in [mm1,mm2]:
                ax.axvline(x, color='yellow', linestyle='--')
            ax.grid()
            if i==0:
                ax.text(0.1,0.85,tit2,transform=ax.transAxes)
            if i==1:
                ax.text(0.1,0.85,'n=%d'%p.shape[0],transform=ax.transAxes)


#...!...!..................
    def params_vs_expTime(self,P,bigD,figId=4):  # only for experimental data       
        metaD=self.inpMD
        parName=metaD['parName']
        nPar=metaD['numPar']
        nrow,ncol=3,5
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(2.6*ncol,2.2*nrow))

        tit0=self.sumRec['short_name'][:27]
        j=1
        mm2=1.2; mm1=-mm2

        sweepTA=bigD['sweep_trait']
        wallT=[]
        bbV=[]
        for rec in sweepTA:
            sweepId, sweepTime, serialRes=rec
            wallT.append(sweepTime/60.) # now in minutes         
        wallT=np.array(wallT)

        if 0:
            ix=[6,7]
            wallT=np.delete(wallT,ix)
            P=np.delete(P,ix,axis=0)
            print('skip %s measurement !!!'%str(ix),P.shape)
           
        binsX= np.linspace(mm1,mm2,30)
        for i in range(0,nPar):            
            ax=self.plt.subplot(nrow,ncol,j)
            j+=1
            uval=P[:,i]
            hcol=get_arm_color(parName[i])
            ax.plot(uval,wallT,'*-',color=hcol)            
            ax.set(title=parName[i], xlabel='pred Upar %d'%(i),ylabel='wall time (min)')
                       
            for x in [-1.,1.]:
                ax.axvline(x, color='C2', linestyle='--')
            ax.grid()
            yy=0.9
            if i==0:
                ax.text(-0.05,yy,tit0,transform=ax.transAxes, color='r')
            if i==1:
                ax.text(0.05,yy,'n=%d'%uval.shape[0],transform=ax.transAxes, color='r')

