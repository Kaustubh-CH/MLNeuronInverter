__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
this data loader reads one shard of data from a common h5-file upon start, there is no distributed sampler

reads all data at once and serves them from RAM
- optimized for mult-GPU training
- only used block of data  from each H5-file
- reads data from common file for all ranks
- allows for in-fly transformation

Shuffle: only  all samples after read is compleated

'''

import time,  os
import random
import h5py, json
import numpy as np

import copy
from torch.utils.data import Dataset, DataLoader
import torch 
import logging
from pprint import pprint


#...!...!..................
def get_data_loader(params,domain, verb=1):
  assert type(params['cell_name'])==type('abc')  # Or change the dataloader import in Train

  conf=copy.deepcopy(params)  # the input is reused later in the upper level code
  
  conf['domain']=domain
  conf['h5name']=os.path.join(params['data_conf']['data_path'],params['cell_name']+'.mlPack1.h5')
  shuffle=conf['shuffle']

  dataset=  Dataset_h5_neuronInverter(conf,verb)
  
  # return back some of info
  params[domain+'_steps_per_epoch']=dataset.sanity()
  params['model']['inputShape']=list(dataset.data_frames.shape[1:])
  params['model']['outputSize']=dataset.data_parU.shape[1]
  params['full_h5name']=conf['h5name']

  dataloader = DataLoader(dataset,
                          batch_size=dataset.conf['local_batch_size'],
                          num_workers=params['data_conf']['num_data_workers'],
                          shuffle=shuffle,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())
  return dataloader


#-------------------
#-------------------
#-------------------
class Dataset_h5_neuronInverter(object):
#...!...!..................    
    def __init__(self, conf,verb=1):
        self.conf=conf
        self.verb=verb

        self.openH5()
        if self.verb and 0:
            print('\nDS-cnst name=%s  shuffle=%r BS=%d steps=%d myRank=%d numSampl/hd5=%d'%(self.conf['name'],self.conf['shuffle'],self.localBS,self.__len__(),self.conf['myRank'],self.conf['numSamplesPerH5']),'H5-path=',self.conf['dataPath'])
        assert self.numLocFrames>0
        assert self.conf['world_rank']>=0

        if self.verb :
            logging.info(' DS:load-end %s locSamp=%d, X.shape: %s type: %s'%(self.conf['domain'],self.numLocFrames,str(self.data_frames.shape),self.data_frames.dtype))
            #print(' DS:Xall',self.data_frames.shape,self.data_frames.dtype)
            #print(' DS:Yall',self.data_parU.shape,self.data_parU.dtype)
            

#...!...!..................
    def sanity(self):      
        stepPerEpoch=int(np.floor( self.numLocFrames/ self.conf['local_batch_size']))
        if  stepPerEpoch <1:
            print('\nDS:ABORT, Have you requested too few samples per rank?, numLocFrames=%d, BS=%d  name=%s'%(self.numLocFrames, localBS,self.conf['name']))
            exit(67)
        # all looks good
        return stepPerEpoch
        
#...!...!..................
    def openH5(self):
        cf=self.conf
        dcf=cf['data_conf']
        inpF=cf['h5name']        
        dom=cf['domain']
        if self.verb>0 : logging.info('DS:fileH5 %s  rank %d of %d '%(inpF,cf['world_rank'],cf['world_size']))
        
        if not os.path.exists(inpF):
            print('DLI:FAILED, missing HD5',inpF)
            exit(22)

        startTm0 = time.time()
        
        # = = = READING HD5  start
        h5f = h5py.File(inpF, 'r')
            
        Xshape=h5f[dom+'_volts_norm'].shape
        totSamp,timeBins,mxProb,mxStim=Xshape

        assert max( dcf['probs_select']) <mxProb 
        assert max( dcf['stims_select']) <mxStim
        # TypeError: Only one indexing vector or array is currently allowed for fancy indexing
        numProb=len( dcf['probs_select'])
        numStim=len( dcf['stims_select'])
        
        if 'max_glob_samples_per_epoch' in cf['data_conf']:            
            max_samp= dcf['max_glob_samples_per_epoch']
            if dom=='valid': max_samp//=4
            totSamp,oldN=min(totSamp,max_samp),totSamp
            if totSamp<oldN and  self.verb>0 :
              logging.warning('GDL: shorter dom=%s max_glob_samples=%d from %d'%(dom,totSamp,oldN))
                   

        if dom=='exper':  # special case for exp data
            cf['local_batch_size']=totSamp

        locStep=int(totSamp/cf['world_size']/cf['local_batch_size'])
        locSamp=locStep*cf['local_batch_size']
        logging.info('DLI:locSamp=%d numStim=%d locStep=%d BS=%d rank=%d dom=%s'%(locSamp,numStim,locStep,cf['local_batch_size'],self.conf['world_rank'],dom))
        assert locStep>0
        maxShard= totSamp// locSamp
        assert maxShard>=cf['world_size']
                    
        # chosen shard is rank dependent, wraps up if not sufficient number of ranks
        myShard=self.conf['world_rank'] %maxShard
        sampIdxOff=myShard*locSamp
        
        if self.verb : logging.info('DS:file dom=%s myShard=%d, maxShard=%d, sampIdxOff=%d allXshape=%s  probs=%s stims=%s'%(cf['domain'],myShard,maxShard,sampIdxOff,str(Xshape), dcf['probs_select'],dcf['stims_select']))

        serializedStim=dcf['serialize_stims']
        
        #********* data reading starts .... is compact to save CPU RAM
        # TypeError: Only one indexing vector or array is currently allowed for fancy indexing
        volts=h5f[dom+'_volts_norm'][sampIdxOff:sampIdxOff+locSamp,:,:,dcf['stims_select']] .astype(np.float32)  # input=fp16 is not working for Model - fix it one day
        parU=h5f[dom+'_unit_par'][sampIdxOff:sampIdxOff+locSamp]
        #... chose how to re-shape the ML input
       
        if not  serializedStim: # probs*1stm--> M*timeBins
            volts=volts[:,:,dcf['probs_select']]
            if self.verb : print('WT1 numStim=%d volts:'%(numStim),volts.shape)
            volts=np.swapaxes(volts,2,3).reshape(locSamp,numStim*timeBins,-1)
            if self.verb : print('WT2 locSamp=%d, volts:'%locSamp,volts.shape,' dom=',dom)
        if 0: # probs*stims--> M*channel
            assert not serializedStim
            volts=volts[:,:,cf['probs_select']].reshape(locSamp,timeBins,-1)
        

        if serializedStim: # stacking stims as independent samples requires clonning of parU as well
            shp=parU.shape+(numStim,)
            parU2=np.zeros(shp,dtype=np.float32)  # WARN: slightly increases RAM usage
            for i in range(numStim): parU2[...,i]=parU.copy()
            if self.verb : print('WS1 numStim=%d volts:'%(numStim),volts.shape,', parU2:',parU2.shape)
            locSamp*=numStim
            volts=np.moveaxis(volts,-1,0).reshape(locSamp,timeBins,-1)
            parU=np.moveaxis(parU2,-1,0).reshape(locSamp,-1)
            if self.verb : print('WS2 locSamp=%d, volts:'%locSamp,volts.shape,', parU:',parU.shape,', dom=',dom)
            
        self.data_frames=volts
        self.data_parU=parU

        #print('AA2 volts:',self.data_frames.shape,dom,self.data_parU.shape) ; okok11

        if cf['world_rank']==0:
            blob=h5f['meta.JSON'][0]
            self.metaData=json.loads(blob)

        h5f.close()
        #******* READING HD5  done
        
        if self.verb>0 :
            startTm1 = time.time()
            if self.verb: logging.info('DS: hd5 read time=%.2f(sec) dom=%s '%(startTm1 - startTm0,dom))
            
        # .......................................................
        #.... data embeddings, transformation should go here ....

        # none
            
        #.... end of embeddings ........
        # .......................................................

        if 0 : # check X normalizations            
            X=self.data_frames
            xm=np.mean(X,axis=1)  # average over 1600 time bins
            xs=np.std(X,axis=1)
            print('DLI:X=volts_norm',X[0,::500,0],X.shape,xm.shape)

            print('DLI:Xm',xm[:10],'\nXs:',xs[:10],myShard,'dom=',cf['domain'],'X:',X.shape)
            
        if 0:  # check Y avr
            Y=self.data_parU
            ym=np.mean(Y,axis=0)
            ys=np.std(Y,axis=0)
            print('DLI:U',myShard,cf['domain'],Y.shape,ym.shape,'\nUm',ym[:10],'\nUs',ys[:10])
            pprint(self.conf)
            end_test_norm
        
        self.numLocFrames=self.data_frames.shape[0]

#...!...!..................
    def __len__(self):        
        return self.numLocFrames

#...!...!..................
    def __getitem__(self, idx):
        # print('DSI:',idx,self.conf['name'],self.cnt); self.cnt+=1
        assert idx>=0
        assert idx< self.numLocFrames
        X=self.data_frames[idx]
        Y=self.data_parU[idx]
        return (X,Y)

