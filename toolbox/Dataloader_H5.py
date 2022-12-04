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
#from toolbox.Util_IOfunc import read_yaml
from pprint import pprint


#...!...!..................
def get_data_loader(params,domain, verb=1):
  assert type(params['cell_name'])==type('abc')  # Or change the dataloader import in Train

  conf=copy.deepcopy(params)  # the input is reused later in the upper level code
  
  conf['domain']=domain
  conf['h5name']=os.path.join(params['data_path'],params['cell_name']+'.simRaw.h5')
  shuffle=conf['shuffle']

  dataset=  Dataset_h5_neuronInverter(conf,verb)
  
  # return back some of info
  params[domain+'_steps_per_epoch']=dataset.sanity()
  params['model']['inputShape']=list(dataset.data_frames.shape[1:])
  params['model']['outputSize']=dataset.data_parU.shape[1]
  params['full_h5name']=conf['h5name']

  dataloader = DataLoader(dataset,
                          batch_size=dataset.conf['local_batch_size'],
                          num_workers=params['num_data_workers'],
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
        inpF=cf['h5name']        
        dom=cf['domain']
        if self.verb>0 : logging.info('DS:fileH5 %s  rank %d of %d '%(inpF,cf['world_rank'],cf['world_size']))
        
        if not os.path.exists(inpF):
            print('DLI:FAILED, missing HD5',inpF)
            exit(22)

        startTm0 = time.time()
        
        # = = = READING HD5  start
        h5f = h5py.File(inpF, 'r')
            
        Xshape=h5f['volts_'+dom].shape
        totSamp,timeBins,mxProb,mxStim=Xshape

        assert max( cf['probs_select']) <mxProb 
        assert max( cf['stims_select']) <mxStim
        # TypeError: Only one indexing vector or array is currently allowed for fancy indexing
        
        if 'max_glob_samples_per_epoch' in cf:            
            max_samp= cf['max_glob_samples_per_epoch']
            if dom=='valid': max_samp//=4
            totSamp,oldN=min(totSamp,max_samp),totSamp
            if totSamp<oldN and  self.verb>0 :
              logging.warning('GDL: shorter dom=%s max_glob_samples=%d from %d'%(dom,totSamp,oldN))
                   

        if dom=='exper':  # special case for exp data
            cf['local_batch_size']=totSamp

        locStep=int(totSamp/cf['world_size']/cf['local_batch_size'])
        locSamp=locStep*cf['local_batch_size']
        logging.info('DLI:locSamp=%d locStep=%d BS=%d rank=%d'%(locSamp,locStep,cf['local_batch_size'],self.conf['world_rank']))
        assert locStep>0
        maxShard= totSamp// locSamp
        assert maxShard>=cf['world_size']
                    
        # chosen shard is rank dependent, wraps up if not sufficient number of ranks
        myShard=self.conf['world_rank'] %maxShard
        sampIdxOff=myShard*locSamp
        
        if self.verb : logging.info('DS:file dom=%s myShard=%d, maxShard=%d, sampIdxOff=%d allXshape=%s  probs=%s stims=%s'%(cf['domain'],myShard,maxShard,sampIdxOff,str(Xshape), cf['probs_select'],cf['stims_select']))

         
        #********* data reading starts .... is compact to save CPU RAM
        # TypeError: Only one indexing vector or array is currently allowed for fancy indexing
        volts=h5f['volts_'+dom][sampIdxOff:sampIdxOff+locSamp,:,:,cf['stims_select']].astype(np.float32)
        self.data_frames=volts[:,:,cf['probs_select']].reshape(locSamp,timeBins,-1)
        #print('AA2',volts.shape,self.data_frames.shape,dom) ; ok9
        self.data_parU=h5f['unit_par_'+dom][sampIdxOff:sampIdxOff+locSamp]

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
        
        if self.verb:  # WARN: it double RAM for a brief time
           logging.info('DLI:per_waveform_norm=%r dom=%s'%(cf['data_conf']['per_wavform_norm'],cf['domain']))
        if cf['data_conf']['per_wavform_norm']:
            Ta = time.time()
            #print('WW1',self.data_frames.shape,self.data_frames.dtype)
            
            # for breadcasting to work the 1st dim must be skipped
            X=np.swapaxes(self.data_frames,0,1)# returns view, no data duplication           #print('WW2',X.shape)
            xm=np.mean(X,axis=0) # average over 1600 time bins
            xs=np.std(X,axis=0)
            elaTm=(time.time()-Ta)/60.
            if self.verb>1: print('DLI:PWN xm:',xm.shape,'Xswap:',X.shape,'dom=',cf['domain'],'elaT=%.2f min'%elaTm)

            
            nZer=np.sum(xs==0)
            if nZer>0: logging.warning('DLI: nZer=%d %s rank=%d : corrected  mu'%(nZer,xs.shape, self.conf['world_rank']))
            # to see indices of frames w/ 0s:   result = np.where(xs==0)  
            xs[xs==0]=1  # hack - for zero-value samples use mu=1
            X=(X-xm)/xs
            self.data_frames=np.swapaxes(X ,0,1)# returns the initial view
            
        #.... end of embeddings ........
        # .......................................................

        if 0 : # check X normalizations            
            X=self.data_frames
            xm=np.mean(X,axis=1)  # average over 1600 time bins
            xs=np.std(X,axis=1)
            print('DLI:X',X[0,::500,0],X.shape,xm.shape)

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

