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
import h5py
import numpy as np

import copy
from torch.utils.data import Dataset, DataLoader
import torch 
import logging
from toolbox.Util_IOfunc import read_yaml

#...!...!..................
def get_data_loader(params, inpMD,domain, verb=1):
  assert type(params['cell_name'])==type('abc')  # Or change the dataloader import in Train

  conf=copy.deepcopy(params)  # the input is reused later in the upper level code
  
  conf['domain']=domain
  conf['h5name']=os.path.join(params['data_path'],inpMD['h5nameTemplate'].replace('*',params['cell_name']))
  if params['num_inp_chan']!=None: #user wants a change
    assert params['num_inp_chan']>0
    assert params['num_inp_chan']<=inpMD['numFeature']
    conf['numInpChan']=params['num_inp_chan']
  else:  # just copy the meta-data value
    conf['numInpChan']=inpMD['numFeature']
  shuffle=conf['shuffle']

  #print('bb inpChan=',conf['numInpChan'],params['num_inp_chan']); ok11
  dataset=  Dataset_h5_neuronInverter(conf,verb)
  if 'max_samples_per_epoch' in params:
        max_samp= params['max_samples_per_epoch']
        print('GDL: WARN, shorter %s max_samples=%d from %d'%(domain,max_samp,dataset.numLocFrames))
        dataset.numLocFrames=min(dataset.numLocFrames,max_samp)

  
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
        inpFeat=cf['numInpChan'] # this is what user wants
        dom=cf['domain']
        if self.verb>0 : logging.info('DS:fileH5 %s  rank %d of %d '%(inpF,cf['world_rank'],cf['world_size']))
        
        if not os.path.exists(inpF):
            print('DLI:FAILED, missing HD5',inpF)
            exit(22)

        startTm0 = time.time()
        
        # = = = READING HD5  start
        h5f = h5py.File(inpF, 'r')
        Xshape=h5f[dom+'_frames'].shape
        totSamp=Xshape[0]
        
        if dom=='exper':  # special case for exp data
          cf['local_batch_size']=totSamp

        locStep=int(totSamp/cf['world_size']/cf['local_batch_size'])
        locSamp=locStep*cf['local_batch_size']
        #print('DLI:totSamp=%d locStep=%d BS=%d'%(totSamp,locStep,cf['local_batch_size']))
        assert locStep>0
        maxShard= totSamp// locSamp
        assert maxShard>=cf['world_size']
                    
        # chosen shard is rank dependent, wraps up if not sufficient number of ranks
        myShard=self.conf['world_rank'] %maxShard
        sampIdxOff=myShard*locSamp
        
        if self.verb: logging.info('DS:file dom=%s myShard=%d, maxShard=%d, sampIdxOff=%d allXshape=%s  inpFeat=%d'%(cf['domain'],myShard,maxShard,sampIdxOff,str(Xshape),inpFeat))       
        
        # data reading starts ....
        assert inpFeat<=Xshape[2]
        if inpFeat==Xshape[2]:
            self.data_frames=h5f[dom+'_frames'][sampIdxOff:sampIdxOff+locSamp]
        else:
            self.data_frames=h5f[dom+'_frames'][sampIdxOff:sampIdxOff+locSamp,:,:inpFeat]
        self.data_parU=h5f[dom+'_unitStar_par'][sampIdxOff:sampIdxOff+locSamp]
        h5f.close()
        # = = = READING HD5  done
        
        if self.verb>0 :
            startTm1 = time.time()
            if self.verb: logging.info('DS: hd5 read time=%.2f(sec) dom=%s '%(startTm1 - startTm0,dom))
            
        # .......................................................
        #.... data embeddings, transformation should go here ....
        useUpar=cf['train_conf']['recover_upar_from_ustar']
        if self.verb: print('DLI:recover_upar_from_ustar',useUpar,cf['cell_name'],self.data_parU.dtype)
        if useUpar:
          assert 'bbp' in cf['cell_name']  # rescaling makes no sense for ontra data
          inpF2=inpF.replace('.data.h5','.meta.yaml')
          blob=read_yaml( inpF2,verb=self.verb)
          utrans_bias=blob['packInfo']['utrans_bias']
          utrans_scale=blob['packInfo']['utrans_scale']
          self.data_parU=self.data_parU*utrans_scale + utrans_bias
          self.data_parU=self.data_parU.astype('float32')
          
        if self.verb: print('DLI:per_waveform_norm=',cf['train_conf']['per_wavform_norm'],'dom=',cf['domain'])
        if cf['train_conf']['per_wavform_norm']:
            Ta = time.time()
            # for breadcasting to work the 1st dim must be skipped
            X=np.swapaxes(self.data_frames,0,1)# returns view, no data duplication            
            xm=np.mean(X,axis=0) # average over 1600 time bins
            xs=np.std(X,axis=0)
            elaTm=(time.time()-Ta)/60.
            if self.verb: print('DLI:PWN xm:',xm.shape,'Xswap:',X.shape,'dom=',cf['domain'],'elaT=%.2f min'%elaTm)

            nZer=np.sum(xs==0)
            if nZer>0: print('DLI:WARN nZer:',nZer,xs.shape, 'rank=%d corrected  mu'%self.conf['world_rank'])
            # to see indices of frames w/ 0s:   result = np.where(xs==0)  
            xs[xs==0]=1  # hack - for zero-value samples use mu=1
            X=(X-xm)/xs
            self.data_frames=np.swapaxes(X,0,1)# returns view
            
        #.... end of embeddings ........
        # .......................................................

        if 0 : # check X normalizations            
            X=self.data_frames
            xm=np.mean(X,axis=1)  # average over 1600 time bins
            xs=np.std(X,axis=1)
            print('DLI:X',X[0,::80,0],X.shape,xm.shape)

            print('DLI:Xm',xm[:10],'\nXs:',xs[:10],myShard,'dom=',cf['domain'],'X:',X.shape)
            
        if 0:  # check Y avr
            Y=self.data_parU
            ym=np.mean(Y,axis=0)
            ys=np.std(Y,axis=0)
            print('DLI:U',myShard,cf['domain'],Y.shape,ym.shape,'\nUm',ym[:10],'\nUs',ys[:10])
            print(self.conf)
            ok99
        
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

