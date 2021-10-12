__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
this data loader reads one shard of data from each of many common h5-files upon start, there is no distributed sampler.
HD5-files should have comparable sizes because malloc is done once at the beginning

reads all data at once and serves them from RAM
- optimized for mult-GPU training
- only used block of data  from each H5-file
- reads data from common file for all ranks
- allows for in-fly transformation

Shuffle: only once, all samples after read is compleated

'''

import time,  os
import random
import h5py
import numpy as np

import copy
import logging

#from torch.utils.data import Dataset, DataLoader
#import torch 

#...!...!..................
def get_data_loader(params, inpMD,domain, verb=1,onlyDataset=False):
  assert type(params['cell_name'])==type([])  # Or change the dataloader import in Train
  cf={}

  cf['dataPath']=params['data_path']
  cf['domain']=domain
  cf['h5nameTemplate']=params['data_path']+'/'+inpMD['h5nameTemplate']
  cf['myRank']=params['world_rank']
  cf['local_batch_size']=params['local_batch_size']
  cf['numRanks']=params['world_size']
  cf['h5Names']=params['cell_name']
  cf['numLocalSamples']=params['numLocalSamples']
  cf['shuffle']=params['shuffle']
  
  if params['num_inp_chan']!=None: #user wants a change
    assert params['num_inp_chan']>0
    assert params['num_inp_chan']<=inpMD['numFeature']
    cf['numInpChan']=params['num_inp_chan']
  else:  # just use all avaliable features
    cf['numInpChan']=inpMD['numFeature']
  cf['numTimeBin']=inpMD['numTimeBin']
  cf['numOutChan']=inpMD['numPar']

  dataset=Dataset_multiH5_neuronInverter(cf,verb)
  # retreive some dataset info for post-processing
  params['steps_per_epoch']=dataset.sanity()
  params['model']['inputShape']=list(dataset.data_frames.shape[1:])
  params['model']['outputSize']=dataset.data_parU.shape[1]

  if onlyDataset: return dataset # used in special case of data repacking
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params['local_batch_size']),
                          num_workers=params['num_data_workers'],
                          shuffle=cf['shuffle'],
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader


#-------------------
#-------------------
#-------------------
class Dataset_multiH5_neuronInverter(object):
#...!...!..................    
    def __init__(self, conf0,verb=1):
        self.conf=copy.deepcopy(conf0)  # the input conf0 is reused later in the upper level code, can be modiffied
        self.verb=verb
        cf=self.conf  # shortcut
        h5NameL=cf['h5Names']
        numH5N=len( h5NameL)
        
        localSamplesPerFile=cf['numLocalSamples']//numH5N
        
        if self.verb>0 : logging.info('DSM:inp dom=%s read %.1fk samples/file from  %d files: %s '%(cf['domain'],localSamplesPerFile/1024,numH5N,','.join(h5NameL)))
        assert numH5N>=1 # at least one HD5 file is needed
        assert localSamplesPerFile>0
        assert cf['myRank']>=0
        
        # malloc storage for the data, to avoid concatenation of long lists
        self.data_frames=np.zeros((cf['numLocalSamples'],cf['numTimeBin'],cf['numInpChan']),dtype='float32')
        self.data_parU=np.zeros((cf['numLocalSamples'],cf['numOutChan']),dtype='float32')

        # RAM-efficient pre-shuffling of target indexes
        idxA=np.arange(cf['numLocalSamples']) # needa also if unshuffled
        if cf['shuffle']:
            np.random.shuffle(idxA)
            if self.verb>0 : logging.info('DSM:pre-shufle, dom=%s, all local %d samplesmyRank %d of %d'%(cf['domain'],cf['numLocalSamples'], cf['myRank'],cf['numRanks']))

        # read all data from disc   ... takes ~5 sec/file
        startTm0 = time.time()
        for ic in range(numH5N):
            idxOff=ic*localSamplesPerFile
            goalIdxL=idxA[idxOff:idxOff+localSamplesPerFile]
            self.openH5(h5NameL[ic],goalIdxL,localSamplesPerFile,ic)

        startTm1 = time.time()

        if 0: # check normalization
            xm=np.mean(self.data_frames)
            xs=np.std(self.data_frames)
            print('xm',xm,xs,myShard,cf['domain'])
            ok99
                
        if self.verb :
            logging.info('DSM:load-end %s locSamp=%.1fk, X.shape: %s type: %s'%(cf['domain'],cf['numLocalSamples']/1024,str(self.data_frames.shape),self.data_frames.dtype))
            #print(' DS:Xall',self.data_frames.shape,self.data_frames.dtype)
            #print(' DS:Yall',self.data_parU.shape,self.data_parU.dtype)
            

#...!...!..................
    def sanity(self):      
        stepPerEpoch= len(self)// self.conf['local_batch_size']
        if  stepPerEpoch <1:
            print('\nDS:ABORT, Have you requested too few samples per rank?, numLocFrames=%d, BS=%d  domain=%s'%(self.numLocFrames, self.conf['local_batch_size'],self.conf['domain']))
            exit(67)
        # all looks good
        return stepPerEpoch
        
#...!...!.................. 
    def openH5(self,h5name,goalIdxL,numSamp,ic):
        cf=self.conf
        inpF=cf['h5nameTemplate'].replace('*',h5name)
        inpChan=cf['numInpChan'] # this is what will be used
        dom=cf['domain']

        pr=self.verb>0 and (ic<5 or ic%20==0)
        #if self.verb>1 : print('IG:cell %s name=%s, idxOff[0..3]='%(cellN,self.conf['name']),goalIdxL[:3],'inpF=',inpF)

        
        if pr : logging.info('DSM:fileH5 %s '%(inpF))
        
        if not os.path.exists(inpF):
            print('FAILD, missing HD5',inpF)
            exit(22)

        startTm0 = time.time()

        # = = = READING HD5  start
        h5f = h5py.File(inpF, 'r')
        Xshape=h5f[dom+'_frames'].shape
        totSamp=Xshape[0]
        maxShard=totSamp//numSamp

        if  maxShard <1:
            print('\nABORT, Have you requested too many samples=%d per rank?, one cell Xshape:'%numSamp,Xshape,'dom=',self.conf['domain'],', file=',inpF)
            exit(66)
        # chosen shard is rank dependent, it wraps up if not sufficient number of shards
        myShard=self.conf['myRank'] %maxShard
        sampIdxOff=myShard*numSamp
        if pr: logging.info('DSM:  myShard=%d, maxShard=%d, numSamp/shard=%.1fk, sampIdxOff=%d, X.shape:%s'%(myShard,maxShard,numSamp/1024,sampIdxOff,str(Xshape)))

        assert self.conf['numTimeBin']==Xshape[1]
        assert inpChan<=Xshape[2]
        
        # data reading starts
        if inpChan==Xshape[2]:
            self.data_frames[goalIdxL]=h5f[dom+'_frames'][sampIdxOff:sampIdxOff+numSamp]
        else:
            self.data_frames[goalIdxL]=h5f[dom+'_frames'][sampIdxOff:sampIdxOff+numSamp,:,:inpChan]
        self.data_parU[goalIdxL]=h5f[dom+'_unitStar_par'][sampIdxOff:sampIdxOff+numSamp]
        h5f.close()
        # = = = READING HD5  done

                
        if pr :
            startTm1 = time.time()
            logging.info('DSM:  %s hd5 read time=%.2f(sec) dom=%s '%(h5name,startTm1 - startTm0,dom))
            
        # .......................................................
        #.... data embeddings, transformation should go here ....
        
        #self.data_parU*=1.2
        #.... end of embeddings ........
        # .......................................................

        self.numLocFrames=self.data_frames.shape[0]

#...!...!..................
    def __len__(self):        
        return self.data_frames.shape[0]

#...!...!..................
    def __getitem__(self, idx):
        #assert idx>=0
        #assert idx< self.numLocFrames
        X=self.data_frames[idx]
        Y=self.data_parU[idx]
        return (X,Y)
 
