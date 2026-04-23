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
import efel


def _safe_genfromtxt(csv_path):
    arr = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    # Drop header row if parsed as NaN values
    if arr.shape[0] > 0 and np.isnan(arr[0]).any():
        arr = arr[1:]
    return arr.astype(np.float32)

#...!...!..................
def get_data_loader(params,domain, verb=1):
  assert type(params['cell_name'])==type('abc')  # Or change the dataloader import in Train

  conf=copy.deepcopy(params)  # the input is reused later in the upper level code
  
  conf['domain']=domain
  #   conf['h5name']=os.path.join(params['data_conf']['data_path'],params['cell_name']+'.mlPack1.h5')
  conf['h5name']=os.path.join(params['data_path_temp'],params['cell_name']+'.mlPack1.h5')
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
    _train_stats_cache = {}

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
            print('\nDS:ABORT, Have you requested too few samples per rank?, numLocFrames=%d, BS=%d  name=%s'%(self.numLocFrames, self.conf['local_batch_size'],self.conf['name']))
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
        appendStim=dcf['append_stim']
        parallelStim=dcf['parallel_stim']
        
        #********* data reading starts .... is compact to save CPU RAM
        # TypeError: Only one indexing vector or array is currently allowed for fancy indexing
        if dom=="train" or not serializedStim:
            volts=h5f[dom+'_volts_norm'][sampIdxOff:sampIdxOff+locSamp,:,:,dcf['stims_select']] .astype(np.float32)  # input=fp16 is not working for Model - fix it one day
        else:
            volts=h5f[dom+'_volts_norm'][sampIdxOff:sampIdxOff+locSamp,:,:,dcf['valid_stims_select']] .astype(np.float32)
        parU=h5f[dom+'_unit_par'][sampIdxOff:sampIdxOff+locSamp]
        #... chose how to re-shape the ML input
       
        if appendStim: #not  serializedStim: # probs*1stm--> M*timeBins
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
            if dom=="train":
                if self.verb : print('WS1 numStim=%d volts:'%(numStim),volts.shape,', parU2:',parU2.shape)
                locSamp*=numStim
                volts=volts[:,:,dcf['probs_select']]
                volts=np.moveaxis(volts,-1,0).reshape(locSamp,timeBins,-1)
                parU=np.moveaxis(parU2,-1,0).reshape(locSamp,-1)
                # rand_idx=torch.randperm(volts.elements())
                # volts=volts.view(-1)[rand_idx].view(volts.size())
                # parU = parU.view(-1)[rand_idx].view(volts.size())
                rand_idx=np.random.permutation(len(volts))
                volts=volts[rand_idx]
                parU=parU[rand_idx]

                if self.verb : print('WS2 locSamp=%d, volts:'%locSamp,volts.shape,', parU:',parU.shape,', dom=',dom)
            else:
                if self.verb : print('WS1 numStim=%d volts:'%(numStim),volts.shape,', parU2:',parU2.shape)
                volts=volts[:,:,dcf['probs_select']].reshape(locSamp,timeBins,-1)

                if self.verb : print('WS2 locSamp=%d, volts:'%locSamp,volts.shape,', parU:',parU.shape,', dom=',dom)
        
        if parallelStim:
            if self.verb : print('WT1 numStim=%d volts:'%(numStim),volts.shape)
            volts=volts[:,:,dcf['probs_select']] #.reshape(locSamp,timeBins,-1)
            # volts=volts[:,:,dcf['probs_select']].reshape(locSamp,numStim*numProb,timeBins)
            if self.verb : print('WT2 locSamp=%d, volts:'%locSamp,volts.shape,' dom=',dom)
            
        self.data_frames=volts
        self.data_parU=parU

        self.use_manual_features = cf.get('use_manual_features', False)

        if self.use_manual_features:
            if self.verb:
                logging.info("DS: Loading precomputed manual features...")

            # Load domain features and select same local shard as voltages
            local_feature_slice = slice(sampIdxOff, sampIdxOff + locSamp)
            self.data_extras = self._load_features_for_domain(dom, sample_slice=local_feature_slice)

            # Normalize with stats from all TRAIN features
            self.feat_mean, self.feat_std = self._get_train_feature_stats()
            self.data_extras = np.nan_to_num(self.data_extras, nan=0.0, posinf=0.0, neginf=0.0)
            self.data_extras = (self.data_extras - self.feat_mean) / self.feat_std
            self.data_extras = self.data_extras.astype(np.float32)

            # Keep feature rows aligned with data reshuffling in serialized train mode
            if serializedStim and dom == "train":
                self.data_extras = np.repeat(self.data_extras, numStim, axis=0)
                self.data_extras = self.data_extras[rand_idx]

            if self.data_extras.shape[0] != self.data_frames.shape[0]:
                raise ValueError(
                    f"Feature/data sample mismatch: features={self.data_extras.shape[0]} "
                    f"vs frames={self.data_frames.shape[0]} for domain={dom}"
                )

            if self.verb:
                logging.info(
                    'DS: loaded extras %s normalized by TRAIN stats shape=%s',
                    dom,
                    self.data_extras.shape
                )
        
        # if self.use_manual_features:
        #     if self.verb: logging.info("DS: Extracting manual features (e.g. AP count)...")
        #     self.data_extras = self.extract_features(self.data_frames)
        #     # Normalize
        #     mean = np.mean(self.data_extras, axis=0)
        #     std = np.std(self.data_extras, axis=0) + 1e-6
        #     self.data_extras = (self.data_extras - mean) / std
        #     self.data_extras = self.data_extras.astype(np.float32)

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
            pass
        
        self.numLocFrames=self.data_frames.shape[0]

    def _resolve_feature_csv_path(self, domain):
        cf = self.conf
        dcf = cf['data_conf']

        templ = (
            cf.get('manual_features_csv')
            or dcf.get('manual_features_csv')
            or cf.get('manual_features_csv_template')
            or dcf.get('manual_features_csv_template')
        )
        if templ is None:
            raise ValueError(
                'Missing feature CSV configuration. Set one of: '
                '`manual_features_csv` or `manual_features_csv_template` '
                '(template may use {domain}).'
            )

        path = templ.format(domain=domain)
        if not os.path.isabs(path):
            path = os.path.join(cf['data_path_temp'], path)
        return path

    def _get_probe_select(self):
        dcf = self.conf['data_conf']
        # Backward compatibility for typo in config key
        return dcf.get('prove_select', dcf['probs_select'])

    def _load_features_for_domain(self, domain, sample_slice=None):
        cf = self.conf
        dcf = cf['data_conf']
        if sample_slice is None:
            sample_slice = slice(None)

        # Prefer precomputed features stored in HDF5
        efel_key = domain + '_efel_features'
        with h5py.File(cf['h5name'], 'r') as h5f:
            if efel_key in h5f:
                feats = h5f[efel_key][sample_slice].astype(np.float32)  # (N, probes, stims, features)

                probe_select = self._get_probe_select()
                feats = feats[:, probe_select, :, :]

                if domain == "train" or not dcf['serialize_stims']:
                    stim_select = dcf['stims_select']
                else:
                    stim_select = dcf['valid_stims_select']
                feats = feats[:, :, stim_select, :]

                # Flatten to 2D per sample, expected by model concat path
                feats = feats.reshape(feats.shape[0], -1)
                return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Fallback: load manual features from CSV
        csv_path = self._resolve_feature_csv_path(domain)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Manual feature CSV not found: {csv_path}')
        feats = _safe_genfromtxt(csv_path)
        if feats.shape[0] == 0:
            raise ValueError(f'No rows found in manual feature CSV: {csv_path}')
        if sample_slice.stop is not None and sample_slice.stop > feats.shape[0]:
            raise ValueError(
                f"Feature CSV too short for domain={domain}: need {sample_slice.stop}, have {feats.shape[0]}"
            )
        feats = feats[sample_slice]
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_train_feature_stats(self):
        cf = self.conf
        h5_stats_key = cf['h5name'] + '::train_efel_features::' + str(self._get_probe_select())
        train_csv = self._resolve_feature_csv_path('train') if (
            cf.get('manual_features_csv')
            or cf['data_conf'].get('manual_features_csv')
            or cf.get('manual_features_csv_template')
            or cf['data_conf'].get('manual_features_csv_template')
        ) else None

        stats_key = h5_stats_key
        if train_csv is not None:
            stats_key = train_csv

        if stats_key in Dataset_h5_neuronInverter._train_stats_cache:
            return Dataset_h5_neuronInverter._train_stats_cache[stats_key]

        train_feats = self._load_features_for_domain('train', sample_slice=slice(None))
        mean = np.mean(train_feats, axis=0).astype(np.float32)
        std = (np.std(train_feats, axis=0) + 1e-6).astype(np.float32)
        Dataset_h5_neuronInverter._train_stats_cache[stats_key] = (mean, std)
        return mean, std

#...!...!..................
    def extract_features(self,volts_batch, dt=0.1):
        """
        Extracts AP count using EFEL.
        Expects volts_batch in shape (N, TimeBins) or (N, TimeBins, 1).
        Assumes data is in mV.
        """
        # Handle input shape to ensure (N, TimeBins, 1)
        if volts_batch.ndim == 1:
            volts_batch = volts_batch[np.newaxis, :, np.newaxis]
        elif volts_batch.ndim == 2:
            volts_batch = volts_batch[:, :, np.newaxis]
            
        num_samples = volts_batch.shape[0]
        num_time_bins = volts_batch.shape[1]
        
        # Create time array
        time_array = np.arange(num_time_bins) * dt
        
        # Define stimulus window (using full trace for AP counting)
        stim_start = 0.0
        stim_end = num_time_bins * dt
        
        # Prepare Data for EFEL
        trace_data_list = []
        for i in range(num_samples):
            trace = volts_batch[i, :, 0]
            
            trace_data = {
                'T': time_array,
                'V': trace,
                'stim_start': [stim_start],
                'stim_end': [stim_end]
            }
            trace_data_list.append(trace_data)
            
        # Run EFEL
        # efel.setThreshold(-20) # Uncomment to force a specific threshold
        feature_names = [
            'mean_frequency', 'AP_amplitude', 'AHP_depth_abs_slow',
            'fast_AHP_change', 'AHP_slow_time',
            'spike_half_width', 'time_to_first_spike', 'inv_first_ISI', 'ISI_CV',
            'ISI_values','adaptation_index'
        ]
        
        # getFeatureValues processes list of traces efficiently
        try:
            features_list = efel.getFeatureValues(trace_data_list, feature_names)
        except Exception as e:
            logging.error(f"EFEL failed: {e}")
            return np.zeros((num_samples, len(feature_names) + 1))
        
        # Parse Results
        # We add 1 column for AP_Count at the end
        features_out = np.zeros((num_samples, len(feature_names) + 1), dtype=np.float32)
        
        for i, result_dict in enumerate(features_list):
            # 1. Compute mean of requested features
            for j, name in enumerate(feature_names):
                val = result_dict.get(name)
                if val is not None and len(val) > 0:
                    features_out[i, j] = np.mean(val)
                else:
                    features_out[i, j] = 0.0
            
            # 2. Compute AP_Count (length of AP_amplitude)
            ap_amp = result_dict.get('AP_amplitude')
            if ap_amp is not None:
                features_out[i, -1] = len(ap_amp)
            else:
                features_out[i, -1] = 0.0
                
        return features_out
        # getFeatureValues processes list of traces efficiently

#...!...!.............
    def __len__(self):
        return self.numLocFrames

#...!...!..................
    def __getitem__(self, idx):
        # print('DSI:',idx,self.conf['name'],self.cnt); self.cnt+=1
        assert idx>=0
        assert idx< self.numLocFrames
        X=self.data_frames[idx]
        Y=self.data_parU[idx]
        
        if self.use_manual_features:
            # Return preloaded and normalized manual features from memory
            Z=self.data_extras[idx]
            return (X, Z, Y)
        else:
            return (X, Y)

