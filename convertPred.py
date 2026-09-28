from pyibt.read_ibt import Read_IBT
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf
import csv
import h5py
import os,sys,time
import json
import efel


EFEL_FEATURE_NAMES = [
    'mean_frequency', 'AP_amplitude', 'AHP_depth_abs_slow',
    'fast_AHP_change', 'AHP_slow_time',
    'spike_half_width', 'time_to_first_spike', 'inv_first_ISI', 'ISI_CV',
    'ISI_values', 'adaptation_index'
]


def extract_efel_features_from_volts(volts, dt=0.1):
    """
    volts shape: (samples, time_bins, probes, stims)
    returns: (samples, probes, stims, num_features_plus_ap_count)
    """
    n_samp, n_time, n_probe, n_stim = volts.shape
    n_feat = len(EFEL_FEATURE_NAMES) + 1  # + AP_count
    out = np.zeros((n_samp, n_probe, n_stim, n_feat), dtype=np.float32)

    time_array = np.arange(n_time) * dt
    trace_data_list = []
    trace_map = []
    for i in range(n_samp):
        for p in range(n_probe):
            for s in range(n_stim):
                trace_data_list.append({
                    'T': time_array,
                    'V': volts[i, :, p, s],
                    'stim_start': [0.0],
                    'stim_end': [n_time * dt]
                })
                trace_map.append((i, p, s))

    try:
        features_list = efel.getFeatureValues(trace_data_list, EFEL_FEATURE_NAMES)
    except Exception as e:
        print('EFEL failed:', e)
        return out

    for idx, res in enumerate(features_list):
        i, p, s = trace_map[idx]
        for j, name in enumerate(EFEL_FEATURE_NAMES):
            val = res.get(name)
            out[i, p, s, j] = np.mean(val) if (val is not None and len(val) > 0) else 0.0

        ap_amp = res.get('AP_amplitude')
        out[i, p, s, -1] = len(ap_amp) if ap_amp is not None else 0.0

    return out


def normalize_volts(volts,name='',verb=1):  # slows down the code a lot
    Ta = time.time()
    #print('WW1',volts.shape,volts.dtype)

    #... for breadcasting to work the 1st dim (=timeBins) must be skipped
    # X=np.swapaxes(volts,0,1).astype(np.float32) # important for correct result
    #print('WW2',X.shape)
    X=volts    
    xm=np.mean(X,axis=0) # average over time bins
    xs=np.std(X,axis=0)
    elaTm=(time.time()-Ta)/60.
    print('Volts norm, xm:',xm.shape,'Xswap:',X.shape,'elaT=%.2f min'%elaTm)

    nZer=np.sum(xs==0)
    zerA=xs==0
    print('   nZer=%d %s  : name=%s'%(nZer,xs.shape,name))
    
    #... to see indices of frames w/ volts==const
    result = np.where(xs==0)  
    xs[xs==0]=1  #hack:  for zero-value samples use mu=1 to allow scaling
    X=(X-xm)/xs

    #... revert indices and reduce bit-size
    # volts_norm=np.swapaxes(X,0,1).astype(np.float16)
    volts_norm=X
    del X
    #print('WW3',volts_norm.shape,volts_norm.dtype)

    if verb>1: # report flat volts for each sample
        na,nb,nc=zerA.shape    
        for i,A in enumerate(zerA):
            if np.sum(A)==0: continue
            zSt=np.sum(A,axis=0)
            zBo=np.sum(A,axis=1)
            print('zer', i,np.sum(A),'stims:',zSt,' body:',zBo)
            #assert nZer==0  # to stop at 1st case
 
    return volts_norm,nZer



def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal



# file = open("/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/Stims/cahotic_50khz.csv","r")
file = open("/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/Stims/chaotic_50khz.csv","r")
data2 = list(csv.reader(file, delimiter=","))
file.close()
stim = [float(row[0]) for row in data2]

ibt = Read_IBT('/global/homes/k/ktub1999/ExperimentalData/PyForEphys/Data/012722B2.ibt')

params = range(66,77)

sweep = ibt.sweeps[75]
data=resample_by_interpolation(sweep.data[:20000],20000,4000)
print(np.count_nonzero(data[:1000]==0))

volts_reduce = np.zeros(shape=(11,4000,1,6), dtype=np.float32)
volts_exact = np.zeros(shape=(11,4000,1,6), dtype=np.float32)
raw_volts_reduce = np.zeros(shape=(11,4000,1,6), dtype=np.float32)
raw_volts_exact = np.zeros(shape=(11,4000,1,6), dtype=np.float32)
# volts= np.empty(shape=(1,4000,1))
sample=0
probe =0
mean_tot=[]
std_tot=[]
for sample in range(11):
    # for c,p in enumerate(params):
        c = 5
        sweep = ibt.sweeps[params[sample]]
        data = resample_by_interpolation(sweep.data[:20000],20000,4000)
        data = np.array(data, dtype=np.float32)
        data_exact = data.copy()
        data_reduce = data.copy() - 6.0
        #FEATURE WISE
        # data_load=np.load('/global/homes/k/ktub1999/Neuron/neuron4/neuroninverter/packBBP3/Stats.npz')
        # xm=data_load['Mean'][:4000]
        # xs=data_load['Std'][:4000]
        meanD = data_reduce.mean()
        stdD = data_reduce.std()
        
        # meanD=xm
        # stdD=xs
        meanD=-60.09519969696969
        stdD=18.950556714187503
        # meanD=0
        # stdD=1
        data_reduce.resize((4000,1))
        data_exact.resize((4000,1))
        # data_norm=(data-xm)/xs

        data_norm_reduce=(data_reduce-meanD)/stdD
        data_norm_exact=(data_exact-meanD)/stdD

        # store raw (pre-normalization) traces
        raw_volts_reduce[sample,:4000,probe,c]=np.squeeze(data_reduce)
        raw_volts_exact[sample,:4000,probe,c]=np.squeeze(data_exact)

        # store normalized traces
        volts_reduce[sample,:4000,probe,c]=np.squeeze(data_norm_reduce)
        volts_exact[sample,:4000,probe,c]=np.squeeze(data_norm_exact)
        
        mean_tot.append(data_norm_reduce.mean())
        std_tot.append(data_norm_reduce.std())
        # volts[sample][:4000][probe][c]=data_norm
        # for d1,d in enumerate(data):
        #     volts[sample][d1][probe][c]=  (d-meanD)/stdD
        # data,nFlat = normalize_volts(data)

    # volts=np.append(volts,np.reshape(resample_by_interpolation(sweep.data[:20000],20000,4000),(1,4000,1)))
# volts[12,:4000,probe,c]=
par = np.zeros((11,19))
meta={"message":"Experimentail data","Stim":'chaotic_50khz_interpolated',"params":[66,77]}
metaJ=json.dumps(meta)
array_data=[metaJ]
# dataset = hdf5_file.create_dataset(dataset_name, shape=(0,), maxshape=(None,), )
structured_array = np.array([json.dumps(meta) for obj in array_data], dtype=h5py.special_dtype(vlen=str))
print("EXP MEAN",np.array(mean_tot).mean())
print("EXP STD",np.array(std_tot).mean())
print("EXP MEAN",mean_tot)
print("EXP MEAN",std_tot)
# dataD['meta.JSON']=metaJ

# -------- EFEL extraction BEFORE normalization (for both variants) --------
test_efel_features_reduce = extract_efel_features_from_volts(raw_volts_reduce)
test_efel_features_exact = extract_efel_features_from_volts(raw_volts_exact)

# -------- Write REDUCED (-6) variant --------
out_reduce = '/global/homes/k/ktub1999/ExperimentalData/PyForEphys/ChaoticA_Data_Reduce_TotalNorm/L5_TTPC1cADpyr2.mlPack1.h5'
os.makedirs(os.path.dirname(out_reduce), exist_ok=True)
meta_reduce = dict(meta)
meta_reduce['data_variant'] = 'reduced_minus_6'
meta_reduce['efel_feature_names'] = EFEL_FEATURE_NAMES + ['AP_count']
meta_reduce_j = json.dumps(meta_reduce)
meta_reduce_arr = np.array([meta_reduce_j], dtype=h5py.special_dtype(vlen=str))

hf = h5py.File(out_reduce,'w')
hf.create_dataset('test_volts_norm', data=volts_reduce)
hf.create_dataset('test_unit_par', data=par)
hf.create_dataset('test_efel_features', data=test_efel_features_reduce)
hf.create_dataset('meta.JSON', data=meta_reduce_arr,dtype=h5py.special_dtype(vlen=str))
hf.close()

# -------- Write EXACT (no -6) variant --------
out_exact = '/global/homes/k/ktub1999/ExperimentalData/PyForEphys/ChaoticA_Data_Exact_TotalNorm/L5_TTPC1cADpyr2.mlPack1.h5'
os.makedirs(os.path.dirname(out_exact), exist_ok=True)
meta_exact = dict(meta)
meta_exact['data_variant'] = 'exact_no_minus_6'
meta_exact['efel_feature_names'] = EFEL_FEATURE_NAMES + ['AP_count']
meta_exact_j = json.dumps(meta_exact)
meta_exact_arr = np.array([meta_exact_j], dtype=h5py.special_dtype(vlen=str))

hf = h5py.File(out_exact,'w')
hf.create_dataset('test_volts_norm', data=volts_exact)
hf.create_dataset('test_unit_par', data=par)
hf.create_dataset('test_efel_features', data=test_efel_features_exact)
hf.create_dataset('meta.JSON', data=meta_exact_arr,dtype=h5py.special_dtype(vlen=str))
hf.close()

