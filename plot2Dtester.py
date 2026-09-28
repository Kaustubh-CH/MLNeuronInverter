import yaml
from toolbox.Plotter import Plotter_NeuronInverter
from toolbox.Util_IOfunc import read_yaml
import numpy as np
class args1():
    def __init__(self) -> None:
        self.formatVenue="Paper"
        self.noXterm=True
        self.outPath="./"
        self.prjName="testing_plots"


# inpMD=yaml.load( "/global/homes/k/ktub1999/Neuron/neuron4/neuroninverter/packBBP3/aa.yaml", Loader=yaml.CLoader)
inpMD = read_yaml( "/global/homes/k/ktub1999/Neuron/neuron4/neuroninverter/packBBP3/aa.yaml")
npar = len(inpMD["include"])

residualL = np.random.uniform(-1,1,(npar,3))
trueU = np.random.uniform(-1,1,(100,npar))

args = args1()

sumRec={}
sumRec['residual_mean_std']=residualL
sumRec['jobId']=1
sumRec['lossThrHi']=10
sumRec['domain']='test'
sumRec['testLossMSE']=1
sumRec['inpShape']='(10,23)'
sumRec['short_name']='testing'
sumRec['modelDesign']='Bt'
sumRec['trainTime']=1
sumRec['train_stims_select']=1
sumRec['train_glob_sampl']=1
sumRec['loss_valid']=1
sumRec['pred_stims_select']=1












plot=Plotter_NeuronInverter(args,inpMD ,sumRec )
plot.param_residua2D(trueU,trueU)
plot.display_all( png=1)  