import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tune

#-------------------
#-------------------
#-------------------
class MyModel(nn.Module):
#...!...!..................
    def __init__(self,hpar,verb=0):
        super(MyModel, self).__init__()
        if verb: print('CNNandFC_Model hpar=',hpar)
        
        timeBins,inp_chan,stim_number=hpar['inputShape']

        self.inp_shape=(inp_chan,timeBins) # swap order
        
        self.verb=verb
        if verb: print('CNNandFC_Model inp_shape=',self.inp_shape,', verb=%d'%(self.verb))
        bn_cnn_slot=hpar['batch_norm_cnn_slot']
        # if('num_cnn_blocks' in hpar.keys()):
        #     self.num_stim=hpar['num_cnn_blocks']
        # else:
        #     self.num_stim=1
        self.num_stim=stim_number
        self.cnn_block_all=nn.ModuleList()
        self.flat_dim_all=[]
        self.flat_dim_total=0
        for stim in range(self.num_stim):
        # .....  CNN layers
            hpar1=hpar['conv_block'] #Can be stim specific
            cnn_block = nn.ModuleList()
            cnn_stride=1
            timeBins,inp_chan,_=hpar['inputShape']
            for out_chan,cnnker,plker in zip(hpar1['filter'],hpar1['kernel'],hpar1['pool']):
                # class _ConvMd( in_channels, out_channels, kernel_size, stride,
                # CLASS torch.nn.MaxPoolMd(kernel_size, stride=None,                
                cnn_block.append( nn.Conv1d(inp_chan, out_chan, cnnker, cnn_stride))
                cnn_block.append( nn.MaxPool1d(plker))
                cnn_block.append( nn.ReLU())
                if len(cnn_block)==bn_cnn_slot:
                    cnn_block.append( torch.nn.BatchNorm1d( out_chan))
                if len(cnn_block)==hpar['instance_norm_slot']:
                    cnn_block.append( nn.InstanceNorm1d(out_chan))
                inp_chan=out_chan

            ''' Automatically compute the size of the output of CNN+Pool block,  
            needed as input to the first FC layer 
            '''
            self.cnn_block_all.append(cnn_block)
            with torch.no_grad():
                # process 2 fake examples through the CNN portion of model
                x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
                y1=self.forwardCnnOnly(x1,stim)
                flat_dim=np.prod(y1.shape[1:])
                self.flat_dim_total+=flat_dim
                if verb>1: print('myNet flat_dim=',flat_dim)
            
            self.flat_dim_all.append(flat_dim)

        # here are all the preexisting normalization layers: https://pytorch.org/docs/stable/nn.html#normalization-layers

            if hpar['layer_norm']:
                cnn_block.append( nn.LayerNorm(y1.shape[1:]))
            
            
            

        self.flat_bn=None
        if hpar['batch_norm_flat']:
                self.flat_bn=torch.nn.BatchNorm1d(self.flat_dim_total)
    
        # .... add FC  layers
        hpar2=hpar['fc_block']
        self.fc_block  = nn.ModuleList()
        inp_dim=self.flat_dim_total
        for i,dim in enumerate(hpar2['dims']):
            self.fc_block.append( nn.Linear(inp_dim,dim))
            inp_dim=dim
            self.fc_block.append( nn.ReLU())
            if hpar2['dropFrac']>0 : self.fc_block.append( nn.Dropout(p= hpar2['dropFrac']))

        #.... the last FC layer may have different activation and no Dropout
        self.fc_block.append(nn.Linear(inp_dim,hpar['outputSize']))
        # here I have chosen no tanh activation so output range is unconstrained
  

#...!...!..................
    def forwardCnnOnly(self, x,no_of_stims):
        # flatten 2D image  .contiguous() for Some issue memory view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead 
        x=x.contiguous().view((-1,)+self.inp_shape )
        cnn_block=self.cnn_block_all[no_of_stims]
        if self.verb>2: print('J: inp2cnn',x.shape,x.dtype)
        for i,lyr in enumerate(cnn_block):
            if self.verb>2: print('Jcnn-lyr: ',i,lyr)
            x=lyr(x)
            if self.verb>2: print('Jcnn: out ',i,x.shape)
        return x
        
#...!...!..................
    def forward(self, x):
        if self.verb>2: print('J: inF',x.shape,'numLayers CNN=',len(self.cnn_block),'FC=',len(self.fc_block))
        xnew_all=None
        for no_of_stims in range(self.num_stim):
            x_temp=self.forwardCnnOnly(x[:,:,:,no_of_stims],no_of_stims)
            if xnew_all is None:
                xnew_all=x_temp
            else:
                xnew_all=torch.cat((xnew_all,x_temp),dim=1)


        xnew_all = xnew_all.view(-1,self.flat_dim_total)
        
        if self.flat_bn!=None:
            xnew_all=self.flat_bn(xnew_all);
            
        for i,lyr in enumerate(self.fc_block):
            xnew_all=lyr(xnew_all)
            if self.verb>2: print('Jfc: ',i,xnew_all.shape)
        if self.verb>2: print('J: y',xnew_all.shape)
        return xnew_all

#...!...!..................
    def summary(self):
        numLayer=sum(1 for p in self.parameters())
        numParams=sum(p.numel() for p in self.parameters())
        return {'modelWeightCnt':numParams,'trainedLayerCnt':numLayer,'modelClass':self.__class__.__name__}

