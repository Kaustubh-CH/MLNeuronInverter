import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CnnBlock(nn.Module):
    def _init_(self,hpar,verb=0):
        super(CnnBlock, self).__init__()
        if verb: print("CNNBlock Number=",hpar) #Print Block number pls
        if verb: print("CNNBlock hpar=",hpar)

        timeBins,inp_chan=hpar['inputShape']
        self.inp_shape=(inp_chan,timeBins)

        self.verb=verb
        if verb: print("CnnBlock Inp Shape=",self.inp_shape)
        bn_cnn_slot=hpar['batch_norm_cnn_slot']
        in_cnn_slot=hpar['instance_norm_slot']
        
        self.cnn_block=nn.ModuleList()
        #Layers loading
        cnn_stride=1
        for out_chan,cnnker,plker in zip(hpar['filter'],hpar['kernel'],hpar['pool']):
            self.cnn_block.append( nn.Conv1d(inp_chan, out_chan, cnnker, cnn_stride))
            self.cnn_block.append( nn.MaxPool1d(plker))
            self.cnn_block.append( nn.ReLU())
            if len(self.cnn_block)==bn_cnn_slot:
                self.cnn_block.append( torch.nn.BatchNorm1d( out_chan))
            if len(self.cnn_block)==in_cnn_slot:
                self.cnn_block.append( nn.InstanceNorm1d(out_chan))
            inp_chan=out_chan
    
        ''' Automatically compute the size of the output of CNN+Pool block,  
        needed as input to the first FC layer 
        '''
             
        with torch.no_grad():
            # process 2 fake examples through the CNN portion of model
            x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
            y1=self.forwardCnnOnly(x1)
            self.flat_dim=np.prod(y1.shape[1:]) 
            if verb>1: print('myNet flat_dim=',self.flat_dim)

        if hpar['layer_norm']:
            self.cnn_block.append( nn.LayerNorm(y1.shape[1:]))

        # self.flat_bn=None
        # if hpar['batch_norm_flat']:
        #         self.flat_bn=torch.nn.BatchNorm1d(self.flat_dim)
    
    def forward(self,x):
        if self.verb:
              print('J: inF',x.shape,'numLayers CNN=',len(self.cnn_block))
        #flatten 2D image:
        x=x.view((-1,)+self.inp_shape)
        for i,lyr in enumerate(self.cnn_block):
            if self.verb>2: print('Jcnn-lyr: ',i,lyr)
            x=lyr(x)
            if self.verb>2: print('Jcnn: out ',i,x.shape)
        return x

class FcBlock(nn.Module):
    def __init__(self,hpar,verb=0):
        super(FcBlock, self).__init__()
        self.fc_block  = nn.ModuleList()
        self.flat_dim=hpar['flat_dim']
        if verb:
            print("FC Hpar=",hpar)
        inp_dim=self.flat_dim
        for i,dim in enumerate(hpar['dims']):
            self.fc_block.append( nn.Linear(inp_dim,dim))
            inp_dim=dim
            self.fc_block.append( nn.ReLU())
            if hpar['dropFrac']>0 : self.fc_block.append( nn.Dropout(p= hpar['dropFrac']))

        self.fc_block.append(nn.Linear(inp_dim,hpar['outputSize']))
    

    def forward(self,x):
        if self.verb>2: print('F: inF',x.shape,'FC=',len(self.fc_block))
        for i,lyr in enumerate(self.fc_block):
            x=lyr(x)
            if self.verb>2: print('Jfc: ',i,x.shape)
        if self.verb>2: print('J: y',x.shape)
        return x
    
class MyModel(nn.Module):
    ## Add in Flat Dim from ConV Block to Hpar before FC Block
    def __init__(self,hpar,verb=0):
        super(MyModel, self).__init__()
        if verb: print('MyModel hpar=',hpar)
        self.cnn_blocks=[]
        num_cnn_blocks=hpar['num_cnn_blocks'] #Number of Stims
        self.num_cnn_blocks=num_cnn_blocks
        self.flat_dim=0
        self.flat_dim_all=[]
        for cnn in range(num_cnn_blocks):
            hpar1=hpar['conv_block']
            hpar1['batch_norm_cnn_slot']=hpar['batch_norm_cnn_slot']
            hpar1['instance_norm_slot']=hpar['instance_norm_slot']
            hpar1['layer_norm']= hpar['layer_norm']
            cnn=CnnBlock(hpar=hpar1,verb=verb)
            self.cnn_blocks.append(cnn)
            self.flat_dim+=self.cnn_blocks[-1].flat_dim
            self.flat_dim_all.append(self.cnn_blocks[-1].flat_dim)

        
        self.flat_bn=None
        ##After Contatenation
        if hpar['batch_norm_flat']:
            self.flat_bn=torch.nn.BatchNorm1d(self.flat_dim)
        hpar2=hpar['fc_block']
        hpar2['flat_dim']=self.flat_dim
        self.fc=FcBlock(hpar2,verb)
    
    def forward(self,x):
        #x.shape (400k,4000,4,2)
        xnew_all=[]
        for j in range(self.num_cnn_blocks):
            xnew=self.cnn_blocks[j](x[:,:,:,j])
            xnew = xnew.view(-1,self.flat_dim_all[j])
            xnew_all=torch.cat((xnew_all,xnew))
        xnew_all=xnew.view(-1,self.flat_dim)##Check again

        xnew_all = FcBlock(xnew_all)
        return xnew_all



        
