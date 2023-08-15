import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len,device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.pe=self.pe.to(device)
        
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        if(x.device!=self.pe.device):
            self.pe=self.pe.to(x.device)
        x = x + self.pe[:, :x.size(1)]
        return x,self.pe

class TransformerEncoder(nn.Module):
    def __init__(self, d_model,nhead, num_encoder_layers, dim_feedforward, max_seq_len):
        super(TransformerEncoder, self).__init__()
        device = torch.cuda.current_device() 
        # self.embedding = nn.Linear(num_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len,device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # self.output_layer = nn.Linear(d_model*max_seq_len, 19)
        self.flatten = nn.Flatten()
        self.fc_block  = nn.ModuleList()
        inp_dim=d_model*max_seq_len
        dims=[ 512, 512, 512, 256, 128 ]
        drp=0.02
        for i,dim in enumerate(dims):
            self.fc_block.append( nn.Linear(inp_dim,dim))
            inp_dim=dim
            self.fc_block.append( nn.ReLU())
            if drp>0 : self.fc_block.append( nn.Dropout(p= drp))
        self.fc_block.append(nn.Linear(inp_dim,19))
    def forward(self, x):
        x = self.positional_encoding(x)[0]
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        # x = self.output_layer(x)
        for i,lyr in enumerate(self.fc_block):
            x=lyr(x)
            
        return x

# # Parameters
# d_model = 4
# nhead = 4
# num_encoder_layers = 6
# dim_feedforward = 64
# max_seq_len = 4000
# num_channels = d_model

# # Create the model
# model = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len)

# # Generate dummy input
# input_data = torch.rand(max_seq_len, num_channels)

# # Forward pass
# output = model(input_data.unsqueeze(0))  # Adding batch dimension

# print(output.shape)  # Output shape: (batch_size, seq_len, num_features)
