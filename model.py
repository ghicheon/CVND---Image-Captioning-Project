import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size , hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size , vocab_size)

    def forward(self,features,captions):
        arg1 = features.view(len(features),1,-1)
        arg2 = self.embed(captions[:,:-1])
        #print("DEBUG1:", arg1.shape , arg2.shape) #DEBUG1: torch.Size([10, 1, 256]) torch.Size([10, 14, 256])
        in_ = torch.cat(( arg1,arg2),1)
        #print("DEBUG2 ", in_.shape) #DEBUG2  torch.Size([10, 15, 256])
        out,h = self.lstm(in_)
        #print("DEBUG3 ",out.shape , h[0].shape) #DEBUG3  torch.Size([10, 15, 512]) torch.Size([1, 10, 512])
        out = self.fc(out)
        #print("DEBUG4 ",out.shape ) #DEBUG4  torch.Size([10, 15, 8856])
        return out


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #print("input.shape  ",  inputs.shape)   # 1 1 512
  
        ret=[]
        for i in range(max_len):
           out,states = self.lstm(inputs,states)
           #print("sample1:", out.shape)  # 1 1 1024
           out = self.fc(out.squeeze(1))
           #print("sample2:", out.shape)   # 1 8855
            
           idx = out.max(1)[1]
           #print("sample3:",idx, idx.shape)  
            
           ret.append(idx.item())
           
           inputs = self.embed(idx).unsqueeze(1)   
  
        return ret

