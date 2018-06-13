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
        self.lstm = nn.LSTM(embed_size , hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size , vocab_size)

    def forward(self,features,captions):
        #print("DEBUG  ",  type(features),type(captions))
        #print("DEBUG  ",  features.shape,captions.shape)
        arg1 = features.view(len(features),1,-1)
        arg2 = self.embed(captions[:,:-1])
        #print("DEBUG3 ", arg1.shape , arg2.shape)
        in_ = torch.cat(( features.view(len(features),1,-1),   self.embed(captions[:,:-1]) ),1)
        #print("DEBUG4 ", in_.shape)
        #in_ = torch.cat(( features.view(len(features),1,-1),   self.embed(captions) ),1)
        out,h = self.lstm(in_)
        #print("DEBUG5 ",out.shape , h[0].shape)
        out = self.fc(out)
        #print("DEBUG6 ",out.shape )
        return out
    


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        print("input.shape  ",  inputs.shape)
        print("len(inputs)  ", len(inputs))

        ret=[]

        for i in range(max_len):
           out,states = self.lstm(inputs,states)
           out = self.fc(out.squeeze(1))
           idx = out.max(1)[1]
           ret.append(idx.item())
           inputs = self.embed(idx).unsqueeze(1)
  
        return ret





"""
out = self.embed(out.long())


arg1 = inputs.view(len(out),1,-1)
out,h = self.lstm(arg1)
out = self.fc(out)
print(out)
print("out.shape ", out.shape)
idx = out.argmax()
print(idx)
result.append( idx)

return result

"""
