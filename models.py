"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform

import numpy as np

from math import floor, sqrt, sin, cos
import random
import sys
import time
# a base class with basic parameters for initialization
class BaseModel(nn.Module):
    def __init__(self, Y, word_embeddings_matrix, vocab_size, dropout=0.5, gpu=True, embed_size=100, embed_freeze=False, hier=False):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.hier = hier
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        
        print('loading model')
        #make embedding layer
        if word_embeddings_matrix is not None:               
            W = torch.Tensor(word_embeddings_matrix)
            self.embed = nn.Embedding.from_pretrained(W, freeze=embed_freeze)
            self.embed.padding_idx = 0
        else:
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target):
        #calculate the BCE011
        return F.binary_cross_entropy_with_logits(yhat, target)

class Attention(torch.nn.Module): # Attention layer used in model deep dilation 
    def __init__(self, n_dim, n_labels, embed_desc=False, desc_dim=300):

        super(Attention, self).__init__()
        
        self.embed_desc = embed_desc
        self.softmax = nn.Softmax(dim=2)
        
        if self.embed_desc:
            self.linear1 = nn.Linear(desc_dim, n_dim)
            self.linear2 = nn.Linear(n_dim, n_dim)
            xavier_uniform(self.linear1.weight)
            xavier_uniform(self.linear2.weight)
            self.activation = nn.Tanh()
        else:
            self.U = nn.Parameter(torch.FloatTensor(n_labels, n_dim))
            xavier_uniform(self.U)

    def forward(self, x, desc_data=None):

        if self.embed_desc:
            desc_data = self.activation(self.linear1(desc_data))
            desc_data, _ = desc_data.max(dim=1, keepdim=False)
            desc_data = self.activation(self.linear2(desc_data))
            alpha = self.softmax(desc_data.matmul(x.transpose(1,2))) # from dot-product attention
        else:
            #alpha = self.softmax(self.U(x).transpose(1,2))
            alpha = self.softmax(self.U.matmul(x.transpose(1,2)))

        return alpha.matmul(x), alpha

            
class DeepDilated(BaseModel):

    def __init__(self, Y, dims, kernel_size, dilation, embed_matrix, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_freeze=False, dropout=[0.5], hier=False, embed_desc=False, fine2coarse = None):
        super(DeepDilated, self).__init__(Y, embed_matrix, vocab_size, dropout=.0, gpu=gpu, embed_size=dims[0], embed_freeze=embed_freeze, hier=hier)
        
        assert len(dropout) == len(dims)-1
        
        self.drops = nn.ModuleList([nn.Dropout(p=drop) for drop in dropout])
        
        self.convs = nn.ModuleList([nn.Conv1d(dims[i], dims[i+1], kernel_size=kernel_size, dilation=dilation[i], padding=floor(kernel_size/2)*dilation[i]) for i in range(len(self.drops))])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(torch.Size([dims[i+1]])) for i in range(0,len(self.drops))])
        
        self.activation = nn.Tanh()
        
        self.attention = Attention(dims[-1], Y, embed_desc, desc_dim=self.embed_size)
        
        self.embed_desc = embed_desc
        
        self.final = nn.Linear(dims[-1], Y, bias=True)
        xavier_uniform(self.final.weight)
        
        self.sigmoid = nn.Sigmoid()
        
        if self.hier:
            self.attention_coarse = Attention(dims[-1], Y_coarse)
            xavier_uniform(self.attention_coarse.weight)
        
            self.final_coarse = nn.Linear(dims[-1], Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)

    def forward(self, x, target, target_coarse=None, desc_data=None, get_attention=True):
        
        x = self.embed(x)
        
        for i in range(len(self.convs)):
            x = self.drops[i](x)
            x = x.transpose(1,2)
            x = self.convs[i](x)
            x = x.transpose(1,2)
            x = self.activation(x)
            x = self.layer_norms[i](x)

        if self.hier:
            m_coarse, alpha_coarse = self.attention_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
            
            m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            mask = torch.round(self.sigmoid(yhat_coarse[:,self.fine2coarse]))
            yhat = yhat * mask
            
            loss = self._get_loss(yhat, target) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat = self.sigmoid(yhat)
            
            return (yhat, yhat_coarse), loss, (alpha, alpha_coarse)
        else:    
            m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            loss = self._get_loss(yhat, target)
            yhat = self.sigmoid(yhat)
            
            return yhat, loss, alpha
