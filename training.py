#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:03:50 2018

@author: janinanu
"""

import torch
import torch.utils.data
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn 
import preprocess
import data
import model

#%%

encoder_model = model.Encoder(
        input_size = model.embedding_dim,
        hidden_size = 300,
        vocab_size = data.vocab_size)

dual_encoder = model.DualEncoder(encoder_model)

loss_func = torch.nn.BCELoss()

#loss_func = torch.nn.BCEWithLogitsLoss() #input: bilinear_output (batch_size x 1)

learning_rate = 0.001
epochs = 10000
batch_size = 50

parameters = dual_encoder.parameters() #LSTM weights and M?

optimizer = optim.Adam(dual_encoder.parameters(),
                       lr = learning_rate)

for i in range(epochs):
    
    context_matrix, response_matrix, y = data.make_matrix('train_shuffled_onethousand.csv')

    y_pred = model(context_matrix, response_matrix)

    loss = loss_func(y_pred, y)
    
    if i % 10 == 0:
        print("Epoch: ", i, ", Loss: ", loss.data[0])
        
    #evaluation metrics...
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
    
    optimizer.step()
    
    
    
    
    
    
    
    
    
    












