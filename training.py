#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:03:50 2018

@author: janinanu
"""

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.utils.rnn 
from vocab_and_dicts import vocab 
import ids_and_matrices
import model

#%%
vocab_len = len(vocab)

encoder_model = model.Encoder(
        input_size = 100,
        hidden_size = 300,
        vocab_size = vocab_len)
#%%
dual_encoder = model.DualEncoder(encoder_model)

loss_func = torch.nn.BCELoss()

#loss_func = torch.nn.BCEWithLogitsLoss() #input: bilinear_output (batch_size x 1)

learning_rate = 0.001
epochs = 10000
batch_size = 50

optimizer = optim.Adam(dual_encoder.parameters(),
                       lr = learning_rate)

for i in range(epochs):
    
    context_matrix, response_matrix, y = ids_and_matrices.make_matrices()

    y_preds = dual_encoder(context_matrix, response_matrix)

    loss = loss_func(y_preds, y)
    
    if i % 10 == 0:
        print("Epoch: ", i, ", Loss: ", loss.data[0])
        
    #evaluation metrics...
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
    
    optimizer.step()
    
    
    
    
    
    
    
    
    
    











