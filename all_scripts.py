#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:18:46 2017

@author: janinanu
"""

import csv
import torch.nn as nn
import numpy as np
import torch
import torch.autograd as autograd
import pandas as pd
from torch.nn import init
import torch.nn.utils.rnn 
import torch.optim as optim

#%%
def create_vocab(csvfile):
    vocab = []
    
    word_freq = {}
    
    infile = csv.DictReader(open(csvfile))
    
    for row in infile:
        #returns dict: {context: "...", "response": "...", label: "."}
        
        context_cell = row["Context"]
        response_cell = row["Utterance"]
        
        context_str = str(context_cell)
        response_str = str(response_cell)
        context_words = context_str.split()
        response_words = response_str.split()
        
        train_words = context_words + response_words
        
        for word in train_words:
          
            if word.lower() not in vocab:
                vocab.append(word)         
                       
            if word.lower() not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    
    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = [pair[0] for pair in word_freq_sorted]
    vocab = ["<PAD>"] + vocab
    
#    outfile = open('my_vocab.txt', 'w')  
#
#    for word in vocab:
#        outfile.write(word + "\n")
#    outfile.close()
#    
    return vocab

#%%
    
def create_word_to_id(csvfile):
    
    vocab = create_vocab(csvfile)
        
    enumerate_list = [(id, word) for id, word in enumerate(vocab)]
        
    word_to_id = {pair[1]: pair[0] for pair in enumerate_list}
    
    return word_to_id
#%%

def create_id_to_vec(csvfile, glovefile): 
    word_to_id = create_word_to_id(csvfile)
    
    lines = open(glovefile, 'r').readlines()
    
    id_to_vec = {}
    
    vector = None
        
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
 
        for word in word_to_id:
            if word_to_id[word] not in id_to_vec:
                v = np.zeros(*vector.shape, dtype='float32')
                v[:] = np.random.randn(*v.shape)
                #v = np.random.randn(*vector.shape).astype('float32') #32
                id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
     
    vec = np.zeros(*vector.shape, dtype='float32')

    id_to_vec[0] = torch.FloatTensor(torch.from_numpy(vec))
        
    vec_length = id_to_vec[1].shape[0]
    
    return id_to_vec, vec_length

#%%  

def load_batch(csvfile, batch_size, epoch):
   
    rows = pd.read_csv(csvfile)
       
    batch = rows[batch_size*epoch:batch_size*(epoch+1)]
      
    return batch

#%%

def load_ids_and_labels(csvfile, batch_size, epoch):
    
    word_to_id = create_word_to_id(csvfile)

    batch = load_batch(csvfile, batch_size, epoch)

    context_id_list = []
    response_id_list = []
     
    batch_context = batch['Context']
    batch_response = batch['Utterance']
    batch_label = batch['Label'] #int

    for cell in batch_context:
        context_ids = [word_to_id[word] for word in cell.split()]
        context_id_list.append(context_ids)
    
    for cell in batch_response:
        response_ids = [word_to_id[word] for word in cell.split()]
        response_id_list.append(response_ids)
        
    return batch, context_id_list, response_id_list, batch_label


#%% 
    
def make_tensors(csvfile, batch_size, epoch):
    
    batch, context_id_list, response_id_list, batch_label = load_ids_and_labels(csvfile, batch_size, epoch)

    max_len_context = max(len(c) for c in context_id_list)
    max_len_response = max(len(r) for r in response_id_list)
    max_len = max(max_len_context, max_len_response)
    
        
    for list in context_id_list:
        if (len(list) < max_len):
            list += [0] * (max_len - len(list))
      
    context_batch_tensor = torch.LongTensor(context_id_list)
    
    for list in response_id_list:
        if (len(list) < max_len):
           list += [0] * (max_len - len(list))
    
    response_batch_tensor = torch.LongTensor(response_id_list)
    
    len_batch = len(batch)
    label_array = np.array(batch_label)
    np.reshape(label_array, (len_batch, 1))
    label_float = label_array.astype('float32')
    label_batch_tensor = torch.FloatTensor(label_float).view(len_batch, 1)
    
    return context_batch_tensor, response_batch_tensor, label_batch_tensor

#%%
    
def get_emb_dim(glovefile):
    lines = open(glovefile, 'r').readlines()        
    vector = np.array(lines[0].split()[1:], dtype='float32')
    embedding_dim = len(vector)
    
    return embedding_dim


#%% MODEL
    
class Encoder(nn.Module):

    def __init__(self, 
             input_size, 
             hidden_size, 
             vocab_size, 
             num_layers = 1, 
             num_directions = 1, 
             dropout = 0, 
             bidirectional = False,
             rnn_type = 'lstm'): 
    
             super(Encoder, self).__init__()
             
             self.input_size = input_size
             self.hidden_size = hidden_size
             self.vocab_size = vocab_size
             self.num_layers = 1
             self.num_directions = 1
             self.dropout = 0,
             self.bidirectional = False
#        
             self.embedding = nn.Embedding(vocab_size, input_size, sparse = False, padding_idx = 0)
             self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout = dropout, bidirectional=False).cuda()
    
             self.init_weights()
             
    def init_weights(self):
        init.orthogonal(self.lstm.weight_ih_l0)
        init.uniform(self.lstm.weight_hh_l0, a=-0.01, b=0.01)
         
        embedding_weights = np.random.randn(self.vocab_size, self.input_size)
        
        id_to_vec, emb_dim = create_id_to_vec('/data/train_shuffled_onethousand.csv','/data/glove.6B.100d.txt')
    
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
    
        embedding_weights_tensor = torch.FloatTensor(embedding_weights).cuda()
    
        self.embedding.weight.data.copy_(embedding_weights_tensor)
    
    
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, hiddens = self.lstm(embeddings)
        return outputs, hiddens
    
#%%
        
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
         super(DualEncoder, self).__init__()
         self.encoder = encoder
         self.number_of_layers = 1
        
         M = torch.FloatTensor(self.encoder.hidden_size, self.encoder.hidden_size).cuda()
         
         init.normal(M)
         
         self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_batch, response_batch):
        
        context_out, context_hn = self.encoder(context_batch)
        
        response_out, response_hn = self.encoder(response_batch)
                
        scores_list = []
        
        len_batch = context_out.shape[0]
        
        for example in range(len_batch):
        
            context_h = context_out[example][-1].view(1, self.encoder.hidden_size)
            response_h = response_out[example][-1].view(self.encoder.hidden_size, 1)
        
            dot_var = torch.mm(torch.mm(context_h, self.M), response_h)[0][0]#gives 1x1 variable floattensor

            dot_tensor = dot_var.data #gives 1x1 floattensor
            dot_tensor.cuda()
        
            score_tensor = torch.sigmoid(dot_tensor)
            scores_list.append(score_tensor)
            
        y_preds_tensor = torch.stack(scores_list, 0).cuda()  
        y_preds = autograd.Variable(y_preds_tensor, requires_grad = True).cuda()
    
        return y_preds 
    
    
#%% TRAINING

torch.backends.cudnn.enabled = False
#%%
vocab = create_vocab('/data/train_shuffled_onethousand.csv')
vocab_len = len(vocab)
emb_dim = get_emb_dim('/data/glove.6B.100d.txt')
#%%

encoder_model = Encoder(
        input_size = emb_dim,
        hidden_size = 300,
        vocab_size = vocab_len)

encoder_model.cuda()
#%%
dual_encoder = DualEncoder(encoder_model)

dual_encoder.cuda()
#%%
loss_func = torch.nn.BCELoss()

loss_func.cuda()
#%%

learning_rate = 0.001

optimizer = optim.Adam(dual_encoder.parameters(),
                       lr = learning_rate)

#%%

epochs = 50
batch_size = 20

for epoch in range(epochs):
        
    context_batch_tensor, response_batch_tensor, label_batch_tensor = make_tensors('/data/train_shuffled_onethousand.csv', batch_size, epoch)
 
    context_batch = autograd.Variable(context_batch_tensor, requires_grad=False).cuda()
   
    response_batch = autograd.Variable(response_batch_tensor, requires_grad=False).cuda()
     
    y_preds = dual_encoder(context_batch.detach(), response_batch.detach())
                
    y = autograd.Variable(label_batch_tensor, requires_grad = False).cuda()
        
    loss = loss_func(y_preds, y)
        
    print("Epoch: ", epoch, ", Loss: ", loss.data[0])
        
    dual_encoder.zero_grad()
    loss.backward()
    
    optimizer.step()

#%%

torch.save(dual_encoder.state_dict(), 'SAVED_MODEL.pt')
    
#%%


