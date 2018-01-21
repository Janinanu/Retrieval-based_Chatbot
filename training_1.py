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
import torch.utils.data as data_utils


#%%

def create_dataframe(csvfile):
    dataframe = pd.read_csv(csvfile)
    return dataframe


#%% 
def create_vocab(dataframe):
    vocab = []
    
    word_freq = {}
    
    for index, row in dataframe.iterrows():
        
        context_cell = row["Context"]
        response_cell = row["Utterance"]
        
        context_str = str(context_cell)
        response_str = str(response_cell)
        context_words = context_str.split()
        response_words = response_str.split()
        
        train_words = context_words + response_words
        
        for word in train_words:
          
            if word.lower() not in vocab:
                vocab.append(word.lower())         
                       
            if word.lower() not in word_freq:
                word_freq[word.lower()] = 1
            else:
                word_freq[word] += 1
    
    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = ["<UNK>"] + [pair[0] for pair in word_freq_sorted]
    
#    outfile = open('my_vocab.txt', 'w')  
#
#    for word in vocab:
#        outfile.write(word + "\n")
#    outfile.close()
#    
    return vocab

#%%
    
def create_word_to_id(vocab):
            
    enumerate_list = [(id, word) for id, word in enumerate(vocab)]
        
    word_to_id = {pair[1]: pair[0] for pair in enumerate_list}
    
    return word_to_id
#%%

def create_id_to_vec(word_to_id, glovefile): 
    
    lines = open(glovefile, 'r').readlines()
    
    id_to_vec = {}
    
    vector = None
    
    count_glove = 0
    
    count_non_glove = 0
    
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
            count_glove += 1 #1046 from 1579 words are pretrained for 100 examples!!!
            
    for word, id in word_to_id.items(): #!!!
        
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01 #random also for <UNK>
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
            count_non_glove += 1
            
    embedding_dim = id_to_vec[0].shape[0]
    
    return id_to_vec, embedding_dim, count_glove, count_non_glove


#%%

def load_ids_and_labels(dataframe, word_to_id):
    
    rows_shuffled = dataframe.reindex(np.random.permutation(dataframe.index))
    
    
    context_id_list = []
    response_id_list = []

    
    context_column = rows_shuffled['Context']
    response_column = rows_shuffled['Utterance']
    label_column = rows_shuffled['Label'] #int

    max_context_len = 160
    
    for cell in context_column:
        context_words = cell.split()
        if len(context_words) > max_context_len:
            context_words = context_words[:max_context_len]
        context_ids = [word_to_id[word] for word in context_words]
        context_id_list.append(context_ids)

    for cell in response_column:
        response_ids = [word_to_id[word] for word in cell.split()]
        response_id_list.append(response_ids)
    
    label_array = np.array(label_column).astype(np.float32)

    return context_id_list, response_id_list, label_array


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
             self.embedding = nn.Embedding(self.vocab_size, self.input_size, sparse = False)
             self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout = dropout, bidirectional=False)
    
             self.init_weights()
             
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
            
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
        
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
            
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, hiddens = self.lstm(embeddings)

        return outputs, hiddens




#%%
        
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal(M)#?
        self.M = nn.Parameter(M, requires_grad = True)

        
         
    def forward(self, context_tensor, response_tensor):
        
        context_out, context_hc_tuple = self.encoder.forward(context_tensor)
        
        response_out, response_hc_tuple = self.encoder.forward(response_tensor)
        
        
        context_h = context_hc_tuple[0]
        
        response_h = response_hc_tuple[0]
        
        
        context_h_layer = context_h[0] #batch x hidden_size
        
        response_h_layer = response_h[0] #batch x hidden_size
        
        
        context = context_h_layer.mm(self.M) #batch x hidden_size
        context = context.view(-1, 1, self.hidden_size) #batch x 1 x hidden_size
        
        response = response_h_layer.view(-1, self.hidden_size, 1) #batch x hidden_size x 1
        
        score = torch.bmm(context, response).view(-1, 1) # batch x 1 x 1 --> batch x 1

        return score
    
#%% TRAINING
dataframe = create_dataframe('train_shuffled_tenthousand.csv')
vocab = create_vocab(dataframe)
word_to_id = create_word_to_id(vocab)
id_to_vec, emb_dim, count_glove, count_non_glove = create_id_to_vec(word_to_id, 'glove.6B.100d.txt')
vocab_len = len(vocab)

#%%
encoder_model = Encoder(
        input_size = emb_dim,
        hidden_size = 200,
        vocab_size = vocab_len)

dual_encoder = DualEncoder(encoder_model)

#%%

def train_model(learning_rate, epochs): 
    
     dual_encoder.train()
        
     optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = learning_rate)
       
     loss_func = torch.nn.BCEWithLogitsLoss()
     
     for epoch in range(epochs): 
                             
            context_id_list, response_id_list, label_array = load_ids_and_labels(dataframe, word_to_id)
            
            loss_sum = 0.0
                
            for i in range(len(label_array)):
                context = autograd.Variable(torch.LongTensor(context_id_list[i]).view(len(context_id_list[i]),1), requires_grad = False)
                
                response = autograd.Variable(torch.LongTensor(response_id_list[i]).view(len(response_id_list[i]), 1), requires_grad = False)
                
                label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label_array[i]).reshape(1,1))), requires_grad = False)
                
                score = dual_encoder(context, response)
        
                loss = loss_func(score, label)
                
                loss_sum += loss.data[0]
                
                loss.backward()
        
                optimizer.step()
               
                optimizer.zero_grad()
                
                torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
                
            print("Epoch: ", epoch, ", Loss: ", (loss_sum/len(label_array)))
            
#%%

#train_model(learning_rate = 0.001, epochs = 100)

#%%
#torch.save(dual_encoder.state_dict(), 'SAVED_MODEL_1000.pt')

#%%


