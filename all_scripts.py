

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

def create_dataframe(csvfile):
    dataframe = pd.read_csv(csvfile)
    return dataframe


def create_vocab(dataframe):
    vocab = []
    
    word_freq = {}
    
    for index, row in dataframe.iterrows():
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
        
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
 
        for word, id in word_to_id.items(): #!!!
            
            if word_to_id[word] not in id_to_vec:
                v = np.zeros(*vector.shape, dtype='float32')
                v[:] = np.random.randn(*v.shape)*0.01
                id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
    
        
    embedding_dim = id_to_vec[0].shape[0]
    
    return id_to_vec, embedding_dim


#%%

def load_ids_and_labels(dataframe, word_to_id):
   
    rows_shuffled = dataframe.reindex(np.random.permutation(dataframe.index))
    
    context_id_list = []
    response_id_list = []

    context_column = rows_shuffled['Context']
    response_column = rows_shuffled['Utterance']
    label_column = rows_shuffled['Label'] #int
    
    for cell in context_column:
        context_ids = [word_to_id[word] for word in cell.split()]
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
        init.uniform(self.lstm.weight_ih_l0)
        init.uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        #init.uniform(embedding_weights, a = -0.25, b= 0.25)
            
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
        
         
    def forward(self, context_tensor, response_tensor):
        #Outputs: output, (h_n, c_n)
        
        #output (seq_len, batch, hidden_size * num_directions): 
        #tensor containing the output features (h_t) from the last layer 
        #of the RNN, for each t. 
        
        #h_n (num_layers * num_directions, batch, hidden_size): 
        #tensor containing the hidden state for t=seq_len
                
        context_out, context_hc_tuple = self.encoder.forward(context_tensor)
           
        response_out, response_hc_tuple = self.encoder.forward(response_tensor)
        
        
        context_h = context_hc_tuple[0]
        
        response_h = response_hc_tuple[0]
        
        
        context_h_layer = context_h[-1] #batch x hidden_size
        
        response_h_layer = response_h[-1] #batch x hidden_size
        
        
        context = context_h_layer.view(-1, 1, self.hidden_size) #batch x 1 x hidden_size
        
        response = response_h_layer.view(-1, self.hidden_size, 1) #batch x hidden_size x 1
        
         
        M = torch.FloatTensor(context.shape[0], self.hidden_size, self.hidden_size) # batch x hidden x hidden     
        
        init.xavier_normal(M)
         
        self.M = nn.Parameter(M, requires_grad = True)
        
        score_M = torch.bmm(torch.bmm(context, self.M), response).view(-1, 1) # batch x 1
        
        return score_M
        
    
              
#%% TRAINING

#torch.backends.cudnn.enabled = False

dataframe = create_dataframe('train_shuffled_onethousand.csv')
vocab = create_vocab(dataframe)
word_to_id = create_word_to_id(vocab)
id_to_vec, emb_dim = create_id_to_vec(word_to_id, 'glove.6B.100d.txt')
vocab_len = len(vocab)


encoder_model = Encoder(
        input_size = emb_dim,
        hidden_size = 200,
        vocab_size = vocab_len)

dual_encoder = DualEncoder(encoder_model)


#%%
#loss_func = torch.nn.functional.binary_cross_entropy_with_logits() #input: bilinear_output (batch_size x 1)
optimizer = torch.optim.SGD(dual_encoder.parameters(), lr = 0.0001)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)



epochs = 100
for epoch in range(epochs): 
    
    context_id_list, response_id_list, label_array = load_ids_and_labels(dataframe, word_to_id)
    
    loss_acc = 0

    optimizer.zero_grad()
    
    for i in range(len(label_array)):
        context = autograd.Variable(torch.LongTensor(context_id_list[i]).view(len(context_id_list[i]),1), requires_grad = False)
        
        response = autograd.Variable(torch.LongTensor(response_id_list[i]).view(len(response_id_list[i]), 1), requires_grad = False)
        
        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label_array[i]).reshape(1,1))), requires_grad = False)
        
        score = dual_encoder(context, response)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, label) #loss for 1 example
        
        loss_acc += loss.data[0]
        
        loss.backward()

        scheduler.step()
        
        #torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
        
    print("Epoch: ", epoch, ", Loss: ", (loss_acc/len(label_array)))
   
    #for name, param in dual_encoder.named_parameters():
       #if param.grad is None:
           #print(name, param.shape)
        
        #if ((epoch == 0) or (epoch == 1)) and (name == "encoder.embedding.weight"):
          # print(name, param.grad) 
    

torch.save(dual_encoder.state_dict(), 'SAVED_MODEL.pt')
    
#%%



