#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:18:46 2017

@author: janinanu
"""

# vocab, word to id, id to vec, padding

import torch.nn as nn
import numpy as np
import torch
import torch.utils.data
import torch
import torch.autograd as autograd
import pandas as pd
from torch.nn import init
import torch.nn.utils.rnn 
import torch.optim as optim


#%% METHOD: MAKE WORD_TO_ID DICTIONARY FROM VOCAB FILE

def make_word_to_id(file):

    lines = open(file, 'r').readlines()
    
    for line in lines:
        line.strip("\n")

    word_to_id = {word: id for id, word in enumerate(lines)}
   
    
    return word_to_id
    
    #word_to_id_sorted = sorted(word_to_id.items(), key=lambda item: item[1])

#%% METHOD: MAKE ID_TO_VEC DICTIONARY FROM WORD_TO_ID AND GLOVE

def make_id_to_vec(glovefile): 
    word_to_id = make_word_to_id('/data/my_vocab.txt')
    
    lines = open(glovefile, 'r').readlines()
    
    word_vecs = {}
    
    vector = None
        
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if (word+"\n") in word_to_id:
            word_vecs[word_to_id[word+"\n"]] = torch.FloatTensor(torch.from_numpy(vector)).cuda()
     
        for word in word_to_id:
            if word_to_id[word] not in word_vecs:
               v = np.zeros(*vector.shape, dtype='float32')
               v[:] = np.random.randn(*v.shape)
               word_vecs[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v)).cuda()
     
    vec = np.zeros(*vector.shape, dtype='float32')
    #vec = np.zeros(*vector.shape).astype('float32') #32
    word_vecs[0] = torch.FloatTensor(torch.from_numpy(vec)).cuda()
        
    vec_length = vector.shape[0]
    
    return word_vecs, vec_length
#%%

word_to_id = make_word_to_id('/data/my_vocab.txt')
id_to_vec, embedding_dim = make_id_to_vec('/data/glove.6B.100d.txt')
#%%

def load_ids_and_labels():
    file = '/data/train_shuffled_onethousand.csv'
    
    context_id_list = []
    response_id_list = []
    
    rows = pd.read_csv(file)
    
    context_column = rows.iloc[:, 1]
    response_column = rows.iloc[:, 2]
    label_column = rows.iloc[:, 3]
    
    for context_cell in context_column:
        context_ids = [word_to_id[word+"\n"] for word in context_cell.split()]
        context_id_list.append(context_ids)
    
    for response_cell in response_column:
        response_ids = [word_to_id[word+"\n"] for word in response_cell.split()]
        response_id_list.append(response_ids)
     
    labels_array = np.array(label_column)    
    
    return context_id_list, response_id_list, labels_array


#%% AS LOOKUP: FROM WORD ID IN MATRIX FIND CORRESPONDING WORD VECTOR IN EMBEDDING
def make_matrices():
    context_id_list, response_id_list, labels_array = load_ids_and_labels()
    
    context_id_list.sort(key=len, reverse=True)
    response_id_list.sort(key=len, reverse=True)    
    
    max_len_context = max(len(c) for c in context_id_list)
    max_len_response = max(len(r) for r in response_id_list)
    max_len = max(max_len_context, max_len_response)
    
        
    for list in context_id_list:
        if (len(list) < max_len):
            list += [0] * (max_len - len(list))
      
    context_matrix = torch.LongTensor(context_id_list).cuda()
    context_matrix = autograd.Variable(context_matrix).cuda()
    
    for list in response_id_list:
        if (len(list) < max_len):
           list += [0] * (max_len - len(list))
    
    response_matrix = torch.LongTensor(response_id_list).cuda()
    response_matrix = autograd.Variable(response_matrix).cuda()
    
    
    labels_tor = torch.from_numpy(labels_array)
    labels_var = autograd.Variable(torch.LongTensor(labels_tor))
    labels_var.cuda()

    return context_matrix, response_matrix, labels_var
    
#%%MODEL

#dtype = torch.cuda.FloatTensor
    
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
                 self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout = dropout, bidirectional=False)
        
                 self.init_weights()

    def init_weights(self):
        init.orthogonal(self.lstm.weight_ih_l0)
        init.uniform(self.lstm.weight_hh_l0, a=-0.01, b=0.01)
         
        embedding_weights = torch.FloatTensor(self.vocab_size, 100).cuda()
        init.uniform(embedding_weights, a = -0.25, b= 0.25)
        
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
        
        self.embedding.weight.data.copy_(embedding_weights)

            
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, hiddens = self.lstm(embeddings)

#%%
        
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
         super(DualEncoder, self).__init__()
         self.encoder = encoder
         self.number_of_layers = 2
         #h_0 (num_layers * num_directions, batch, hidden_size): 
         #tensor containing the initial hidden state for each element in the batch.
         dual_hidden_size = self.encoder.hidden_size * self.encoder.num_directions
         
         M = torch.FloatTensor(dual_hidden_size, dual_hidden_size).cuda()
         
         init.normal(M)
         
         self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, contexts, responses):
        #output (seq_len, batch, hidden_size * num_directions): 
        #tensor containing the output features (h_t) from the last layer 
        #of the RNN, for each t. 
        
        #h_n (num_layers * num_directions, batch, hidden_size): 
        #tensor containing the hidden state for t=seq_len
        context_out, context_hn = self.encoder(contexts)
        context_hn = context_hn[0] #final hidden state in layer 0
        
        response_out, response_hn = self.encoder(responses)
        response_hn = response_hn[0]
        
        dual_hidden_size = self.encoder.hidden_size * self.encoder.num_directions
        
        scores_list = []
        
        for e in range(len(context_hn[0])): #context_hn = context_hn[0] #over all examples
            #context_out[e][-1].view(1, dual_hidden_size)
            context_h = context_out[-1][e].view(1, dual_hidden_size)#context_out[-1] = context_hn
            #spread hidden_size*num_directions over hidden_size*num_directions?
            
            response_h = response_out[-1][e].view(dual_hidden_size,1)
            #gives vectors of hidden_size for each example
            
            
            dot = torch.mm((context_h, self.M), response_h)#gives 1x1 vector
         
            score = nn.Sigmoid(dot)
            
            scores_list.append(score)
            
        y_preds = torch.stack(scores_list)
        
        return y_preds #to be used in training to compare with label
     
#%% TRAINING

#torch.backends.cudnn.enabled = False


vocab_len = 0
vocab_lines = open('/data/my_vocab.txt', 'r').readlines()
for line in vocab_lines:
	vocab_len +=1  
#%%

encoder_model = Encoder(
        input_size = embedding_dim,
        hidden_size = 300,
        vocab_size = vocab_len)

encoder_model.cuda()

#%%
dual_encoder = DualEncoder(encoder_model)

dual_encoder.cuda()


loss_func = torch.nn.BCELoss()

loss_func.cuda()

#loss_func = torch.nn.BCEWithLogitsLoss() #input: bilinear_output (batch_size x 1)

learning_rate = 0.001
epochs = 100
#batch_size = 50

optimizer = optim.Adam(dual_encoder.parameters(),
                       lr = learning_rate)

for i in range(epochs):
    #batches needed!
    context_matrix, response_matrix, y = make_matrices()

    y_preds = dual_encoder(context_matrix, response_matrix)

    loss = loss_func(y_preds, y)
    
    if i % 10 == 0:
        print("Epoch: ", i, ", Loss: ", loss.data[0])
        
    #evaluation metrics...
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
    
    optimizer.step()

torch.save(dual_encoder.state_dict(), 'SAVED_MODEL.pt')
    
    


