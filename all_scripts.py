#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:18:46 2017

@author: janinanu
"""

# vocab, word to id, id to vec, padding
import csv
import torch.nn as nn
import numpy as np
from math import floor
import torch
import torch
import torch.autograd as autograd
import pandas as pd
from torch.nn import init
import torch.nn.utils.rnn 
import torch.optim as optim


# make batches
# go through examples in batches
# split example into context, response, label
# turn each example's context and response into id tensor

#%% CREATE VOCABULARY
def create_vocab(csvfile):
    vocab = []
    
    word_freq = {}
    
    infile = csv.DictReader(open(csvfile))
    
    for row in infile:
        #returns dict: {context: "...", "response": "...", label: "."}
        
        context_cell = row["Context"]
        response_cell = row["Utterance"]
        #label_cell = row["Label"]
        
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
    #vec = np.zeros(*vector.shape).astype('float32') #32
    id_to_vec[0] = torch.FloatTensor(torch.from_numpy(vec))
        
    vec_length = id_to_vec[1].shape[0]
    
    return id_to_vec, vec_length

#%% make batches 

def load_batch(csvfile, batch_size, epoch):
   
    rows = pd.read_csv(csvfile)
    num_train_examples = len(rows)    
    num_complete_batches = floor(num_train_examples/batch_size)
    
    batch = rows[batch_size*epoch:batch_size*(epoch+1)]
    
    if(batch_size*epoch != batch_size*num_complete_batches):
        batch = rows[batch_size*epoch:num_train_examples]
    
    return batch


#%%

def load_ids_and_labels(batch_example):
    #file = '/data/train_shuffled_onethousand.csv'
    #batch = load_batch(csvfile)
    
    context_id_list = []
    response_id_list = []
     
    example_context = batch_example['Context']
    example_response = batch_example['Utterance']
    example_label = batch_example['Label']

    context_ids = [word_to_id[word] for word in example_context.split()]
    context_id_list.append(context_ids)
    
    response_ids = [word_to_id[word] for word in example_response.split()]
    response_id_list.append(response_ids)
 
    label_array = np.array(example_label)
    label_float = label_array.astype('float32')
       
    return context_id_list, response_id_list, label_float


#%% 
    
def make_tensors(batch_example):
    
    context_id_list, response_id_list, label_float = load_ids_and_labels(batch_example)
    
    max_len_context = max(len(c) for c in context_id_list)
    max_len_response = max(len(r) for r in response_id_list)
    max_len = max(max_len_context, max_len_response)
    
        
    for list in context_id_list:
        if (len(list) < max_len):
            list += [0] * (max_len - len(list))
      
    context_tensor = torch.LongTensor(context_id_list)  
    
    for list in response_id_list:
        if (len(list) < max_len):
           list += [0] * (max_len - len(list))
    
    response_tensor = torch.LongTensor(response_id_list)
    
    label_tensor = torch.FloatTensor(torch.from_numpy(label_float))
    
    return context_tensor, response_tensor, label_tensor
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
         
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size).cuda()
        init.uniform(embedding_weights, a = -0.25, b= 0.25)
        
        id_to_vec, emb_dim = create_id_to_vec('/data/train_shuffled_onethousand.csv','/data/glove.6B.100d.txt')

        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
        
        #del self.embedding.weight
        #self.embedding.weight = nn.Parameter(embedding_weights)
        #self.embedding.weight.requires_grad = True
        
        self.embedding.weight.data.copy_(torch.from_numpy(self.embedding_weights))
            
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
         #h_0 (num_layers * num_directions, batch, hidden_size): 
         #tensor containing the initial hidden state for each element in the batch.
         #dual_hidden_size = self.encoder.hidden_size * self.encoder.num_directions
         
         M = torch.FloatTensor(self.encoder.hidden_size, self.encoder.hidden_size).cuda()
         
         init.normal(M)
         
         self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, contexts, responses):
        #output (seq_len, batch, hidden_size * num_directions): 
        #tensor containing the output features (h_t) from the last layer 
        #of the RNN, for each t. 
        
        #h_n (num_layers * num_directions, batch, hidden_size): 
        #tensor containing the hidden state for t=seq_len
        context_out, context_hn = self.encoder(contexts)
        
        response_out, response_hn = self.encoder(responses)
                
        scores_list = []
        
        y_preds = None
        
       # context_hn_tensor = context_hn.data
        # iter = context_hn_tensor[0].shape[0]
        
        #iter = context_out.shape[0] #to iterate over 999 examples

        for e in range(999): 
            context_h = context_out[e][-1].view(1, self.encoder.hidden_size)
            response_h = response_out[e][-1].view(self.encoder.hidden_size,1)
            #gives vectors of hidden_size for each example
            
            dot_var = torch.mm(torch.mm(context_h, self.M), response_h)[0][0]#gives 1x1 variable floattensor
    
            dot_tensor = dot_var.data #gives 1x1 floattensor
            dot_tensor.cuda()
            
            score = torch.sigmoid(dot_tensor)
            scores_list.append(score)
            
        y_preds_tensor = torch.stack(scores_list).cuda()  
        y_preds = autograd.Variable(y_preds_tensor).cuda()
        
        return y_preds #to be used in training to compare with label
     
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
#loss_func = torch.nn.BCEWithLogitsLoss() #input: bilinear_output (batch_size x 1)

learning_rate = 0.001
epochs = 100

optimizer = optim.Adam(dual_encoder.parameters(),
                       lr = learning_rate)
#%%

word_to_id = create_word_to_id('/data/train_shuffled_onethousand.csv')

epochs = 10
for epoch in range(epochs):
       
    batch = load_batch('/data/train_shuffled_onethousand.csv', 128, epoch)
    
    example = None 
    
    for i, example in batch.iterrows():        
        context_tensor, response_tensor, label_tensor = make_tensors(example)
     
        context = autograd.Variable(context_tensor, requires_grad=False)
       
        response = autograd.Variable(response_tensor, requires_grad=False)
     
        y = autograd.Variable(label_tensor, requires_grad=True)

        y_preds = dual_encoder(context, response)
        
        loss = loss_func(y_preds, y)
        
        if epoch % 10 == 0:
            print("Epoch: ", epoch, ", Loss: ", loss.data[0])
        
    #evaluation metrics...

        dual_encoder.zero_grad()
        loss.backward()
    #torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
    
        optimizer.step()
#%%

torch.save(dual_encoder.state_dict(), 'SAVED_MODEL.pt')
    
    

