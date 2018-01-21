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
import torch
import torch.autograd as autograd
import pandas as pd
from torch.nn import init
import torch.nn.utils.rnn 
import torch.utils.data as data_utils

import torch.optim as optim

#%%

def create_dataframe(csvfile):
    #dataframe = csv.DictReader(open(csvfile))
    dataframe = pd.read_csv(csvfile)
    return dataframe


#%% CREATE VOCABULARY
def create_vocab(dataframe):
    vocab = []
    
    word_freq = {}

    #rows_df = pd.read_csv(csvfile)
    
    for index, row in dataframe.iterrows():
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
    vocab = ["<PAD>"] + [pair[0] for pair in word_freq_sorted]

    
#    outfile = open('my_vocab.txt', 'w')  
#
#    for word in vocab:
#        outfile.write(word + "\n")
#    outfile.close()
#    
    return vocab

#%%
    
def create_word_to_id(vocab):
    
    #vocab = create_vocab(csvfile)
        
    enumerate_list = [(id, word) for id, word in enumerate(vocab)]
        
    word_to_id = {pair[1]: pair[0] for pair in enumerate_list}
    
    return word_to_id
#%%

def create_id_to_vec(word_to_id, glovefile): 
    #word_to_id = create_word_to_id(csvfile)
    
    lines = open(glovefile, 'r').readlines()
    
    id_to_vec = {}
    
    vector = None
    
    count_glove = 0
    
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector)) #glove
            count_glove += 1
            
    for word, id in word_to_id.items(): #!!!
        
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            #v = np.random.randn(*vector.shape).astype('float32') #32
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v)) #not in glove, random
     
    vec = np.zeros(*vector.shape, dtype='float32')

    id_to_vec[0] = torch.FloatTensor(torch.from_numpy(vec))
        
    embedding_dim = id_to_vec[0].shape[0]
    
    #id_to_vec[word_to_id["<PAD>"]] = torch.FloatTensor(torch.from_numpy(vec)) #pads, zeros
        
    #embedding_dim = vector.shape
    
    return id_to_vec, embedding_dim, count_glove


#%%

def load_ids_and_labels(dataframe, word_to_id):
    
    #word_to_id = create_word_to_id(csvfile)

    #rows = pd.read_csv(csvfile)
    
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
    #context_id_tensor = torch.LongTensor(context_id_list)
    
    for cell in response_column:
        response_ids = [word_to_id[word] for word in cell.split()]
        response_id_list.append(response_ids)
    #response_id_tensor = torch.LongTensor(response_id_list)
    
    #label_array = np.array(label_column).astype(np.float32)
    
#    label_array = np.reshape(label_array, (len(label_column), 1))
#    label_array = label_array.astype(np.float32)
#    label_tensor = torch.FloatTensor(torch.from_numpy(label_array))
#    
    

    return context_id_list, response_id_list, label_column
#%%
#context_id_list, response_id_list, label_array = load_ids_and_labels('train_shuffled_hundred.csv')
#for i in range(len(label_array)):
#    label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label_array[i][0]).reshape(1,1))), requires_grad = False)
#    print(label)

#%%
    

#def get_lengths(csvfile):
#
#    context_id_list, response_id_list, label_column = load_ids_and_labels(csvfile)
#    
#    context_len_list = []
#    response_len_list = []
#    
#    for list in context_id_list:
#        context_len_list.append(len(list))
#        
#    for list in response_id_list:
#        response_len_list.append(len(list))
#        
#    return context_len_list, response_len_list

#%% 
    
def make_tensors(context_id_list, response_id_list, label_column):

#    max_len_context = len(context_id_list[0])
#    max_len_response = len(response_id_list[0])

    max_context_len = 160
    max_response_len = max(len(r) for r in response_id_list)
    max_len = max(max_context_len, max_response_len) #shorten sequence length??? 

    #max_len = min(300, max(max_len_context, max_len_response))
    
    
    for list in context_id_list:
        if (len(list) < max_len):
            list += [0] * (max_len - len(list))
      
    context_matrix = torch.LongTensor(context_id_list)
    
    for list in response_id_list:
        if (len(list) < max_len):
           list += [0] * (max_len - len(list))
    
    response_matrix = torch.LongTensor(response_id_list)
    
    
    label_array = np.array(label_column)
    np.reshape(label_array, (len(label_column), 1))
    label_float = label_array.astype('float32')
    label_vector = torch.FloatTensor(label_float).view(len(label_column), 1)
    
    return context_matrix, response_matrix, label_vector

#%%
    
#context_matrix, response_matrix, label_vector = make_tensors('train_shuffled_hundred.csv')
#%%
    
#def get_emb_dim(glovefile):
#    
#    lines = open(glovefile, 'r').readlines()        
#    vector = np.array(lines[0].split()[1:], dtype='float32')
#    embedding_dim = len(vector)
#    
#    return embedding_dim
#


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
             
             self.embedding = nn.Embedding(self.vocab_size, self.input_size)
             self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout = dropout, bidirectional=False)
    
             self.init_weights()
             
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        #init.uniform(embedding_weights, a = -0.25, b= 0.25)
            
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
    
        #del self.embedding.weight
    
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
        
        #self.embedding.weight.data.copy_(embedding_weights)
    
        #self.embedding.weight.requires_grad = True

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        #outputs, hiddens = self.lstm(embeddings)

        outputs, hiddens = self.lstm(embeddings.view(inputs.shape[1],inputs.shape[0], self.input_size))
        return outputs, hiddens
    #inputs.shape[1],inputs.shape[0], input_size




#%%
        
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(1, self.hidden_size, self.hidden_size)     
        init.xavier_normal(M)#?
        self.M = nn.Parameter(M, requires_grad = True)

        
         
    def forward(self, context_batch_matrix, response_batch_matrix):
        #Outputs: output, (h_n, c_n)
        
        #output (seq_len, batch, hidden_size * num_directions): 
        #tensor containing the output features (h_t) from the last layer 
        #of the RNN, for each t. 
        
        #h_n (num_layers * num_directions, batch, hidden_size): 
        #tensor containing the hidden state for t=seq_len
        
        #context_matrix = context_batch_matrix.view(context_batch_matrix.shape[1],context_batch_matrix.shape[0], 0)
        
        context_out, context_hc_tuple = self.encoder.forward(context_batch_matrix)
        
        
        #response_matrix = response_batch_matrix.view(response_batch_matrix.shape[1],response_batch_matrix.shape[0], 0)

        response_out, response_hc_tuple = self.encoder.forward(response_batch_matrix)
        
        
        context_h = context_hc_tuple[0]
        
        response_h = response_hc_tuple[0]
        
        
        context_h_layer = context_h[0] #batch x hidden_size
        
        response_h_layer = response_h[0] #batch x hidden_size
        
        context = context_h_layer.view(-1, 1, self.hidden_size) #batch x 1 x hidden_size
        
        response = response_h_layer.view(-1, self.hidden_size, 1)#batch x hidden_size x 1
        
        scores = torch.bmm(torch.bmm(context, self.M), response).view(-1, 1) # batch x 1 x 1 --> batch x 1
        
        #scores_M.requires_grad =True
        
    
        
        #scores = autograd.Variable(scores.view(-1, 1), requires_grad =True) # batch x 1 
        
        
        #scores = torch.sigmoid(scores_M) # batch x 1 
    
        #y_preds = autograd.Variable(score_sigmoid, requires_grad = True)
    
        return scores
#%% TRAINING

#torch.backends.cudnn.enabled = False
#%%
dataframe = create_dataframe('train_shuffled_hundred.csv')
vocab = create_vocab(dataframe)
word_to_id = create_word_to_id(vocab)
id_to_vec, emb_dim, count_glove = create_id_to_vec(word_to_id, 'glove.6B.100d.txt')
context_id_list, response_id_list, label_column = load_ids_and_labels(dataframe, word_to_id)

vocab_len = len(vocab)


#%%
encoder_model = Encoder(
        input_size = emb_dim,
        hidden_size = 200,
        vocab_size = vocab_len)

dual_encoder = DualEncoder(encoder_model)

#%%

#count = 0
#for parameter in parameters:
#    count += 1
#    print(parameter.shape)
#print(count)    

#%%
loss_func = torch.nn.BCEWithLogitsLoss()


#%%
#loss_func = torch.nn.functional.binary_cross_entropy() 
optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = 0.0001)

#lr = 0.0005: overshoots at some point
#lr = 0.0001: gets stuck in local minimum
#%%

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)



#def exp_lr_scheduler(optimizer, epoch, lr_decay=0.05, lr_decay_epoch=7):
#    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
#    if epoch % lr_decay_epoch:
#        return optimizer
#    
#    for param_group in optimizer.param_groups:
#        param_group['lr'] *= lr_decay
#    return optimizer


#optimizer = torch.optim.SGD(parameters, lr = learning_rate)
#%%


context_matrix, response_matrix, label_vector = make_tensors(context_id_list, response_id_list, label_column)

concat_matrix = torch.cat((context_matrix, response_matrix), 1)

data_tensor = concat_matrix
target_tensor = label_vector

batch_size = 1

train = data_utils.TensorDataset(data_tensor, target_tensor)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

num_train_examples = len(label_vector)
num_batches = num_train_examples//batch_size

            
for epoch in range(200): 
        
    loss_acc = 0.0

    #scheduler.step()
    
    for batch_matrix, batch_labels in train_loader: #for each batch
        
        split_at = batch_matrix.shape[1]//2
        
        context_matrix = batch_matrix[:, :split_at]
        response_matrix = batch_matrix[:, split_at:]
        
        context = autograd.Variable(context_matrix.long(), requires_grad = False)
        
        response = autograd.Variable(response_matrix.long(), requires_grad = False)
        
        labels = autograd.Variable(batch_labels.float(), requires_grad = False)
        
        scores = dual_encoder(context, response)
        
#        if (scores.requires_grad == True):
#            print("requires_grad")
#            print(scores.type)
#            print(scores)

        loss = loss_func(scores, labels) #average loss per batch
        
        loss_acc += loss.data[0] #sum of average losses over all batches per epoch
        
        
        loss.backward()

        optimizer.step()
       
        optimizer.zero_grad()
        
        #torch.nn.utils.clip_grad_norm(dual_encoder.parameters(), 10)
        
    print("Epoch: ", epoch, ", Loss: ", loss_acc/num_batches)
    
   
    
#    for name, param in dual_encoder.named_parameters():
#          if (param.requires_grad == True):
#              print(name, param.shape)

       #if param.grad is None:
           #print(name, param.shape)
        
        #if ((epoch == 0) or (epoch == 1)) and (name == "encoder.embedding.weight"):
          # print(name, param.grad) 
    #param.grad: None
    #param.grad.data: NoneType' object has no attribute 'data' 
    
    

    

#print(count)
#%%

torch.save(dual_encoder.state_dict(), 'SAVED_MODEL.pt')
    
#%%


