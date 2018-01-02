#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:25:04 2017

@author: janinanu
"""

#shuffle and subsample training data, read in training examples, batch function, same for validation
import numpy as np
import csv
import torch
import torch.utils.data
import torch.autograd as autograd
import preprocess

#%% create sorted training vocab as list and file
#change into preprocess method when adding test vocab 
vocab = []

word_freq = {}

infile = csv.DictReader(open('train_shuffled_onethousand.csv'))

for row in infile:
    #returns dict: {context: "...", "response": "...", label: "."}
    
    context_cell = row["Context"]
    response_cell = row["Utterance"]
    label_cell = row["Label"]
    
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

outfile = open('my_vocab.txt', 'w')  

for word in vocab:
    outfile.write(word + "\n")
outfile.close()

vocab_size = len(vocab)
#print(len(vocab))

#%% create two dictionaries to map from word to id and from id to embedding vector
file = 'my_vocab.txt'
word_to_id = preprocess.make_word_to_id(file)
id_to_vec, vec_length = preprocess.make_id_to_vec('glove.6B.100d.txt', word_to_id)

#%% create list of id tensors for context, sort in order to prepare for pad_sequence method 
#first pad, then LongTensor

def load_batch(trainset, batch_size):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        
    context_id_list, response_id_list = []
    labels_array = np.empty(batch_size, 1)

    for i, row in enumerate(trainloader):
            context, response, label = row
            
            context_ids = [word_to_id[word+"\n"] for word in context.split()]
            #context_tensor = torch.LongTensor(context_ids)
            context_id_list.append(context_ids)
            
            response_ids = [word_to_id[word+"\n"] for word in response.split()]
            #response_tensor = torch.LongTensor(response_ids)
            response_id_list.append(response_ids)
            
            labels_array[i] = int(label)
         
    labels_vec = autograd.Variable(torch.from_numpy(labels_array))
    
    return context_id_list, response_id_list, labels_vec
#%%
def make_matrices(trainset):
    import data
    context_id_list, response_id_list, labels_vec = data.load_batch(trainset)
    
    context_id_list.sort(key=len, reverse=True)
    response_id_list.sort(key=len, reverse=True)    
    
    max_len_context = max(len(c) for c in context_id_list)
    max_len_response = max(len(r) for r in response_id_list)
    max_len = max(max_len_context, max_len_response)
        
    for c in context_id_list:
        if (len(c) < max_len):
            c += [0] * (max_len - len(c))
        c = autograd.Variable(torch.LongTensor(c))
    
    for r in response_id_list:
        if (len(r) < max_len):
            r += [0] * (max_len - len(r))
        r = autograd.Variable(torch.LongTensor(r))
        
    context_matrix = preprocess.pad_sequence(context_id_list, batch_first = True)
    response_matrix = preprocess.pad_sequence(response_id_list, batch_first = True)
    
    return context_matrix, response_matrix, labels_vec  


  
