#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:18:46 2017

@author: janinanu
"""

# vocab, word to id, id to vec, padding


import csv
import numpy as np
import torch
import torch.utils.data
#import torch.nn.utils.rnn 


#%% CREATE VOCABULARY

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

vocab_len = len(vocab)
#print(len(vocab))

#%% METHOD: MAKE WORD_TO_ID DICTIONARY FROM VOCAB FILE

def make_word_to_id(file):
    #import data
    lines = open(file, 'r').readlines()
    
    for line in lines:
        line.strip("\n")

    word_to_id = {word: id for id, word in enumerate(lines)}
   #word_to_id = {word: id for id, word in enumerate(lines.strip("\n"))}
 
   
#    for key, value in word_to_id.iteritems():
#        key.strip("\n")
    
    return word_to_id
    
    #word_to_id_sorted = sorted(word_to_id.items(), key=lambda item: item[1])

#%% METHOD: MAKE ID_TO_VEC DICTIONARY FROM WORD_TO_ID AND GLOVE

def make_id_to_vec(glovefile): 
    word_to_id = make_word_to_id('my_vocab.txt')
    
    lines = open(glovefile, 'r').readlines()
    
    word_vecs = {}
    
    vector = None
        
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if (word+"\n") in word_to_id:
            word_vecs[word_to_id[word+"\n"]] = torch.FloatTensor(torch.from_numpy(vector))
     
        for word in word_to_id:
            if word_to_id[word] not in word_vecs:
               v = np.zeros(*vector.shape, dtype='float32')
               v[:] = np.random.randn(*v.shape)
               word_vecs[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
     
    vec = np.zeros(*vector.shape, dtype='float32')
    #vec = np.zeros(*vector.shape).astype('float32') #32
    word_vecs[0] = torch.FloatTensor(torch.from_numpy(vec))
        
    vec_length = vector.shape[0]
    
    return word_vecs, vec_length
#%%
    
file = 'my_vocab.txt'
word_to_id = make_word_to_id(file)
id_to_vec, embedding_dim = make_id_to_vec('glove.6B.100d.txt')




