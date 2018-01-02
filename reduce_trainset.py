#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:21:38 2018

@author: janinanu
"""

import pandas as pd
import random

#%% shuffle and subsample training data, create file:

train_orig = "train.csv"

num_lines = sum(1 for line in open(train_orig)) #count lines

size = int(num_lines/1000) #0.1% = 1,000 sample size

skip_idx = random.sample(range(1, num_lines), num_lines - size) #indices of randomly sampled rows to skip (99.9% to skip)

shuffle = pd.read_csv(train_orig, skiprows=skip_idx) #reads in random sample of 0.1%

train_shuffled_writer = shuffle.to_csv("train_shuffled_onethousand.csv")

#%% read in shuffled and reduced training examples from new file:

train_shuffled_file = 'train_shuffled_onethousand.csv'

train_shuffled_reader = pd.read_csv(train_shuffled_file, header=None)

rows = train_shuffled_reader

#%%
##%%extract training features and labels
#context_train_column = rows.iloc[1:, 1]
##print(context_train)
#
#response_train_column = rows.iloc[1:, 2]
##print(response_train)
#
#label_train_column = rows.iloc[1:, 3]
#
##%% create list of id tensors for context, sort in order to prepare for pad_sequence method 
#context_id_list = []
#
#for _, line in context_train_column.iteritems():
#
#    context_ids = [word_to_id[word+"\n"] for word in line.split()]
#    tensor = torch.LongTensor(context_ids)
#    context_id_list.append(tensor)
#    
#context_id_list.sort(key=len, reverse=True)
#    
#id_length_list = [len(var) for var in context_id_list]
#
##%% pad index vectors and turn them into matrix
#context_padded_matrix = preprocess.pad_sequence(context_id_list)
#
##%% find max_length to check dimensions
#max_length = 0
#
#for _, line in context_train_column.iteritems():
#    act_length = len(line.split())
#    if act_length > max_length:
#        max_length = act_length
#%% reshape padded_matrix to get dimensions (number of examples x max_length)
#padded_matrix_reshape = padded_matrix.permute(1,0)
