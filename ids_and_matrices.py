#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:25:04 2017

@author: janinanu
"""

#shuffle and subsample training data, read in training examples, batch function, same for validation
import numpy as np
import torch
import torch.utils.data
import torch.autograd as autograd
import pandas as pd
from vocab_and_dicts import word_to_id


#%%

def load_ids_and_labels():
    file = 'train_shuffled_onethousand.csv'
    
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
      
    context_matrix = torch.LongTensor(context_id_list)
    context_matrix = autograd.Variable(context_matrix)
    
    for list in response_id_list:
        if (len(list) < max_len):
           list += [0] * (max_len - len(list))
    
    response_matrix = torch.LongTensor(response_id_list)
    response_matrix = autograd.Variable(response_matrix)
    
    
    labels_tor = torch.from_numpy(labels_array)
    labels_var = autograd.Variable(torch.LongTensor(labels_tor))

    return context_matrix, response_matrix, labels_var
#%%
    
#context_matrix, response_matrix, y = make_matrices()




