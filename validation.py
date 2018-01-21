#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:44:20 2018

@author: janinanu
"""


import torch
import NOBATCHES
import pandas as pd
import numpy as np
import torch.autograd as autograd
from collections import Counter
#%%

encoder_model = NOBATCHES.Encoder(
        input_size = NOBATCHES.emb_dim,
        hidden_size = 200,
        vocab_size = NOBATCHES.vocab_len)

dual_encoder = NOBATCHES.DualEncoder(encoder_model)

dual_encoder.load_state_dict(torch.load('SAVED_MODEL_1000.pt'))

dual_encoder.eval()
dual_encoder.training = False

#%%
def check_params(param_name):
    for name, param in dual_encoder.named_parameters():
        if name == param_name: 
            print(name, param.data)            
        
#%%

def load_eval_data(csvfile):
    
    dataframe_eval = pd.read_csv(csvfile)

    dataframe_eval_reduced = dataframe_eval[:100] 
    
    return dataframe_eval_reduced

#%%
dataframe_eval_reduced = load_eval_data('valid.csv')
#%% 
    
def load_ids_eval(dataframe_eval_reduced, word_to_id):
    
    rows_shuffled = dataframe_eval_reduced.reindex(np.random.permutation(dataframe_eval_reduced.index))

    max_context_len = 160
    
    id_lists_eval_dict = {}
    
    count_unk = 0
    
    count_known = 0
    
    count_words = 0
    
    for column_name, column in rows_shuffled.iteritems():
        id_list_eval = []

        for cell in column:
            ids_eval = []
            words_eval = cell.split()
            if len(words_eval) > max_context_len:
                words_eval = words_eval[:max_context_len]
    
            for word in words_eval:
                count_words += 1
                if word in word_to_id:
                    ids_eval.append(word_to_id[word])
                    count_known += 1
                else: 
                    ids_eval.append(0) #UNK  
                    count_unk += 1
    
            id_list_eval.append(ids_eval)
    
        id_lists_eval_dict[column_name] = id_list_eval
    
    return id_lists_eval_dict, count_unk, count_known, count_words

#%%

id_list_eval_dict, count_unk, count_known, count_words = load_ids_eval(dataframe_eval_reduced, NOBATCHES.word_to_id)


#%%
    
def validate_model_1in10(): 
    
    dual_encoder.eval()
    dual_encoder.training = False
    
    id_list_eval_dict, _, _, _ = load_ids_eval(dataframe_eval_reduced, NOBATCHES.word_to_id)
    
    scores_per_example_dict = {}
        
    for example in range(len(id_list_eval_dict['Context'])): #loop over examples
        
        score_per_candidate_dict = {}
 
        for column_name, id_list in sorted(id_list_eval_dict.items()): #sorted order: Context, Distractors 1-8, Ground Truth Utterance

            if column_name != 'Context':
        
                context = autograd.Variable(torch.LongTensor(id_list_eval_dict['Context'][example]).view(len(id_list_eval_dict['Context'][example]),1), requires_grad = False)
                
                candidate_response = autograd.Variable(torch.LongTensor(id_list_eval_dict[column_name][example]).view(len(id_list_eval_dict[column_name][example]), 1), requires_grad = False)
                        
                score = torch.sigmoid(dual_encoder(context, candidate_response))
                
                score_per_candidate_dict["Score with " + column_name] = score.data[0][0]
        
        scores_per_example_dict[example] = score_per_candidate_dict
    
    return scores_per_example_dict


#%%
    
def validate_model_1in2(): 
    
    dual_encoder.eval()
    dual_encoder.training = False
    
    id_list_eval_dict, _, _, _ = load_ids_eval(dataframe_eval_reduced, NOBATCHES.word_to_id)
    
    scores_per_example_dict = {}
        
    for example in range(len(id_list_eval_dict['Context'])): #loop over examples
        
        score_true_response_dict = {}
 
        context = autograd.Variable(torch.LongTensor(id_list_eval_dict['Context'][example]).view(len(id_list_eval_dict['Context'][example]),1), requires_grad = False)
        
        true_response = autograd.Variable(torch.LongTensor(id_list_eval_dict['Ground Truth Utterance'][example]).view(len(id_list_eval_dict['Ground Truth Utterance'][example]), 1), requires_grad = False)
                
        score = torch.sigmoid(dual_encoder(context, true_response))
        
        score_true_response_dict["Score with " + 'Ground Truth Utterance'] = score.data[0][0]
        
        scores_per_example_dict[example] = score_true_response_dict
    
    return scores_per_example_dict 


#%%

scores_dict_1in10 = validate_model_1in10()

scores_dict_1in2 = validate_model_1in2()

#%%
#for 1 in 10 R@k: look at all 10 candidate responses, k= 1,2,5

def one_in_ten_recall_at_k(k): #next utterance classification in %
    # k can be 1, 2 or 5 for R@1, R@2, R@5
    count_true_hits = 0
    
    for example, example_dict in sorted(scores_dict_1in10.items()):  #Context, Distractors, Ground Truth Utterance
    
        top_k = dict(Counter(example_dict).most_common(k))
        
        if 'Score with Ground Truth Utterance' in top_k:
            count_true_hits += 1
    
    number_of_examples = len(scores_dict_1in10)
    
    one_in_ten_recall_at_k = count_true_hits/number_of_examples
    
    return one_in_ten_recall_at_k
#%%
    
recall_at_5 = one_in_ten_recall_at_k(k = 5)      
recall_at_2 = one_in_ten_recall_at_k(k = 2)      
recall_at_1 = one_in_ten_recall_at_k(k = 1)      


             
                
        







    
    
    
    
    
    
    
    
    
    