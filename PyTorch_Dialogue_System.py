#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:04:32 2018

@author: janinanu
"""

#%%
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn 
import datetime
import operator
#%%
def create_dataframe(csvfile):
    dataframe = pd.read_csv(csvfile)
    return dataframe

def shuffle_dataframe(dataframe):
    dataframe.reindex(np.random.permutation(dataframe.index))

def create_vocab(dataframe):
    vocab = []
    word_freq = {}
    
    for index, row in dataframe.iterrows():
        
        context_cell = row["Context"]
        response_cell = row["Utterance"]
        
        train_words = str(context_cell).split() + str(response_cell).split()
        
        for word in train_words:
          
            if word.lower() not in vocab:
                vocab.append(word.lower())         
                       
            if word.lower() not in word_freq:
                word_freq[word.lower()] = 1
            else:
                word_freq[word] += 1
    
    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = ["<UNK>"] + [pair[0] for pair in word_freq_sorted]
    
    return vocab


def create_word_to_id(vocab):     
    enumerate_list = [(id, word) for id, word in enumerate(vocab)]
        
    word_to_id = {pair[1]: pair[0] for pair in enumerate_list}
    
    return word_to_id


def create_id_to_vec(word_to_id, glovefile): 
    lines = open(glovefile, 'r').readlines()
    id_to_vec = {}
    vector = None
    
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
            
    for word, id in word_to_id.items(): 
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
            
    embedding_dim = id_to_vec[0].shape[0]
    
    return id_to_vec, embedding_dim


def load_ids_and_labels(row, word_to_id):
    context_ids = []
    response_ids = []

    context_cell = row['Context']
    response_cell = row['Utterance']
    label_cell = row['Label']

    max_context_len = 160
    
    context_words = context_cell.split()
    if len(context_words) > max_context_len:
        context_words = context_words[:max_context_len]
    for word in context_words:
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else: 
            context_ids.append(0) #UNK
    
    response_words = response_cell.split()
    for word in response_words:
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else: 
            response_ids.append(0)
    
    label = np.array(label_cell).astype(np.float32)

    return context_ids, response_ids, label

#%%
class Encoder(nn.Module):

    def __init__(self, 
             input_size, 
             hidden_size, 
             vocab_size, 
             num_layers, 
             num_directions, 
             dropout, 
             bidirectional,
             rnn_type,
             p_dropout): 
    
             super(Encoder, self).__init__()
             
             self.input_size = input_size
             self.hidden_size = hidden_size
             self.vocab_size = vocab_size
             self.num_layers = num_layers
             self.num_directions = num_directions
             self.dropout = dropout 
             self.bidirectional = False
             self.rnn_type = 'lstm'
             self.p_dropout = p_dropout
       
             self.embedding = nn.Embedding(self.vocab_size, self.input_size)
             self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout = dropout, bidirectional=False)
             self.dropout_layer = nn.Dropout(self.p_dropout) 

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
        outputs, hiddens_tuple = self.lstm(embeddings)
        
        last_hidden = hiddens_tuple[0] #access first tuple element to get last hidden state, dimensions: (num_layers * num_directions x batch_size x hidden_size)
        last_hidden = last_hidden[-1] #access last lstm layer, dimensions: (batch_size x hidden_size)
        last_hidden = self.dropout_layer(last_hidden) #dimensions: (batch_size x hidden_size)

        return last_hidden

    
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor) #dimensions: (batch_size x hidden_size)
        response_last_hidden = self.encoder(response_tensor) #dimensions: (batch_size x hidden_size)
        
        #context = context_last_hidden.mm(self.M).cuda()
        context = context_last_hidden.mm(self.M) #dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size) #dimensions: (batch_size x 1 x hidden_size)
        
        response = response_last_hidden.view(-1, self.hidden_size, 1) #dimensions: (batch_size x hidden_size x 1)
        
        #score = torch.bmm(context, response).view(-1, 1).cuda()
        score = torch.bmm(context, response).view(-1, 1) #dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        return score
#%%  
def creating_variables(num_training_examples, num_validation_examples, embedding_dim):

    print(str(datetime.datetime.now()).split('.')[0], "Creating variables for training and validation...")

    training_dataframe = create_dataframe('/home/janinanu/Local_Chatbot/FinalData/training_%d.csv' %num_training_examples)
    vocab = create_vocab(training_dataframe)
    word_to_id = create_word_to_id(vocab)
    id_to_vec, emb_dim = create_id_to_vec(word_to_id, '/home/janinanu/Local_Chatbot/FinalData/glove.6B.%dd.txt' %embedding_dim)

    validation_dataframe = create_dataframe('/home/janinanu/Local_Chatbot/FinalData/validation_%d.csv' %num_validation_examples)

    print(str(datetime.datetime.now()).split('.')[0], "Variables created.\n")
    
    return training_dataframe, vocab, word_to_id, id_to_vec, emb_dim, validation_dataframe
#%%    
def creating_model(hidden_size, p_dropout):

    print(str(datetime.datetime.now()).split('.')[0], "Calling model...")

    encoder = Encoder(
            input_size = emb_dim,
            hidden_size = hidden_size,
            vocab_size = len(vocab),
            num_layers = 1,
            dropout = 0.0,
            num_directions = 1,
            bidirectional = False,
            rnn_type = 'lstm',
            p_dropout = p_dropout)
    #encoder.cuda()

    dual_encoder = DualEncoder(encoder)
    #dual_encoder.cuda()

    print(str(datetime.datetime.now()).split('.')[0], "Model created.\n")
    print(dual_encoder)
    
    return encoder, dual_encoder
#%%
def increase_count(correct_count, score, label):
    if ((score.data[0][0] >= 0.5) and (label.data[0][0] == 1.0)) or ((score.data[0][0] < 0.5) and (label.data[0][0]  == 0.0)):
       correct_count +=1  
   
    return correct_count

def get_accuracy(correct_count, dataframe):
    accuracy = correct_count/(len(dataframe))
        
    return accuracy
#%%
def train_model(learning_rate, epochs): 
    print(str(datetime.datetime.now()).split('.')[0], "Starting training and validation...\n")
    print("====================Data and Hyperparameter Overview====================\n")
    print("Number of training examples: %d, Number of validation examples: %d" %(len(training_dataframe), len(validation_dataframe)))
    print("Learning rate: %.5f, Embedding Dimension: %d, Hidden Size: %d, Dropout: %.2f\n" %(learning_rate, emb_dim, encoder.hidden_size, encoder.p_dropout))
    print("================================Results...==============================\n")

    optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = learning_rate)
       
    loss_func = torch.nn.BCEWithLogitsLoss()
    #loss_func.cuda()
     
    best_validation_accuracy = 0.0
     
    for epoch in range(epochs): 
                     
            shuffle_dataframe(training_dataframe)
                        
            sum_loss_training = 0.0
            
            training_correct_count = 0
            
            dual_encoder.train()

            for index, row in training_dataframe.iterrows():            
            
                context_ids, response_ids, label = load_ids_and_labels(row, word_to_id)
                
                context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False) #.cuda()
                
                response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad = False) #.cuda()
                                
                label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1))), requires_grad = False) #.cuda()
                             
                score = dual_encoder(context, response)
        
                loss = loss_func(score, label)
                
                sum_loss_training += loss.data[0]
                
                loss.backward()
        
                optimizer.step()
               
                optimizer.zero_grad()
                
                training_correct_count = increase_count(training_correct_count, score, label)
                                                    
            training_accuracy = get_accuracy(training_correct_count, training_dataframe)
                
            shuffle_dataframe(validation_dataframe)
            
            validation_correct_count = 0

            sum_loss_validation = 0.0

            dual_encoder.eval()

            for index, row in validation_dataframe.iterrows():
                
                context_ids, response_ids, label = load_ids_and_labels(row, word_to_id)
                
                context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1)) #.cuda()
                
                response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1)) #.cuda()
                                
                label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1)))) #.cuda()
                
                score = dual_encoder(context, response)
                
                loss = loss_func(score, label)
                
                sum_loss_validation += loss.data[0]
                
                validation_correct_count = increase_count(validation_correct_count, score, label)
                    
            validation_accuracy = get_accuracy(validation_correct_count, validation_dataframe)
                        
            print(str(datetime.datetime.now()).split('.')[0], 
                  "Epoch: %d/%d" %(epoch,epochs),  
                  "TrainLoss: %.3f" %(sum_loss_training/len(training_dataframe)), 
                  "TrainAccuracy: %.3f" %(training_accuracy), 
                  "ValLoss: %.3f" %(sum_loss_validation/len(validation_dataframe)), 
                  "ValAccuracy: %.3f" %(validation_accuracy))
            
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(dual_encoder.state_dict(), 'saved_model_%d_examples.pt' %(len(training_dataframe)))
                print("New best found and saved.")
                
    print(str(datetime.datetime.now()).split('.')[0], "Training and validation epochs finished.")
#%%
training_dataframe, vocab, word_to_id, id_to_vec, emb_dim, validation_dataframe = creating_variables(num_training_examples = 10000, 
                                                                                                     embedding_dim = 50, 
                                                                                                     num_validation_examples = 10)
#%%
encoder, dual_encoder = creating_model(hidden_size = 50, 
                                       p_dropout = 0.75)
#%%
#train_model(learning_rate = 0.0001, 
            #epochs = 100)
#%%
dual_encoder.load_state_dict(torch.load('saved_model_10000_examples.pt'))
dual_encoder.eval()
#%%
test_dataframe_same_structure = pd.read_csv('/home/janinanu/Local_Chatbot/FinalData/testing_same_structure_10.csv')
#%%
def testing_same_structure():
    
    test_correct_count = 0

    for index, row in test_dataframe_same_structure.iterrows():

        context_ids, response_ids, label = load_ids_and_labels(row, word_to_id)

        context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1)) #.cuda()

        response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1)) #.cuda()

        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1)))) #.cuda()

        score = dual_encoder(context, response)

        test_correct_count = increase_count(test_correct_count, score, label)

    test_accuracy = get_accuracy(test_correct_count, test_dataframe_same_structure)
    
    return test_accuracy
#%%
test_accuracy = testing_same_structure()
print("Test accuracy for %d training examples and %d test examples: %.2f" %(len(training_dataframe),len(test_dataframe_same_structure),test_accuracy))
#%%
test_dataframe_different_structure = pd.read_csv('/home/janinanu/Local_Chatbot/FinalData/testing_different_structure_10.csv')
#%%
def load_ids(test_dataframe_different_structure, word_to_id):
    
    print(str(datetime.datetime.now()).split('.')[0], "Loading test IDs...")

    max_context_len = 160
    
    ids_per_example_and_candidate = {}
    
    for i, example in test_dataframe_different_structure.iterrows():
        
        ids_per_candidate = {}
      
        for column_name, cell in  example.iteritems():
            
                id_list = []
            
                words = str(cell).split()
                if len(words) > max_context_len:
                    words = words[:max_context_len]
    
                for word in words:
                    if word in word_to_id:
                        id_list.append(word_to_id[word])
                    else: 
                        id_list.append(0) #UNK  
                    
                ids_per_candidate[column_name] = id_list
    
        ids_per_example_and_candidate[i] = ids_per_candidate
    
    print(str(datetime.datetime.now()).split('.')[0], "Test IDs loaded.")
    
    return ids_per_example_and_candidate
#%%
ids_per_example_and_candidate = load_ids(test_dataframe_different_structure, word_to_id)
#%%

def load_scores(): 
    print(str(datetime.datetime.now()).split('.')[0], "Computing test scores...")
    
    scores_per_example_and_candidate = {}
                 
    for example, utterance_ids_dict in sorted(ids_per_example_and_candidate.items()): 
        
        score_per_candidate = {}

        for utterance_name, ids_list in sorted(utterance_ids_dict.items()):
        
            context = autograd.Variable(torch.LongTensor(utterance_ids_dict['Context']).view(-1,1))#.cuda()
            
            if utterance_name != 'Context':

                candidate_response = autograd.Variable(torch.LongTensor(utterance_ids_dict[utterance_name]).view(-1, 1))#.cuda()
                        
                score = torch.sigmoid(dual_encoder(context, candidate_response))
                
                score_per_candidate["Score with " + utterance_name] = score.data[0][0]
    
        scores_per_example_and_candidate[example] = score_per_candidate

    print(str(datetime.datetime.now()).split('.')[0], "Test scores computed.")
    
    return scores_per_example_and_candidate
#%%
scores_per_example_and_candidate = load_scores()
#%%
def get_recall_at_k(k):
    count_true_hits = 0
    
    for example, score_per_candidate_dict in sorted(scores_per_example_and_candidate.items()): 
    
        top_k = dict(sorted(score_per_candidate_dict.items(), key=operator.itemgetter(1), reverse=True)[:k])
        
        if 'Score with Ground Truth Utterance' in top_k:
            count_true_hits += 1
    
    number_of_examples = len(scores_per_example_and_candidate)
    
    recall_at_k = count_true_hits/number_of_examples
    
    return recall_at_k
#%%
print("recall_at_5 =",get_recall_at_k(k = 5)) #Baseline expectation: 5/10 = 0.5
print("recall_at_2 =",get_recall_at_k(k = 2)) #Baseline expectation: 2/10 = 0.2
print("recall_at_1 =",get_recall_at_k(k = 1)) #Baseline expectation: 1/10 = 0.1