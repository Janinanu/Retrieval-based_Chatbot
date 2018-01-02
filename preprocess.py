#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:18:46 2017

@author: janinanu
"""

# vocab, word to id, id to vec, padding



import numpy as np
import torch
import torch.utils.data
import torch.autograd as autograd
import torch.nn.utils.rnn 
#%%create word_to_id dictionary from vocab file

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

#%%

def make_id_to_vec(glovefile, word_to_id): 
    #word_to_id = make_word_to_id('my_vocab.txt')
    
    lines = open(glovefile, 'r').readlines()
    
    word_vecs = {}
    
    vector = None
        
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            word_vecs[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
 
        for word in word_to_id:
            if word_to_id[word] not in word_vecs:
                v = np.zeros(*vector.shape, dtype='float32')
                v[:] = np.random.randn(*v.shape)
                #v = np.random.randn(*vector.shape).astype('float32') #32
                word_vecs[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
     
    vec = np.zeros(*vector.shape, dtype='float32')
    #vec = np.zeros(*vector.shape).astype('float32') #32
    word_vecs[0] = torch.FloatTensor(torch.from_numpy(vec))
        
    vec_length = word_vecs[1].shape[0]
    
    return word_vecs, vec_length


#%% cannot import pad_sequence

def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = Variable(torch.ones(25, 300))
        >>> b = Variable(torch.ones(22, 300))
        >>> c = Variable(torch.ones(15, 300))
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = autograd.Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable
