# UDC_Chatbot

# Retrieval-based Dialogue System on the Ubuntu Dialogue Corpus | PyTorch LSTM 

This project is a retrieval-based dialogue system, that means a model which classifies whether an utterance is the correct response to a given context utterance or not. Ideally, it is suppose to retrieve the best response to a conversational input from a whole pool of candidate utterances.

**Data:**

*Please find the necessary data files within this GDrive folder: https://drive.google.com/open?id=1RIIbsS-vxR7Dlo2_v6FWHDFE7q1XPPgj*

You do not need all of those files. It depends on which sample size you would like to choose. As a starter, it might be a good idea to first download the following files only:

    - glove.6B.50d.txt (Subfolder GloVe)
    - training_10000.csv (Subfolder MAIN FILES)
    - validation_10.csv (Subfolder MAIN FILES)
    - testing_different_structure_10.csv (Subfolder MAIN FILES)
    - testing_same_structure_10.csv (Subfolder MAIN FILES)
    - saved_model_10000.pt (Subfolder SAVED MODELS)
    
**Summary:**

The following model tends to heavy overfitting. Applying regularization techniques such as dropout or reducing the model size showed only limited effectiveness. Only increasing the amount of training data showed a promising regularizing effect. The paper which I took the idea for the model from (see paragraph below) also describes this problem, they achieved good results by training on at least 100.000 up to 1 million examples.

Due to restrictions in the computational resources available to me, I mostly trained the model on 10.000 as a bare minimum. It was possible to achieve acceptably good validation results only with very small validation sets, and the testing results were expectedly weak. 

Two saved models are worth looking at in the subfolder SAVED MODELS:
(Note: both were trained once on an external cloud GPU. Please un-comment the ".cuda()" if you want to use these saved models on your GPU.) 

- saved_model_10000.pt : 10.000 training examples, 10 validation examples, validation accuracy of 0.9 (classified 9 from 10 correctly)

- saved_model_500000.pt : (run once), 500.000 training examples, 100 validation examples, validation accuracy of ~0.7 after 4 epochs...


**Some important remarks:**

I am using Python 3.5 and PyTorch 0.3.0 to implement the LSTM model, taking inspiration from the following two papers:

 - https://arxiv.org/pdf/1506.08909.pdf (Note that this paper refers to the old version of the data, see below.)

 - https://arxiv.org/pdf/1510.03753v2.pdf


Ubuntu Dialogue Corpus (UDC) Data taken from ...

 - (Older version: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/)

 - New version: https://github.com/rkadlec/ubuntu-ranking-dataset-creator

Note that there is an old and a new version of the data. The older version comes with less preprocessing done and a different structure of the test and validation data. I am only working with the new version to take advantage of the additional preprocessing.

The data generation files for the new version can be found in this github repository: https://github.com/rkadlec/ubuntu-ranking-dataset-creator

Those new, original data files are as follows and can be found in above GDrive folder, too:

- train.csv (463Mb with 1,000,000 examples): 
It is separated into 3 columns: the context of the conversation, the candidate response or 'utterance', and a flag or 'label' (= 0 or 1) denoting whether the response is a 'true response' to the context (flag = 1), or a randomly drawn response from elsewhere in the dataset (flag = 0). 

- valid.csv (27Mb with 19,561 examples):
Separated into 11 columns: the context, the true response or 'ground truth utterance', and 9 false responses or 'distractors' that were randomly sampled from elsewhere in the dataset. 

- test.csv (27Mb with 18,921 examples): 
Formatted in the same way as the validation set.


However, I did not strictly stick with their approach: 

I split the train.csv into three parts (80%/10%/10%) to create my own training/validation/test files that have the same column structure as the original train.csv (context, response, label). 
These files can be found in the GDrive folder as split_training.csv, split_validation.csv and split_testing.csv.

For training and validation, I then used smaller subsamples of split_training.csv and split_validation.csv.
You can find those subsamples in the GDrive folder, like for example **training_10000.csv** for 10000 training examples and **validation_100.csv** for 100 validation examples.

For testing, I tried two different approaches:

1) I used a subsample of my split_testing.csv (with the same structure as in the training and validation). It can be found as **testing_same_structure_20.csv** for 20 testing examples in the folder.

2) I used a subsample of the original test.csv (with a different structure than in the training and validation as explained before). It can be found as **testing_different_structure_20.csv**.
