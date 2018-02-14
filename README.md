# Chatbot | Retrieval-based Dialog System on the Ubuntu Dialog Corpus | LSTM | PyTorch

<img src="https://github.com/Janinanu/UDC_Chatbot/blob/master/src/drib_blink_bot.gif" width="160" height="120" />

**1) Idea**

One of the top goals in artificial intelligence research is to enable a computer to hold natural, coherent conversations with humans - ideally in such a way that the computer tricks the human into thinking that it is talking to another human being. As of now, none of these so-called chatbots can pass the Turing test yet and research is still very far from developing a generative open-domain dialog system, meaning one that can freely compose sentences to hold human-style conversations about all kinds of topics. Despite that, more and more companies make use of chatbots to create value e.g. in their technical customer support or as personal assistants. Often, these are closed-domain retrieval-based chatbots: using some kind of heuristic, they pick an appropriate response from a fixed repository of responses predefined by humans within a narrowly specified range of topics. **Taking inspiration from such retrieval-based chatbots, the goal in this project was to build a model that, at best, is able to retrieve the most appropriate response to a conversational input from a whole pool of candidate responses.**

**2) Corpus**

In the paper [“The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems”](https://arxiv.org/pdf/1506.08909.pdf), the hypothesis is made that the lack of progress in recent years in building sophisticated dialogue systems is due to the lack of sufficiently large datasets. The authors of the paper thus aimed at compiling a new large corpus to exploit the opportunities of deep learning for dialog systems and came up with the so-called Ubuntu Dialog Corpus (UDC). It is a collection of chat logs from Ubuntu-internal chat rooms, in which users of the operating system seek and give Ubuntu-related technical advice. Containing around 930.000 dialogs, it is the largest freely available corpus with the following favourable characteristics: it is targeting a task-specific domain, namely technical support, with conversations being two-way oriented and going back and forth between the two users at least three times. As such, it is much more representative of natural human dialogs than, for example, common microblogging datasets like from Twitter or Weibo, in which the exchanges lack a natural conversational structure and length.

**3) Data & task** 

The data comes fully tokenized, stemmed and lemmatized and entities like names, locations, organizations, URLs, and system paths were replaced with special tokens.

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Original%20datasets.png "Original Datasets Overview")

Mostly importantly, the training set is separated into 3 columns: the context of the conversation (all previous utterances in the conversation up to the point at which the next response is to be found), the candidate response, and a label denoting whether that candidate response is a true response to the context (label = 1), or a randomly drawn response from elsewhere in the dataset (label = 0), with a 50%/50% distribution of positive and negative examples. Given this nature of the data, the task is to classify whether an utterance is a true response to a given context utterance or not. It is thus a binary classification problem of context-response pairs.
However, I did not strictly stick with the original approach of the train/validation/test setup: I split the train.csv into three parts (80%/10%/10%) to create my own training/validation/test files that all share the same column structure as the original train.csv. In addition, I took a subsample of the original test.csv as a second test file with a different structure, as will be explained in greater detail in the testing paragraph.

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Subsample%20data.png "Subsample data overview")

**4) Model architecture**

Python 3.5 and PyTorch 0.3.0 was used to implement a neural network that can be described as a dual encoder model. The main component is an encoder model composed of three layers:
- Embedding layer: initialized with pre-trained GloVe vectors for those words available, otherwise randomly initialized (from standard normal distribution), fine-tuned during training. Dimension: vocabulary size x embedding dimension.
- LSTM layer: unidirectional, single-layer LSTM with input-to-hidden weights initialized from a uniform distribution and hidden-to-hidden weights with orthogonal initialization, following the original paper’s recommendations. At each time step, one word vector of the input utterance is fed into the LSTM and the hidden state is updated. For this classification task, we are only interested in  the last hidden state of each input sequence, which can be interpreted as a numerical summary of the input utterance. 
- Dropout layer: According to PyTorch documentation and forum, the in-built dropout in the LSTM layer will not apply effectively if num_layers = 1 (since it does by definition not apply to the last layer)., Therefore, the in-built dropout was set to 0.0 and an additional dropout layer was added to the model such that is applies directly to the last hidden state of each input sequence.

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Model.png "LSTM dual encoder model")

To obtain the dual encoder model, one instance of the encoder model is applied to the context utterance and subsequently to the response utterance for each training example. The two outputs (the last hidden state of the context input sequence, denoted as c, and the last hidden state of the response input sequence, denoted as r) are then used to calculate the probability that this is a valid pair: a weight matrix M is initialized as another learnable parameter (in addition to the embedding weights and the LSTM weights) and used to map between c and r.  This can be interpreted as a predictive approach: given some context c, by multiplying c with M we predict what a possible response r’ could look like. We then measure the similarity of r’ to the actual response r using the dot product and convert it to a probability using the sigmoid function. 

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Equation.png "Sigmoid probability")

A high similarity will result in a high dot product and a sigmoid value that goes towards 1. The model was trained by minimizing the binary cross entropy loss.

**5) Training & validation results**

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Loss.png "Training and validation loss")

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Accuracy.png "Training and validation accuracy")

These results are based on 10,000 training examples & 1,000 validation examples. While training loss and accuracy behaved as desired, the course of the validation loss and accuracy show clearly that the model tends to heavy overfitting. The validation loss never decreases; even worse, it increases unstoppably. Parallelly, the validation accuracy increases by only a small amount, is relatively unstable and then falls rapidly. 
Attempts to regularize this overfitting situation by optimizing for high values for dropout and L2 weight decay showed limited effectiveness. Other attempts to prevent overfitting, such as drastically reducing the model’s hidden and embedding size and freezing the word embedding parameters (so that the LSTM weights and M remain the only learnable parameters) did not improve the validation results neither. 
The best validation accuracy was achieved with the hyperparameter configuration shown in the table.

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Hyperparameters.png "Hyperparameter configuration")

**6) Test data & test results**

Testing was done with two different approaches.
- I used a subsample of my split_testing.csv which has the same data structure as the data used for training and validation (context - response - label). The appropriate testing metric used was accuracy - it simply measures what is the chance that the label 1 or 0 is classified correctly.
- I used a subsample of the original test.csv which has a different structure than the training and validation data:

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Test%20columns.png "Test approach 2 data sructure")

It contains 11 columns: 1 context utterance to which we want to find an appropriate response, 1 ground truth response utterance which is the true appropriate response, 9 distractor utterances which, together with the 1 ground truth utterance, represent in total 10 candidate responses, for each one of which the model computes the sigmoid scores (the probability for each candidate response being the ground truth response) and is challenged to rank them accordingly. For this testing approach, the appropriate metric is recall, specifically recall at k for k = 5, 2, 1. The system’s ranking is considered correct if the ground truth utterance is among the k highest scored response candidates. 
Given the above validation results, the test results are expectedly weak. The model basically performs no better than taking random guesses:

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Test%20results.png "Test results")

**7) Key findings**

As explained before, the most distinct characteristic of the UDC is its size, which makes it so well-suited to accelerate progress in dialog system research by making use of deep learning. In their paper [“Improved Deep Learning Baselines for Ubuntu Corpus Dialogs”](https://arxiv.org/pdf/1510.03753v2.pdf), the authors set the benchmark performance for the dual encoder model architecture on the UDC dataset. To achieve good recall rates, they trained the model on at least 100,000 up to the full 1,000,000 training examples available. 

![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Benchmark%20recall%401.png "Benchmark recall")

This sheds some light on why training on 10,000 examples leads the model into an overfitting situation that cannot be regularized with the methods applied in this project. It implies that a maximum amount of training data is crucial for good performance in dialog systems and that only increasing the amount of training data might finally show a promising regularizing effect on the LSTM model developed in this project.

Taking a closer look at the benchmark test results, it is interesting that even with the maximum of training data used, the recall for k = 1 is at 63,8% and does not reach its peak yet. In other words: even with 1,000,000 training examples, in around one third of all cases the user of a chatbot, be it a customer seeking technical advice, will not receive the most appropriate response to his question.This illustrates one of the biggest shortcomings of chatbots as of now - their need for vast amounts of data. In practice, only few companies can provide those amounts of data when they first start putting chatbots in dialog with their customers. That’s why many chatbots are first released with mediocre performance, gathering data to learn over time, yet frustrating and driving away many users.
The authors of the benchmark paper therefore propose three ways of enhancing performance of dialog systems: 
- improved pre-processing approaches
- sophisticated model ensembles (combinations of multiple model architectures)
- extended models with memory (e.g. providing external sources of information from user manuals) 

