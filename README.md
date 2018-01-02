# UDC_Chatbot

In this work-in-progress project, I am building a retrieval-based dialogue system, that means a chatbot that picks the best response to a conversational input from a pool of candidate responses. 

I am using Python 3.5 and PyTorch 0.3.0 to implement the LSTM model, taking inspiration from the following two papers and a blog tutorial:

- https://arxiv.org/pdf/1506.08909.pdf (Note that this paper refers to the old version of the data, see below.)

- https://arxiv.org/pdf/1510.03753v2.pdf

- http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow (Tensorflow implementation for this is available on github: https://github.com/dennybritz/chatbot-retrieval/, by Denny Britz)

UDC Data taken from ...
- Older version: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/

- New version: https://github.com/rkadlec/ubuntu-ranking-dataset-creator 

(Note that there is an older and a newer version of the data.  The older version comes with less preprocessing done and a different structure of the test and evaluation data.)

The GloVe file can be found here: https://nlp.stanford.edu/projects/glove/
