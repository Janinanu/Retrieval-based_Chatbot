# UDC_Chatbot

# Retrieval-based Dialogue System on the Ubuntu Dialogue Corpus | PyTorch LSTM 

1) Idea

One of the top goals in artificial intelligence research is to enable a computer to hold natural, coherent conversations with humans - ideally in such a way that the computer tricks the human into thinking that it is talking to another human being. As of now, none of these so-called chatbots can pass the Turing test yet and research is still very far from developing a generative open-domain dialog system, meaning one that can freely compose sentences to hold human-style conversations about all kinds of topics. Despite that, more and more companies make use of chatbots to create value e.g. in their technical customer support or as personal assistants. Often, these are closed-domain retrieval-based chatbots: using some kind of heuristic, they pick an appropriate response from a fixed repository of responses predefined by humans within a narrowly specified range of topics.,, Taking inspiration from such retrieval-based chatbots, the goal in this project was to build a model that, at best, is able to retrieve the most appropriate response to a conversational input from a whole pool of candidate responses.

2) Corpus

In the paper “The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems”, the hypothesis is made that the lack of progress in recent years in building sophisticated dialogue systems is due to the lack of sufficiently large datasets. The authors of the paper thus aimed at compiling a new large corpus to exploit the opportunities of deep learning for dialog systems and came up with the so-called Ubuntu Dialog Corpus (UDC). It is a collection of chat logs from Ubuntu-internal chat rooms, in which users of the operating system seek and give Ubuntu-related technical advice. Containing around 930.000 dialogs, it is the largest freely available corpus with the following favourable characteristics: it is targeting a task-specific domain, namely technical support, with conversations being two-way oriented and going back and forth between the two users at least three times. As such, it is much more representative of natural human dialogs than, for example, common microblogging datasets like from Twitter or Weibo, in which the exchanges lack a natural conversational structure and length.

3) Data & task 
The data comes fully tokenized, stemmed and lemmatized and entities like names, locations, organizations, URLs, and system paths were replaced with special tokens.
![alt text](https://github.com/Janinanu/UDC_Chatbot/blob/master/src/Original%20datasets.png "Original Datasets Overview")

