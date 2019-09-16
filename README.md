# Deep Learning Techniques for Natural Language Mining on Quora Questions

Text mining often depends on the type of documents that you are analysing,
and in the case of user generated content and natural language in general
semantics are particularly relevant. 

We present here several text mining techniques and we compare novel approaches 
based on deep learning with more common non-neural techniques like TF-IDF 
search and LSH indexing on both the text understanding performance and the 
computational time performance.

We run our experiments on a particular class of documents namely Quora 
questions which have the property of being short and semantically dense, and 
show how more advanced techniques with stronger language understanding 
capabilities manage to yield better results while maintaining acceptable 
speeds.

## Usage

The class `PretrainedLMForQQP` is the base for training the duplicate question 
network.

The script `evaluation_tests.py` reproduces the results presented in the report.
(Notice that if you don't have the pre-computeed dataset vectors the calculation
of such vectors can take a long time).

The search servers are all implemented in the `questions_recommend.py` module.
While in the `semantic_sim.py` module you can find a server using the pre-trained
models to compute the sentence vectors.
