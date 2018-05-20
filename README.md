# Question Encoding for Robust QA Networks
==========================================

## Introduction
There are many ways to syntactically ask the same question. Can we train a machine learning model to map similar questions to a single syntax-invariant representation? We propose using neural techniques to learn a mapping from a typical word-embedding question representation to a syntax-invariant one.

We propose to learn these embeddings by training on sentence similarity tasks. Using a Self-Attentive RNN model [1], we construct an output sequence of new embeddings represented as matrix $M$. We propose an addition to this network which condenses variable length embedding sequences to a single vector representation, which we call the summary vector, allowing explicit comparisons between two questions. We hypothesize that this intermediate embedding sequence $M$ produced by the model will result in a more syntactically invariant representation of a question.

To evaluate the utility of our embeddings, we attempt to use them to train Question-Answering (QA) network. QA networks are tasked with finding the answer to a given question within an article. We hypothesize that a QA network might converge faster or achieve more robust performance if fed our new embeddings. For this project we used the Dynamic Coattention Network available on github.

For more information check out the paper we wrote for CS194-126 called question-encoding-robust.pdf. 

## Instructions
Our up-to-date branch is named 'key' (not 'master'). Follow the instructions on the README within paraphrase-id-tensorflow-master and dynamic-coattention-network-plus to train the question encoding and QA networks. 

### Dependencies

We used python 3.5 with Tensorflow v.1.4 for this project. Other dependencies are outlined in the individual subrepos.

## Authors
Arnav Vaid / (https://github.com/vaidarnav)
Chris Correa / (https://github.com/chriscorrea14)
Anaga Rajan / (https://github.com/rajananaga)
Nikhil Sharma / (https://github.com/sharmaster96)
