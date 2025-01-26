---
title: NLP - Getting started [Part 6]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-26 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 6]

## Information extraction

Information extraction in natural language processing
(NLP) is the process of automatically extracting structured
information from unstructured text data.

The goal of information extraction is to transform
unstructured text data into structured data that can be
easily analyzed, searched, and visualized.

Billions of data are generated everyday, ranging
from social media posts to scientific literature, so
information extraction has become an essential tool
for processing this data.

As a result, many organizations rely on Information
Extraction techniques to use solid NLP algorithms to
automate manual tasks.

Information extraction involves identifying specific
entities, relationships, and events of interest in text
data, such as named entities like people,
organizations, dates and locations, and converting
them into a structured format that can be analyzed
and used for various applications.

Information extraction is a challenging task that
requires the use of various techniques, including
named entity recognition (NER), regular expressions,
and text matching, among others.

There is a wide range of applications involved,
including search engines, chatbots, recommendation
systems, and fraud detection, among others, making
it a vital tool in the field of NLP.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-1.png?raw=true)

Example:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-2.png?raw=true)

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-3.png?raw=true)

### Information Extraction Works

There are many techniques involved in the process
of information extraction.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-4.png?raw=true)

- Basic information extraction using named entity
recognition (NER) with the spaCy library.

General Pipeline of the Information Extraction Process

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-5.png?raw=true)

## Information Retrieval

Information retrieval (IR) in computing and information science is the task of
identifying and retrieving information system resources that are relevant to an
information need.

The information need can be specified in the form of a search query. In the case
of document retrieval, queries can be based on full-text or other content-based
indexing.

Process of accessing and retrieving the most appropriate information from text
based on a particular query given by the user, with the help of context-based
indexing or metadata.

These days we frequently think first of web search, but
there are many other cases:

- E-mail search

- Searching your laptop

- Corporate knowledge bases

- Legal information retrieval

### Basic assumptions of Information Retrieval

<b> Collection: </b> A set of documents

- Assume it is a static collection for the moment

<b> Goal: </b> Retrieve documents with information that is
relevant to the user’s information need and helps
the user complete a task

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-6.png?raw=true)

### Types of Information Retrieval Models

- Classic IR Model

It is the most basic and straightforward IR model. This paradigm is
founded on mathematical information that was easily recognized
and comprehended. The three traditional IR models are Boolean,
Vector, and Probabilistic.

- Non-Classic IR Model

It is diametrically opposed to the traditional IR model. Addition than
probability, similarity, and Boolean operations, such IR models are
based on other ideas. Non-classical IR models include situation
theory models, information logic models, and interaction models.

- Alternative IR Model

It is an improvement to the traditional IR model that makes use of
some unique approaches from other domains. Alternative IR models
include fuzzy models, cluster models, and latent semantic indexing
(LSI) models.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-7.png?raw=true)

### Information Retrieval applications

- Search engines, searching for text documents,
images, videos, and so on.

- Question answering over a set of documents (e.g.
with a chatbot or a smart speaker).

- Recommender systems.

- Summarization of a set of documents.

### Note

Basic information retrieval using the TF-IDF (Term
Frequency-Inverse
Document
Frequency)
vectorization method. This code demonstrates how
to create a simple information retrieval system with
a set of documents, and then how to search through
them using a query.

```python
# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Sample documents to form the corpus
documents = [
    "The quick brown fox jumped over the lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Python is a high-level, interpreted, and general-purpose programming language.",
    "Artificial intelligence and machine learning are technologies that use algorithms to simulate human intelligence."
]
# Initialize a TF-IDF Vectorizer
# This vectorizer converts a collection of raw documents into a matrix of TF-IDF features.
tfidf_vectorizer = TfidfVectorizer()
# Fit the vectorizer on the documents and transform the documents into their respective TF-IDF
vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# Define a query to search within the documents
query = "programming and algorithms"
# Convert the query into the TF-IDF vector using the same vectorizer
query_tfidf = tfidf_vectorizer.transform([query])
# Compute cosine similarity between the query TF-IDF vector and the document TF-IDF vectors
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
# Get the index of the most similar document in the corpus
most_similar_document_index = np.argmax(cosine_similarities)
# Print the most similar document and its similarity score
print("Most Similar Document:", documents[most_similar_document_index])
print("Similarity Score:", cosine_similarities[most_similar_document_index])
```

## Summarization

Text summarization is the process of generating
short, fluent, and most importantly accurate
summary of a respectively longer text document.

The main idea behind automatic text summarization
is to be able to find a short subset of the most
essential information from the entire set and present
it in a human-readable format.

As online textual data grows, automatic text
summarization methods have the potential to be
very helpful because more useful information can
be read in a short time.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-8.png?raw=true)

There are broadly two different approaches that
are used for text summarization:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-9.png?raw=true)

### Extractive Summarization

We identify the important sentences or phrases
from the original text and extract only those from
the text. Those extracted sentences would be our
summary.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-10.png?raw=true)

### Abstractive Summarization

Here, we generate new sentences from the original
text. This is in contrast to the extractive approach
we saw earlier where we used only the sentences
that were present. The sentences generated through
abstractive summarization might not be present in
the original text:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-11.png?raw=true)


### Example TextRank Algorithm for Extractive Text Summarization

```python
# Import the spaCy library to use for NLP tasks.
import spacy
# Import the pyTextRank library, which adds additional text processing capabilities on top of spaCy.
import pytextrank
# Load the large English language model in spaCy.
nlp = spacy.load("en_core_web_lg")
# Add the text ranking algorithm from pyTextRank into the spaCy pipeline for processing documents.
nlp.add_pipe("textrank")
# Define a string containing a lengthy and complex text about deep learning.
example_text = """Deep learning (also known as deep structured learning) is part of a
    broader family of machine learning methods based on artificial neural networks with
    representation learning. Learning can be supervised, semi-supervised or unsupervised.
    Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning,
    recurrent neural networks and convolutional neural networks have been applied to
    fields including computer vision, speech recognition, natural language processing,
    machine translation, bioinformatics, drug design, medical image analysis, material
    inspection and board game programs, where they have produced results comparable to
    and in some cases surpassing human expert performance. Artificial neural networks
    (ANNs) were inspired by information processing and distributed communication nodes
    in biological systems. ANNs have various differences from biological brains. Specifically,
    neural networks tend to be static and symbolic, while the biological brain of most living organisms
    is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
    layers in the network. Early work showed that a linear perceptron cannot be a universal classifier,
    but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
    Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size,
    which permits practical application and optimized implementation, while retaining theoretical universality
    under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely
    from biologically informed connectionist models, for the sake of efficiency, trainability and understandability,
    whence the structured part."""
# Output the original size of the document.
print('Original Document Size:', len(example_text))
# Process the example_text through the spaCy pipeline which now includes text ranking.
doc = nlp(example_text)
# Iterate through sentences identified by pyTextRank as the top 2 key phrases and sentences,
# and print each sentence along with its length.
for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
    print(sent)
    print('Summary Length:', len(sent))
```

Output:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-12.png?raw=true)

### Abstractive Summarization Example

```python
# Import necessary libraries from transformers
from transformers import pipeline
# Create a summarization pipeline using the "facebook/bart-large-cnn" model.
# This model is pretrained for abstractive summarization tasks.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# This is a simple text about machine learning.
text = """
    Machine learning is a field of artificial intelligence that uses statistical
    techniques to give computer systems the ability to "learn" (i.e., progressively
    improve performance on a specific task) from data, without being explicitly
    programmed. The name Machine Learning was coined in 1959 by Arthur Samuel.
    Evolving from the study of pattern recognition and computational learning theory
    in artificial intelligence, machine learning explores the study and construction
    of algorithms that can learn from and make predictions on data. These algorithms
    operate by building a model from sample inputs and using that model to make
    predictions or decisions, rather than following strictly static program
    instructions.
"""
# Call the summarizer on the text.
# The summary will try to condense the information in the text into a shorter
form, maintaining the essential details.
summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
# Print the generated summary
print("Summary:", summary[0]['summary_text'])
```

## Attention Mechanisms and Transformers

### Attention Mechanisms

The concept of attention mechanism was first
introduced in a 2014 paper on neural machine
translation.

Prior to this, RNN encoder-decoder frameworks
encoded variable-length source sentences into
fixed-length vectors that would then be decoded
into variable-length target sentences. This approach
not only restricts the network’s ability to cope with
large sentences but also results in performance drop
for long input sentences.

Rather than trying to force-fit all the information
from an input sentence into a fixed-length vector,
the paper proposed the implementation of a
mechanism of attention in the decoder.

In this approach, the information from an input
sentence is encoded across a sequence of vectors,
instead of a fixed-length vector, with the attention
mechanism allowing the decoder to adaptively
choose a subset of these vectors to decode the
translation.

### Transformer

The Transformer was the first transduction model to
implement self-attention as an alternative to
recurrence and convolutions.

<b> Attention allows models to dynamically focus on
appropriate parts of the input data </b>, akin to the
way humans pay attention to certain aspects of a
visual scene or conversation. This selective focus is
particularly crucial in tasks where context is key,
such as language understanding or image
recognition.

The Transformer was the first transduction model to
implement self-attention as an alternative to
recurrence and convolutions.

In the context of transformers, attention
mechanisms serve to weigh the influence of
different input tokens when producing an output.

This is not purely a replication of human attention
but an enhancement, enabling machines to
surpass human performance in certain tasks.

### Transformer Architecture

Transformer consists of
two primary components:
an encoder and a
decoder.
Each
component is designed to
perform distinct yet
complementary functions
in the processing of input
and
generation
of
output.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-6-14.png?raw=true)

Inputs and Outputs:

- Input Embedding: Converts input data into a fixed-size numerical format that the
model can process.
- Output Embedding: Similar to input embedding, but for the output data.

Positional Encoding: Adds information about the position of each element in the
sequence to the embeddings, helping the model understand the order of the data.

Nx (Layers):
This part of the diagram is repeated several times (N times). Each layer consists of
the following components:

- Masked Multi-Head Attention: Focuses on different positions of the input to better
understand relationships between elements.

- Multi-Head Attention: Similar to masked version but used in different contexts
within the model.

- Add & Norm: A step that combines the outputs of attention with the original input
(add) and normalizes the result (norm).

- Feed Forward: A neural network that processes each position of the input
independently.

- Softmax and Linear:

    - Linear: A transformation that prepares the outputs for probability prediction.
    - Softmax: Converts the linear outputs into probabilities, indicating the likelihood of
each possible output element.
