---
title: NLP - Getting started [Part 4]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2025-01-26 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 4]

## Vector Space Model

### The Search Task

Given a <i> query </i> and a <i> corpus </i>, find <i> relevant </i> items

<i> query: </i> a textual description of the user’s information need 

<i> corpus: </i> a repository of textual documents

<i> relevance: </i> satisfaction of the user’s information need

### Retrieval Model

A formal method that predicts the degree of relevance of a document to a query.

#### Basic Retrieval Process

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-1.png?raw=true)

### Vector Space Model

The Vector Space Model (VSM) is a way of
representing documents through the words that
they contain.

It is a standard technique in Information
Retrieval.

The VSM allows decisions to be made about
which documents are similar to each other and
to keyword queries.

Formally, a <b> vector space </b> is defined by a set of <u> linearly
independent </u> basis vectors.

The <b> basis vectors </b> correspond to the dimensions or
directions of the vector space.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-2.png?raw=true)

The Vector Space Model represents documents and terms as vectors in a multi-
dimensional space. Each dimension corresponds to a unique term in the entire
corpus of documents.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-3.png?raw=true)

#### How it works:

- Each document is broken down into a word
frequency table.

- The tables are called vectors and can be stored
as arrays.

- A vocabulary is built from all the words in all
documents in the system.
 
- Each document is represented as a vector
based against the vocabulary.

Example: 

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-4.png?raw=true)

Queries can be represented as vectors in the same way as
documents:

Dog = [0, 0, 0, 1, 0]

### Document-Term Matrix

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-5.png?raw=true)

### Similarity measures

There are many different ways to measure how
similar two documents are, or how similar a
document is to a query.

The cosine measure is a very common similarity
measure.

Using a similarity measure, a set of documents can be
compared to a query and the most similar document
returned.

#### The cosine measure

For two vectors d and d’ the cosine similarity between
d and d’ is given by:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-6.png?raw=true)

Here d X d’ is the vector product of d and d’,
calculated by multiplying corresponding frequencies
together.

The cosine measure calculates the angle between the
vectors in a high-dimensional virtual space.

#### Ranking documents

A user enters a query.

The query is compared to all documents using a similarity
measure.

The user is shown the documents in decreasing order of
similarity to the query term.

### Cosine Similarity

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-7.png?raw=true)

## VSM Variations

### Vocabulary

### Stopword lists

Commonly occurring words are unlikely to
give useful information and may be removed
from the vocabulary to speed processing.

Stopword lists contain frequent words to be
excluded

Stopword lists need to be used carefully

- E.g. “to be or not to be”

### Term weighting

Not all words are equally useful

A word is most likely to be highly relevant to
document A if it is:

- Infrequent in other documents
- Frequent in document A
 
The cosine measure needs to be modified to
reflect this

### Normalised term frequency (tf)

A normalised measure of the importance of a
word to a document is its frequency, divided by
the maximum frequency of any term in the
document

This is known as the tf factor.

This stops large documents from scoring
higher.

### Inverse document frequency (idf)

A calculation designed to make rare words
more important than common words

The idf of word i is given by

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-8.png?raw=true)

Where N is the number of documents and n_i is
the number that contain word i

### tf-idf

The tf-idf weighting scheme is to multiply each
word in each document by its tf factor and idf
factor.

Different schemes are usually used for query
vectors.

Different variants of tf-idf are also used.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-4-9.png?raw=true)














