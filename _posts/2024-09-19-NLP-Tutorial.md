---
title: NLP - Getting started [Part 1]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-09-19 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 1]

### Introduction

<i> Ultimate goal: human-to-computer communication.</i>

The goal of natural language processing (NLP) is to design and build computer systems that are able to analyze natural languages like vietnamese, korean, german, and that generate their outputs in a natural language, too.

In natural language understanding, the objective is to extract the meaning of an input sentence or an input text. Usually, the meaning is represented in a suitable formal representation language so that it can be processed by a computer.

NLP techniques enable businesses to extract structured, machine-readable information from unstructured text data. This empowers companies to process vast textual sources like feedback, discovering patterns and insights to inform strategic decisions.

NLP allows real-time spoken language translation, facilitating smooth cross-cultural communication. Accurate, accessible translations enable businesses to tap into international markets and engage a global audience.

NLP combined with Machine Learning streamlines market research, examining vast amounts of text data to detect customer sentiment, predict trends, and provide valuable strategic insights to outperform competitors.

### Phases of NLP

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-1-1.png?raw=true)

#### Lexical Analysis

This phase scans the source text as a stream of characters and converts it into meaningful lexemes (tokens or phrases).

Example: The sentence "The quick brown fox jumps over the lazy dog." is broken down into tokens like "The", "quick", "brown", "fox", etc.

#### Syntactic Analysis (Parsing)

Syntactic Analysis is used to check grammar, word arrangements, and shows the relationship among the words. Also known as parsing, it involves analyzing the tokens' syntactic structure, according to the rules of grammar.

Example: Identifying that "The quick brown fox" is a noun phrase and "jumps over the lazy dog" is a verb phrase.

#### Semantic Analysis

Semantic analysis is concerned with the meaning representation. It mainly focuses on the literal meaning of words, phrases, sentences by syntactic structure and vocabulary.

Example: Recognizing that "apple" can mean both a fruit and a company, depending on context. In the sentence "I ate an apple," semantic analysis helps understand that "apple" refers to the fruit.

#### Discourse Integration

Discourse Integration depends upon the sentences that proceeds it and also invokes the meaning of the sentences that follow it. This involves understanding how the immediate sentence relates to preceding and following sentences to ensure that the text makes sense as a whole.

Example: In the text "She lifted the violin. She played a beautiful melody." Discourse integration helps us understand that "she" in the second sentence refers to the same person who lifted the violin.

#### Pragmatic Analysis

It helps you to discover the intended effect by applying a set of rules that characterize cooperative dialogues. This is the understanding of language beyond the literal meaning, including implied meanings, politeness, and other aspects of language that depend on context.

For Examples: "Open the door" is interpreted as a request instead of an order.

If someone says, "Can you pass the salt?" during dinner, pragmatic analysis understands that it's a request, not just a question about one's ability to pass the salt.

### NLP Note

<b> NLP is difficult due to various reasons such as Ambiguity and Uncertainty exist in the language. </b>

#### Ambiguity

There are the following three ambiguity -

##### Lexical Ambiguity 

Lexical Ambiguity exists in the presence of two or more possible meanings of the sentence within a single word. Example: Manya is looking for a match. In this, the word match refers to that either Manya is looking for a partner or Manya is looking for a match. (Cricket or other match).

##### Syntactic Ambiguity
Syntactic Ambiguity exists in the presence of two or more possible meanings within the sentence.

Example:
I saw the girl with the binocular.
In the above example, did I have the binoculars? Or did
the girl have the binoculars?
1. You saw the girl and you were using binoculars to see
her. In this interpretation, you have the binoculars.
2. You saw a girl who was holding binoculars. In this
interpretation, the girl has the binoculars.

###### Referential Ambiguity
Referential Ambiguity exists when you are referring
to something using the pronoun.
Example: Kiran went to Sunita. She said, "I am
hungry."
In the above sentence, you do not know that who is
hungry, either Kiran or Sunita.
The ambiguity lies in not knowing which person the
pronoun is referencing, making it unclear who is
expressing hunger.

### Text pre-processing

Text pre-processing is the process of transforming
unstructured text to structured text to prepare it
for analysis. When you pre-process text before feeding it to
algorithms, you increase the accuracy and
efficiency of said algorithms by removing noise and
other inconsistencies in the text that can make it
hard for the computer to understand. Making the text easier to understand also helps to
reduce the time and resources required for the
computer to pre-process data. Making the text easier to understand also helps to
reduce the time and resources required for the
computer to pre-process data.

Now, we discuss some basics pre-processing steps.









