---
title: NLP - Getting started [Part 3]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-10-09 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 3]

## Word Sense Disambiguation
 
Word sense disambiguation (WSD) is the
task of determining the correct meaning of
a word in a given context. It is a common problem in natural language
processing (NLP) because many words have
multiple meanings, and the meaning of a
word can change depending on the context
in which it is used.

For example, the word “bass”
can refer to a type of fish or
a musical instrument.

Disambiguating the word
“bass” in the sentence “I
caught a bass while fishing”
would involve determining
that it refers to a type of fish,
while in the sentence “I play
the bass in a band,” it would
refer to the musical
instrument.

```
In computational linguistics, word-sense disambiguation is an open problem
concerned with identifying which sense of a word is used in a sentence.

(WSD) has been a basic and on-going issue since its introduction in natural language
processing (NLP) community.
```

### Two variants of WSD task

- Lexical Sample task
    - Small pre-‐selected set of target words (line, plant)
    - And inventory of senses for each word
    - Supervised machine learning: train a classifier for each word
-  All-wordstask
    - Every word in an entire text
    - A lexicon with senses for each word
    - Data sparseness: can’t train word-‐specific classifiers

### WSD Methods

- Dictionary and knowledge-based methods
- Supervised methods
- Semi-supervised Methods
- Unsupervised Methods

#### Dictionary and knowledge-based methods

As the name suggests, for disambiguation, these
methods primarily rely on dictionaries, treasures
and lexical knowledge base. They do not use corpora evidences for
disambiguation. The Lesk method is the seminal dictionarybased method introduced by Michael Lesk in
1986.

The Lesk definition, on which the Lesk algorithm
is based is “measure overlap between sense
definitions for all words in context”. However, in 2000, Kilgarriff and Rosensweig
gave the simplified Lesk definition as “measure
overlap between sense definitions of word and
current context”, which further means identify
the correct sense for one word at a time. Here
the current context is the set of words in
surrounding sentence or paragraph.

<b> Lesk Algorithm in NLP </b>

It is founded on the idea that words used in a text
are related to one another, and that this
relationship can be seen in the definitions of the
words and their meanings. The pair of dictionary senses having the highest
word overlap in their dictionary meanings are used
to disambiguate two (or more) terms.

```python
# Importing necessary modules from NLTK
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
# Downloading NLTK data necessary for processing. Consider commenting this out after the first run to save time.
nltk.download('all')

def get_semantic(seq, key_word):
  # Tokenizes the input sequence into words.
  tokens = word_tokenize(seq)
  # Applying the Lesk algorithm to find the best sense of the key_word given the context of the tokens.
  best_sense = lesk(tokens, key_word)
  # Returning the definition of the identified best sense. If no sense was found, handle NoneType error.
  return best_sense.definition() if best_sense else None
# Defining the keyword and sequence for the word 'book'.
keyword = 'book'
seq1 = 'I love reading books on coding.'
seq2 = 'The table was already booked by someone else.'
# Printing the semantic meanings of 'book' in different contexts.
print(get_semantic(seq1, keyword))
print(get_semantic(seq2, keyword))
# Defining the keyword and sequence for the word 'jam'.
keyword = 'jam'
seq1 = 'My mother prepares very yummy jam.'
seq2 = 'Signal jammers are the reason for no signal.'
# Printing the semantic meanings of 'jam' in different contexts.
print(get_semantic(seq1, keyword))
print(get_semantic(seq2, keyword))
```

<b> Explaination </b>

1. "a number of sheets (ticket or stamps etc.) bound together on one edge"

Context: seq1 = "I love reading books on coding."

Keyword: book

Explanation: This definition matches the typical meaning of "book" when referring to
a physical or digital item consisting of multiple pages bound together, which people
read for information or enjoyment. The context "reading books on coding" clearly
indicates the standard noun sense of "book" as a collection of sheets of paper.

2. "arrange for and reserve (something for someone else) in advance"

Context: seq2 = "The table was already booked by someone else."

Keyword: book

Explanation: This sense of "book" is a verb meaning "to reserve or arrange in
advance." The context of a table being "booked" refers to reserving a spot at a
restaurant or other venue, which aligns with this definition.

3. "press tightly together or cram"

Context: seq1 = "My mother prepares very yummy jam."

Keyword: jam

Explanation: This sense might seem odd in the given context, but it was chosen
because the word "jam" as a noun can mean something tightly packed together.
While the context refers to "jam" as a fruit preserve, the Lesk algorithm may have
chosen this definition based on the similarity between "jam" and "cram" (possibly due
to overlaps in the context).

4. "deliberate radiation or reflection of electromagnetic energy for the purpose
of disrupting enemy use of electronic devices or systems"
Context: seq2 = "Signal jammers are the reason for no signal."

Keyword: jam

Explanation: This sense is relevant to the context, as the sentence refers to "signal
jammers." The word "jam" here is used in a technical sense, referring to the
intentional disruption of signals, which matches the definition provided.

#### Supervised Methods

For disambiguation, machine learning methods
make use of sense-annotated corpora to train. These methods assume that the context can
provide enough evidence on its own to
disambiguate the sense. In these methods, the words knowledge and
reasoning are deemed unnecessary.

The context is represented as a set of “features”
of the words. It includes the information about
the surrounding words also. Support vector machine and memory-based
learning are the most successful supervised
learning approaches to WSD. These methods rely on substantial amount of
manually sense-tagged corpora, which is very
expensive to create.

<b> Classification Methods: Supervised Machine Learning </b>

Input:
• a word w in a text window d (which we’ll call a “document”)
• a fixed set of classes C = {c1
, c2
,…, cJ
}
• A training set of m hand-‐labeled text windows again called
“documents” (d1
,c1
),....,(dm,cm)
• Output:
• a learned classifier γ:d → c

#### Semi-supervised Methods

Due to the lack of training corpus, most of the
word sense disambiguation algorithms use
semi-supervised learning methods. It is because semi-supervised methods use both
labelled as well as unlabeled data. These
methods require very small amount of
annotated text and large amount of plain
unannotated text. The technique that is used by semi-supervised
methods is bootstrapping from seed data.

#### Unsupervised Methods

These methods assume that similar senses
occur in similar context. That is why the senses can be induced from text
by clustering word occurrences by using some
measure of similarity of the context. This task is
called word sense induction or discrimination.

### Word-Sense Disambiguation (Example)

```python
from nltk import wsd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')
nlp = load('en_core_web_sm')
# in the below example the word die has a different meanings in each sentence.
# only by understanding the context the of the word the NLP can further improvise.
X = 'The die is cast.'
Y = 'Roll the die to get a 6.'
Z = 'What is dead may never die.'
# in this we will use wordnet from princeton university to get the word different context sentences with part of speech attached.
# Wordnet : WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of
# cognitive synonyms (synsets), each expressing a distinct concept
# In python wordnet data is loaded with NLTK.
# In the below will pass word die to wordnet and try to get the different unqiue sentences that wordnet have for die
# output : wordnet gave many different definations of die that include Verbs and nouns
wn.synsets("die")
# check noun related details
wn.synsets('die', pos=wn.NOUN)
# print all the definations of nouns
i =0
for syn in wn.synsets('die', pos=wn.NOUN):
  print("defination {0} : {1}".format(i, syn.definition()))
  i+=1
# print all the definations of verbs
i =0
for syn in wn.synsets('die', pos=wn.VERB):
  print("defination {0} : {1}".format(i, syn.definition()))
  i+=1


# Word-Sense Disambiguation with Lesk Algorithm
# input the sentence X i.e 'The die is cast.' and check if lesk is able to find the correct similar sentence.
print(X)
print(wsd.lesk(X.split(), 'die'))
print(wsd.lesk(X.split(), 'die').definition())
# # For the input sentence X,lesk have found a similar matching sentence the type of that is verb, and which is not correct. In the next will explicitly pass the part of speech (POS) and check the output
# with passing a POS we got the correct defiantion.
print(X)
wsd.lesk(X.split(), 'die', pos=wn.NOUN).definition()
# for sentence X i.e 'Roll the die to get a 6.' is again a noun.
print(Y)
wsd.lesk(Y.split(), 'die').definition()
# with passing a POS we got the correct defination.
wsd.lesk(Y.split(), 'die', pos=wn.NOUN).definition()
# Similar observations with sentance Z
print(Z)
wsd.lesk(Z.split(), 'die').definition()
wsd.lesk(Z.split(), 'die', pos=wn.VERB).definition()
```









