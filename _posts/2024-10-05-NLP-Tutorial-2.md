---
title: NLP - Getting started [Part 2]
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2024-10-05 11:11:14 +0700
categories: [Tutorial]
tags: [NLP, Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# NLP - Getting started [Part 2]

## Bag of Words Model

Bag of Words (BoW) is a Natural Language
Processing strategy for converting a text document
into numbers that can be used by a computer
program.

This method involves converting text into a vector
based on the frequency of words in the text,
without considering the order or context of the
words.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-1.png?raw=true)

Example:

Social media platform that aims to analyze customer
reviews and understand the popularity of services among
users. This platform decides to employ the Bag of Words
method for processing customer reviews.

- Data Collection: The first step involves collecting and
storing customer reviews, which consist of text written
by customers about various services.
- Preprocessing: Text data is cleaned by removing
punctuation marks, numbers and unnecessary
whitespace.
- Bag of Words Model: A word list is created for BoW.
This list includes all the unique words in the dataset.
- Text Representation: Each customer review is
represented using the BoW method. The
frequency of each word is recorded within a vector
based on its position in the word list. For example,
the BoW representation for the phrase “great
service” could be as follows: [service: 1, great: 1,
other_words: 0].
- Analysis and Classification

```python
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')
vectorizer = CountVectorizer()

def preprocessing_text(text):
    """
    Function to preprocess text by performing lemmatization, removing stopwords,
    removing numbers, removing special characters, removing emojis, and converting
    text to lowercase.
    Args:
    text (str): Input text to be preprocessed.
    Returns:
    str: Preprocessed text.
    """
    lemmatizer = WordNetLemmatizer()

    emoji_pattern = r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]| [\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$'

    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(emoji_pattern, '', text)
    text= re.sub(r'\s+', ' ', text)
    text= text.lower().strip()
    return text

# Download 'punkt' tokenizer
nltk.download('punkt')
paragraph = """I am really disappointed this product. I would not use it again. It has really bad feature.
I love this product! It has some good features"""
sentences_list = nltk.sent_tokenize(paragraph)
corpus = [preprocessing_text(sentence) for sentence in sentences_list]
print(corpus)


X = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()

X_array = X.toarray()

print("Unique Word List: \n", feature_names)

print("BoW Matrix: \n", X_array)
```
Output:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-2.png?raw=true)

## TF-IDF for Feature Extraction

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-3.png?raw=true)

TF-IDF is a commonly used technique in Natural
Language Processing (NLP) to evaluate the
importance of a word in a document or corpus. It
works by assigning weights to words based on
their frequency. The term TF stands for term frequency, and the
term IDF stands for inverse document frequency.

- <b> Term Frequency (TF): </b> Term frequency refers to the frequency of a word
in a document. For a specified word, it is defined
as the ratio of the number of times a word appears
in a document to the total number of words in the
document.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-4.png?raw=true)

- <b> Inverse document frequency (IDF): </b> Inverse document frequency measures the
importance of the word in the corpus. It measures
how common a particular word is across all the
documents in the corpus.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-5.png?raw=true)

- <b> TF-IDF Score: </b> The TF-IDF score for a term in a document is
calculated by multiplying its TF and IDF values. This
score reflects how important the term is within the
context of the document and across the entire
corpus. Terms with higher TF-IDF scores are
considered more significant.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-6.png?raw=true)

Example:

- Preprocessing: The text data is preprocessed by
removing stop words, punctuation, and other non-
alphanumeric characters.
- Tokenization: The text is tokenized into individual
words.
- Instantiate TfidfVectorizer and fit the corpus
- Transform that corpus to get the representation

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# Sample corpus of documents
corpus = ['The quick brown fox jumps over the lazy dog.',
        'The lazy dog likes to sleep all day.',
        'The brown fox prefers to eat cheese.',
        'The red fox jumps over the brown fox.',
        'The brown dog chases the fox'
        ]

# Define a function to preprocess the text
def preprocess_text(text):
    # Remove punctuation and other non-alphanumeric characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # Join the words back into a string
    return ' '.join(words)

# Preprocess the corpus
corpus = [preprocess_text(doc) for doc in corpus]
print('Corpus: \n{}'.format(corpus))
# Create a TfidfVectorizer object and fit it to the preprocessed corpus
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
# Transform the preprocessed corpus into a TF-IDF matrix
tf_idf_matrix = vectorizer.transform(corpus)
# Get list of feature names that correspond to the columns in the TF-IDF matrix
print("Words:\n", vectorizer.get_feature_names_out())
# Print the resulting matrix
print("TF-IDF Matrix:\n",tf_idf_matrix.toarray())
```

Output:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/nlp/part-2-7.png?raw=true)