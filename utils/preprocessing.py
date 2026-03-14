# utils/preprocessing.py

import re
import nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except LookupError:
        sentence_words = re.findall(r"\b\w+\b", sentence.lower())
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag
