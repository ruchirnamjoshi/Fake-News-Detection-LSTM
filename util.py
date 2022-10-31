import pickle
import json
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras.preprocessing import text,sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Dropout

import warnings
warnings.filterwarnings('ignore')

import numpy as np
data = pd.read_csv('Processed_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state=0)
_data_columns = None
_model = None

from bs4 import BeautifulSoup
nltk.download("stopwords")
from nltk.corpus import stopwords


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)


# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Z]", " ", text)


# Removal of stopwords
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)

    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word)
            final_text.append(word)
    return " ".join(final_text)


# Total function
def cleaning(text):
    text = remove_html(text)
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text


def predict_fake_news(news):
    cleaning(news)
    news = [news]
    max_features = 10000
    maxlen = 300
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    tokenized_train = tokenizer.texts_to_sequences(news)
    news = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    ans = FND_model.predict(news) > 0.5



    return ans


def load_artifacts():
    global FND_model

    print('Loading Artifacts...')

    FND_model = pickle.load(open('Fake_News_detection.pickle', 'rb'))

    print('Artifacts...Loaded')


load_artifacts()