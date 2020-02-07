import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from collections import Counter
pd.options.display.max_rows = 4000
from sklearn.model_selection import train_test_split
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pre_train.py
import train.py

vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = vectorizer.fit_transform(train_title_genre_director['text'].values)
classifier = MultinomialNB()
targets = train_title_genre_director['profitable'].values
classifier.fit(counts, targets)

test = test_title_genre_director['text']
example_counts = vectorizer.transform(test)
predictions = classifier.predict(example_counts)

tested_data = test_title_genre_director.copy()

tested_data['prediction'] = predictions

tested_data['correct'] = np.where(tested_data['prediction'] == tested_data['profitable'], 1, 0)

true_positives = np.where((tested_data['prediction'] == 1) & (tested_data['profitable'] == 1), 1, 0)
true_negatives = np.where((tested_data['prediction'] == 0) & (tested_data['profitable'] == 0), 1, 0)
false_positives = np.where((tested_data['prediction'] == 1) & (tested_data['profitable'] == 0), 1, 0)
false_negatives = np.where((tested_data['prediction'] == 0) & (tested_data['profitable'] == 1), 1, 0)

acc = (true_positives.sum()+true_negatives.sum())/tested_data.count()
precision = true_positives.sum()/(true_positives.sum()+false_positives.sum())
recall = true_positives.sum()/(true_positives.sum()+false_negatives.sum())