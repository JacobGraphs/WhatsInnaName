import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from collections import Counter
import os
import io

# This pulls in the cleaned file, removed nulls etc.
final_set = pd.from_csv('/Users/jacobtryba/DSI/assignments/capstone2/data/cleaned_data.csv')

# This initiates a function to remove punctuation, and is then used by the three text columns
# to be used in the analysis - description, title, genre
punct_to_remove = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('','', punct_to_remove))
final_set['description'] = final_set['description'].apply(lambda text: remove_punctuation(text))
final_set['title'] = final_set['title'].apply(lambda text: remove_punctuation(text))
final_set['genre'] = final_set['genre'].apply(lambda text: remove_punctuation(text))

# This initiates a function to remove stop words and is then used on the description column
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])
final_set['description'] = final_set['description'].apply(lambda text: remove_stopwords(text))

# This creates my train and test dataframes, split to 70% train, 30% test
# I used a random seed associated with the year I met my wife
train, test = train_test_split(final_set, test_size =0.3, random_state=2015)

