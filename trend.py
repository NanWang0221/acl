
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, os
import random
import pickle

from nltk.probability import FreqDist
from nltk.corpus import state_union

import re
import string
from nltk.stem import LancasterStemmer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from google.colab import files


from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Colab/data/wos.csv')
df['Abstract'] = df['Abstract'].astype(str)
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
no_n = lambda x: re.sub('\n', '', x)
no_r = lambda x: re.sub('\r', '', x)
no_hyphen = lambda x: re.sub('-', ' ', x)
df['Abstract'] = df.Abstract.map(alphanumeric).map(punc_lower).map(no_n).map(no_r).map(no_hyphen)
df['Split'] = df.Abstract.str.split()
stemmer = LancasterStemmer()
df['Stemmed'] = df['Split'].apply(lambda x: [stemmer.stem(y) for y in x])



dates = pd.date_range(start='2000', end='2026', freq='Y').year
dict_subjects = {
                 'repetition':['repeat', 'repetition', 'repetitions', 'repeated' ],
                 'talker variability':['recorder', 'speaker', 'recorders', 'speakers'],
                 'context variability':['context', 'contexts', 'contextual'],
                 'instruction': ['instruct', 'instruction', 'instructed'],
                 'presentation': ['interleaved','block', 'blocked', 'interleave'],
                 
                 'feedback':['feedback', 'correct', 'wrong'],
                 'modality': ['cross-modal', 'gestures', 'modal', 'modality', 'gesture'],
                 'learn mode': ['active', 'passive'],
                 
                 'duration':['intensity', 'duration', 'days'],
                 'consolidation':['sleep', 'consolidation'],
                 }



# Initialize a dictionary to store total frequency for each subject and each year
subject_frequency_by_year = {subject: [0] * len(dates) for subject in dict_subjects}
for subject, words in dict_subjects.items():
    for i, year in enumerate(dates):
        total_frequency = sum((' '.join(df.loc[df['Publication Year'] == year].Abstract.values)).count(word) for word in words)
        subject_frequency_by_year[subject][i] = total_frequency
total_frequency_by_year = [sum(subject_frequency_by_year[subject][i] for subject in dict_subjects)/10 for i in range(len(dates))]



# subject_frequency_by_year = {subject: [0] * len(dates) for subject in dict_subjects}
# for subject, words in dict_subjects.items():
#     for i, year in enumerate(dates):
#         # Extract and clean the abstracts for the specified year
#         abstracts = df.loc[df['Publication Year'] == year, 'Abstract'].fillna('').astype(str).values
#         total_frequency = sum((' '.join(abstracts)).count(word) for word in words)        
#         subject_frequency_by_year[subject][i] = total_frequency
# total_frequency_by_year = [
#     sum(subject_frequency_by_year[subject][i] for subject in dict_subjects) / 10 
#     for i in range(len(dates))
# ]




gradient_group_1 = sns.light_palette("green", n_colors=6)
gradient_group_1 = gradient_group_1[1:]  
gradient_group_2 = sns.light_palette("orange", n_colors=4)
gradient_group_2 = gradient_group_2[1:]  
gradient_group_3 = sns.light_palette("purple", n_colors=3)
gradient_group_3 = gradient_group_3[1:]  

line_colors = []
for idx, subject in enumerate(dict_subjects):
    idu =  idx + 1
    if subject in ['repetition', 'talker variability', 'context variability', 'modality', 'presentation']:
        line_colors.append(gradient_group_1[idu % len(gradient_group_1)])  
    elif subject in [  'feedback', 'instruction', 'learn mode']:
        line_colors.append(gradient_group_2[idu % len(gradient_group_2)])
    elif subject in [  'duration', 'consolidation']:
        line_colors.append(gradient_group_3[idu % len(gradient_group_3)])
bar_color = sns.color_palette("coolwarm", 1)[0]
plt.figure(figsize=(25, 8))
bars = plt.bar(dates, total_frequency_by_year, color=bar_color, alpha=0.4, label='Total Frequency/10')
for idx, (subject, frequencies) in enumerate(subject_frequency_by_year.items()):
    plt.plot(dates, frequencies, label=subject, marker='o', linestyle='-', color=line_colors[idx])

plt.title(' ')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.xticks(dates)
plt.legend()
plt.grid(True)
plt.show()


file_name = "subject_frequency_over_time.svg"
plt.savefig(file_name, format="svg", bbox_inches="tight")
plt.show()
files.download(file_name)

