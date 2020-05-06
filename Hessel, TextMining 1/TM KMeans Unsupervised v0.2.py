# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:18:04 2020

@author: H. Bijlsma
"""
# importeren van de benodigde packages
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
# load the dataset

# for i in range(0,861):
#     fileseq = '000' + str(i)
#     filename = r'data\Incidenten_SD_2018_2019_totaal-' + fileseq[-3:] + '.csv'
#     print(fileseq[-3:])
#     data = pd.read_csv(filename, header=0, index_col=0, parse_dates=True, squeeze=True)

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')
# data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm', header=0, parse_dates=False, squeeze=True)
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v2-subset2.csv', header=0, parse_dates=False, squeeze=True)
# data.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv')
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
# data_subset = data[1:10000]
# data_subset.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv')

# Nu eerst even werken met een subset.

data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
data = data[['Incidentnummer', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]

# KMeans zorgt voor het onderkennen groepen (unsupervised) in data.
# Helaas werkt KMeans alleen met cijfers en getallen. Daarom is het
# eerst noodzakelijk de teksten de vectorizeren.

documentstxt = data['Korte omschrijving Details']
documentslst = list()
for d in documentstxt:
    documentslst.append(word_tokenize(d))
nltk.download('stopwords')
my_stopwords = stopwords.words('dutch')

f = open(r'data/stopwords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for stopword in x:
   if stopword not in my_stopwords:
       my_stopwords.append(stopword)

vectorizer = TfidfVectorizer(stop_words=my_stopwords)
X = vectorizer.fit_transform(documentstxt)

df_freq_objectid = data.groupby("Object ID", sort=True).count().nlargest(20, columns=('Incidentnummer')).astype(np.uintc)['Incidentnummer']

true_k = len(df_freq_objectid)
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

# Nu gaan we op basis van teksten voorspellen in welk cluster deze valt.
Y = vectorizer.transform(["problemen met de sessie"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Mijn Outlook is ermee gestopt !"])
prediction = model.predict(Y)
print(prediction)

# from nltk.corpus import movie_reviews
# nltk.download('movie_reviews')

# Bepaald de woordfrequentie binnen alle documenten
all_words = list()
for x in documentslst:
    for y in x:
        all_words.append(y)
        
# Bepaal de meest gebruikte woorden.
all_words = nltk.FreqDist(w.lower() for w in all_words)
word_features = list(all_words)[:20]
def document_features(document):
    document_words = set()
    for d in document:
        document_words.add(d)
    features = {}
    for word in word_features:
        if word in document_words:
            features['bevat({})'.format(word)] = (word in document_words)
    return features

for i in range(0, len(documentslst)):
    print(document_features(documentslst[i]))
