# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:18:04 2020

@author: H. Bijlsma
"""
# importeren van de benodigde packages
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

print("Load dataset . . .")
mapGebruikers = "C:\\Users\\"
mapGebruiker = os.getlogin()
mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
os.chdir(hoofdmap)

# for i in range(0,861):
#     fileseq = '000' + str(i)
#     filename = r'data\Incidenten_SD_2018_2019_totaal-' + fileseq[-3:] + '.csv'
#     print(fileseq[-3:])
#     data = pd.read_csv(filename, header=0, index_col=0, parse_dates=True, squeeze=True)

# Scikit learn
# Elbow methode ...
# sparse matrices 

# data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm', header=0, parse_dates=False, squeeze=True)
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v2-subset2.csv', header=0, parse_dates=False, squeeze=True)
# data.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv')
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
# data_subset = data[1:10000]
# data_subset.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv')

# Nu eerst even werken met een subset.

# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
data = data[['Incidentnummer', 'Aanmelddatum', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]
# Filter de events eruit
data=data.loc[data.index[data['Soort incident']!="Event"]]

df_freq_objectid = data.groupby('Object ID', sort=True).count().nlargest(50, columns=('Incidentnummer')).astype(np.uintc)['Incidentnummer']
df_freq_inctype = data.groupby('Soort incident', sort=True).count().nlargest(1000, columns=('Incidentnummer')).astype(np.uintc)['Incidentnummer']

# We hebben het nog steeds over erg veel data en erg veel intern geheugen dat 
# moet worden gealloceerd. We zijn dus verplicht om met een steekproef verder te 
# gaan

data = data[data['Object ID'].isin(df_freq_objectid.index)]

steekproefgrootte = 40000
chosen_idx = np.random.choice(len(data), replace=False, size=((len(data)>steekproefgrootte)*steekproefgrootte)+((len(data)<=steekproefgrootte)*len(data)))
data = data.iloc[chosen_idx]

# KMeans zorgt voor het onderkennen groepen (unsupervised) in data.
# Helaas werkt KMeans alleen met cijfers en getallen. Daarom is het
# eerst noodzakelijk de teksten de vectorizeren.

my_stopwords=list()
f = open(r'data/stopwords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for stopword in x:
   if stopword not in my_stopwords:
       my_stopwords.append(stopword)

documentstxt = data['Korte omschrijving Details']
documentstxtclean = list()
for d in documentstxt:
    cleanw = ''
    res = d.split()
    for w in res:
        if (w.lower() not in my_stopwords) & (len(w)>2):
            cleanw = cleanw + ' ' + w
    documentstxtclean.append(cleanw.lower())

documentslstall = list()
for d in documentstxtclean:
    documentslstall.append(word_tokenize(d))

documentslst = list()
for x in documentslstall:
    if x not in my_stopwords:
        documentslst.append(x)
        
vectorizer = TfidfVectorizer(stop_words=my_stopwords,max_features=300)
X = vectorizer.fit_transform(documentstxt)
Xdf = pd.DataFrame(X.toarray())

# Bepaal optimaal aantal clusters via elbow methode
def calculate_wcss(data):
    wcss = {}
    for n in range(2, 51, 4):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss[n] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
        print(n)
    return wcss


# X = pd.DataFrame(iris.data, columns=iris['feature_names'])
# print(X)
elbow = calculate_wcss(Xdf)
plt.figure()
plt.plot(list(elbow.keys()), list(elbow.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

# Bepaal clusters bij incidentengegevens
sse = {}
k = 20
kmeans = KMeans(n_clusters=k, max_iter=15).fit(Xdf)
data["clusters"] = kmeans.labels_
sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

print("Top terms per cluster:")

f = open(r'data/uitvoerclusters_korte_omschrijving_details.txt', 'w')

true_k = 20 # berekend in de stap hiervoor
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    s = str(i) + ' '
    f.write("Cluster %s" % s)
    for ind in order_centroids[i, :15]:
        f.write(terms[ind] + ','),
    f.write("\n")

f.close()

print("\n")
print("Prediction")

# Nu gaan we op basis van teksten voorspellen in welk cluster deze valt.
Y = vectorizer.transform(["problemen met de sessie"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Mijn Outlook is ermee gestopt !"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Kan mijn netwerkschijven niet meer vinden !"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Kan niet inloggen !"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Outlook hangt bij afdrukken !"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Wachtwoord is verlopen !"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["CW is verlopen !"])
prediction = model.predict(Y)
print(prediction)

# from nltk.corpus import movie_reviews
# nltk.download('movie_reviews')

# Bepaal de woordfrequentie binnen alle documenten
all_words = list()
for x in documentslst:
    for y in x:
        all_words.append(y)
        
# Bepaal de meest gebruikte woorden.
all_words = nltk.FreqDist(w.lower() for w in all_words)
word_features = list(all_words)[:100]
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

data['counter'] = 1
group_data = data.groupby(['clusters','Object ID'])['counter'].sum() #sum function

