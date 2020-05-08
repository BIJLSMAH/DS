# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:18:04 2020

@author: H. Bijlsma
"""
# Aan aan dataset hangen alle labels OBJECT ID
# Proberen van een aantal modelleringstechnieken waaruit op basis van
# de tekst van de omschrijving het OBJECT ID kan worden voorspeld.

# 1. Probleemdefinitie en oplossingsrichting
# 2. Input Data
# 3. Aanmaken van de initiele dataset
# 4. Eploratory Data Analysis
# 5. Feature Engineering
# 6. Predictive Models

# Ad 1 Problem Definition and solution approach
# As we have said, we are talking about a supervised learning problem. 
# This means we need a labeled dataset so the algorithms can learn the 
# patterns and correlations in the data. We fortunately have one available, 
# but in real life problems this is a critical step since we normally have 
# to do the task manually. Because, if we are able to automate the task of 
# labeling some data points, then why would we need a classification model?

# DATA PREPARATION
# importeren van de benodigde packages
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code for hiding seaborn warnings
import warnings
warnings.filterwarnings("ignore")

# load the dataset
os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')
# data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm', header=0, parse_dates=False, squeeze=True)
# data.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv')
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
# data_subset = data[1:10000]
# data_subset.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv')

# Nu eerst even werken met een subset.

df = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
df = df[['Incidentnummer', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]

df.head()
#%% DATA EXPLORATION

df2 = df.groupby('Object ID').count()
df2 = df2.nlargest(20, 'Incidentnummer')

# set font
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

df2 = df2.sort_values(by='Incidentnummer')

# we first need a numeric placeholder for the y axis
my_range=list(range(1,len(df2.index)+1))

fig, ax = plt.subplots(figsize=(10,5))

# create for each expense type an horizontal line that starts at x = 0 with the length 
# represented by the specific expense percentage value.
plt.hlines(y=my_range, xmin=0, xmax=df2['Incidentnummer'], color='#007ACC', alpha=0.2, linewidth=5)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df2['Incidentnummer'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)

# set labels
ax.set_xlabel('# VOORKOMENS', fontsize=12, fontweight='black', color = '#333F4B')
ax.set_ylabel('')

# set axis
ax.tick_params(axis='both', which='major', labelsize=8)
plt.yticks(my_range, df2.index)

# add an horizonal label for the y axis 
fig.text(-0.23, 0.96, 'OBJECT ID', fontsize=12, fontweight='black', color = '#333F4B')

# change the style of the axis spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

# set the spines position
ax.spines['bottom'].set_position(('axes', -0.04))
ax.spines['left'].set_position(('axes', 0.015))

# plt.savefig('hist2.png', dpi=300, bbox_inches='tight')
#%% DATA EXPLORATION

df['lengte'] = df['Korte omschrijving Details'].str.len()
# plot frequency
plt.figure(figsize=(12.8, 6))
sns.distplot(df['lengte']).set_title('Distributie Lengte Meldingstekst')

df['lengte'].describe()
# ga uit van de 95% en laat uitzonderingsgevallen erbuiten.
quantile_95 = df['lengte'].quantile(0.95)
df_95 = df[df['lengte'] < quantile_95]

plt.figure(figsize=(12.8, 6))
sns.distplot(df_95['lengte']).set_title('Distributie Lengte Meldingstekst')

plt.figure(figsize=(10, 5))
# Veel te veel OBJECT ID. Pak alleen de relevante er uit
df1 = df_95[df_95['Object ID'].isin(df2.index)]
chart =sns.boxplot(data=df1, x='Object ID', y='lengte', width=.5)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

#%% FEATURE ENGINEERING
# De volgende stap is om vanuit de ruwe tekst relevante eigenschappen te creeren.
# Feature engineering is het proces om data te transformeren naar eigenschappen 
# die als input kunnen dienen voor het trainen van modellen hierop.
# De stappen die we hierin nemen zijn:
# 1. Text cleaning and Preparation
# 2. Label Coding: Het aanmaken van een dictionary zodat iedere categorie een code krijgt.
# 3. Train-test split zodat het model kan worden getest op 'ongeziene' data.
# 4. Text representation. Gebruikt TF-IDF scores om de tekst te representeren.

# Text Representatie
# a.    Word Count Vectors
#       Iedere term is een kolom in de dataset. Per rij wordt de term frequency weeergegeven
# b.    TF-IDF Vectors
#       Is een score de relatieve zwaarte van een term in het document, maar ook in alle
#       documenten (corpus) weergeeft. TFIDF(t,d)=TF(t,d) * log(N/DF(t))
#           t       = term (word in document)
#           d       = document
#           TF(t)   = termfrequency in document
#           N       = Number of documents in corpus
#           DF(t)   = Number of documents in corpus containing the term t
#       De waarde voor TF-IDF neemt proportioneel toe hoe vaker een woord in een
#       document voorkomt en wordt verrekend met het aantal documenten
#       waarin het woord in de corpus voorkomt (log)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

# De dataset gaan we eerst laden
os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')
# Nu eerst even werken met een subset.

df = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
df = df[['Incidentnummer', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]
df['Korte omschrijving Details'][1]

# Vervangen sturingskarakters.
df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace("\r"," ")
df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace("\n"," ")
df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace("  "," ")
df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace('"','')

# Maak alles kleine letters

df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.lower()

# Verwijder punctuatie
punctiation_signs = list("?:!.,;")
for punct_sign in punctiation_signs:
    df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace(punct_sign,'')

# Stemming en Lemmatazation laten we even achterwege
# Geen gewenst effect op Nederlandse tekst op dit moment

# Stopwoorden verwijderen
# Laden stopwoorden
nltk.download('stopwords')
my_stopwords = stopwords.words('dutch')

f = open(r'data/stopwords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for stopword in x:
   if stopword not in my_stopwords:
       my_stopwords.append(stopword)

for stop_word in my_stopwords:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Korte omschrijving Details'] = df['Korte omschrijving Details'].str.replace(regex_stopword, '')

# Nu het splitsen van de gegevensverzameling in train en test

X_train, X_test, y_train, y_test = train_test_split(df['Korte omschrijving Details'],
                                                    df['Object ID'], 
                                                    test_size = 0.15,
                                                    random_state = 8)

# Text Representation
# Verschillende keuzes.
# 1. Count Vectors as features
# 2. TF-IDF Vectors as features
# 3. Word Embeddings as features
# 4. Text/NLP based features
# 5. Topic models as features

# Wij gaan voor optie 2.
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

df2 = df.groupby('Object ID').count()
df2 = df2.nlargest(20, 'Incidentnummer')

for objectid in sorted(df2.index):
    features_chi2 = chi2(features_train, labels_train == objectid)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' OBJECTID:".format(objectid))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

import pickle
# X_train
with open('data/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('data/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('data/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('data/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('data/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# df2
with open('data/df2.pickle', 'wb') as output:
    pickle.dump(df2, output)
    
# features_train
with open('data/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('data/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('data/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('data/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('data/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
#%% MODEL TRAINING
# Haal gegevens op
os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')

# Dataframe
path_df = "data/df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# Dataframe unique top ObjectID
path_df2 = "data/df2.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# features_train
path_features_train = "data/features_train.pickle"
with open(path_features_train, 'rb') as data:
    features_train = pickle.load(data)

# labels_train
path_labels_train = "data/labels_train.pickle"
with open(path_labels_train, 'rb') as data:
    labels_train = pickle.load(data)

# features_test
path_features_test = "data/features_test.pickle"
with open(path_features_test, 'rb') as data:
    features_test = pickle.load(data)

# labels_test
path_labels_test = "data/labels_test.pickle"
with open(path_labels_test, 'rb') as data:
    labels_test = pickle.load(data)

print(features_train.shape)
print(features_test.shape)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit

rf_0 = RandomForestClassifier(random_state = 8)

print('Parameters currently in use:\n')
pprint(rf_0.get_params())

# Aanpassen tune parameters

# n_estimators
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Perform Random Search
rfc = RandomForestClassifier(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=rfc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
random_search.fit(features_train, labels_train)


#sns.boxplot(data=df1, x='Object ID', y='lengte', width=.5)

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

