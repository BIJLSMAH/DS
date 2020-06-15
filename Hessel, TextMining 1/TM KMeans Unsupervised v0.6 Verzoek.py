# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:18:04 2020

@author: H. Bijlsma
"""
def ask_user(questionyn):
    check = str(input(questionyn + " ? (y/n): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Verkeerde waarde ingevoerd !')
            return ask_user()
    except Exception as error:
        print("Voer een correcte waarde in AUB !")
        print(error)
        return ask_user()

# importeren van de benodigde packages
import os
import pandas as pd
import numpy as np

from yellowbrick.cluster import KElbowVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

#%% Laden dataset
if ask_user('Wilt u dataset laden'):
    print("Load dataset . . .")
    mapGebruikers = "C:\\Users\\"
    mapGebruiker = os.getlogin()
    mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
    hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
    os.chdir(hoofdmap)

    data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm')
    data = data[['Incidentnummer', 'Aanmelddatum', 'Korte omschrijving Details', 'verzoek_clean', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]

    # Filter de events eruit
    data=data.loc[data.index[data['Soort incident']!="Event"]]

    # Bepaal de frequenties per gevonden Object ID in de top 50
    df_freq_objectid = data.groupby('Object ID', sort=True).count().nlargest(50, columns=('Incidentnummer')).astype(np.uintc)['Incidentnummer']
    df_freq_inctype = data.groupby('Soort incident', sort=True).count().nlargest(1000, columns=('Incidentnummer')).astype(np.uintc)['Incidentnummer']

    # We hebben het nog steeds over erg veel data en erg veel intern geheugen dat 
    # moet worden gealloceerd. We zijn dus verplicht om met een steekproef verder te 
    # gaan

    data = data[data['Object ID'].isin(df_freq_objectid.index)]

    steekproefgrootte = len(data)
    chosen_idx = np.random.choice(len(data), replace=False, size=((len(data)>steekproefgrootte)*steekproefgrootte)+((len(data)<=steekproefgrootte)*len(data)))
    data = data.iloc[chosen_idx]
else:    
    print("Dataset niet opnieuw geladen !")
#%% Tekstbewerking, ruwe naar schone text
# KMeans zorgt voor het onderkennen groepen (unsupervised) in data.
# Helaas werkt KMeans alleen met cijfers en getallen. Daarom is het
# eerst noodzakelijk de teksten de vectorizeren en text mining uit te voeren.

if ask_user('Wilt u de textmining uitvoeren'):
    print('Textmining wordt uitgevoerd !')
    my_stopwords=list()
    f = open(r'data/stopwords_nl.txt', 'r')
    x = f.read().splitlines()
    f.close()
    for stopword in x:
        if stopword not in my_stopwords:
            my_stopwords.append(stopword)

    data = data[data.index.isin(data['verzoek_clean'].dropna().index)]

    documentstxt = data['verzoek_clean']
    documentstxtclean = list()
    for d in documentstxt:
        cleanw = ''
        res = d.split()
        for w in res:
            if (w.lower() not in my_stopwords) & (len(w)>2):
                cleanw = cleanw + ' ' + w.lower()
        documentstxtclean.append(cleanw.lower())

    documentslstall = list()
    for d in documentstxtclean:
        documentslstall.append(word_tokenize(d))

    documentslst = list()
    for x in documentslstall:
        documentslstelement = list()
        for y in range(0, len(x)):
            if (x[y] not in my_stopwords) & (len(x[y])>2):
                documentslstelement.append(x[y])
        documentslst.append(documentslstelement)

    documentstextclean = list()
    for x in documentslst:
        documentstxtelement = ''
        for y in range(0, len(x)):
            if (x[y] not in my_stopwords) & (len(x[y])>2):
                documentstxtelement = documentstxtelement + ' ' + x[y]
        documentstextclean.append(documentstxtelement)
else:
    print('Textmining niet uitgevoerd !')

#%%
# Instantiate the clustering model and visualizer
if ask_user('Bepalen optimaal # clusters met elbow'):
    print('Optimaal aantal clusters voor KMeans bepalen !')

    vectorizer = TfidfVectorizer(stop_words=my_stopwords,max_features=300)
    X = vectorizer.fit_transform(documentstxtclean)
    Xdf = pd.DataFrame(X.toarray())

    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=range(2, 51, 4), metric='calinski_harabasz', timings=False)

    visualizer.fit(Xdf)        # Fit the data to the visualizer
    visualizer.show()          # Finalize and render the figure
else:
    print('Optimaal aantal clusters voor KMeans niet bepaald !')

#%% Toevoegen bepaalde clusters aan incidentgegevens
# Resultaat van de elbow is een breekpunt bij 10 en tussen de 18 en 22

if ask_user('Toevoegen clusters aan data'):
    print('Clustergegevens worden aan de dataset toegevoegd!')
    sse = {}
    k = 10
    kmeans = KMeans(n_clusters=k, max_iter=50).fit(Xdf)
    data['clusters'] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
else:
    print('Clustergegevens zijn niet aan de dataset toegevoegd!')
#%% Wegschrijven woorden per cluster
if ask_user('Matrix woorden per cluster wegschrijven'):
    true_k = k # berekend in de stap hiervoor
    print('Top 15 woorden per cluster weggeschreven naar \'uitvoerclusters_verzoek_k_'+ str(true_k) + '.csv\'')

    f = open(r'data/uitvoerclusters_verzoek_k_'+ str(true_k) + '.csv', 'w')

    kmeans.fit(X)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        s = str(i)
        f.write("Cluster,%s" % s)
        for ind in order_centroids[i, :15]:
            f.write(',' + terms[ind] ),
        f.write("\n")
    f.close()
else:
    print('top 15 woorden per cluster niet weggeschreven!')
#%% Voorspelling van clusters op basis van KOD
print("\n")
print("Prediction")

# Nu gaan we op basis van teksten voorspellen in welk cluster deze valt.
vectorizer = TfidfVectorizer(stop_words=my_stopwords,max_features=300)
X = vectorizer.fit_transform(documentstxtclean)
Xdf = pd.DataFrame(X.toarray())

Y = vectorizer.transform(["printer probleem"])
prediction = kmeans.predict(Y)
print(prediction)

#%% Bepaal aantal Object ID per cluster t.b.v. Pivot
if ask_user('Maken input voor object id\'s per cluster'):
    true_k = k
    data['freq'] = 1 
    group_data = data.groupby(['clusters','Object ID'])['freq'].sum() #sum function
    group_data.reset_index()
    print('Object ID\'s per cluster wegschrijven naar \'uitvoer_object_ids_per_cluster_vrz_k_' + str(true_k) + '.csv\'')
    group_data.to_csv(r'data/uitvoerobject_ids_per_cluster_vrz_k_'+ str(true_k) + '.csv')

else:
    print('Geen verwerking voor Object ID\'s per cluster uitgevoerd !')
