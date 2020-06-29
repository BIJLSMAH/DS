# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:46:33 2020

@author: Laptop
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

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
import numpy as np
import pandas as pd
import xgboost, numpy
from datetime import datetime
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os
import nltk
nltk.download('averaged_perceptron_tagger')

#%% DATASET PREPARATION
# load the dataset voorbeeld
print("Load dataset . . .")
mapGebruikers = "C:\\Users\\"
mapGebruiker = os.getlogin()
mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
os.chdir(hoofdmap)

# load the dataset incidenten
data = pd.read_excel(r'data\Topdesk Incidenten Totaal Overzicht 2019-2020.xlsx')
# data = pd.read_csv(r'data\Topdesk Incidenten Totaal Overzicht 2019-2020.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
# data = data[['Incidentnummer', 'Aanmelddatum', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]
data = data[['Incidentnummer', 'Datum aangemeld', 'verzoek_clean', 'Impact', 'Soort incident', 'Categorie', 'Subcategorie', 'Object']]
#   De volgende stap is nodig om de wijzigende kolomnamen te converteren voor een
#   eenduidige naamgeving in het rest van het script.
data.columns = ['Incidentnummer', 'Aanmelddatum', 'verzoek_clean', 'Impact', 'Soort incident','Categorie', 'Subcategorie', 'Object ID']

# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
df = data[['Incidentnummer', 'Aanmelddatum', 'verzoek_clean', 'Soort incident', 'Categorie', 'Object ID']]
# Filter de events eruit
df=df.loc[data.index[data['Soort incident']!="Event"]]
# replacing na values in OBJECT ID with Onbekend 
df['Object ID'].fillna("Onbekend", inplace = True) 
df['Aanmelddatum'] = df['Aanmelddatum'].apply(lambda x: x.date())
df = df.dropna(subset=['verzoek_clean'])

# Er zijn verzoekvelden zonder omschrijving. Deze gaan we eerst verwijderen.
# Voor 179967, na 179178

# Vanwege performance overwegingen, werken met een steekproef (sample),

# chosen_idx = np.random.choice(len(df), replace=False, size=int(len(df)/2))
# Er zijn objectid waar heel weinig incidenten betrekking op hebben. Deze 
# vervuilen de analyse. Daarom worden deze in onderstaande code gefilterd.

df2 = df.groupby('Object ID').count()
df2 = df2.nlargest(50, 'Incidentnummer')
df = df[df['Object ID'].isin(df2.index)]

steekproefgrootte = 500000
chosen_idx = np.random.choice(len(df), replace=False, size=((len(df)>steekproefgrootte)*steekproefgrootte)+((len(df)<=steekproefgrootte)*len(df)))
df = df.iloc[chosen_idx]
# inclabels, inctexts = [], []
# for objid in df['Object ID']:
#     inclabels.append(objid)

# for omschr in df['Korte omschrijving Details']:
#     inctexts.append(omschr)

print('Textmining wordt uitgevoerd !')
my_stopwords=list()
f = open(r'data/stopwords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for stopword in x:
    if stopword not in my_stopwords:
        my_stopwords.append(stopword)

df['VRZ'] = df['verzoek_clean'].str.lower().str.split()
df['VRZ'] = df['VRZ'].apply(lambda x: [item for item in x if item not in my_stopwords])
df['VRZ'] = df['VRZ'].apply(lambda x: [item for item in x if len(item)>2])
df['VRZ'] = df['VRZ'].apply(lambda x: ' '.join(map(str,x)))

#%% OPDELEN IN TRAIN EN TEST
# split the dataset into training and validation datasets 
print("Split dataset in train- en testset . . .")
manier = input('Op basis van: \n1. datum\n2. train_test_split functie\nGeef keuze: ')

if (manier=='1'): 
    # Er is voor gekozen om de laatste maand als testset te gebruiken en 
    # de anderhalf jaar daarvoor als trainingset
    # Verdelen van de train en testset
    # laatste maand is de testset

    datestr = '2020-05-23'
    dftrain = df[(df['Aanmelddatum'].astype(str) < datestr)]
    dftest = df[(df['Aanmelddatum'].astype(str) >= datestr)]
    
    inctrainDF = pd.DataFrame()
    inctrainDF['label'] = dftrain['Object ID']
    inctrainDF['text'] = dftrain['VRZ']
    
    inctestDF = pd.DataFrame()
    inctestDF['label'] = dftest['Object ID']
    inctestDF['text'] = dftest['VRZ']
    
    inctrain_x_original = inctrainDF['text']
    inctrain_y_original = inctrainDF['label']
    incvalid_x_original = inctestDF['text']
    incvalid_y_original = inctestDF['label']
    inctrain_x = inctrainDF['text']
    inctrain_y = inctrainDF['label']
    incvalid_x = inctestDF['text']
    incvalid_y = inctestDF['label']
   
    incencoder = preprocessing.LabelEncoder()
    inctrain_y = incencoder.fit_transform(inctrain_y)
    incvalid_y = incencoder.fit_transform(incvalid_y)
    
elif (manier=='2'):
    # split the dataset into training and validation datasets 
    print("Split dataset in train- en testset . . .")

    inctrain_x, incvalid_x, inctrain_y, incvalid_y = model_selection.train_test_split(inctrainDF['text'], inctrainDF['label'])
    inctrain_x_original = inctrain_x
    inctrain_y_original = inctrain_y
    incvalid_x_original = incvalid_x
    incvalid_y_original = incvalid_y
    
    incencoder = preprocessing.LabelEncoder()
    inctrain_y = incencoder.fit_transform(inctrain_y)
    incvalid_y = incencoder.fit_transform(incvalid_y)

    # label encode the target variable 
else:
    print('Verkeerde keuze, start opnieuw !')
    
#%% FEATURE ENGINEERING
# 2.1 COUNT VECTORS
# create a count vectorizer object 
print("2.1. FEATURE ENGINEERING: Generate Count Vectors . . .")
trainDF = inctrainDF
train_x = inctrain_x
valid_x = incvalid_x
train_y = inctrain_y
valid_y = incvalid_y
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_x)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# 2.2 TF-IDF
# word level tf-idf
print("2.2. FEATURE ENGINEERING: Generate TF-IDF Vectors . . .")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_x)
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_x)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_x)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# create a tokenizer 
print("2.3. FEATURE ENGINEERING: Word Embeddings Tokenize Vectors . . .")

token = text.Tokenizer()
token.fit_on_texts(train_x)
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=300)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=300)

# create token-embedding mapping
embedding_index={}
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 2.4 TEXT NLP BASED FEATURES
print("2.4. FEATURE ENGINEERING: TEXT NLP FEATURES . . .")
print ("2.4. Niet verder gebruikt, meer op Engelse taal gebaseerd")
# trainDF['char_count'] = trainDF['text'].apply(len)
# trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
# trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
# trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
# trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()])) 

# pos_family = {
#     'noun' : ['NN','NNS','NNP','NNPS'],
#     'pron' : ['PRP','PRP$','WP','WP$'],
#     'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
#     'adj' :  ['JJ','JJR','JJS'],
#     'adv' : ['RB','RBR','RBS','WRB']
# }

# function to check and get the part of speech tag count of a words in a given sentence
# def check_pos_tag(x, flag):
#     cnt = 0
#     try:
#         wiki = textblob.TextBlob(x)
#         for tup in wiki.tags:
#             ppo = list(tup)[1]
#             if ppo in pos_family[flag]:
#                 cnt += 1
#     except:
#         pass
#     return cnt

# trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
# trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
# trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
# trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
# trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

# 2.5 TOPIC MODELLING
# train a LDA Model
print("2.5. FEATURE ENGINEERING: TOPIC MODELLING . . .")
print ("2.5. Niet verder gebruikt")
# lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
# X_topics = lda_model.fit_transform(xtrain_count)
# topic_word = lda_model.components_ 
# vocab = count_vect.get_feature_names()

# view the topic models
# n_top_words = 15
# topic_summaries = []
# for i, topic_dist in enumerate(topic_word):
#    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
#    topic_summaries.append(' '.join(topic_words))
#%% MODEL BUILDING
# The final step in the text classification framework is to train a classifier 
# using the features created in the previous step. There are many different 
# choices of machine learning models which can be used to train a final model. 
# We will implement following different classifiers for this purpose:

# 1. Naive Bayes Classifier
# 2. Linear Classifier
# 3. Support Vector Machine
# 4. Bagging Models
# 5. Boosting Models
# 6. Shallow Neural Networks
# 7. Deep Neural Networks
# 8. Convolutional Neural Network (CNN)
# 9. Long Short Term Modelr (LSTM)
# 10. Gated Recurrent Unit (GRU)
# 11. Bidirectional RNN
# 12. Recurrent Convolutional Neural Network (RCNN)
# 13. Other Variants of Deep Neural Networks

# Lets implement these models and understand their details. The following 
# function is a utility function which can be used to train a model. 
# It accepts the classifier, feature_vector of training data, labels of 
# training data and feature vectors of valid data as inputs. Using 
# these inputs, the model is trained and accuracy score is computed.

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

#%% 3.1. NAIVE BAYES
# Implementing a naive bayes model using sklearn implementation with 
# different features Naive Bayes is a classification technique based on 
# Bayes’ Theorem with an assumption of independence among predictors. 
# A Naive Bayes classifier assumes that the presence of a particular feature 
# in a class is unrelated to the presence of any other feature.

# Naive Bayes on Count Vectors
f = open(r'data/SL_VRZ_NB.txt', 'w')

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: %.10f" % accuracy)
f.write("NB, Count Vectors: %.10f" % accuracy)
f.write('\n')

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF:  %.10f" % accuracy)
f.write("NB, WordLevel TF-IDF:  %.10f" % accuracy)
f.write('\n')

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors:  %.10f" % accuracy)
f.write("NB, N-Gram Vectors:  %.10f" % accuracy)
f.write('\n')

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: %.10f" % accuracy)
f.write("NB, CharLevel Vectors: %.10f" % accuracy)
f.write('\n')

f.close()

#%% 3.2. LINEAR CLASSIFIER
# Logistic regression measures the relationship between the categorical 
# dependent variable and one or more independent variables by estimating 
# probabilities using a logistic/sigmoid function. 

f = open(r'data/SL_VRZ_LR.txt', 'w')
# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(max_iter=4000), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: %.10f" % accuracy)
f.write("LR, Count Vectors: %.10f" % accuracy)
f.write('\n')

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(max_iter=4000), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF:  %.10f" % accuracy)
f.write("LR, WordLevel TF-IDF:  %.10f" % accuracy)
f.write('\n')

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(max_iter=4000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors:  %.10f" % accuracy)
f.write ("LR, N-Gram Vectors:  %.10f" % accuracy)
f.write('\n')

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(max_iter=4000), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: %.10f" % accuracy)
f.write("LR, CharLevel Vectors: %.10f" % accuracy)
f.close()

#%% 3.3. SVM 

f = open(r'data/SL_VRZ_SVM.txt', 'w')

# SVM on Ngram Level Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print ("SVM, Count Vectors: %.10f" % accuracy)
f.write ("SVM, Count Vectors: %.10f" % accuracy)
f.write('\n')

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("SVM, WordLevel TF-IDF:  %.10f" % accuracy)
f.write ("SVM, WordLevel TF-IDF:  %.10f" % accuracy)
f.write('\n')

# SVM on Ngram Level Ngram Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors:  %.10f" % accuracy)
f.write ("SVM, N-Gram Vectors:  %.10f" % accuracy)
f.write('\n')

# SVM on Ngram Level Charlevel Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("SVM, CharLevel Vectors: %.10f" % accuracy)
f.write ("SVM, CharLevel Vectors: %.10f" % accuracy)


f.close()
#%% 3.4 RF on Word Level TF IDF Vectors

f = open(r'data/SL_VRZ_RF.txt', 'w')
# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: %.10f" % accuracy)
f.write("RF, Count Vectors: %.10f" % accuracy)
f.write('\n')
# RF on Wordlevel TF-IDF
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: %.10f" % accuracy)
f.write("RF, WordLevel TF-IDF: %.10f" % accuracy)
f.write('\n')
# RF on Ngram Level Ngram vector
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, N-Gram Vectors: %.10f" % accuracy)
f.write("RF, N-Gram Vectors: %.10f" % accuracy)
f.write('\n')
# RF on Ngram Level Charlevel Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("RF, CharLevel Vectors: %.10f" % accuracy)
f.write("RF, CharLevel Vectors: %.10f" % accuracy)
f.close()

#%% Uitwerken model met de beste score
import _pickle as cPickle

if ask_user('Herberekenen model'):

#   train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    model=svm.SVC()
    model.fit(xtrain_tfidf_ngram_chars,train_y)
    # bewaren berekende optimale model
    with open(r'data/optimodel_SVM_SL_VRZ.pck', 'wb') as f:
        cPickle.dump(model, f)
else:
    print('Model wordt geladen uit picklebestand  . . .')    

with open(r'data/optimodel_SVM_SL_VRZ.pck', 'rb') as f:
    model=cPickle.load(f)

# predictions = model.predict_proba(xvalid_tfidf_ngram_chars)
predictions = model.predict(xvalid_tfidf_ngram_chars)
xtest = pd.DataFrame(xvalid_tfidf_ngram_chars.copy().toarray())
xtest['actual'] = incencoder.inverse_transform(valid_y)
xtest['prediction']=incencoder.inverse_transform(predictions)
xtest['VRZ'] = incvalid_x_original.reset_index().text
xtest['actual2'] = incvalid_y_original.reset_index().label

xtest.to_csv(r'data/resultaat_SVM_SL_VRZ.csv')
#%% 3.5. Boosting model
# Implementing Xtreme Gradient Boosting Model
# Boosting models are another type of ensemble models part of tree 
# based models. Boosting is a machine learning ensemble meta-algorithm 
# for primarily reducing bias, and also variance in supervised learning, 
# and a family of machine learning algorithms that convert weak learners to 
# strong ones. A weak learner is defined to be a classifier that is only 
# slightly correlated with the true classification (it can label examples 
# better than random guessing). 

# Extreme Gradient Boosting on Count Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print("Xgb, Count Vectors: %.10f" % accuracy)

# Extreme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("Xgb, Wordlevel TF-IDF: %.10f" % accuracy)

# Extreme Gradient Boosting on NGram Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y, xvalid_tfidf_ngram.tocsc())
print("Xgb, N-Gram Vectors: %.10f" % accuracy)

# Extreme Gradient Boosting on Character Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: %.10f" % accuracy)

#%% 4.1. CONVOLUTIONAL NEURAL NETWORKS

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((300, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)

print("CNN, Word Embeddings: %.10f" % accuracy)

#%% 4.2. RECURRENT NEURAL NETWORK (LSTM)

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers                                            
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-LSTM, Word Embeddings: %.10f" % accuracy)

#%% 4.2. RECURRENT NEURAL NETWORK (GRU = GRADIENT RECURRENT UNITS)
def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-GRU, Word Embeddings: %.10f" % accuracy)

#%% BIDIRECTIONAL RECURRENT NEURAL NETWORK (GRU)
def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-Bidirectional, Word Embeddings: %.10f" % accuracy)
