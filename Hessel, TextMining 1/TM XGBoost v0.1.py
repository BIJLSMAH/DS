# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:46:33 2020

@author: Laptop
"""


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd
import xgboost, numpy, textblob, string
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

vbdata = open('data/corpus', encoding="utf-8").read()
vblabels, vbtexts = [], []
for i, line in enumerate(vbdata.split("\n")):
    content = line.split()
    vblabels.append(content[0])
    vbtexts.append(" ".join(content[1:]))

vbtrainDF = pd.DataFrame()
vbtrainDF['text'] = vbtexts
vbtrainDF['label'] = vblabels

# load the dataset incidenten
df = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
df = df[['Incidentnummer', 'Korte omschrijving Details', 'Verzoek', 'Soort binnenkomst', 'Soort incident', 'Categorie', 'Object ID']]
# replacing na values in OBJECT ID with Onbekend 
df['Object ID'].fillna("Onbekend", inplace = True) 

# inclabels, inctexts = [], []
# for objid in df['Object ID']:
#     inclabels.append(objid)

# for omschr in df['Korte omschrijving Details']:
#     inctexts.append(omschr)

inctrainDF = pd.DataFrame()
inctrainDF['text'] = df['Object ID']
inctrainDF['label'] = df['Korte omschrijving Details']

#%% OPDELEN IN TRAIN EN TEST
# split the dataset into training and validation datasets 
print("Split dataset in train- en testset . . .")

vbtrain_x, vbvalid_x, vbtrain_y, vbvalid_y = model_selection.train_test_split(vbtrainDF['text'], vbtrainDF['label'])

# label encode the target variable 
vbencoder = preprocessing.LabelEncoder()
vbtrain_y = vbencoder.fit_transform(vbtrain_y)
vbvalid_y = vbencoder.fit_transform(vbvalid_y)     

inctrain_x, incvalid_x, inctrain_y, incvalid_y = model_selection.train_test_split(inctrainDF['text'], inctrainDF['label'])

# label encode the target variable 
incencoder = preprocessing.LabelEncoder()
inctrain_y = incencoder.fit_transform(inctrain_y)
incvalid_y = incencoder.fit_transform(incvalid_y)

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
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# 2.2 TF-IDF
# word level tf-idf
print("2.2. FEATURE ENGINEERING: Generate TF-IDF Vectors . . .")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# 2.3 WORD EMBEDDINGS
print("2.3. FEATURE ENGINEERING: Word Embeddings Pretrained Vectors . . .")

# load the pre-trained word-embedding vectors 
embeddings_index = {}
# for i, line in enumerate(open('data/wiki-news-300d-1M.vec', encoding="utf-8")):
#     values = line.split()
#     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
print("2.3. FEATURE ENGINEERING: Word Embeddings Tokenize Vectors . . .")

token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 2.4 TEXT NLP BASED FEATURES
print("2.4. FEATURE ENGINEERING: TEXT NLP FEATURES . . .")

trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()])) 

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

# 2.5 TOPIC MODELLING
# train a LDA Model
print("2.5. FEATURE ENGINEERING: TOPIC MODELLING . . .")

lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    
#%% MODEL SELECTION
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