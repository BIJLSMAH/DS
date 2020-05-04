from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import os
import pandas as pd
import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')
data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm', header=0, index_col=None, squeeze=True)
