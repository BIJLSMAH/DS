# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:21:56 2020

@author: Administrator
"""


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

# Scikit learn
# Elbow methode ...
# sparse matrices 

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TextMining 1')
# data = pd.read_excel(r'data\Incidenten_SD_2018_2019_totaal.xlsm', header=0, parse_dates=False, squeeze=True)
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v2-subset2.csv', header=0, parse_dates=False, squeeze=True)
# data.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv')
# data = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
# data_subset = data[1:10000]
# data_subset.to_csv(r'data\Incidenten_SD_2018_2019_totaal-v3-subset.csv')

# Nu eerst even werken met een subset.

data = pd.read_csv(r'data\resultaat_SVM_SL_VRZ.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
data = data[['actual', 'prediction', 'VRZ', 'actual2', ]]

data.to_csv(r'data\resultaat_SVM_SL_VRZ2.csv')