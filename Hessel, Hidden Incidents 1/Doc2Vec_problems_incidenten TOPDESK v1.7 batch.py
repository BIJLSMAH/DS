# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:35:10 2020

@author: cimanbijlsmah

Laatste wijzigingen:
1. Dubbele waarden eruit.
2. Alle unieke parents eruit van niet gekoppelde incidenten.

"""
import sys
import os
import numpy as np
if len(sys.argv) >= 3:
    # er zijn parameters meegeleverd.
    # controleer of er 2 parameters zijn meegeleverd
    # 1. De naam van dit Pythonscript
    # 2. De naam van het te gebruiken invoerbestand Major Incidenten
    # 3. De naam van de originele map van waaruit het script moet worden
    #    gestart
    #    Controleer of het invoerbestand, opgegeven in de eerste parameter
    #    bestaat. Controleer ook of de mapnaam van waaruit het script moet 
    #    worden gestart ook bestaat.
    invoerbestandincidenten = sys.argv[1]
    invoerbestandproblemen = sys.argv[2]
    hoofdmap = sys.argv[3]
else:
    # er zijn geen parameters meegegeven, ga uit van de default waarden
    mapGebruikers = "C:\\Users\\"
    mapGebruiker = os.getlogin()
    mapTM = r'\Documents\GitHub\DS\Hessel, Hidden Incidents 1'
    hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
    invoerbestandincidenten = "%s%s%s%s%s" %(mapGebruikers, mapGebruiker, mapTM, '\\', r'data\Brongegevens\Incidenten en Gekoppelde Problemen.csv' )
    invoerbestandproblemen = "%s%s%s%s%s" %(mapGebruikers, mapGebruiker, mapTM, '\\', r'data\Brongegevens\Problemen.csv' )

if os.path.exists(hoofdmap):
    print('Hoofdmap is gevonden . . .')
else:
    print ('Hoofdmap is niet gevonden . . .')
    print ('Verwerking wordt afgebroken !')
    exit()

if os.path.isfile(invoerbestandincidenten):
    print('Invoerbestand incidenten is gevonden . . .')
else:
    print('Invoerbestand incidenten is niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

if os.path.isfile(invoerbestandproblemen):
    print('Invoerbestand problemen is gevonden . . .')
else:
    print('Invoerbestand problemen is niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

# importeren van resterende voor de analyse benodigde packages

import nltk
import pandas as pd
import pickle
import multiprocessing
import re, unicodedata
from unidecode import unidecode
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import chardet

import gensim
from gensim.models.doc2vec import Doc2Vec

# nltk.set_proxy('http://145.12.1.220:8080')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')

pd.options.mode.chained_assignment = None  # default='warn'

cores = multiprocessing.cpu_count()
os.chdir(hoofdmap)

#%% Functies voor voorbewerken tekst t.b.v. vectorisering

def opschonen_data(data, verzoekveld):
    # Verwijder vreemde tekens uit het uit verzoekveld
    # Verwijder einde regels uit verzoekveld
    # Verwijder return uit het verzoekveld
    # Verwijder tabs uit het verzoekveld
    
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(" :", ":", regex=True)
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(r"\n", " ", regex=True)
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(r"\r", " ", regex=True)
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(r"\t", " ", regex=True)
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(r"~"," ", regex=True)        
    return data

def verwijder_datum_naam(data, verzoekveld):
    data[verzoekveld] = data[verzoekveld]\
            .str.replace(r"^\d\d-\d\d-\d\d\d\d\s\d\d:\d\d .+?, .+?:",
                    "", regex=True)\
            .str.strip()
    return data

def apply_regex(incident, regex_expr, verzoekveld):
    if incident['Omschrijving_verzoek'] is not None:
        omschrijving = incident['Omschrijving_verzoek']
    else:
        # Detecteer de regex.
        zoekresultaat = re.search(regex_expr, incident[verzoekveld].lower())
        # Als de regex gevonden is, geef dan de eerste capture group terug
        omschrijving = zoekresultaat.group(1).strip() if zoekresultaat\
                is not None else None
    return omschrijving

def apply_regex_rest(incident,regex_expr,verzoekveld):
    # Als zowel Omschrijving_verzoek en -rest leeg is, dan originele Verzoek
    incverzoek = str(incident[verzoekveld])
    incomschrijving = str(incident['Omschrijving_verzoek'])
    incomschrijvingrest = str(incident['Omschrijving_verzoek_rest'])
#   print('a. '+ incverzoek[0:10] + '-----'+ incomschrijving[0:10] + '-----' + incomschrijvingrest[0:10])
    if (incomschrijving in (None,'','None')) and (incomschrijvingrest in (None,'','None')):
        # Bepaal het zoekresultaat uit het originele verzoek
#       print('a->. '+ incverzoek[0:10] + '-----'+ incomschrijving[0:10] + '-----' + incomschrijvingrest[0:10])
        zoekresultaat = re.search(regex_expr, incverzoek.lower())
        # Als de regex gevonden is, geef dan de eerste capture group terug
        # en anders het volledige zoekveld
        if zoekresultaat:
            if zoekresultaat.group(2):
 #              print('1. Fout ' + incverzoek.lower() + '|' + regex_expr)
                omschrijving = zoekresultaat.group(2).strip()
            else:
                omschrijving = incverzoek
        else:
            omschrijving = incverzoek
 #  Als rest gevuld, ga dan met rest aan de gang
    else: 
#       print('b. ' + incomschrijving + '-----' + incomschrijvingrest)
        if not (incomschrijvingrest in (None,'', 'None')):
            # Bepaal het zoekresultaat uit het bewerkte rest verzoek
            zoekresultaat = re.search(regex_expr, incomschrijvingrest.lower())
                
            # Als de regex gevonden is, geef dan de eerste capture group terug
            # en anders het volledige zoekveld-rest
            if zoekresultaat:
                if zoekresultaat.group(2):
#                   print('2. Fout ' + incomschrijvingrest.lower() + '|' + regex_expr)
                    omschrijving = zoekresultaat.group(2).strip()
                else:
                    omschrijving = incomschrijvingrest
            else:
                omschrijving = incomschrijvingrest
        else:
            # Bepaal het restveld uit het Omschrijving_verzoekveld       
            zoekresultaat = re.search(regex_expr, incomschrijving.lower())
            # Als de regex gevonden is, geef dan de eerste capture group terug
            # en anders het volledige zoekveld-rest
            if zoekresultaat:
                if zoekresultaat.group(2):
#                   print('3. Fout ' + incomschrijving.lower() + '|' + regex_expr)
                    omschrijving = zoekresultaat.group(2).strip()
                else:
                    omschrijving = incomschrijving
            else:
                omschrijving = incomschrijving
    return omschrijving

def construct_query(oplossing):
    oplossing_labels = [x for x in oplossing.index\
            if oplossing[x] is not None or not np.isnan(oplossing[x])]
    opl_query = ["`" + x + "` == @" + x.lower().replace(" ", "")
            for x in oplossing_labels]
    query_string = " & ".join(opl_query)
    return query_string

def detecteer_regex_incidenten(incidenten, oplossingen):
    if not 'Omschrijving_verzoek' in incidenten.columns:       
        incidenten.insert(incidenten.shape[1], 'Omschrijving_verzoek', None)
    if not 'Omschrijving_verzoek_rest' in incidenten.columns:       
        incidenten.insert(incidenten.shape[1], 'Omschrijving_verzoek_rest', None)

    oplossingen_selectie = oplossingen
    len_regex = oplossingen_selectie['Verzoek_regex'].str.len()\
            .sort_values(ascending=False).index
    oplossingen_selectie = oplossingen_selectie.reindex(len_regex)\
            .reset_index(drop=True)
    oplossingen_selectie = oplossingen_selectie.drop_duplicates()
    incidenten_selectie = incidenten

    for opl_idx, oplossing in oplossingen_selectie.iterrows():
        verzoek_reg = oplossing['Verzoek_regex']
        incidenten_selectie['Omschrijving_verzoek'] =\
                    incidenten_selectie\
                    .apply(lambda row: apply_regex(row, verzoek_reg, 'Verzoek (I)'), axis=1)
 
        incidenten.loc[incidenten_selectie.index, 'Omschrijving_verzoek'] =\
                incidenten_selectie['Omschrijving_verzoek']

    print("Incidenten met een standaardoplossing verwerkt, nu de restgroep")
    len_regex = oplossingen['Verzoek_regex'].str.len()\
            .sort_values(ascending=False).index
    oplossingen = oplossingen.reindex(len_regex).reset_index(drop=True)

    # Begin bovenaan in de lijst met oplossingen.
    rest_regex= list()
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)inlognaam[\s]*[bas]*[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*?:([\s\S]*)e[-]*?mail[\s\S]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)volledige[\s]*naam[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)\(graag volledige, 10-cijferige\)[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)applicatie[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)telefoonnummer[\s\S]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(vraag)[\s]*:([\s\S]*)')
    rest_regex.append('[\s\S]*(vraag)[\s\S]*:([\s\S]*)antwoord[\s\S]*')
    for verzoek_reg in rest_regex:
        incidenten['Omschrijving_verzoek_rest'] =\
                incidenten.apply(lambda row: apply_regex_rest(row, verzoek_reg,'Verzoek (I)'), axis = 1)
    return incidenten

def detecteer_regex_problemen(problemen, oplossingen):
    if not 'Omschrijving_verzoek' in problemen.columns:
        problemen.insert(problemen.shape[1], 'Omschrijving_verzoek', None)
    if not 'Omschrijving_verzoek_rest' in problemen.columns:
        problemen.insert(problemen.shape[1], 'Omschrijving_verzoek_rest', None)

    oplossingen_selectie=oplossingen
    len_regex = oplossingen_selectie['Verzoek_regex'].str.len()\
            .sort_values(ascending=False).index
    oplossingen_selectie = oplossingen_selectie.reindex(len_regex)\
            .reset_index(drop=True)
    oplossingen_selectie = oplossingen_selectie.drop_duplicates()
    problemen_selectie = problemen   
    for opl_idx, oplossing in oplossingen_selectie.iterrows():

        verzoek_reg = oplossing['Verzoek_regex']
        problemen_selectie['Omschrijving_verzoek'] =\
                    problemen_selectie\
                    .apply(lambda row: apply_regex(row, verzoek_reg, 'Verzoek'), axis=1)
 
        problemen.loc[problemen_selectie.index, 'Omschrijving_verzoek'] =\
                problemen_selectie['Omschrijving_verzoek']

    print("Problemen met een standaardoplossing verwerkt, nu de restgroep")

    len_regex = oplossingen['Verzoek_regex'].str.len()\
            .sort_values(ascending=False).index
    oplossingen = oplossingen.reindex(len_regex).reset_index(drop=True)

    # Begin bovenaan in de lijst met oplossingen.
    rest_regex= list()
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)inlognaam[\s]*[bas]*[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*?:([\s\S]*)e[-]*?mail[\s\S]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)volledige[\s]*naam[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)\(graag volledige, 10-cijferige\)[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)applicatie[\s]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(probleem|storing|foutmelding)[\s]*:([\s\S]*)telefoonnummer[\s\S]*:')
    rest_regex.append('[\s\S]*omschrijving[\s\S]*?(vraag)[\s]*:([\s\S]*)')
    rest_regex.append('[\s\S]*(vraag)[\s\S]*:([\s\S]*)antwoord[\s\S]*')

    for verzoek_reg in rest_regex:
        problemen['Omschrijving_verzoek_rest'] =\
                problemen.apply(lambda row: apply_regex_rest(row, verzoek_reg, 'Verzoek'), axis = 1)

    return problemen

def opschonen_oplossingen(oplossingen):
    # In het bestand van de oplossingen zijn er enkele mogelijkheden voor het
    # 'Verzoek'-veld. Slechts de eerste kolom heeft de label 'Verzoek'; de anderen
    # zijn ongelabeld. Deze zijn genoteerd als bijv "Unnamed: 23".
    # Houdt alleen de kolommen 'Binnenkomst', 'Soort incident', 'Categorie'
    # en 'Verzoek'
    print("Selecteer kolommen")
    keep_columns = ['Soort incident', 'Verzoek'] \
            + [x for x in oplossingen.columns.values if 'Unnamed:' in x]
    oplossingen = oplossingen[keep_columns]

    # Geef de kolommen rechts van 'Verzoek' de labels 
    # 'Verzoek_1' t/m 'Verzoek_n'.
    print("Hernoem 'Unnamed' kolommen")
    unnamed_columns = [x for x in oplossingen.columns.values if 'Unnamed:' in x]
    verzoek_columns = ['Verzoek_' + str(x) for x in range(1, len(unnamed_columns) + 1)]
    for k, unname in enumerate(unnamed_columns):
        oplossingen = oplossingen.rename(columns = {unname : verzoek_columns[k]})
  
    # Neem alleen de rijen waarin een Verzoekveld staat met 'omschrijving' staat
    verzoekvelden = oplossingen[['Verzoek'] + verzoek_columns]\
            .apply(lambda x: x.str.lower())
    verzoekvelden = verzoekvelden.apply(lambda x: x.str.contains('omschrijving'))
    bevat_omschrijving = verzoekvelden.apply(lambda row: row.any(), axis=1)
    bevat_omschrijving_idx = bevat_omschrijving[bevat_omschrijving].index
    oplossingen = oplossingen.iloc[bevat_omschrijving_idx, :]

    print("Maak varianten van Verzoek-oplossingen, verschillende regels")
    oplossingen_nieuw = oplossingen.copy()
    # Maak voor ieder mogelijk verzoekveld een regel in het bestand
    for m in range(oplossingen.shape[0]):
        row = oplossingen.iloc[[m]].copy()
        verzoek_indices = [x for x in row.columns.values if "Verzoek" in x]
        verzoek_entries = row[verzoek_indices].values[0]
        row = pd.concat([row]*len(verzoek_indices), ignore_index=True)
        row.loc[:, 'Verzoek'] = verzoek_entries
        oplossingen_nieuw = oplossingen_nieuw.append(row, ignore_index=True)

    oplossingen = oplossingen_nieuw
    keep_columns = [x for x in oplossingen.columns.values if "Verzoek_" not in x]
    oplossingen = oplossingen[keep_columns]
    oplossingen = oplossingen.drop_duplicates()
    oplossingen = oplossingen[oplossingen.Verzoek.notnull()]
    oplossingen.reset_index(drop=True, inplace=True)
    return oplossingen

def filter_janee(oplossingen):
    oplossingen['Verzoek'] = oplossingen['Verzoek'].str.replace(r'Ja/Nee', '')
    return oplossingen

def detecteer_omschrijvingzin(oplossingen):
    omschrijving_zinnen = [None] * oplossingen.shape[0]
    
    for m in range(oplossingen.shape[0]):
        verzoek_text = oplossingen.at[m, 'Verzoek']
        # Splits het sjabloon op '\n'.
        text_zinnen = verzoek_text.splitlines()
        zin_bool = [True if 'omschrijving' in zin.lower() else False 
                    for zin in text_zinnen]
        zin_idx = np.where(zin_bool)[0]
        if zin_idx.size > 0:
            omschrijving_zinnen[m] = zin_idx[0]
        else: 
            omschrijving_zinnen[m] = np.nan

    return omschrijving_zinnen

def genereer_regex(oplossingen):
    # Detecteer de zin met 'omschrijving' erin om te kijken waar de capture group moet komen
    omschrijvingszinnen = detecteer_omschrijvingzin(oplossingen)
    oplossingen = oplossingen.copy()
    oplossingen.insert(oplossingen.shape[1], 'Verzoek_regex', None)
    for m in range(oplossingen.shape[0]):
        verzoek_text = oplossingen.at[m, 'Verzoek']
        omschrijvingszin = omschrijvingszinnen[m]
        text_zinnen = verzoek_text.splitlines()
        regex_zinnen = text_zinnen
        regex_zinnen = [zin.strip() for zin in regex_zinnen]
        regex_zinnen = [zin.lower() for zin in regex_zinnen]
        regex_zinnen = [re.escape(zin) for zin in regex_zinnen]
        for i in range(len(regex_zinnen)):
            if i == omschrijvingszin:
                regex_zinnen[i] += "(.+?)"
            elif regex_zinnen[i] == "":
                pass
            else:
                regex_zinnen[i] += ".*?"
        if np.isnan(omschrijvingszin):
            regex_zinnen = ["(.+?)"] + regex_zinnen
        regex_zinnen = ["^"] + regex_zinnen + ["$"]
        regex_zinnen = [x for x in regex_zinnen if x != ""]
        regex_patroon = ''.join(regex_zinnen)
        oplossingen.loc[m, 'Verzoek_regex'] = regex_patroon

    # Sorteer op hoeveelheid regels in verzoekveld
    verzoek_lines = oplossingen['Verzoek'].apply(lambda x: len(x.splitlines()))
    verzoek_lines = verzoek_lines.sort_values(ascending=False).index
    oplossingen = oplossingen.reindex(verzoek_lines)
    oplossingen.reset_index(drop=True, inplace=True)
    return oplossingen

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
#   p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = ''
#           new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word.lower() in my_replacewords:
            new_words.append(my_replacewords[word])
        if word.lower() not in my_stopwords:
            new_words.append(word.lower())
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
#   Tokenize
    words = nltk.word_tokenize(sample)
    return words

print('Stopwoordenlijst opbouwen . . .')    
my_stopwords=list()
f = open(r'data/stopwords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for stopword in x:
    if stopword not in my_stopwords:
        my_stopwords.append(stopword)

print('Vervangende woordenlijst opbouwen . . .')    
my_replacewords={}
f = open(r'data/replacewords_nl.txt', 'r')
x = f.read().splitlines()
f.close()
for replaceline in x:
    replacewords = replaceline.split('=')
    if replacewords[0].strip() not in my_replacewords:
        my_replacewords[replacewords[0].strip()] = replacewords[1].strip()

#%% Inlezen benodigde bronbestanden m.b.t. incidenten en problemen
# In voorkomende gevallen worden de csv-bestanden in verschillende encoding
# formaten aangeleverd. Aangezien de encoding in de aanlevering kan variëren
# dient hiermee bij het inlezen van de csv rekening worden gehouden.

print('Inlezen brongegevens . . . ')

rawdata = open(invoerbestandincidenten, 'rb').read()
formaat = chardet.detect(rawdata)
csvencoding=formaat['encoding']

incidenten = pd.read_csv(invoerbestandincidenten, parse_dates=False, squeeze=True, low_memory=False, encoding=csvencoding, sep='\t')
incidenten.columns=['Incidentnummer', 'Korte omschrijving (Details) (I)', 'Verzoek (I)', 'DatumTijd aangemaakt (I)', 'Probleemnummer', 'DatumTijd aangemaakt (P)', 'Korte omschrijving (Details) (P)', 'Verzoek (P)', 'RK', 'Soort Incident']
print("Periode incidenten van %s - %s" % (str(incidenten['DatumTijd aangemaakt (I)'].min()), str(incidenten['DatumTijd aangemaakt (I)'].max())))
# Filter alle events eruit
incidenten[incidenten['Soort Incident']!='Event']
# Gebruik alleen de benodigde kolommen
incidenten=incidenten[['Incidentnummer', 'Korte omschrijving (Details) (I)', 'Verzoek (I)', 'Probleemnummer', 'Korte omschrijving (Details) (P)', 'Verzoek (P)', 'RK']]
          
rawdata = open(invoerbestandproblemen, 'rb').read()
formaat = chardet.detect(rawdata)
csvencoding=formaat['encoding']

problemen = pd.read_csv(invoerbestandproblemen, parse_dates=False, squeeze=True, low_memory=False, encoding=csvencoding, sep='\t')
problemen.columns=['Probleemnummer', 'DatumTijd aangemaakt', 'Korte omschrijving (Details)','Verzoek', 'Aantal']
print("Periode problemen van %s - %s" % (str(problemen['DatumTijd aangemaakt'].min()), str(problemen['DatumTijd aangemaakt'].max())))
problemen = problemen[['Probleemnummer', 'Korte omschrijving (Details)', 'Verzoek']]

#   Alle incidenten met woa (workarounds) zijn niet meer verborgen.  
#   Interessant om te kijken of er incidenten met workarounds zijn, maar die
#   nog niet zijn gekoppeld aan een probleem of major incident.   

print('Inlezen brongegevens afgerond . . . ')

#%% Voorbewerken Sanering Verzoekveld op basis van Standaardoplossingen

print('Laden Standaardoplossingen . . .')

oplossingen = pd.read_excel(r'data/Verzoek/Standaardoplossingen42.xlsx')
oplossingen = opschonen_oplossingen(oplossingen)
# Filter 'Ja/Nee' uit de oplossingen om een meer uniforme aanpak voor de regex te creeeren
oplossingen = filter_janee(oplossingen)
oplossingen = genereer_regex(oplossingen)
oplossingen = oplossingen[['Verzoek', 'Verzoek_regex']]

oplossingen.to_csv("data/Standaardoplossingen_processed_regex_16.csv", 
        index = False, encoding="UTF-8")

print('Conversie van Standaardoplossingen naar Regex gereed . . .')

#%% Converteren verzoekveld, toevoegen aan de bestaande gegevensverzamelingen
#   incidenten en problemen

print("Sanering verzoekveld incidenten . . .")

# In eerste instantie het verzoekveld opschonen (vreemde tekens ed.)
incidenten = opschonen_data(incidenten, 'Verzoek (I)')
incidenten = verwijder_datum_naam(incidenten, 'Verzoek (I)')

# Dan nu de standaardsjablonen toepassen en de resterende regex_rest
# De gegenereerde regexen in standaardoplossingen_processed_regext.csv nu inlezen
oplossingen = pd.read_csv(r'data/Standaardoplossingen_processed_regex_16.csv', sep = ",", low_memory=False, encoding='UTF-8')

# Dan nu de standaardsjablonen toepassen 
incidenten = detecteer_regex_incidenten(incidenten, oplossingen)

clean_incidenten = list() 
for d in incidenten['Omschrijving_verzoek_rest']:
    words = d.split()
    sentence = ""
    for w in words:
        w = unidecode(w)
        sentence = sentence + ' ' + str(w)
    clean_incidenten.append(sentence)
incidenten['Omschrijving_verzoek_rest'] = clean_incidenten

# Onderstaande code voor testdoeleinden om de verhouding tussen standaardoplossingen en vrij gekozen 
# meldingen weer te geven.
# print(incidenten['Omschrijving_verzoek'].apply(lambda x: True if x is not None else False).sum())
# print(incidenten['Omschrijving_verzoek_rest'].apply(lambda x: True if x is not None else False).sum())

# dan nu het verzoekveld onderwerpen aan stopwoorden en tekstvervangingen.

print('Tekst Verzoek schonen incidenten. . . ')    
incidenten['LSTVRZ'] = incidenten['Omschrijving_verzoek_rest']
incidenten['LSTVRZ'] = incidenten['LSTVRZ'].dropna()    
incidenten['LSTVRZ'] = incidenten['LSTVRZ'].apply(lambda x: normalize(str(x).split()) )
incidenten['LSTVRZ'].apply(lambda x: preprocess(' '.join(x)))
incidenten['VRZ'] = incidenten['LSTVRZ'].apply(lambda x: ' '.join(x))

# Verwijder overbodige kolommen.
# Alleen het originele verzoekveld en het eindresultaat moet blijven bewaard
# incidenten = incidenten[['Incidentnummer', 'Korte omschrijving (Details)', 'Verzoek', 'LSTVRZ', 'VRZ']]

incidenten.to_csv(r'data/Incidenten Processed Export - Omschrijvingen_16.csv', index = False, encoding="UTF-8")
print('CSV Incidenten gesaneerd verzoekveld aangemaakt . . .')

# PROBLEMEN

print("Sanering verzoekveld problemen . . .")

# In eerste instantie het verzoekveld opschonen (vreemde tekens ed.)
problemen  = opschonen_data(problemen, 'Verzoek')
problemen = verwijder_datum_naam(problemen, 'Verzoek')

# Dan nu de standaardsjablonen toepassen 
problemen = detecteer_regex_problemen(problemen, oplossingen)

clean_problemen = list() 
for d in problemen['Omschrijving_verzoek_rest']:
    words = d.split()
    sentence = ""
    for w in words:
        w = unidecode(w)
        sentence = sentence + ' ' + str(w)
    clean_problemen.append(sentence)
problemen['Omschrijving_verzoek_rest'] = clean_problemen

# dan nu het verzoekveld onderwerpen aan stopwoorden en tekstvervangingen.

print('Tekst Verzoek schonen problemen. . . ')    
problemen['LSTVRZ'] = problemen['Omschrijving_verzoek_rest']
problemen['LSTVRZ'] = problemen['LSTVRZ'].dropna()    
problemen['LSTVRZ'] = problemen['LSTVRZ'].apply(lambda x: normalize(str(x).split()) )
problemen['LSTVRZ'].apply(lambda x: preprocess(' '.join(x)))
problemen['VRZ'] = problemen['LSTVRZ'].apply(lambda x: ' '.join(x))

# Verwijder overbodige kolommen.
# Alleen het originele verzoekveld en het eindresultaat moet blijven bewaard
problemen = problemen[['Probleemnummer', 'Korte omschrijving (Details)', 'Verzoek', 'LSTVRZ', 'VRZ']]

problemen.to_csv(r'data/Problemen Processed Export - Omschrijvingen_16.csv', index = False, encoding="UTF-8")
print('CSV Problemen gesaneerd verzoekveld aangemaakt . . .')
#%% Handel af Korte omschrijving (Details)
print('Voorbewerking data t.b.v. vectorisering, Korte omschrijving (Details)')

#   incidenten zonder omschrijving verwijderen

incidenten = incidenten[incidenten['Korte omschrijving (Details) (I)'].notna()]
incidenten.reindex()

#   problemen zonder omschrijving verwijderen    

problemen = problemen[problemen['Korte omschrijving (Details)'].notna()]
problemen.reindex()

print('Korte omschrijving (Details) schonen incidenten . . . ')    
incidenten['LSTKOD'] = incidenten['Korte omschrijving (Details) (I)']
incidenten['LSTKOD'] = incidenten['LSTKOD'].dropna()    
incidenten['LSTKOD'] = incidenten['LSTKOD'].apply(lambda x: normalize(str(x).split()) )
incidenten['LSTKOD'].apply(lambda x: preprocess(' '.join(x)))
incidenten['KOD'] = incidenten['LSTKOD'].apply(lambda x: ' '.join(x))

print('Korte omschrijving (Probleem) schonen problemen . . . ')    
problemen['LSTKOD'] = problemen['Korte omschrijving (Details)']
problemen['LSTKOD'] = problemen['LSTKOD'].dropna()    
problemen['LSTKOD'] = problemen['LSTKOD'].apply(lambda x: normalize(str(x).split()) )
problemen['LSTKOD'].apply(lambda x: preprocess(' '.join(x)))
problemen['KOD'] = problemen['LSTKOD'].apply(lambda x: ' '.join(x))
 
print('Eindresultaat incidenten vastleggen in pickle bestand . . .')
f = open('pckl_df_incidenten_16.pkl', 'wb')
pickle.dump(incidenten, f)
f.close()
print('Eindresultaat problemen vastleggen in pickle bestand . . .')
f = open('pckl_df_problemen_16.pkl', 'wb')
pickle.dump(problemen, f)
f.close()
print('Gereed !')
  
   
#%% Trainen DOC2VECmodel KOD Incidenten en Problemen

# Deze cel is tijdrovend en moet in het kader van de herstartbaarheid
# controleren of een model al gegenereerd is. Dit kan door in de naam 
# van het gegenereerde pickle bestand (model) te verwijzen naar de naam
# van het bronbestand en ook het tijdstip van de laatste update van dat 
# bestand.

# Is er een pickle bestand met betrekking tot de bewerkte gegevens ?
# 

if os.path.isfile('pckl_df_incidenten_16.pkl'):
    print('Bewerkte brongegevens incidenten ophalen . . .')
    f = open('pckl_df_incidenten_16.pkl', 'rb')
    incidenten = pickle.load(f)
    f.close()
else:
    print('Bewerkte brongegevens incidenten niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

if os.path.isfile('pckl_df_problemen_16.pkl'):
    print('Bewerkte brongegevens problemen ophalen . . .')
    f = open('pckl_df_problemen_16.pkl', 'rb')
    problemen = pickle.load(f)
    f.close()
else:
    print('Bewerkte brongegevens problemen niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

 
print('Start training modellen KOD . . .')

lstKOD = []
lstKODID = []

# Vastleggen benodigde gegevens voor DOC2VEC en ook terugvinden van
# de originele gegevens.
# Maak een vocabulaire met alle KOD, zowel uit incidenten als problemen

# Incidenten
for index, row in incidenten.iterrows():
    lstKOD.append(row['LSTKOD'])
    lstKODID.append(row['Incidentnummer'])

# Problemen
for index, row in problemen.iterrows():
    lstKOD.append(row['LSTKOD'])
    lstKODID.append(row['Probleemnummer'])

#   Vastleggen LSTKOD en LSTKODID als input voor training model
print('LSTKOD vastleggen in pickle bestand . . .')
f = open('pckl_LSTKOD_16.pkl', 'wb')
pickle.dump(lstKOD, f)
f.close()
print('LSTKODID vastleggen in pickle bestand . . .')
f = open('pckl_LSTKODID_16.pkl', 'wb')
pickle.dump(lstKODID, f)
f.close()
 
# woa = workarounds zijn niet verborgen, maar zijn al aan een probleem toegekend

labeled_incidenten_KOD=[]
labeled_incidenten_KOD=[gensim.models.doc2vec.TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(lstKOD)]

max_epochs = 100
vec_size = 32
alpha = 0.025

# Note: dm defines the training algorithm. If dm=1 means ‘distributed memory’ 
# (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). 
# Distributed Memory model preserves the word order in a document whereas 
# Distributed Bag of words just uses the bag of words approach, 
# which doesn’t preserve any word order.

# Volgorde woorden van belang dm=1
# model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=1, epochs=max_epochs, alpha=alpha, min_alpha=0.00025, window=2, dm=1, workers = 4)
# Volgorde woorden niet van belang dm=0
model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=1, epochs=max_epochs, alpha=alpha, min_alpha=0.00025, window=2, dm=0, workers=cores )

model.build_vocab(labeled_incidenten_KOD) 
model.train(labeled_incidenten_KOD, total_examples=model.corpus_count, epochs=model.epochs)

for epoch in range(2):
     print('iteration {0}'.format(epoch))
     model.train(labeled_incidenten_KOD, 
                 total_examples=model.corpus_count,
                 epochs=model.epochs)
#    decrease the learning rate
     model.alpha -= 0.0002
#    fix the learning rate, no decay
     model.min_alpha = model.alpha

model.save("KOD DOC2VEC PROBLEMS 6 MAAND_16.model")

print("Model KOD Bewaard . . . ")

#%% Trainen DOC2VECmodel VRZ Incidenten en Problemen

# Deze cel is tijdrovend en moet in het kader van de herstartbaarheid
# controleren of een model al gegenereerd is. Dit kan door in de naam 
# van het gegenereerde pickle bestand (model) te verwijzen naar de naam
# van het bronbestand en ook het tijdstip van de laatste update van dat 
# bestand.

# Is er een pickle bestand met betrekking tot de bewerkte gegevens ?
# 

if os.path.isfile('pckl_df_incidenten_16.pkl'):
    print('Bewerkte brongegevens incidenten ophalen . . .')
    f = open('pckl_df_incidenten_16.pkl', 'rb')
    incidenten = pickle.load(f)
    f.close()
else:
    print('Bewerkte brongegevens incidenten niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

if os.path.isfile('pckl_df_problemen_16.pkl'):
    print('Bewerkte brongegevens problemen ophalen . . .')
    f = open('pckl_df_problemen_16.pkl', 'rb')
    problemen = pickle.load(f)
    f.close()
else:
    print('Bewerkte brongegevens problemen niet gevonden . . . ')
    print('Verwerking wordt afgebroken !')
    exit()

print('Start training modellen VRZ . . .')

lstVRZ = []
lstVRZID = []

# Vastleggen benodigde gegevens voor DOC2VEC en ook terugvinden van
# de originele gegevens.
# Maak een vocabulaire met alle KOD, zowel uit incidenten als problemen

# Incidenten
for index, row in incidenten.iterrows():
    lstVRZ.append(row['LSTVRZ'])
    lstVRZID.append(row['Incidentnummer'])
# Problemen
for index, row in problemen.iterrows():
    lstVRZ.append(row['LSTVRZ'])
    lstVRZID.append(row['Probleemnummer'])

#   Vastleggen LSTKOD en LSTKODID als input voor training model
print('LSTVRZ vastleggen in pickle bestand . . .')
f = open('pckl_LSTVRZ_16.pkl', 'wb')
pickle.dump(lstVRZ, f)
f.close()
print('LSTVRZID vastleggen in pickle bestand . . .')
f = open('pckl_LSTVRZID_16.pkl', 'wb')
pickle.dump(lstVRZID, f)
f.close()

# woa = workarounds zijn niet verborgen, maar zijn al aan een probleem toegekend

labeled_incidenten_VRZ=[]
labeled_incidenten_VRZ=[gensim.models.doc2vec.TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(lstVRZ)]

max_epochs = 100
vec_size = 50
alpha = 0.025

# Note: dm defines the training algorithm. If dm=1 means ‘distributed memory’ 
# (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). 
# Distributed Memory model preserves the word order in a document whereas 
# Distributed Bag of words just uses the bag of words approach, 
# which doesn’t preserve any word order.

# Volgorde woorden van belang dm=1
# model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=1, epochs=max_epochs, alpha=alpha, min_alpha=0.00025, window=2, dm=1, workers = 4)
# Volgorde woorden niet van belang dm=0
model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=1, epochs=max_epochs, alpha=alpha, min_alpha=0.00025, window=2, dm=0, workers=cores )

model.build_vocab(labeled_incidenten_VRZ) 
model.train(labeled_incidenten_VRZ, total_examples=model.corpus_count, epochs=model.epochs)

for epoch in range(2):
    print('iteration {0}'.format(epoch))
    model.train(labeled_incidenten_VRZ, 
                total_examples=model.corpus_count,
                epochs=model.epochs)
#   decrease the learning rate
    model.alpha -= 0.0002
#   fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("VRZ DOC2VEC PROBLEMS 6 MAAND_16.model")

print("Model VRZ Bewaard . . . ")
#%% Model is gereed, gebruik deze nu voor de voorspellingen
# haal de gegevens op uit pickle bestand

# Schoon eerst de variabelen / ruimte
incidenten = None
problemen = None
clean_incidenten = None
clean_problemen = None

print('Haal incidenten uit pickle bestand . . .')
f = open('pckl_df_incidenten_16.pkl', 'rb')
incidenten = pickle.load(f)
f.close()

print('Haal problemen uit pickle bestand . . . ')
f = open('pckl_df_problemen_16.pkl', 'rb')
problemen = pickle.load(f)
f.close()

print('LSTVRZ ophalen uit pickle bestand . . .')
f = open('pckl_LSTVRZ_16.pkl', 'rb')
lstVRZ = pickle.load(f)
f.close()

print('LSTVRZID ophalen uit pickle bestand . . .')
f = open('pckl_LSTVRZID_16.pkl', 'rb')
lstVRZID = pickle.load(f)
f.close()

print('Laad het KOD model . . . ')
modelKOD = Doc2Vec.load("KOD DOC2VEC PROBLEMS 6 MAAND_16.model")

print('Laad het VRZ model . . . ')
modelVRZ = Doc2Vec.load("VRZ DOC2VEC PROBLEMS 6 MAAND_16.model")

# We hebben de basis voor de analyse.
# Hierbij gaan we ervanuit dat de incidenten en de problemen die 
# zijn aangeleverd betrekking hebben op de periode van het halfjaar
# Gegevensverzamelingen hebben nu de vorm
#   incidenten
#   1. Incidentnummer
#   2. Korte omschrijving (Details)
#   3. Verzoek
#   4. LSTVRZ
#   5. VRZ
#   6. LSTKOD
#   7. KOD
#   8. Gekoppeld Probleemnummer
#   9. Wijze van Koppeling
#
#   problemen
#   1. Probleemnummer
#   2. Korte omschrijving (Details)
#   3. Verzoek
#   4. LSTVRZ
#   5. VRZ
#   6. LSTKOD
#   7. KOD

# Doorloop nu de problemen en bekijk de overeenkomsten

# Schrijf kopgegevens
# f.write("ProbleemNummer|Probleemmelddatum|Probleem_KOD_Origineel|Problem_KOD|RelatedIncident_ID|RelatedIncident_Melddatum|RelatedIncident_KOD_origineel|RelatedIncident_KOD|RelatedIncident_PROBLEM|RelatedIncident_SIM|Inclevel1|Inclevel2|Inclevel3|Via|ProbleemNummers")

# Tijdens de verwerking gaan we aan incidenten nog aanvullende kolommen toevoegen
# namelijk:
#   10. Overeenkomstscore o.b.v. KOD
#   11. Overeenkomstscore o.b.v. VRZ
#   12. Overeenkomend Level1
#   13. Overeenkomend Level2
#   14. Overeenkomend Level3

#   Maak eerst even een steekproef voor testdoeleinden

f = open(r'data/uitvoer_similarity_doc2vec_topdesk_problems_incidenten_16.csv', 'w')

f.write("Probleemnummer~Korte omschrijving Details (P)~Verzoek (P)~Gerelateerd Incidentnummer~Korte omschrijving Details (I)~Verzoek (I)~Overeenkomstscore KOD~Overeenkomstscore VRZ~Type koppeling~Inclevel1~Inclevel2~Inclevel3~Probleem KOD~Probleem VRZ~Incident KOD~Incident VRZ")
f.write("\n")

lstVerwerkteIncidenten = []

#   STAP 1.
#   Bepaal de overeenkomst van de probleembeschrijving en de incidentbeschrijving
#   van problemen en incidenten die niet gekoppeld zijn.
#   Bepaal de niet gekoppelde incidenten bij een probleem en doorloop deze incidenten    

maskNietGekoppeldeIncidenten=(incidenten['Probleemnummer'].isna()) 
NietGekoppeldeIncidenten = incidenten.loc[maskNietGekoppeldeIncidenten]
dictNietGekoppeldeIncidenten = NietGekoppeldeIncidenten.set_index('Incidentnummer').to_dict(orient='index')
dictProblemen = problemen.set_index('Probleemnummer').to_dict(orient='index')

voortgangsteller = 0
maxIncidenten = len(incidenten)
maxvoortgangsteller = len(problemen)

for x in dictProblemen.keys():
# for index, probleem in problemen.head(10).iterrows():
    probleem = dictProblemen[x]
    print('....' + x + ' (' + str(voortgangsteller) + ' van ' + str(maxvoortgangsteller) + ')')
    voortgangsteller+=1
#     if voortgangsteller>10:
#        break

#   Zoek het specifieke probleem erbij
    
    strProbleemOriginalKOD = probleem['Korte omschrijving (Details)']
    strProbleemOriginalVRZ = probleem['Verzoek']

    cleanProbleemKOD = list(filter(lambda x: x in modelKOD.wv.vocab, probleem['LSTKOD']))
    cleanProbleemVRZ = list(filter(lambda x: x in modelVRZ.wv.vocab, probleem['LSTVRZ']))
    
#   for index, NietGekoppeldIncident in NietGekoppeldeIncidenten.iterrows():
    for y in dictNietGekoppeldeIncidenten.keys():
        NietGekoppeldIncident = dictNietGekoppeldeIncidenten[y]
#       Bepaal overeenkomstscores
        if y not in lstVerwerkteIncidenten:
            scoreKOD = 0    
            if cleanProbleemKOD != [] and NietGekoppeldIncident['LSTKOD'] != []:
                scoreKOD = modelKOD.wv.n_similarity(cleanProbleemKOD, NietGekoppeldIncident['LSTKOD'])
            scoreVRZ = 0    
            if cleanProbleemVRZ != [] and NietGekoppeldIncident['LSTVRZ'] != []:
                scoreVRZ = modelVRZ.wv.n_similarity(cleanProbleemVRZ, NietGekoppeldIncident['LSTVRZ'])
                if (scoreKOD > 0.75) or (scoreVRZ > 0.75):
                    f.write("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (x,  \
                                strProbleemOriginalKOD.encode("ascii", "ignore").decode(), \
                                strProbleemOriginalVRZ.encode("ascii", "ignore").decode(), \
                                y, \
                                NietGekoppeldIncident['Korte omschrijving (Details) (I)'].encode("ascii", "ignore").decode(), \
                                NietGekoppeldIncident['Verzoek (I)'].encode("ascii", "ignore").decode(), \
                                scoreKOD,  \
                                scoreVRZ, \
                                NietGekoppeldIncident['RK'],\
                                x, \
                                y, \
                                y, \
                                probleem['KOD'], \
                                probleem['VRZ'], \
                                NietGekoppeldIncident['KOD'], \
                                NietGekoppeldIncident['VRZ']))
#                   Als het toegevoegde incident eenmaal is toegevoegd dan niet nogmaals
                    lstVerwerkteIncidenten.append(y)
                    f.write("\n")

f.close()

#%%
#   STAP 2
#   Bepaal de overeenkomst van de probleembeschrijving en de incidentbeschrijving
#   van problemen en incidenten die gekoppeld zijn.

maskGekoppeldeIncidenten=(incidenten['Probleemnummer'].notna()) 

#   De incidenten gekoppeld aan de probleem zal op similarity worden gecontroleerd
GekoppeldeIncidenten = incidenten.loc[maskGekoppeldeIncidenten]

#   Niet gekoppelde incidenten staan al in NietGekoppeldeIncidenten (voorgaande bewerking)

voortgangsteller = 0
maxvoortgangsteller = len(GekoppeldeIncidenten)
for index, GekoppeldIncident in GekoppeldeIncidenten.iterrows():
    x = GekoppeldIncident['Incidentnummer']
    print('....' + x + ' (' + str(voortgangsteller) + ' van ' + str(maxvoortgangsteller) + ')')
    voortgangsteller+=1

#   Zoek het specifieke gegevens erbij
  
    GekoppeldIncidentKOD = GekoppeldIncident['Korte omschrijving (Details) (I)']
    GekoppeldIncidentVRZ = GekoppeldIncident['Verzoek (I)']

    cleanGekoppeldIncidentKOD = list(filter(lambda x: x in modelKOD.wv.vocab, GekoppeldIncident['LSTKOD']))
    cleanGekoppeldIncidentVRZ = list(filter(lambda x: x in modelVRZ.wv.vocab, GekoppeldIncident['LSTVRZ']))

    for index, NietGekoppeldIncident in NietGekoppeldeIncidenten.iterrows():
#       Bepaal overeenkomstscores
        if NietGekoppeldIncident['Incidentnummer'] not in lstVerwerkteIncidenten:
            scoreKOD = 0    
            if cleanProbleemKOD != [] and NietGekoppeldIncident['LSTKOD'] != []:
                scoreKOD = modelKOD.wv.n_similarity(cleanProbleemKOD, NietGekoppeldIncident['LSTKOD'])
            scoreVRZ = 0    
            if cleanProbleemVRZ != [] and NietGekoppeldIncident['LSTVRZ'] != []:
                scoreVRZ = modelVRZ.wv.n_similarity(cleanProbleemVRZ, NietGekoppeldIncident['LSTVRZ'])
                if (scoreKOD > 0.75) or (scoreVRZ > 0.75):
                    f.write("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (GekoppeldIncident['Probleemnummer'], \
                                strProbleemOriginalKOD.encode("ascii", "ignore").decode(), \
                                strProbleemOriginalVRZ.encode("ascii", "ignore").decode(), \
                                NietGekoppeldIncident['Incidentnummer'], \
                                NietGekoppeldIncident['Korte omschrijving (Details) (I)'].encode("ascii", "ignore").decode(), \
                                NietGekoppeldIncident['Verzoek (I)'].encode("ascii", "ignore").decode(), \
                                scoreKOD,  \
                                scoreVRZ, \
                                NietGekoppeldIncident['RK'],\
                                GekoppeldIncident['Probleemnummer'], \
                                GekoppeldIncident['Incidentnummer'], \
                                NietGekoppeldIncident['Incidentnummer']
                                GekoppeldIncident['KOD'], \
                                GekoppeldIncident['VRZ'], \
                                NietGekoppeldIncident['KOD'], \
                                NietGekoppeldIncident['VRZ']))
))
#                   Als het toegevoegde incident eenmaal is toegevoegd dan niet nogmaals
                    lstVerwerkteIncidenten.append(NietGekoppeldIncident['Incidentnummer'])
                    f.write("\n")
 
    
f.close()

# uitvoerbestandcsv = r'C:\Users\cimanbijlsmah\Documents\GitHub\DS\Hessel, Hidden Incidents 1\data\uitvoer_similarity_doc2vec_topdesk_problems_incidenten_kernel2.csv'
# uitvoerbestandcsvnieuw = r'C:\Users\cimanbijlsmah\Documents\GitHub\DS\Hessel, Hidden Incidents 1\data\uitvoer_similarity_doc2vec_topdesk_problems_incidenten_kernel2_beperkt.csv'
# print('Inlezen uitvoer . . . ')

# fr = open(uitvoerbestandcsv, 'r')
# fw = open(uitvoerbestandcsvnieuw, 'w')
# regelnummer = 0
# for line in fr:
#     regelnummer += 1
#     fw.write(line)
#     if regelnummer >=5000:
#         break
# fw.close()
# fr.close()
    

#%% Nabewerken incidenten


#   Lees csv die aangemaakt in de voorgaande stap.

#   Iteratie 1.
#   Tel op niveau 2 en vervolgend op niveau 1 onderliggende niet gekoppelde voorspelde incidenten
#   Als dit aantal gelijk is aan één, dan is er maar één incident die naar aanleiding van 
#   de bovenliggende omschrijving als kandidaat voor een verborgen incident kan worden beschouwd.
#   Als dit het geval is, dan mag deze niet in de lijst met voorspelde verborgen incidenten voorkomen.

#   Lees uitvoer 
print ('Nabewerking verborgen incidenten . . .')
incidenten = pd.read_csv(r'data/uitvoer_similarity_doc2vec_major_1_maand_stap0.csv',delimiter=';')

#   Laat alle gekoppelde incidenten achterwege en concentreer op de potentiele
#   kandidaten voor verborgen incidenten die worden voorspeld.
incidentenNAN = incidenten[incidenten['RelatedIncident_MAJOR'].isna()]

#   Bepaal het aantal onder level 2 gevonden incidenten > 1
incidentenNAN = incidentenNAN[['Inclevel2', 'Inclevel3']]
counts = incidentenNAN['Inclevel2'].value_counts()
counts = counts[counts>1]
incidenten_i1=incidenten[incidenten['Inclevel2'].isin(counts.index)]
incidenten_i1.to_csv(r'data/uitvoer_similarity_doc2vec_major_1_maand_stap1.csv', sep=';')
    
#   Iteratie 2. 
#   Ontdubbel de incidenten op niveau 3 en bewaar het incident met de hoogste overeenkomstscore.

incidenten_i2 = incidenten_i1.loc[incidenten_i1.groupby('Inclevel3')['RelatedIncident_SIM'].idxmax()]
incidenten_i2 = incidenten_i2[['MajorIncident_ID','MajorIncident_KOD_Origineel','RelatedIncident_ID', 'RelatedIncident_KOD_origineel','RelatedIncident_MAJOR','RelatedIncident_SIM', 
                               'Inclevel1', 'Inclevel2', 'Inclevel3']]
incidenten_i2.to_csv(r'data/uitvoer_similarity_doc2vec_major_1_maand_stap2.csv', sep=';')

#   Iteratie 3.
#   Neem alleen de verborgen, niet gekoppelde incidenten

incidenten_i3 = incidenten_i2.loc[incidenten_i2['RelatedIncident_MAJOR'].isna()]
incidenten_i3.to_csv(r'data/uitvoer_similarity_doc2vec_major_1_maand_stap3.csv', sep=';')

#    print('Eindresultaat vastleggen in pickle bestand . . .')
#    f = open('pckl_df_incidenten.pkl', 'wb')
#    pickle.dump(incidenten, f)
#    f.close()
#    print('Gereed !')

# else:
#   print('Haal DF uit pickle bestand !')
#
#    f = open('pckl_df_incidenten.pkl', 'rb')
#    incidenten = pickle.load(f)
#    f.close()
#    print('Gereed !')
