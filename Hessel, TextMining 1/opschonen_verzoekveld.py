import os
import re
import numpy as np
import pandas as pd

def opschonen_incidenten(incidenten):
    incidenten = incidenten[['Incidentnummer', 'Soort binnenkomst',
        'Soort incident', 'Categorie', 'Verzoek']]
    incidenten = incidenten.query('`Soort incident` != "Event"')
    # Pas regex toe op het verzoekveld om deze schoon te maken
    # Verwijder newlines uit verzoekveld
    incidenten['Verzoek'] = incidenten['Verzoek']\
            .str.replace(" :", ":", regex=True)
    incidenten['Verzoek'] = incidenten['Verzoek']\
            .str.replace(r"\n", " ", regex=True)
    incidenten['Verzoek'] = incidenten['Verzoek']\
            .str.replace(r"\r", " ", regex=True)
    incidenten['Verzoek'] = incidenten['Verzoek']\
            .str.replace(r"\t", " ", regex=True)
    return incidenten

def verwijder_datum_naam(incidenten):
    incidenten['Verzoek'] = incidenten['Verzoek']\
            .str.replace(r"^\d\d-\d\d-\d\d\d\d\s\d\d:\d\d .+?, .+?:",
                    "", regex=True)\
            .str.strip()
    return incidenten

def apply_regex(incident, regex_expr):
    if incident['Omschrijving_verzoek'] is not None:
        omschrijving = incident['Omschrijving_verzoek']
    else:
        # Detecteer de regex.
        zoekresultaat = re.search(regex_expr, incident['Verzoek'].lower())
        # Als de regex gevonden is, geef dan de eerste capture group terug
        omschrijving = zoekresultaat.group(1).strip() if zoekresultaat\
                is not None else None
    return omschrijving

def apply_regex_rest(incident, regex_expr):
    if incident['Omschrijving_verzoek'] is not None:
        omschrijving = None
    elif incident['Omschrijving_verzoek_rest'] is not None:
        omschrijving = incident['Omschrijving_verzoek_rest'].lower()
    else:
        # Detecteer de regex.
        zoekresultaat = re.search(regex_expr, incident['Verzoek'].lower())
        # Als de regex gevonden is, geef dan de eerste capture group terug
        omschrijving = zoekresultaat.group(1) if zoekresultaat\
                is not None else incident['Verzoek'].lower()
    return omschrijving

def construct_query(oplossing):
    oplossing_labels = [x for x in oplossing.index\
            if oplossing[x] is not None or not np.isnan(oplossing[x])]
    opl_query = ["`" + x + "` == @" + x.lower().replace(" ", "")
            for x in oplossing_labels]
    query_string = " & ".join(opl_query)
    return query_string

def detecteer_regex(incidenten, oplossingen):
    incidenten.insert(incidenten.shape[1], 'Omschrijving_verzoek', None)
    incidenten.insert(incidenten.shape[1], 'Omschrijving_verzoek_rest', None)
    print(oplossingen.loc[0])
    # Bij de oplossingen zitten een aantal eigenschappen, gebruik deze om
    # de sjablonen toe te passen op de bijbehorende incidenten.
    # Neem ook alle sjablonen mee met dezelfde eigenschappen.
    eigenschappen_opl = [x for x in oplossingen.columns.values 
                         if x not in ['Verzoek', 'Verzoek_regex']]
    oplossingen_uniek = oplossingen[eigenschappen_opl]\
            .drop_duplicates(eigenschappen_opl)

    for opl_idx, oplossing in oplossingen_uniek.iterrows():
        query_string = construct_query(oplossing)
        soortbinnenkomst = oplossing['Soort binnenkomst']
        soortincident = oplossing['Soort incident']
        categorie = oplossing['Categorie']

        # Selecteer incidenten met dezelfde eigenschappen
        incidenten_selectie = incidenten.query(query_string)
        # Neem ook alle sjablonen mee met dezelfde eigenschappen.
        oplossingen_selectie = oplossingen.query(query_string)

        len_regex = oplossingen_selectie['Verzoek_regex'].str.len()\
                .sort_values(ascending=False).index
        oplossingen_selectie = oplossingen_selectie.reindex(len_regex)\
                .reset_index(drop=True)
        oplossingen_selectie = oplossingen_selectie.drop_duplicates()

        # Begin bovenaan in de lijst met oplossingen.
        for verzoek_reg in oplossingen_selectie['Verzoek_regex']:
            incidenten_selectie['Omschrijving_verzoek'] =\
                    incidenten_selectie\
                    .apply(lambda row: apply_regex(row, verzoek_reg), axis=1)
        incidenten.loc[incidenten_selectie.index, 'Omschrijving_verzoek'] =\
                incidenten_selectie['Omschrijving_verzoek']

    print("Incidenten met een standaardoplossing verwerkt, nu de restgroep")
    len_regex = oplossingen['Verzoek_regex'].str.len()\
            .sort_values(ascending=False).index
    oplossingen = oplossingen.reindex(len_regex).reset_index(drop=True)

    # Begin bovenaan in de lijst met oplossingen.
    rest_regex= list()
    rest_regex.append('omschrijving[\s\S]*storing:([\s\S]*)volledige[\s]*naam:[\s\S]*')
    rest_regex.append('omschrijving[\s\S]*probleem:([\s\S]*)volledige[\s\S]*')
    rest_regex.append('^([\s\S]*)volledige[\s\S]*naam:[\s\S]*')
    for verzoek_reg in rest_regex:
        incidenten['Omschrijving_verzoek_rest'] =\
                incidenten.apply(lambda row: apply_regex_rest(row, verzoek_reg), axis=1)

    return incidenten

print("Lezen...")
mapGebruikers = "C:\\Users\\"
mapGebruiker = os.getlogin()
mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
os.chdir(hoofdmap)

incidenten = pd.read_csv(r'data\Incidenten_SD_2018_2019_totaal-v3.csv', header=0, parse_dates=False, squeeze=True, low_memory=False)
incidenten = opschonen_incidenten(incidenten)
incidenten = verwijder_datum_naam(incidenten)

oplossingen = pd.read_csv(r'data/Standaardoplossingen_processed_regex.csv', sep = ",", low_memory=False)

incidenten = detecteer_regex(incidenten, oplossingen)
print(incidenten['Omschrijving_verzoek'].apply(lambda x: True if x is not None else False).sum())
print(incidenten['Omschrijving_verzoek_rest'].apply(lambda x: True if x is not None else False).sum())

incidenten.to_csv(r'data/Incidenten Processed Export - Omschrijvingen.csv', 
        index = False, encoding="UTF-8")
