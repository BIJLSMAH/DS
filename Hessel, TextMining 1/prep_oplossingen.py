import os
import re
import numpy as np
import pandas as pd

def opschonen_oplossingen(oplossingen):
    # In het bestand van de oplossingen zijn er enkele mogelijkheden voor het
    # 'Verzoek'-veld. Slechts de eerste kolom heeft de label 'Verzoek'; de anderen
    # zijn ongelabeld. Deze zijn genoteerd als bijv "Unnamed: 23".
    # Houdt alleen de kolommen 'Binnenkomst', 'Soort incident', 'Categorie'
    # en 'Verzoek'
    print("Selecteer kolommen")
    keep_columns = ['Soort incident', 'Categorie', 'Verzoek'] \
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


print("Load dataset . . .")
mapGebruikers = "C:\\Users\\"
mapGebruiker = os.getlogin()
mapTM = r'\Documents\Github\DS\Hessel, TextMining 1'
hoofdmap = "%s%s%s" %(mapGebruikers, mapGebruiker, mapTM)
os.chdir(hoofdmap)
oplossingen = pd.read_excel("data/Standaardoplossingen42.xlsx")
oplossingen = opschonen_oplossingen(oplossingen)
# Filter 'Ja/Nee' uit de oplossingen om een meer uniforme aanpak voor de regex te creeeren
oplossingen = filter_janee(oplossingen)
oplossingen = genereer_regex(oplossingen)
print(oplossingen)

oplossingen.to_csv("data/Standaardoplossingen_processed_regex.csv", 
        index = False, encoding="UTF-8")
