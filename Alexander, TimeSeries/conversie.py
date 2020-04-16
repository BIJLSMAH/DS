import numpy as np 
import pandas as pd

# Lees data in
input_data = pd.read_csv("../data/raw/20200305 - Incidenten Tijdreeksanalyse - PROD - UTF-8.csv",
        sep="\t", parse_dates=['Datum Aangemeld', 'Datum Afgemeld'])
input_data = input_data[input_data['Datum Aangemeld'] >= "2013-01-01"]
input_data = input_data[input_data['Datum Afgemeld'] <= 
        np.max(input_data['Datum Afgemeld'].values)]

# Maak een kolom met de datum
start_date = np.min(input_data[['Datum Aangemeld', 'Datum Afgemeld']].values)
end_date = np.max(input_data[['Datum Aangemeld', 'Datum Afgemeld']].values)

output_data = pd.DataFrame({"datum" : pd.date_range(start=start_date, 
    end=end_date)})
output_data = output_data.set_index('datum')

soort_list = ['Totaal'] \
        + input_data['Soort Incident'].unique().tolist()
for soort in soort_list:
    data_filter = input_data.copy()
    if soort is not "Totaal":
        data_filter = data_filter.query("`Soort Incident` == @soort")
    # Kolom met het aantal aangemaakte incidenten per dag
    aangemeld = data_filter.groupby("Datum Aangemeld").count()
    aangemeld = aangemeld[['Incidentnummer']]
    aangemeld.columns = [soort + '_Aangemeld']
    output_data = output_data.join(aangemeld, how="left")

    # Kolom met het aantal afgemelde incidenten per dag
    afgemeld = data_filter.groupby("Datum Afgemeld").count()
    afgemeld = afgemeld[['Incidentnummer']]
    afgemeld.columns = [soort + '_Afgemeld']
    output_data = output_data.join(afgemeld, how="left")

    # Kolom met het aantal openstaande incidenten per dag
    output_data[soort + '_Openstaand'] = output_data.apply(
            lambda row: data_filter[data_filter['Datum Aangemeld'].ge(row.name)
                & data_filter['Datum Afgemeld'].le(row.name)].shape[0],
                axis=1)
output_data = output_data.fillna(0)
output_data['Datum'] = output_data.index.values
print(output_data)

output_data.to_csv("../data/processed/Tijdreeksen_Incidenten.csv", encoding="utf-8",
        index=False)
