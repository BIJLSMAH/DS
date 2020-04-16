from datetime import datetime
import numpy as np
import pandas as pd

# Zelf gedefinieerde modellen
import model_definities as md

def mean_error(y_pred, y):
    return np.mean(y_pred - y)

def rmse(y_pred, y):
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    return rmse

def mape(y_pred, y):
    mape = np.mean(np.abs(y_pred - y)/y) * 100
    return mape

# Settings
bestandsnaam = "../data/processed/Tijdreeksen_Incidenten.csv"
tijdreeks_naam = "Totaal_Openstaand"
hulpvar_naam = None

predict_eenheid = "W"
predict_lengte = 26 
tijd_kolom = "Datum"
start_training_set = pd.to_datetime("2013-01-02")
num_horizons = 26

train_model_list = [md.ARIMA1_train, md.ARIMA2_train, md.ExpSmooth_train]
pred_model_list = [md.ARIMA1_pred, md.ARIMA2_pred, md.ExpSmooth_pred]

# Check of er evenveel train als predictie modellen gedefinieerd zijn
assert len(train_model_list) == len(pred_model_list), \
        "Ongelijk aantal train en predictie-functies"

# Laad csv in
input_data = pd.read_csv(bestandsnaam, parse_dates=[tijd_kolom])
input_data = input_data.set_index(tijd_kolom)

# Resample de tijdreeks naar de gewenste frequentie
input_data = input_data.resample(predict_eenheid).sum()

# Kort in tot het begin van de training set
input_data = input_data[input_data.index >= start_training_set]

# Selecteer de te voorspellen tijdreeks
tijdreeks = input_data[[tijdreeks_naam]]

# Selecteer hulpvariabelen
#if hulpvar_naam is not None:
#    pass

# Aantal modellen
num_models = len(train_model_list)
# Het eerste deel van de model-definitie, voor de eerste underscore 
# is de naam van het model
model_names = [func.__name__.split('_')[0] for func in train_model_list]

# Initialiseer list/dataframe voor de resultaten 
resultaten_df = pd.DataFrame(data={"models": model_names},
        columns=["models", "ME", "RMSE", "MAPE"])

# Loop over alle modellen heen
for model_idx in range(num_models):
    print(model_names[model_idx])
    model_train = train_model_list[model_idx]
    model_pred = pred_model_list[model_idx]
    
    resultaten_model_df = pd.DataFrame(columns=["ME", "RMSE", "MAPE"])

    for hor in range(num_horizons):
        start_test_index = tijdreeks.shape[0] - predict_lengte \
                - num_horizons + hor
        train = tijdreeks.iloc[0:start_test_index, :].values
        test = tijdreeks.iloc[start_test_index:-1, :].values

        # Train model op de training set
        model_trained = model_train(train)

        # Maak voorspelling
        predictie = model_pred(model_trained, predict_lengte)

        # Voeg resultaten toe aan de resultaten list/dataframe
        resultaten_model_df.at[hor, "ME"] = mean_error(predictie, test)
        resultaten_model_df.at[hor, "RMSE"] = rmse(predictie, test)
        resultaten_model_df.at[hor, "MAPE"] = mape(predictie, test)
        
    # Print gemiddelde van de resultaten
    resultaten_avg = resultaten_model_df.mean(axis=0)
    resultaten_df.loc[model_idx, ["ME", "RMSE", "MAPE"]] = resultaten_avg

# Sorteer de resultaten dataframe op de gewenste scoremaat
resultaten_df = resultaten_df.sort_values(by=["RMSE"])
print(resultaten_df)
