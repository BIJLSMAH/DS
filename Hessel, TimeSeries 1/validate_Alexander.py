#%% Validate model
#   Validation using validation.csv (testset)
#   1. Load the model and predict the next 12 months
#      The forecast beyond the first one will start
#      to degrade quickly
#   2. Rolling forecast. Updating the transform and model
#      for each time step (preferred). This means that
#      that we will step over lead times in de validation
#      dataset and take the observations as an update to the history.

import warnings

from datetime import datetime
from math import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def monthlist(dates):
    start, end = [datetime.strptime(dt, "%Y-%m-%d") for dt in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = [None] * (total_months(end) - total_months(start) + 1)
    for m_idx, tot_m in enumerate(range(total_months(start)-1, total_months(end))):
        y, m = divmod(tot_m, 12)
        mlist[m_idx] = datetime(y, m+1, 1)
    return mlist

warnings.filterwarnings("ignore")

# Doorloop de maanden en verzamel maandgegevens over de totalen
dates=["2014-01-01", "2019-12-01"]
mndLijst = monthlist(dates)

# load data
series = pd.read_csv('data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values.astype('float32')

# Genereer een sinus om te testen
# x_range = np.arange(len(X))
# X = np.sin(2 * np.pi * x_range / 20)

df = pd.DataFrame(X, index = mndLijst, columns=['open'])
#pyplot.plot(df)
#pyplot.show()
train_size = int(df.shape[0] * 0.66)
train = df['open'].iloc[0:train_size]
test = df['open'].iloc[train_size:]

validation = test.values.tolist()

# load model
# make first prediction
predictions = [None] * len(validation)
confsl = [None] * len(validation)
confsh = [None] * len(validation)
ses = [None] * len(validation)
# rolling forecasts
for i in range(len(validation)):
    # predict
    #history = df['open'].iloc[0:(train_size + 1 + i)]
    history = df['open'].iloc[0:(train_size + i)]
    model = ARIMA(history, order=(6,1,2))
    # model = ARIMA(history, order=(1,0,1)) # Voor de sinus
    model_fit = model.fit(trend='nc', disp=0)
    yhat, se, conf = model_fit.forecast(steps=1)
    predictions[i] = yhat
    ses[i] = se
    confsl[i] = conf[0, 0]
    confsh[i] = conf[0, 1]
    # observation
    obs = validation[i]
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(validation, predictions))
print('RMSE: %.3f' % rmse)

predictionsn = [train.at[max(train.index)]] + predictions

testn = pd.DataFrame(df[train_size-1:])
    
# Als add period to index
dates = [max(train.index).strftime("%Y-%m-%d"), "2019-12-01"]
mndLijst = monthlist(dates)

lower_series = pd.Series(confsl, index=test.index)
upper_series = pd.Series(confsh, index=test.index)
pr = pd.DataFrame(predictionsn, index=mndLijst)

pyplot.figure(figsize=(12, 5), dpi=100)
pyplot.plot(train, label='training', color='black')
pyplot.plot(pr, label='forecast', color='red')
pyplot.plot(testn, label='actual', color='green')
pyplot.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
pyplot.title('Forecast vs Actuals')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()
