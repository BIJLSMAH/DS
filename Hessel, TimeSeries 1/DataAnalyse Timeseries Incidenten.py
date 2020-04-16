import pandas as pd
import pickle 
import matplotlib.pyplot as figuur
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%%
# Laad gegevens uit aangemaakte pickle in de DataManipulatie Incidenten.py
df = pickle.load(open('TSeries', 'rb'))

df['open'] = df.open.astype('float32')
print(df)
tijdr = df
tijdr = tijdr.set_index('maand')
# Resamplen alleen nodig als er ontbrekende waarden zijn in de tijdreeks.
# dat is hier niet het geval
# tijdr = tijdr.open.resample('M').mean()
print(tijdr)
# print(df.head())

# Prepareren van de data
df['datum2'] = pd.to_datetime(df.datum, format='%Y-%m-%d %H:%M:%S')
df['jaar'] = [d.year for d in df.datum2]
df['mnd'] = [d.strftime('%b') for d in df.datum2]
#%%
from statsmodels.tsa.seasonal import seasonal_decompose

# Het is mogelijk om de trend, seizoensinvloeden te isoleren en afzonderlijk te tonen.
# Dit kan met behulp van een additief model waarbij:
#       Waarde = Basis + Trend + Seizoen + Fout
# Of dat kan met behulp van een multiplicatief model waarbij:
#       Waarde = Basis x Trend x Seizoen x Fout

ts = df[['mnd', 'open']]
#ts2 = ts.astype(ts)
#ts.set_index('jjjjdd')

# Additieve decompositie
# result_add = seasonal_decompose(ts['open'], model='additive', freq=9)

# Multiplicatieve decompositie
result_mul = seasonal_decompose(ts['open'], model='multiplicative', freq=9)

# plot het resultaat
decplot = figuur
decplot.rcParams.update({'figure.figsize': (10, 10)})
# result_add.plot().suptitle('Additieve Decompositie', fontsize=22)
result_mul.plot().suptitle('Multiplicatieve Decompositie', fontsize=22)

# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df_reconstructed.head())
#%% Statistische analyse
# Most statistical forecasting methods are designed to work on a stationary time series.
# The first step in the forecasting process is typically to do some transformation to convert
# a non-stationary series to stationary.
# non-stationary timeseries kunnen worden geconverteerd door:
# 1. Differencing the Series(once or more)
# 2. Take the log of the series
# 3. Take the nth root of the series
# 4. Combination of the above

# De meest voorkomende manier is 1. Differencing
# Dus ga je dan uit voor Y(t) van het verschil Y(t) - Y(t-1)

# Er zijn verschillende methodes beschikbaar om te kijken of de tijdreeks stationary, non-stationary is.
#  Dicky Fuller test. Dan test je op non-stationariteit (nulhypothese) als p onder drempelwaarde (0.05) dan verwerp je
#  de nulhypothese. Dit betekent dan dat de tijdreeks stationair is.
#  De nulhypothese = non statonary ADF

# Store in a function to summarize Dickey-Fuller test for stationarity
# Import function
from statsmodels.tsa.stattools import adfuller

# Define function
def adf_check(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic', 'p-value', '#Number of Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print("Nulhypothese WEL verworpen. Tijdreeks heeft GEEN unit root en is STATIONAIR.")
    else:
        print("Nulhypothese NIET verworpen. Tijdreeks heeft WEL unit root en is daarom NIET STATIONAIR.")

# Check 'Value' for stationarity

adf_check(ts['open'])
# Non-stationary, so it must be transformed
# Stationary, so it is not necessary to be transformed
#%%
# import the plotting functions for acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts['open'], lags=40);
plot_pacf(ts['open'], lags=7);

#%%
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse <= best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA %s MSE= %.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA %s MSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = range(0, 3)
d_values = range(1, 2)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(tijdr['open'], p_values, d_values, q_values)
#_____________________________________________

# Andere manier voor het bepalen van de optimale p, d en q

#import pmdarima as pm

#model = pm.auto_arima(tijdr['open'], start_p=1, start_q=1,
#                      test='adf',       # use adftest to find optimal 'd'
#                      max_p=4, max_q=4, # maximum p and q
#                      m=1,              # frequency of series
#                      d=None,           # let model determine 'd'
#                      seasonal=True,    # No Seasonality
#                      start_P=0,
#                      D=0,
#                      trace=True,
#                      error_action='ignore',
#                      suppress_warnings=True,
#                      stepwise=True)

#print(model.summary())

# 1,1,1 ARIMA Model
model = ARIMA(tijdr['open'], order=(1, 1, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
print(model_fit.resid)

# Plot residual errors
figuur.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
plt = figuur
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict('2019', '2020', dynamic=False, plot_insample=True)
plt.show()

# Create Training and Test

cptijdr = tijdr.set_index(pd.DatetimeIndex(tijdr['datum']))
cptijdr.resample('M')
train_size = int(len(cptijdr) * 0.75)
train, test = cptijdr[0:train_size], cptijdr[train_size:]

# Build Model
# model = ARIMA(train, order=(3,2,1))
model = ARIMA(train['open'], order=(0, 1, 2))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(20, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
fc_series.resample('M')
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.gca().set(title="", xlabel='jjjjmm', ylabel='Incidenten')
plt.plot(train['open'], 'b', label='training')
plt.plot(test['open'], 'r', label='actual')
plt.plot(fc_series, 'g', label='forecast')

#plt.plot(train['open'], label='training')
#plt.plot(test['open'], label='actual')
#plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


#decplot.show()
#sbplt.show()
#plt.show()
#slplt.show()


