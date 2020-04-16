#%% Initieren en inlezen van gegevens
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as figuur
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Wijzig de intstellingen voor de te tonen plots

figuur.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
plt = figuur

# seaborn
sns.set(style="whitegrid", color_codes=True)

# Laad gegevens uit maand.xlsx
from pandas import read_csv

# df met index
df = read_csv(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1\data\MAAND OPEN PRD 2014-2019.csv', parse_dates=True, index_col=0)
# df = read_excel("C:\Administratie\Hessel\IncidentenAnalyse SD\TRENDANALYSE\MAAND OPEN.xlsx", parse_dates=['jjjjdd'],
#                 index_col=None)
df['open'] = df.open.astype('float32')
df.index=pd.to_datetime(df.index)
pd.Series(df.open)
tijdr = tijdr.open.resample('M').mean()
print(tijdr)

plt.figure(figsize=(16, 5), dpi=120)
plt.gca().set(title="", xlabel='maand', ylabel='Incidenten')
plt.plot(tijdr, 'r', label='Alle')
plt.legend(loc='upper left')

# Plot nu de seizoensinvloeden
df.reset_index(inplace=True)

# Prepareren van de data
df['incjaar'] = [d.year for d in df.maand]
df['incmaand'] = [d.strftime('%b') for d in df.maand]
jaren = df['incjaar'].unique()

# Selecteren kleuren
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(jaren), replace=False)

# Tekenen van de plot met linegraph
slplt = figuur
slplt.figure(figsize=(16,5), dpi=120)
for i, j in enumerate(jaren):
    if i > 0:
        slplt.plot('incmaand', 'open', data=df.loc[df.incjaar == j, :], color=mycolors[i], label=j)
        slplt.text(df.loc[df.incjaar == j, :].shape[0]-0.9, df.loc[df.incjaar==j, 'open'][-1:].values[0], j, fontsize=12, color=mycolors[i])

# Tekenen van de jaarlijkse trend/seizoensinvloeden
sbplt = figuur
fig, axes = sbplt.subplots(1, 2, figsize=(20, 7), dpi=80)
sns.boxplot(x='incjaar', y='open', data=df, ax= axes[0])
sns.boxplot(x='incmaand', y='open', data=df.loc[~df.incjaar.isin([2014, 2018]),:])

# Zet de titels
axes[0].set_title('Trend (boxplot)', fontsize=18)
axes[1].set_title('Seizoen (boxplot)', fontsize=18)

# Het is mogelijk om de trend, seizoensinvloeden te isoleren en afzonderlijk te tonen.
# Dit kan met behulp van een additief model waarbij:
#       Waarde = Basis + Trend + Seizoen + Fout
# Of dat kan met behulp van een multiplicatief model waarbij:
#       Waarde = Basis x Trend x Seizoen x Fout

ts = df[['maand', 'open']]

# Additieve decompositie
result_add = seasonal_decompose(ts['open'], model='additive', period=12)

# Multiplicatieve decompositie
# result_mul = seasonal_decompose(ts['Openstaand'], model='multiplicative',freq=9)
result_mul = seasonal_decompose(ts['open'], model='multiplicative', period=12)

# plot het resultaat
decplot = figuur
decplot.rcParams.update({'figure.figsize': (10, 10)})
result_add.plot().suptitle('Additive Decompositie', fontsize=22)
result_mul.plot().suptitle('Multiplicatieve Decompositie', fontsize=22)

# Extract the Components ----
# Actual Values = Sum of (Seasonal + Trend + Resid)
df_reconstructed = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df_reconstructed.head())

# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df_reconstructed.head())
#%%
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
ts = df
pd.Series(ts.open)
resultadf = adfuller(ts)
print(resultadf[1])
adf_check(ts)
# import the plotting functions for act and pacf

plot_acf(ts['open'], lags=20);
plot_pacf(ts['open'], lags=7);

ts_stationary = ts.diff(periods=12).dropna()
adf_check(ts_stationary)
decplot = figuur
decplot.rcParams.update({'figure.figsize': (10, 10)})
decplot.plot(ts_stationary)


#%%
import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(ts['Openstaand'], model='multiplicative', 
                                          period=12)
fig = decomposition.plot()
plt.show()

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
# evaluate_models(tijdr, p_values, d_values, q_values)
#_____________________________________________

# Andere manier voor het bepalen van de optimale p, d en q

from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm

#model = pm.auto_arima(tijdr, start_p=1, start_q=1,
#                      test='adf',       # use adftest to find optimal 'd'
#                      max_p=4, max_q=4, # maximum p and q
#                      m=1,              # frequency of series
#                      d=None,           # let model determine 'd'
#                      seasonal=False,   # No Seasonality
#                      start_P=0,
#                      D=0,
#                      trace=True,
#                      error_action='ignore',
#                      suppress_warnings=True,
#                      stepwise=True)

# print(model.summary())

# 1,1,1 ARIMA Model, totale dataset
# model = ARIMA(tijdr, order=(1, 1, 1))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# print(model_fit.resid)

# Plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1, 2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()

# Actual vs Fitted
# model_fit.plot_predict('2019', '2020', dynamic=False, plot_insample=True)
# plt.show()

# from statsmodels.tsa.stattools import acf

# Create Training and Test

train_size = int(len(tijdr) * 0.8)
train, test = tijdr[0:train_size], tijdr[train_size:]

# Build Model
# model = ARIMA(train, order=(3,2,1))
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(test.size, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


#decplot.show()
#sbplt.show()
#plt.show()
#slplt.show()

