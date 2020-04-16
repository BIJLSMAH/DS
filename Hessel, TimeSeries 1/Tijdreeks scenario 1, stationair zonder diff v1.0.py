#%% Versies van gebruikte modules
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
#%% Om de modellen te testen en de data te exploreren
#   verdelen we de dataset in twee gedeelten:
#   1. Validation dataset en een deel om het model op te maken
#   2. Een methode om de modellen te testen
# separate out a validation dataset

import os
from pandas import read_csv

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1')
series = read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# isoleer het laatste jaar (12 maanden) in een afzonderlijke data/testset
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv(r'data\dataset.csv', header=False)
validation.to_csv(r'data\validation.csv', header=False)
#%% Model Evaluation
#   1. Performance Measure
#   2. Test Strategy 

#   It is possible to measure the performance of the
#   model using de RMSE. This will give more weight to
#   the predictions that are significantly wrong and 
#   will use the same units as the original data.

# from sklearn.metrics import mean_squared_error
# from math import sqrt
# ...
# test = ...
# predictions = ...
# mse = mean_squared_error(test, predictions)
# rmse = sqrt(mse)
# print('RMSE: %.3f' % rmse)
#%% Test Strategy
#   Models will be evaluated using the walk forward validation
#   1. The first 50% of the dataset will be iterated to train the model
#   2. The remaining 50% of the dataset will be iterated and test the model
#   3. For each step in de testset:
#       3a. Train a model
#       3b. Make and preserve the prediction for later evaluation
#       3c. Add the actual observation from the testset to the trainingset
#   4. Calculate the difference between obs and forecast and calculate the RMSE
#   5. Choose the best order for the model, based in the lowest RMSE

import os
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt

thisdir = os.getcwd()
print(thisdir)
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

X = series.values
X = X.astype('float32')
train_size = int(len(X)-12)
train, test = X[0:train_size], X[train_size:]    

# Next, we can iterate over the time steps in the test 
# dataset. The train dataset is stored in a Python list 
# as we need to easily append a new observation each 
# iteration and NumPy array concatenation feels 
# like overkill.
# The prediction made by the model is called yhat 
# for convention, as the outcome or observation is 
# referred to as y and yhat (a ‘y‘ with a mark above) 
# is the mathematical notation for the prediction of 
# the y variable.
# The prediction and observation are printed each observation 
# for a sanity check prediction in case there are issues 
# with the model.

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict (based upon previous leg)
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

# Hier wordt berekend welke RMSE er is op het moment dat de voorspelling
# van de volgende periode wordt voorspeld door de huidige periode. Dus het 
# model t=t-1. Ieder ander model zou eigenlijk beter moeten voorspellen dan 
# deze, anders is het geen goed model.

#%% Data Analysis
#   1. Summary Statistics
#   2. Line Plot
#   3. Seasonal Line Plots
#   4. Density Plots
#   5. Box and Whisker Plots

import os
from pandas import read_csv
from matplotlib import pyplot as pp
from pandas import DataFrame
from pandas import Grouper

thisdir = os.getcwd()
print(thisdir)
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

# Samenvatting statistics
print(series.describe())

# Line Plot
series.plot()
pp.show()

# Seasonal Line Plots
groups = series['2014':'2019'].groupby(Grouper(freq='A'))
years = DataFrame()
pp.figure()
i = 1
n_groups = len(groups)
for name, group in groups:
	pp.subplot((n_groups*100) + 10 + i)
	i += 1
	pp.plot(group)
pp.show()

# Density Plots
pp.figure(1)
pp.subplot(211)
series.hist()
pp.subplot(212)
series.plot(kind='kde')
pp.show()

# Box and Whisker Plots
groups = series['2014':'2019'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pp.show()
#%% ARIMA
#   1. Manually Configure ARIMA
#   2. Automatically Configure ARIMA
#   3. Review Residual Results

#   1. Manually Configure ARIMA (p, q, d)
#   1a. Stationary Series ?
#       Make it Stationary by Differencing the values
#       Test is the TimeSeries is Stationary

from pandas import read_csv
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = X.astype('float32')
# check stationarity original
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

if result[1] <= 0.05:
    df = pandas.DataFrame(X)
    df.index = series.index[:]
    df.to_csv(r'data\stationary.csv', header=False)
    stationary = df
else:
# Als p-waarde < 0,05 dan verwerp nulhypothese, wel stationair
# Als p-waarde >= 0,05 dan neem nulhypothese aan, niet stationair

# NB. De originele dataset vertoont al voldoende stationariteit.
#     Verdere stappen richting stationariteit zijn dus overbodig
#     Indien wel nodig zie onderstaande acties

# difference data
    months_in_year = 12
    stationary = difference(X, months_in_year)
    stationary.index = series.index[months_in_year:]
# check if stationary
    result = adfuller(stationary)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv(r'data\stationary.csv', header=False)

# plot
stationary.plot()
pyplot.show()
#%% De te gebruiken differenced and reverse_differenced
#   dataset is nu aangemaakt en kan als input dienen
#   voor het ARIMA model waarbij de d = 0 kan zijn nu.
#   Nu even kijken naar ACF en PACT voor de optimale 
#   andere parameters.
import os
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# ACF  = AR
# PACF = MA
from matplotlib import pyplot as pp

thisdir = os.getcwd()
print(thisdir)
series = read_csv(r'data\stationary.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# series = series[:(len(series)-12)]
pp.figure()
pp.subplot(211)
plot_acf(series, ax=pp.gca())
pp.subplot(212)
plot_pacf(series, ax=pp.gca())
pp.tight_layout(pad=3.0)
pp.show()
#%% Testen ARIMA
#   Uit bovenstaande ACF en PACF

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
# load data
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) - 12)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # no difference, original dataset is stationary
	# predict
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

#%% Grid Search Parameters ARIMA
import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) - 12)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s RMSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
 
# load dataset
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

# Best ARIMA(0, 1, 1) RMSE=1296.254
#%% Review Residual Errors

from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as pp
 
# load data
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) -12)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pp.figure()
pp.subplot(211)
residuals.hist(ax=pp.gca())
pp.subplot(212)
residuals.plot(kind='kde', ax=pp.gca())
pp.show()
#%% The mean is non-zero (-260.164167 residual error)
#   It is possible to (bias) correct this by adding
#   this to each forecast made.

from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

# load data
series = read_csv('data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) - 12)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = 288.361295
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + yhat
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
#%% It is also a good idea to check the time series 
#   of the residual errors for any type of autocorrelation. 
#   If present, it would suggest that the model has more 
#   opportunity to model the temporal structure in the data.
#   The example below re-calculates the residual errors and 
#   creates ACF and PACF plots to check for any significant 
#   autocorrelation.

from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
 
# load data
series = read_csv('data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) - 12)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
bias = 288.361295
history = [x for x in train]
predictions = list()
for i in range(len(test)):
# 	predict
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + yhat
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)

# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()
# The results suggest that what little autocorrelation is present in the 
# time series has been captured by the model.
#%% Model Evaluation
#   After models have been developed and a final 
#   model selected, it must be validated and finalized.
#   Validation is an optional part of the process, 
#   but one that provides a ‘last check’ to ensure 
#   we have not fooled or misled ourselves.
#   This section includes the following steps:
#   1. Finalize Model: Train and save the final model.
#   2. Make Prediction: Load the finalized model and make a prediction.
#   3. Validate Model: Load and validate the final model.

#%% 1. Finalize Model
#   Finalizing the model involves fitting an 
#   ARIMA model on the entire dataset, in this case 
#   on a transformed version of the entire dataset.

from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import numpy
 
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__
 
# load data
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
series = series[:(len(series))]
X = series.values
X = X.astype('float32')
# fit model
model = ARIMA(X, order=(0,1,1))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = 288.361295
# save model
model_fit.save(r'data\model.pkl')
numpy.save(r'data\model_bias.npy', [bias])

# The model has been saved as pickle
# the value bias has been saved

#%% Make Prediction
#   A natural case may be to load the model and make a single forecast.
#   This is relatively straightforward and involves restoring 
#   the saved model and the bias and calling the forecast() 
#   method. To invert the seasonal differencing, the historical data must 
#   also be loaded.
#   The example below loads the model, makes a prediction for the next time 
#   step, and prints the prediction. 
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMAResults
import numpy
 
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + yhat
print('Predicted: %.3f' % yhat)

#%% Validate model
#   Validation using validation.csv (testset)
#   1. Load the model and predict the next 12 months
#      The forecast beyond the first one will start
#      to degrade quickly
#   2. Rolling forecast. Updating the transform and model
#      for each time step (preferred). This means that
#      that we will step over lead times in de validation
#      dataset and take the observations as an update to the history.

from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
import pandas as pd
from datetime import datetime

def monthlist(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1 ,1))
    return mlist

# Doorloop de maanden en verzamel maandgegevens over de totalen
dates=["2014-01-01", "2019-12-01"]
mndLijst = monthlist(dates)

# load data
series = read_csv('data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
df = pd.DataFrame(X, index = mndLijst, columns=['open'])
train_size = int(len(X) - 12)
train, test = df[0:train_size], df[train_size:]

history = [x for x in train['open']] 
validation = [x for x in test['open']]

# load model
model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')
# make first prediction
predictions = list()
confsl = list()
confsh = list()
ses = list()
yhat, se, conf = model_fit.forecast()
yhat = float(yhat)
yhat = bias + yhat
predictions.append(yhat[0])
ses.append(se)
confsl.append(conf[0,0])
confsh.append(conf[0,1])
history.append(validation[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, validation[0]))
# rolling forecasts
for i in range(1, len(validation)):
	# predict
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat, se, conf = model_fit.forecast()
	yhat = bias + yhat
	predictions.append(yhat[0])
	ses.append(se)
	confsl.append(conf[0,0])
	confsh.append(conf[0,1])
	# observation
	obs = validation[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(validation, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

# insert period in forecast en actual (graph)
predictionsn = list()
predictionsn.append(train['open'].loc[max(train.index)])
for i in range(len(predictions)):
    predictionsn.append(predictions[i])

testf = df[train_size-1:train_size]
testn = pd.DataFrame(testf)
testn = testn.append(test)
    
# Als add period to index
dates=[max(train.index).strftime("%Y-%m-%d"), "2019-12-01"]
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

#%% Next

# Build Model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
import pandas as pd
from datetime import datetime

def monthlist(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1 ,1))
    return mlist

# Doorloop de maanden en verzamel maandgegevens over de totalen
Xdates=["2014-01-01", "2018-12-01"]
XmndLijst = monthlist(Xdates)
Ydates=["2019-01-01", "2019-12-01"]
YmndLijst = monthlist(Ydates)

# load data
Xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
Yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = Xseries.values
Y = Yseries.values
X = X.astype('float32')
Y = Y.astype('float32')
train = pd.DataFrame(X, index = XmndLijst, columns=['open'])
test= pd.DataFrame(Y, index = YmndLijst, columns=['open'])

model = ARIMA(train, order=(1,1,1))
model_fit = model.fit(trend='nc', disp=0)
# model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')

# Forecast
fc, se, conf = model_fit.forecast(test.size, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
pyplot.figure(figsize=(12, 5), dpi=100)
pyplot.plot(train, label='training')
pyplot.plot(testn, label='actual')
pyplot.plot(fc_series, label='forecast')
pyplot.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.05)
pyplot.title('Forecast vs Actuals')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()
