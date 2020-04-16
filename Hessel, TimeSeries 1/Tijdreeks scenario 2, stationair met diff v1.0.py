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
import pandas as pd

from matplotlib import pyplot as pp
from pandas import DataFrame
from pandas import Grouper
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#mpl.style.use('fivethirtyeight')
os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries')
series = read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
pp.figure(figsize=(8,3), dpi=100)
pp.title('Openstaande Incidenten Per Maand')
series.plot()
pp.xlabel('jaren')
pp.ylabel('incidenten')
pp.tight_layout(pad=3.0)
pp.show()

groups = series['2014':'2019'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
# Box and Whisker Plots
pp.figure(figsize=(6,4), dpi=100, edgecolor='k')
years.boxplot()
pp.title('Trend')
pp.tight_layout(pad=3.0)
pp.show()

years=years.transpose()
pp.figure(figsize=(6,4), dpi=100, edgecolor='k')
years.boxplot()
pp.tight_layout(pad=3.0)
pp.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['jan', 'feb', 'mrt', 'apr', 'mei', 'jun', 'jul', 'aug','sep','okt','nov','dec'])
pp.title('Seizoen')
pp.show()


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

# prepare data
import os
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt

thisdir = os.getcwd()
print(thisdir)
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)

X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
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
pp.figure(figsize=(8,3), dpi=100)
pp.title('Openstaande Incidenten Per Maand')
series.plot()
pp.xlabel('jaren')
pp.ylabel('incidenten')
pp.tight_layout(pad=3.0)
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


# Density Plots
fig= pp.figure(figsize=(6,8), dpi=100, edgecolor='k')
ax1 = fig.add_subplot(211)
ax1.title.set_text('Histogram Verdeling Maandelijkse Incidenten')
series.hist()
ax2 = fig.add_subplot(212)
ax2.title.set_text('Verdeling Maandelijkse Incidenten')
series.plot(kind='kde')
fig.tight_layout(pad=2.0)
fig.show()

# Box and Whisker Plots
groups = series['2014':'2019'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pp.tight_layout(pad=3.0)
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
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
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

if result[1] <= 0.01:
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
pyplot.xlabel('jaar')
pyplot.ylabel('open')
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
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
# For reference, the seasonal difference operation 
# can be inverted by adding the observation for the 
# same month the year before. This is needed in the 
# case that predictions are made by a model fit on 
# seasonally differenced data. 
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# load data
# series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# X = series.values
# X = X.astype('float32')
# train_size = int(len(X) * 0.50)
# train, test = X[0:train_size], X[train_size:]

xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = xseries.values
X = X.astype('float32')
Y = yseries.values
Y = Y.astype('float32')
train = X
test = Y

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
    # because the seasoneffect will be 12 months look
    # at the observation 12 months ago.
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
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
import pandas as pd
import numpy
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X))
	train, test = X[0:train_size-12], X[train_size-12:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
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
xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
xseries = xseries.append(yseries)

# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(xseries.values, p_values, d_values, q_values)

#%% Review Residual Errors

from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as pp
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# load data
xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
xseries = xseries.append(yseries)

# prepare data
X = xseries.values
X = X.astype('float32')
train_size = int(len(X))
train, test = X[0:train_size-12], X[train_size-12:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
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
#%% The mean is non-zero (17.627382 residual error)
#   It is possible to (bias) correct this by adding
#   this to each forecast made.

from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data 
xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
xseries = xseries.append(yseries)

# prepare data
X = xseries.values
X = X.astype('float32')
train_size = int(len(X))
train, test = X[0:train_size-12], X[train_size-12:]

# walk-forward validation
history = [x for x in train]
predictions = list()
bias = 17.627382
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
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
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# load data 
xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
xseries = xseries.append(yseries)

# prepare data
X = xseries.values
X = X.astype('float32')
train_size = int(len(X))
train, test = X[0:train_size-12], X[train_size-12:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
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
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# load data
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(6,0,6))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = 17.627382
# save model
model_fit.save(r'data\model.pkl')
numpy.save(r'data\model_bias.npy', [bias])

# The model has been saved as pickke
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
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
months_in_year = 12
model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(series.values, yhat, months_in_year)
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
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# load and prepare datasets
series = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
validation = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()

#%% Validate model 2
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

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

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
xseries = read_csv(r'data\dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
yseries = read_csv(r'data\validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
xseries = xseries.append(yseries)

# prepare data
# X = xseries.values
# X = X.astype('float32')
X = xseries
train_size = int(len(X))
train, test = X[0:train_size-12], X[train_size-12:]

history = [x for x in train] 
validation = [x for x in test]

# load model
model_fit = ARIMAResults.load(r'data\model.pkl')
bias = numpy.load(r'data\model_bias.npy')

# make first prediction
months_in_year = 12

predictions = list()
confsl = list()
confsh = list()
ses = list ()
yhat, se, conf = model_fit.forecast()
yhat = float(yhat)
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat[0])
ses.append(se)
confsl.append(inverse_difference(history, conf[0,0], months_in_year))
confsh.append(inverse_difference(history, conf[0,1], months_in_year))
history.append(validation[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat[0], validation[0]))

# rolling forecasts
for i in range(1, len(validation)):
	# difference data
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(6,0,6))
	model_fit = model.fit(trend='nc', disp=0)
	yhat, se, conf = model_fit.forecast()
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat[0])
	ses.append(se)
	confsl.append(inverse_difference(history, conf[0,0], months_in_year))
	confsh.append(inverse_difference(history, conf[0,1], months_in_year))
	# observation
	obs = validation[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat[0], obs))

# report performance
mse = mean_squared_error(validation, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

# insert period in forecast en actual (graph)
predictionsn = list()
predictionsn.append(train.loc[max(train.index)])
for i in range(len(predictions)):
    predictionsn.append(predictions[i])

testf = X[int(len(train))-1:int(len(train))]
testn = testf
# testn = pd.DataFrame(testn)
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

