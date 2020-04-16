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
import warnings
#%%
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
#%%
os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 2')
# series = read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
df = pd.read_excel(r'data\Superstore.xls')
furniture = df.loc[df['Category'] =='Furniture']
furniture['Order Date'].min(), furniture['Order Date'].max()
# Verwijderen van kolommen die niet nodig zijn
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# Indexing met tijdreeksgegevens

furniture = furniture.set_index('Order Date')
furniture.index

# Nu per dag, tijdreeks definieren per maand, uitgaande van het gemiddelde
# voor die maand, met als timestamp de eerste dag van de maand.
y = furniture['Sales'].resample('MS').mean()
y['2017':]

#%%
# plot nu maar even de resultaten.
y.plot(figsize=(15,6))
plt.show()

# Some distinguishable patterns appear when we plot the data. 
# The time-series has seasonality pattern, such as sales are always low 
# at the beginning of the year and high at the end of the year. There is 
# always an upward trend within any single year with a couple of low months 
# in the mid of the year.
# We can also visualize our data using a method called time-series 
# decomposition that allows us to decompose our time series into three 
# distinct components: trend, seasonality, and noise.

from pylab import rcParams
rcParams['figure.figsize'] =18,8

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')
fig = decomposition.plot()
plt.show()


#%% Time series forecasting with ARIMA
# We are going to apply one of the most commonly used method for 
# time-series forecasting, known as ARIMA, which stands for 
# Autoregressive Integrated Moving Average.
# ARIMA models are denoted with the notation ARIMA(p, d, q). These three 
# parameters account for seasonality, trend, and noise in data:

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# This step is parameter Selection for our furniture’s sales ARIMA Time Series Model. 
# Our goal here is to use a “grid search” to find the optimal set of parameters 
# that yields the best performance for our model.
optimal_results_aic = 9999.0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            if results.aic < optimal_results_aic:
                optimal_param, optimal_param_seasonal, optimal_results_aic = param, param_seasonal, results.aic
        except:
            continue
print('Optimaal ARIMA{}x{}12 - AIC:{}'.format(optimal_param, optimal_param_seasonal, optimal_results_aic))

#%% Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1,1,1), seasonal_order=(1,1,0,12),enforce_stationarity = False,enforce_invertibility= False)
results=mod.fit()
print(results.summary().tables[1])

# We should always run model diagnostics to investigate any unusual behavior
results.plot_diagnostics(figsize=(16,8))
plt.show()

#%% Validating forecasts
# To help us understand the accuracy of forecasts, we compare predicted sales to
# real series of the time series, and we set forecasts to start at 2017-01-01 to
# the end of the data

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic = False)
pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha = .7, figsize=(14,7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# the lineplot is showing the observed values compared to the rolling forecast
# predictions. Overall, our forecasts align with the true values very well, showing
# an upward trend starting from the beginning of the year and captured the 
# seasonality toward the end of the year.

y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) **2).mean()
# In statistics, the mean squared error (MSE) of an estimator measures 
# the average of the squares of the errors — that is, the average 
# squared difference between the estimated values and what is estimated. 
# The MSE is a measure of the quality of an estimator — it is always 
# non-negative, and the smaller the MSE, the closer we are to finding 
# the line of best fit.

print('The mse of our forecasts is {}'.format(round(mse,2)))

# Root Mean Square Error (RMSE) tells us that our model was able to forecast 
# the average daily furniture sales in the test set within 151.64 of the real 
# sales. Our furniture daily sales range from around 400 to over 1200. 
# In my opinion, this is a pretty good model so far.

print ('The rmse of our forecasts is {}'.format(round(np.sqrt(mse),2)))


#%% Producing and visualizing forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()