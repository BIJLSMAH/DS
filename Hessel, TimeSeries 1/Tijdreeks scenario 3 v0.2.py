import os
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1')
data = read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# The 'MS' string groups the data in buckets by start of the month
y = data
y = y.resample('MS').mean()

# The term bfill means that we use the value before filling in missing values
# To be secure.The data about incidents do not have missing values. So in this
# case the action is obsolete.
y = y.fillna(y.bfill())

print(y)

y.plot(figsize=(15, 6))
plt.show()

# Grid Search
p = d = q = range(0,3)
parameters = []
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
        except:
            continue
        aic = results.aic
        parameters.append([param,param_seasonal,aic])

result_table = pd.DataFrame(parameters)
result_table.columns = ['parameters',     'parameters_seasonal','aic']
result_table = result_table.sort_values(by='aic',ascending = True).reset_index(drop = True)
minimum = result_table['aic'].min()
a = result_table.loc[result_table['aic'] == minimum]
print(' The best combination that gives the lowest AIC is:')
print(a)



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0,3)
p = 1
d = 2
q = 2 
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

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
        except:
            continue


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 2, 2),
                                seasonal_order=(1, 2, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Our primary concern is to ensure that the residuals of our model are 
# uncorrelated and normally distributed with zero-mean. If the seasonal 
# ARIMA model does not satisfy these properties, it is a good indication 
# that it can be further improved.
# In this case, our model diagnostics suggests that the model residuals 
# are normally distributed based on the following:
# 1.    In the top right plot, we see that the red KDE line follows closely 
#       with the N(0,1) line (where N(0,1)) is the standard notation for a 
#       normal distribution with mean 0 and standard deviation of 1). 
#       This is a good indication that the residuals are normally distributed.
# 2.    The qq-plot on the bottom left shows that the ordered distribution 
#       of residuals (blue dots) follows the linear trend of the samples taken 
#       from a standard normal distribution with N(0, 1). Again, 
#       this is a strong indication that the residuals are normally distributed.
# 3.    The residuals over time (top left plot) don’t display any obvious 
#       seasonality and appear to be white noise. This is confirmed by the 
#       autocorrelation (i.e. correlogram) plot on the bottom right, 
#       which shows that the time series residuals have low correlation with 
#       lagged versions of itself.
# Those observations lead us to conclude that our model produces a 
# satisfactory fit that could help us understand our time series data and 
# forecast future values.
# Although we have a satisfactory fit, some parameters of our seasonal 
# ARIMA model could be changed to improve our model fit. For example, our 
# grid search only considered a restricted set of parameter combinations, 
# so we may find better models if we widened the grid search.

# Step 6 — Validating Forecasts
# We have obtained a model for our time series that can now be used to 
# produce forecasts. We start by comparing predicted values to real values 
# of the time series, which will help us understand the accuracy of our 
# forecasts. The get_prediction() and conf_int() attributes allow us 
# to obtain the values and associated confidence intervals for forecasts
# of the time series.


# predicted_mean
# var_predicted_mean
# var_resid
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()

y.plot(figsize=(15, 6))
ax = y['2019':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

# The code above requires the forecasts to start at January 1998.
# The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
# We can plot the real and forecasted values of the CO2 time series to assess how well we did. Notice how we zoomed in on the end of the time series by slicing the date index.

ax.set_xlabel('Datum')
ax.set_ylabel('Open')
plt.legend()

plt.show()