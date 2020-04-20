import os
import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from chart_studio import plotly as plt_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

plt.style.use('fivethirtyeight')

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1')
data = pd.read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# The 'MS' string groups the data in buckets by start of the month
y = data
y = y.resample('MS').mean()
y = pd.Series(y)
print(y)

y.plot(figsize=(15, 6))
plt.rcParams.update({'font.size': 9})
plt.show()

result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plt.show()

yshow = y[:len(y-11)]
y, ytest = y[:(len(y)-12)], y[(len(y)-12):len(y)]
# The term bfill means that we use the value before filling in missing values
# To be secure.The data about incidents do not have missing values. So in this
# case the action is obsolete.
y = y.fillna(y.bfill())

result = adfuller(y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

def determine_orders(y, maxp, maxd, maxq, freq):
    # Grid Search
    warnings.filterwarnings("ignore") 
    # specify to ignore warning messages
    p = range(0, maxp)
    d = range(0, maxd)
    q = range(0, maxq)
    parameters = []
    # Generate all different combinations of seasonal p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], freq) for x in list(itertools.product(p, d, q))]
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
                parameters.append([param, param_seasonal, aic])
    result_table = pd.DataFrame(parameters)
    result_table.columns = ['parameters',     'parameters_seasonal','aic']
    result_table = result_table.sort_values(by='aic',ascending = True).reset_index(drop = True)
    minimum = result_table['aic'].min()
    a = result_table.loc[result_table['aic'] == minimum]
    print(' The best combination that gives the lowest AIC is:')
    print(a)
    return a

# optparam  = determine_orders(y, 3, 3, 3, 12)
# optp = optparam.parameters[0][0]
# optd = optparam.parameters[0][1]
# optq = optparam.parameters[0][2]
# optP = optparam.parameters_seasonal[0][0]
# optD = optparam.parameters_seasonal[0][1]
# optQ = optparam.parameters_seasonal[0][2]
# optF = optparam.parameters_seasonal[0][3]
optp = 1
optd = 1
optq = 2
optP = 1
optD = 2
optQ = 2
optF = 11

stepwise_model = auto_arima(y, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())
predm12 = stepwise_model.predict(n_periods=12)
mod = ARIMA(diff, order=arima_order)

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



mod = sm.tsa.statespace.SARIMAX(y,
                                order=(optp, optd, optq),
                                seasonal_order=(optP, optD, optQ, optF),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

# results.plot_diagnostics(figsize=(15,6))
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.rcParams.update({'font.size': 9})
# plt.show()

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
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), end=pd.to_datetime('2019-12-01'), dynamic=False)
pred_ci = pred.conf_int()

yshow.plot(figsize=(15, 6))
ax = ytest.plot(label='gerealiseerd')
pred.predicted_mean.plot(ax=ax, label='voorspeld', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

# The code above requires the forecasts to start at January 1998.
# The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
# We can plot the real and forecasted values of the CO2 time series to assess how well we did. Notice how we zoomed in on the end of the time series by slicing the date index.

ax.set_xlabel('Datum')
ax.set_ylabel('Open')
plt.legend(loc='upper left')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.rcParams.update({'font.size': 9})
plt.show()
#
# maand         Voorspeld       Gerealiseerd
# 2019-01-01    13426.539074    14300
# 2019-02-01    14292.103403    13121
# 2019-03-01    15569.793267    12886
# 2019-04-01    13625.614083    13526
# 2019-05-01    13073.407897    12531
# 2019-06-01    10565.367334    11333
# 2019-07-01    11211.175798    12864
# 2019-08-01    14037.935017    10388
# 2019-09-01    12632.575036    11387
# 2019-10-01    15379.729148    13054
# 2019-11-01    11857.994857    11354
# 2019-12-01     7122.817781    9718
#%%
plt.rcParams.update({'font.size': 9})

series = pd.read_csv(r'data\MAAND OPEN PRD 2014-2019.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)
plt.figure(figsize=(8, 3), dpi=100)
plt.title('Openstaande Incidenten Per Maand')
series.plot()
plt.xlabel('jaren')
plt.ylabel('incidenten')
plt.tight_layout(pad=3.0)
plt.show()

groups = series['2014':'2019'].groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
	years[name.year] = group.values
# Box and Whisker Plots
plt.figure(figsize=(6, 4), dpi=100, edgecolor='k')
years.boxplot()
plt.title('Trend')
plt.tight_layout(pad=3.0)
plt.show()

years = years.transpose()
plt.figure(figsize=(6, 4), dpi=100, edgecolor='k')
years.boxplot()
plt.tight_layout(pad=3.0)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          ['jan', 'feb', 'mrt', 'apr', 'mei', 'jun',
           'jul', 'aug', 'sep', 'okt', 'nov', 'dec'])
plt.title('Seizoen')
plt.show()

