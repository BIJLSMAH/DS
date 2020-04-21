import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from datetime import datetime

def monthlist(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1 ,1))
    return mlist

os.chdir(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1')
data = pd.read_csv(r'data\MAAND OPEN PRD 2014-2019.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# The 'MS' string groups the data in buckets by start of the month
y = data
y = y.resample('MS').mean()
y = pd.Series(y)
print(y)
plt.style.use('default')

result = seasonal_decompose(data, model='additive')
plt.figure(figsize=(10, 10), dpi=100)
plt.suptitle('')
result.plot()
plt.show()

result = seasonal_decompose(data, model='multiplicative')
plt.figure(figsize=(10, 10), dpi=100)
plt.suptitle('')
result.plot()
plt.show()

plt.figure(figsize=(8, 2), dpi=100)
# plt.title('Trend')
plt.plot(result.trend)
plt.xlabel('Jaren')
plt.ylabel('Trend')
plt.show()
plt.figure(figsize=(8, 2), dpi=100)
# plt.title('Seasonality')
plt.plot(result.seasonal)
plt.xlabel('Jaren')
plt.ylabel('Seasonality')
plt.show()
plt.figure(figsize=(8, 2), dpi=100)
# plt.title('Residual')
plt.plot(result.resid)
plt.xlabel('Jaren')
plt.ylabel('Residual')
plt.show()

plt.rcParams.update({'figure.figsize': (10, 10)})
result.plot().suptitle('Multiplicatieve Decompositie', fontsize=22)
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

stepwise_model = auto_arima(y, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())
# Doorloop de maanden en verzamel maandgegevens over de totalen
dates=["2019-01-01", "2020-12-01"]
mndLijst = monthlist(dates)

predm12 = pd.Series(stepwise_model.predict(n_periods=len(ytest)*2), index=mndLijst)
lower_series = ytest - 0.15 * ytest
upper_series = ytest + 0.15 * ytest
plt.figure(figsize=(8, 3), dpi=100)
plt.plot(yshow, label='training')
plt.plot(ytest, label='actual')
plt.plot(predm12, label='forecast')
plt.xlabel('jaren')
plt.ylabel('incidenten')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.tight_layout(pad=3.0)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#%% graphs
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

plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 9})
groups = series['2014':'2019'].groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
	years[name.year] = group.values
# Box and Whisker Plots
plt.figure(figsize=(8, 3), dpi=100, edgecolor='k')
years.boxplot()
plt.title('Trend')
plt.tight_layout(pad=3.0)
plt.show()

years = years.transpose()
plt.figure(figsize=(8, 3), dpi=100, edgecolor='k')
years.boxplot()
plt.tight_layout(pad=3.0)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          ['jan', 'feb', 'mrt', 'apr', 'mei', 'jun',
           'jul', 'aug', 'sep', 'okt', 'nov', 'dec'])
plt.title('Seizoen')
plt.show()

