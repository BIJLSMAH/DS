%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels import tsa 
from statsmodels.tsa import stattools as stt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA 

def is_stationary(df, maxlag=15, autolag=None, regression='ct'): 
    """Test if df is stationary using Augmented 
    Dickey Fuller""" 

    adf_test = stt.adfuller(df,maxlag=maxlag, autolag=autolag, regression=regression) 
    adf = adf_test[0]
    cv_5 = adf_test[4]["5%"]

    result = adf < cv_5    
    return result

def d_param(df, max_lag=12):
    d = 0
    for i in range(1, max_lag):
        if is_stationary(df.diff(i).dropna()):
            d = i
            break;
    return d

def ARMA_params(df):
    p, q = tsa.stattools.arma_order_select_ic(df.dropna(),ic='aic').aic_min_order
    return p, q

# read data
# df met index
df = read_csv(r'C:\Users\Administrator\Documents\GitHub\DS\Hessel, TimeSeries 1\data\MAAND OPEN PRD 2014-2019.csv', parse_dates=True, index_col=0)
df['open'] = df.open.astype('float32')
df.index=pd.to_datetime(df.index)
pd.Series(df.open)
# tijdr = tijdr.open.resample('M').mean()


# carsales = pd.read_csv('data/monthly-car-sales-in-quebec-1960.csv', 
#                    parse_dates=['Month'],  
#                    index_col='Month',  
#                    date_parser=lambda d:pd.datetime.strptime(d, '%Y-%m'))
# carsales = carsales.iloc[:,0] 

# get components
df_decomp = seasonal_decompose(df, freq=12)
residuals = df - df_decomp.seasonal - df_decomp.trend 
residuals = residuals.dropna()

# fit model
d = d_param(df, max_lag=12)
p, q = ARMA_params(residuals)
model = ARIMA(residuals, order=(p, d, q)) 
model_fit = model.fit() 

# plot prediction
model_fit.plot_predict(start='2019-01-01', end='2019-12-01', alpha=0.10) 
plt.legend(loc='upper left') 
plt.xlabel('Year') 
plt.ylabel('Sales')
plt.title('Residuals 2019')
print(arimares.aic, arimares.bic)