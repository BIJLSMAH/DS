from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def ARIMA1_train(train_data):
    model = ARIMA(train_data, order=(1, 1, 1)).fit(disp=-1)
    return model

def ARIMA1_pred(getraind_model, predictie_lengte):
    predictie, se, conf = getraind_model.forecast(predictie_lengte, alpha=0.05) 
    return predictie 

def ARIMA2_train(train_data):
    model = ARIMA(train_data, order=(2, 1, 2)).fit(disp=-1)
    return model

def ARIMA2_pred(getraind_model, predictie_lengte):
    predictie, se, conf = getraind_model.forecast(predictie_lengte, alpha=0.05) 
    return predictie 

def ExpSmooth_train(train_data):
    model = ExponentialSmoothing(train_data).fit()
    return model

def ExpSmooth_pred(getraind_model, predictie_lengte):
    predictie = getraind_model.forecast(predictie_lengte)
    return predictie 
