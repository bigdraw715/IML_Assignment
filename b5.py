import matplotlib.pyplot as plt # Draw graph.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from preprocess import *
import pandas as pd # Load and manipulate data and for One-Hot Encoding.
import numpy as np # Data manipulation.
import re

def PCA_feature(X, y, features):

    if features == 'all':
        X_sub = X
    else:
        X_sub = X.loc[:,~X.columns.isin(features)]

    #print(list(X_sub))

    #X_scale = scale_norm(X_sub)

    scaler = StandardScaler()
    scaler.fit(X_sub)
    X_scale = scaler.transform(X_sub)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scale)
    plt.figure(num = 1, figsize = (10,10))
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)

    obj = re.split(r'_', features[0])
    plt.savefig(obj[0]+'.png')
    plt.close()

X, y_lst = data_preprocess(method="onehot")
y = y_lst['Total']

X = X[['year',
   'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
   'temperature',
   'feeling_temperature', 
   'humidity',
   'windspeed', 
   'season_fall', 'season_spring', 'season_summer', 'season_winter',
   'month_Apr', 'month_Aug', 'month_Dec', 'month_Feb','month_Jan', 'month_Jul', 'month_Jun', 'month_Mar', 'month_May', 'month_Nov', 'month_Oct', 'month_Sep',
   'weekday_Fri', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun', 'weekday_Thu', 'weekday_Tue', 'weekday_Wed',
   'weather_Clear or partly cloudy', 'weather_Heavy rain','weather_Light rain', 'weather_Mist']]

PCA_feature(X, y, ['all'])
PCA_feature(X, y, ['year'])
PCA_feature(X, y, ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'])
PCA_feature(X, y, ['temperature'])
PCA_feature(X, y, ['feeling_temperature'])
PCA_feature(X, y, ['humidity'])
PCA_feature(X, y, ['windspeed'])
PCA_feature(X, y, ['season_fall', 'season_spring', 'season_summer', 'season_winter'])
PCA_feature(X, y, ['month_Apr', 'month_Aug', 'month_Dec', 'month_Feb','month_Jan', 'month_Jul', 'month_Jun', 'month_Mar', 'month_May', 'month_Nov', 'month_Oct', 'month_Sep'])
PCA_feature(X, y, ['weekday_Fri', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun', 'weekday_Thu', 'weekday_Tue', 'weekday_Wed'])
PCA_feature(X, y, ['weather_Clear or partly cloudy', 'weather_Heavy rain','weather_Light rain', 'weather_Mist'])