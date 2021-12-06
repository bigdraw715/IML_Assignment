import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import requests
import io
import os

def classify(dataset):
    pd.options.mode.chained_assignment = None
    weather = np.array(['Heavy rain', 'Light rain', 'Mist','Clear or partly cloudy'])
    weekday =np.array(['Mon','Tue', 'Wed','Thu','Fri', 'Sat', 'Sun'])
    season = np.array(['spring','summer',"fall",'winter'])
    for i in range(len(dataset)):
        a = int(np.where( weather==dataset["weather"][i])[0])
        b = int(np.where( season==dataset["season"][i])[0])
        c = int(np.where( weekday==dataset["weekday"][i])[0])
        dataset["weather"][i]=a
        dataset["season"][i]=b
        dataset["weekday"][i]=c       
    return dataset

def scale_norm(X):

    scale = StandardScaler().fit(X)
    X_scale = scale.transform(X)

    norm = Normalizer().fit(X_scale)
    X_norm = norm.transform(X_scale)

    return X_norm

def data_preprocess(method = "scale",visualize= False):
    print("processing data....")

    dataset = pd.read_csv('./Dataset.csv')
    total = np.array(dataset["Total"])
    category = []
    for to in total:
        if to>=0 and to<=27:
            category.append(0)
        elif to>27 and to<=98:
            category.append(1)
        elif to>98 and to<=189:
            category.append(2)
        elif to>189 and to<=321:
            category.append(3)
        elif to>321:
            category.append(4)
        else:
            print("wrong")
    
    
    dataset.insert(15, 'category', category)
    
    if visualize == True:
        visualization(dataset)

    if method == "scale":
        y = dataset[['Subscribed','Non-subscribed','Total','category']]
        onehot_X_dataset = dataset[['season','year','month','hour','weekday','weather','temperature','feeling_temperature','humidity','windspeed']]
        need_onehot = ['season','hour','month','weekday','weather']
        onehot_X_dataset = pd.get_dummies(onehot_X_dataset,columns=need_onehot)

        X_final = scale_norm(onehot_X_dataset)
        print('preprocessing done!')
        return X_final, y

    if method ==  "onehot":
        y = dataset[['Subscribed','Non-subscribed','Total','category']]
        onehot_X_dataset = dataset[['season','year','month','hour','weekday','weather','temperature','feeling_temperature','humidity','windspeed']]
        need_onehot = ['season','hour','month','weekday','weather']
        onehot_X_dataset = pd.get_dummies(onehot_X_dataset,columns=need_onehot)

        print('preprocessing done!')
        return onehot_X_dataset, y
    
    if method == "class":
        y = dataset[['Subscribed','Non-subscribed','Total','category']]
        dates = np.array(dataset["date"])
        monthday = []
        month = []
        year = []
        day = []
        for date in dates:
            splited = date.split("-")
            monthday.append(int(splited[1]+splited[2]))
            month.append(int(splited[1]))
            day.append(int(splited[2]))
            year.append(int(splited[0]))
        dataset.insert(0, 'year1', year)
        dataset.insert(1,"month1",month)
        dataset.insert(2, 'day1', day)
        dataset = classify(dataset)     
        dataset = dataset.drop(['Unnamed: 0','instant','date',"month"],axis=1)
        X = dataset[['year1','month1','day1','season','year','hour','weekday','weather','temperature','feeling_temperature','humidity','windspeed']]
        y =dataset[['Subscribed','Non-subscribed','Total','category']]

        print('preprocessing done!')
        return X, y

def count_B(dataset):
    weather = Counter(np.array(dataset['weather']))
    cate = Counter(np.array(dataset['category']))
    return weather,cate

def visualization(dataset):
    weather, cate = count_B(dataset)

    plt.pie(list(cate.values()), labels=tuple(cate.keys()), 
                    autopct='%1.2f%%')
    plt.show()


    plt.pie(list(weather.values()), labels=tuple(weather.keys()), 
                    autopct='%1.2f%%')
    plt.show()

    plt.hist(np.array(dataset['temperature']),edgecolor = "white")
    plt.ylabel("Frequency")
    plt.xlabel("Temperature Range")
    plt.show()

    plt.hist(np.array(dataset['feeling_temperature']),edgecolor = "white")
    plt.ylabel("Frequency")
    plt.xlabel("feeling_temperature Range")
    plt.show()
    plt.hist(np.array(dataset['humidity']),edgecolor = "white")
    plt.ylabel("Frequency")
    plt.xlabel("humidity Range")
    plt.show()
    plt.hist(np.array(dataset['windspeed']),edgecolor = "white")
    plt.ylabel("Frequency")
    plt.xlabel("windspeed Range")
    plt.show()
    plt.hist(np.array(dataset['Total']),edgecolor = "white")
    plt.ylabel("Frequency")
    plt.xlabel("Total Range")
    plt.show()
    dataset.temperature.plot(kind = 'kde', color = 'r', label = 'temperature')
    dataset.feeling_temperature.plot(kind = 'kde', color = 'g', label = 'feeling_temperature')
    dataset.humidity.plot(kind = 'kde', color = 'b', label = 'humidity')
    dataset.windspeed.plot(kind = 'kde', color = 'pink', label = 'windspeed')
    plt.xlabel("Feature Vlaue Range")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
