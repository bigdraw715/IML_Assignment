import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import data_preprocess
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
import time
from sklearn.linear_model import Perceptron
import matplotlib.pyplot  as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')


def var_name(var,all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]


X_onehot,y_onehot = data_preprocess(method="onehot")
X_class,y_class = data_preprocess(method="class")
year = np.array(X_onehot['year']).reshape(-1,1)
temperature = np.array(X_onehot['temperature']).reshape(-1,1)
feeling_temperature= np.array(X_onehot['temperature']).reshape(-1,1)
humidity = np.array(X_onehot['humidity']).reshape(-1,1)
windspeed = np.array(X_onehot['windspeed']).reshape(-1,1)
season = np.array(X_onehot[['season_fall', 'season_spring', 'season_summer', 'season_winter']])
hour = np.array(X_onehot[['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                            'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                            'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                            'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']])
month = np.array(X_onehot[['month_Apr',
                            'month_Aug', 'month_Dec', 'month_Feb', 'month_Jan', 'month_Jul',
                            'month_Jun', 'month_Mar', 'month_May', 'month_Nov', 'month_Oct',
                            'month_Sep', 'weekday_Fri']])
weekday = np.array(X_onehot[['weekday_Fri', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
                            'weekday_Thu', 'weekday_Tue', 'weekday_Wed']])                         




print("\nTest with all features using Perceptron classifier:")
X = np.concatenate((year,temperature,feeling_temperature,
                    humidity,windspeed,season,hour,month,weekday,np.array(y_onehot)),axis =1)
y = np.array(X_class["weather"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = Perceptron(random_state=0).fit(X_train, y_train)
print("Train Accuracy",clf.score(X_train,y_train))
print("Test Accuracy",clf.score(X_test,y_test))
print("Predict Example")
print(y_test[:30])
print(clf.predict(X_test)[:30])


from scipy import stats
wea = X_class["weather"].to_numpy()
print("\n corrilation between features and weather:(Measured by Pearson correlation coefficient)")
for i in X_class.keys():
    pccs = stats.pearsonr(X_class[i], wea)
    print(i)
    print(pccs)
for i in y_class.keys():
    pccs = stats.pearsonr(y_class[i], wea)
    print(i)
    print(pccs)



print("\nTest all feature combined with humidity:")
for feature in [year,temperature,feeling_temperature,windspeed,season,hour,month,weekday,np.array(y_onehot)]:
    X = np.concatenate((feature,humidity),axis =1)
    print(var_name(feature))
    X = feature
    y = np.array(X_class["weather"]).astype(int)
    # for i in range(len(y)):
    #     if y[i]!=3:
    #         y[i]=0
    #     else:
    #         y[i]=1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=True, random_state=0)
    clf = Perceptron(random_state=0).fit(X_train, y_train)
    print(clf.score(X_train,y_train))
    print(clf.score(X_test,y_test))
    print(y_test[:30])
    print(clf.predict(X_test)[:30])

print("\nTurn into binary task:")
X = np.concatenate((year,temperature,feeling_temperature,
                    humidity,windspeed,season,hour,month,weekday,np.array(y_onehot)),axis =1)
y = np.array(X_class["weather"]).astype(int)
for i in range(len(y)):
    if y[i]!=3:
        y[i]=0
    else:
        y[i]=1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = Perceptron(random_state=0).fit(X_train, y_train)
print("Train Accuracy",clf.score(X_train,y_train))
print("Test Accuracy",clf.score(X_test,y_test))
print("Predict Example")
print(y_test[:30])
print(clf.predict(X_test)[:30])


print("\nTest on OnevsOne classifier:")
X = np.concatenate((temperature,feeling_temperature,
                    humidity,windspeed,season,hour,month,weekday,np.array(y_onehot)),axis =1)
y = np.array(X_class["weather"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = OneVsOneClassifier(Perceptron(random_state=0)).fit(X_train, y_train)
print("Train Accuracy",clf.score(X_train,y_train))
print("Test Accuracy",clf.score(X_test,y_test))
print("Predict Example")
print(y_test[:30])
print(clf.predict(X_test)[:30])


print("\nTest scalars of learning rate in Single task:")
X = np.concatenate((temperature,feeling_temperature,
                    humidity,windspeed,season,hour,month,weekday,np.array(y_onehot)),axis =1)
y = np.array(X_class["weather"]).astype(int)
for i in range(len(y)):
    if y[i]!=3:
        y[i]=2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
trainaccuracy = []
testaccuracy = []
ab = [1e-6,1e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6]
for i in ab:
    clf = SGDClassifier(loss="perceptron", eta0=i, learning_rate="constant", penalty=None).fit(X_train, y_train)
    trainaccuracy.append(clf.score(X_train,y_train))
    testaccuracy.append(clf.score(X_test,y_test))
plt.plot(ab,trainaccuracy,label = "Train")
plt.plot(ab,testaccuracy,label = "Test")
plt.xscale("log")
plt.legend()
plt.show()