import pandas as pd # Load and manipulate data and for One-Hot Encoding.
import numpy as np # Data manipulation.
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import *

X, y_lst = data_preprocess()
y = y_lst['Subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)

def SVM(X_train, X_test, y_train, y_test, kernel):

    mse_lst = []
    r2_lst = []

    model = cross_validate(SVR(kernel=kernel), X_train, y_train, n_jobs=-1, return_estimator = True)
    for m in model['estimator']:
        y_pred = m.predict(X_test)
        mse_lst.append(mean_squared_error(y_pred, y_test))
        r2_lst.append(m.score(X_test, y_test))

    mse = np.mean(mse_lst)
    r2 = np.mean(r2_lst)

    print(kernel+'_mse:', mse, kernel+'_r2:', r2)

SVM(X_train, X_test, y_train, y_test, 'rbf')
SVM(X_train, X_test, y_train, y_test, 'linear')
SVM(X_train, X_test, y_train, y_test, 'poly')