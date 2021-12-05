import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import data_preprocess
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
import time

X_onehot,y_onehot = data_preprocess(method="onehot")
X_class,y_class = data_preprocess(method="class")
print("Use class encoded data")
time.sleep(2)
print("One vs One calssifier:")
X = np.array(X_class)
y = np.array(y_class["category"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = OneVsOneClassifier(
    LogisticRegression(max_iter = 10000)).fit(X_train, y_train)
# clf = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
print("Train Accuracy:",clf.score(X_train,y_train))
print("Test Accuracy:",clf.score(X_test,y_test))

print("One vs All calssifier:")
X = np.array(X_class)
y = np.array(y_class["category"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = OneVsRestClassifier(
    LogisticRegression(max_iter = 10000)).fit(X_train, y_train)
# clf = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
print("Train Accuracy:",clf.score(X_train,y_train))
print("Test Accuracy:",clf.score(X_test,y_test))


print("\n Use onehot encoded data")
time.sleep(2)
print("One vs One calssifier:")
X = np.array(X_onehot)
y = np.array(y_onehot["category"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = OneVsOneClassifier(
    LogisticRegression(max_iter = 10000)).fit(X_train, y_train)
# clf = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
print("Train Accuracy:",clf.score(X_train,y_train))
print("Test Accuracy:",clf.score(X_test,y_test))

print("One vs All calssifier:")
X = np.array(X_onehot)
y = np.array(y_onehot["category"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=0)
clf = OneVsRestClassifier(
    LogisticRegression(max_iter = 10000)).fit(X_train, y_train)
# clf = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
print("Train Accuracy:",clf.score(X_train,y_train))
print("Test Accuracy:",clf.score(X_test,y_test))
