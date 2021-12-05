from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
from preprocess import *

X, y_lst = data_preprocess()
y = y_lst['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
regr = MLPRegressor(random_state=0,
                    hidden_layer_sizes=(100,100,100),
                    max_iter=1000,
                    learning_rate= "adaptive"
                    ).fit(X_train, y_train)
print("rsquared:",regr.score(X_test, y_test))
pred = regr.predict(X_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("mse:",mean_squared_error(pred,y_test))
print(
'''
                            #####       #####  
#####   ##    ####  #    # #     #     #     # 
  #    #  #  #      #   #        #     #     # 
  #   #    #  ####  ####    #####       #####  
  #   ######      # #  #   #       ### #     # 
  #   #    # #    # #   #  #       ### #     # 
  #   #    #  ####  #    # ####### ###  #####  

'''
    )