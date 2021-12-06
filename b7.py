from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from preprocess import *

def RandomForest(X_train, X_test, y_train, y_test):

    # parameter tuning
    param_dict = dict(n_estimators = range(10,50,5),
                        criterion = ['squared_error','absolute_error', 'poisson'],
                        max_depth = range(2,20,2))

    rf = RandomizedSearchCV(RandomForestRegressor(), param_dict, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_dict = rf.best_params_
    print('best param:', p_dict)

    rf_y_pred = rf.predict(X_test)

    rf_mse = mean_squared_error(rf_y_pred, y_test)
    rf_score = rf.score(X_test, y_test)

    print('rf_mse:', rf_mse)
    print('rf_score:', rf_score)

    return p_dict

def rf_oob(X, y, p_dict):

    #out of bag

    rf_oob = RandomForestRegressor(n_estimators = p_dict['n_estimators'], 
                                    criterion = p_dict['criterion'], 
                                    max_depth = p_dict['max_depth'],
                                    oob_score = True,
                                    random_state = 0)
    rf_oob.fit(X, y)

    print('rf_oob_score:', rf_oob.oob_score_)

X, y_lst = data_preprocess()
y = y_lst['Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)

p_dict = RandomForest(X_train, X_test, y_train, y_test)
rf_oob(X, y, p_dict)
print(
'''
                            #####      ####### 
#####   ##    ####  #    # #     #     #    #  
  #    #  #  #      #   #        #         #   
  #   #    #  ####  ####    #####         #    
  #   ######      # #  #   #       ###   #     
  #   #    # #    # #   #  #       ###   #     
  #   #    #  ####  #    # ####### ###   #     
'''
    )
