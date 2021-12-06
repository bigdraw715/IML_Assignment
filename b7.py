from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def RandomForest(X_train, X_test, y_train, y_test):

    # parameter tuning
    param_dict = dict(n_estimators = range(0,100,10),
                        criterion = ['mse','mae', 'poisson'],
                        max_depth = range(2,40,2))
    #'criterion': ('squared_error','absolute_error', 'poisson')
    rf = RandomizedSearchCV(RandomForestRegressor(), param_dict, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_y_pred = rf.predict(X_test)

    rf_mse = mean_squared_error(rf_y_pred, y_test)
    rf_score = rf.score(X_test, y_test)

    #out of bag
    p_dict = rf.best_params_
    rf_oob_y_pred = cross_val_predict(RandomForestRegressor(n_estimators = p_dict[0], 
                                    criterion = p_dict[1], 
                                    max_depth = p_dict[2],
                                    oob_score = True), 
                                X_train, y_train, n_jobs=-1)

    rf_obb_mse = mean_squared_error(rf_oob_y_pred, y_test)
    rf_oob_score = r2_score(rf_oob_y_pred, y_test)


    print('rf_mse:', rf_mse)
    print('rf_score:', rf_score)
    print('rf_obb_mse:', rf_obb_mse)
    print('rf_oob_score:', rf_oob_score)

X, y_lst = data_preprocess()
y = y_lst['Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)

RandomForest(X_train, X_test, y_train, y_test)
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