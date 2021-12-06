from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from preprocess import data_preprocess
from sklearn.model_selection import cross_validate
X_onehot,y_onehot = data_preprocess(method="onehot")
rig = Ridge(alpha = 2)
scores = cross_validate(rig, X_onehot, y_onehot, cv=5, scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
print("Cross Validation Train MSE:",np.mean(scores['train_neg_mean_squared_error']))
print("Cross Validation Test MSE:",np.mean(scores['test_neg_mean_squared_error']))
print("Cross Validation Train R2:",np.mean(scores['train_r2']))
print("Cross Validation Test R2:",np.mean(scores['test_r2']))
print(
'''
                             #        #####  
#####   ##    ####  #    #  ##       #     # 
  #    #  #  #      #   #  # #             # 
  #   #    #  ####  ####     #        #####  
  #   ######      # #  #     #   ###       # 
  #   #    # #    # #   #    #   ### #     # 
  #   #    #  ####  #    # ##### ###  #####  
'''
)
