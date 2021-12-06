from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from preprocess import *

# for decision tree regressor exp

def DecisionTree(X_train, X_test, y_train, y_test):

    regressor = DecisionTreeRegressor(criterion = 'squared_error', max_depth = 40, min_samples_split = 20, min_samples_leaf =2)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)

    score = regressor.score(X_test, y_test)

    print('mse:', mse)
    print('score:', score)

X, y_lst = data_preprocess()
y = y_lst['Subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print('Subscribed task:')
DecisionTree(X_train, X_test, y_train, y_test)

y = y_lst['Non-subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print('Non-subscribed task:')
DecisionTree(X_train, X_test, y_train, y_test)
print(
'''
                             #       #       
#####   ##    ####  #    #  ##       #    #  
  #    #  #  #      #   #  # #       #    #  
  #   #    #  ####  ####     #       #    #  
  #   ######      # #  #     #   ### ####### 
  #   #    # #    # #   #    #   ###      #  
  #   #    #  ####  #    # ##### ###      #  
'''
    )
