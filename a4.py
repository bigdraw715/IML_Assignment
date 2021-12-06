from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
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

def plot_dtr(X_train, X_test, y_train, y_test, name):

    train_list = []
    test_list = []

    idx = [2,3,4,6,8,10,12,14,16,18,20] # for max_depth
    for i in idx:
      regressor = DecisionTreeRegressor(max_depth = i, min_samples_split = 20, min_samples_leaf =2)
      regressor.fit(X_train, y_train)
      #print('train:', regressor.score(X_train, y_train))
      train_list.append(regressor.score(X_train, y_train))
      y_predict = regressor.predict(X_test)
      #print('test', regressor.score(X_test, y_test))
      test_list.append(regressor.score(X_test, y_test))

    arr = np.append([train_list], [test_list], axis = 0)
    pf = pd.DataFrame(arr.T,index=idx,columns=['train set score','test set score'])
    ax = pf.plot(kind = 'bar', title='Prediction on '+name)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Score")
    plt.savefig(name+'.png')


X, y_lst = data_preprocess()
y = y_lst['Subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print('Subscribed task:')
DecisionTree(X_train, X_test, y_train, y_test)
plot_dtr(X_train, X_test, y_train, y_test, 'Subscribed')

y = y_lst['Non-subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print('Non-subscribed task:')
DecisionTree(X_train, X_test, y_train, y_test)
plot_dtr(X_train, X_test, y_train, y_test, 'Non-subscribed')

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
