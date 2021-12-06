import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from preprocess import data_preprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

X_class,y_class = data_preprocess(method="class")
# Define "classifiers" to be used
classifiers = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=0.25),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
        contamination=0.25
    ),
    "OCSVM": OneClassSVM(nu=0.25, gamma=0.35),
}
colors = ["r", "g", "b"]
legend1 = {}
legend2 = {}

# Get data
X1 = np.array(X_class[["temperature",'feeling_temperature']])  # two clusters
print("plotting the first ficture, takes about 2min")
# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(-20, 60, 500), np.linspace(-20, 60, 500))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1,figsize=(10,10))
    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i]
    )

legend1_values_list = list(legend1.values())
legend1_keys_list = list(legend1.keys())



# Plot the results (= shape of the data points cloud)
plt.figure(1,figsize = (10,10))  # two clusters
plt.title("Outlier Detection")
plt.scatter(X1[:, 0], X1[:, 1], color="black")
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate(
    "outlying points",
    xy=(4, 2),
    xycoords="data",
    textcoords="data",
    xytext=(3, 1.25),
    bbox=bbox_args,
    arrowprops=arrow_args,
)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend(
    (
        legend1_values_list[0].collections[0],
        legend1_values_list[1].collections[0],
        legend1_values_list[2].collections[0],
    ),
    (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
    loc="upper center",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.ylabel("temperature"),
plt.xlabel('feeling_temperature')
plt.savefig("outlier1.png")
plt.show()
print("plotting the second ficture ,takes about 2min, don't close the window")
X1 = np.array(X_class[["windspeed",'humidity']])  # two clusters
# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(-20, 130, 500), np.linspace(-20, 130, 500))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1,figsize=(10,10))
    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i]
    )

legend1_values_list = list(legend1.values())
legend1_keys_list = list(legend1.keys())

# Plot the results (= shape of the data points cloud)
plt.figure(1,figsize = (10,10))  # two clusters
plt.title("Outlier Detection")
plt.scatter(X1[:, 0], X1[:, 1], color="black")
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate(
    "outlying points",
    xy=(4, 2),
    xycoords="data",
    textcoords="data",
    xytext=(3, 1.25),
    bbox=bbox_args,
    arrowprops=arrow_args,
)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend(
    (
        legend1_values_list[0].collections[0],
        legend1_values_list[1].collections[0],
        legend1_values_list[2].collections[0],
    ),
    (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
    loc="upper center",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.ylabel("humidity"),
plt.xlabel('windspeed')
plt.savefig("outlier2.png")
plt.show()
print(
'''
                            #####       #####  
#####   ##    ####  #    # #     #     #     # 
  #    #  #  #      #   #        #           # 
  #   #    #  ####  ####    #####       #####  
  #   ######      # #  #   #       ###       # 
  #   #    # #    # #   #  #       ### #     # 
  #   #    #  ####  #    # ####### ###  #####  
'''
)
