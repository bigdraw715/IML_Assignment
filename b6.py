from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from preprocess import *

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def cluster_PCA(X, y):

    model = AgglomerativeClustering(n_clusters=4).fit(X)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(num = 1, figsize = (15,15))
    plt.title("Project Clusters to PCA Space")
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=model.labels_)
    plt.savefig('cluster_PCA.png')
    plt.close()
    print()

def cluster_trans(X, y):

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=4).fit(X_reduced)

    plt.figure(num = 1, figsize = (15,15))
    plt.title("Clustering PCA-transformed Data")
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=model.labels_)
    plt.savefig('cluster_trans.png')
    plt.close()

def cluster(X, y):

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)

    plt.figure(num = 1, figsize = (5,5))
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="level", p=4)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig('cluster.png')
    plt.close()

X, y_lst = data_preprocess()
y = y_lst['Total']

cluster(X, y)
cluster_trans(X, y)
cluster_PCA(X, y)
print(
'''
                            #####       #####  
#####   ##    ####  #    # #     #     #     # 
  #    #  #  #      #   #        #     #       
  #   #    #  ####  ####    #####      ######  
  #   ######      # #  #   #       ### #     # 
  #   #    # #    # #   #  #       ### #     # 
  #   #    #  ####  #    # ####### ###  #####  
'''
    )