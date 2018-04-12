import pickle
import numpy as np
import datetime as dt
from sklearn import preprocessing
from sklearn import cluster
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import itertools


def plot_kmeans(X, attributes):
    #plotting elbow curve to find optimum no of k    
    k_choices = range(1, 7)
    kmeans_model = [cluster.KMeans(n_clusters = k) for k in k_choices]
    #run all models only, empty lhs as data is stored automatically in model objects
    [model.fit(X) for model in kmeans_model]   
    distortions = [sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0] for model in kmeans_model]
    plt.plot()
    plt.plot(k_choices, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum of squared error')
    plt.show()
    #higher values of k are more optimal but k=3 will be easy to visualize and 
    #combine to two classes: ionosphere backscatter and ground backscatter
    #using k = 3
    #ground backscatter is characterised by low Doppler velocity (<~50 m/s) and low Doppler spectral width (<~50 m/s)
    k_means_optimal = kmeans_model[1]
    fig = plt.figure()
    index = 1
    for pair in itertools.combinations(attributes, 2):
        plt.subplot(5, 2, index)
        x_label = pair[0]
        y_label = pair[1]
        plt.scatter(X[:, attributes.index(x_label)], X[:, attributes.index(y_label)], c=k_means_optimal.predict(X), s=1, cmap='viridis')
        centers = k_means_optimal.cluster_centers_
        plt.scatter(centers[:, attributes.index(x_label)], centers[:, attributes.index(y_label)], c='black', s=25, alpha=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        index += 1
    plt.show()
  

def plot_clustering(X, attributes, algorithm):   
    #general function template to run any clustering algorithm
    #in our case, agglomerative clustering and GMM clustering will be run
    algorithm.fit(X)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)    
    fig = plt.figure()
    index = 1        
    for pair in [("gate", "power"),("width", "power"),("elevation", "power")]  :
        plt.subplot(3, 1, index)
        x_label = pair[1]
        y_label = pair[0]
        plt.scatter(X[:, attributes.index(x_label)], X[:, attributes.index(y_label)], c=y_pred, s=1, cmap='viridis')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        index += 1
    plt.show()   


if __name__ == "__main__":
    radar_name = 'sas'
    start_time = dt.datetime(2011,1,1,1,0)
    end_time = dt.datetime(2011,1,1,1,10)
    #get radar data stored in local file
    filename = "radar_data_" + str(str(start_time).split(' ')[0]) + "_" + radar_name + ".pkl"
    data = pickle.load(open(filename, "rb"))  
    #preprocess all data to appropriately scale     
    for key in data.keys():
        data[key] = preprocessing.scale(data[key])    
    #stack data into columns and obtain an attribute name list for future ploting
    X = np.array(data["width"])
    attributes = ["width"]
    X = X.reshape(X.shape[0], 1)        
    for key in data.keys():
        if key== "width":
            continue
        X = np.hstack((X, np.array(data[key]).reshape(X.shape[0], 1)))    
        attributes.append(key)  
    #create clustering objects
    #ward minimizes the variance of the clusters being merged
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
    gmm = GaussianMixture(n_components=2, covariance_type='full')    
    clustering_algorithms = (('GaussianMixture', gmm), ('Ward', ward))
    #run clustering algorithms on data
    for name, algorithm in clustering_algorithms:
        plot_clustering(X, attributes, algorithm)    
    plot_kmeans(X, attributes)
