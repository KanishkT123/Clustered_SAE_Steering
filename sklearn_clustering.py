# Sci-Kit Learn Clusering Documentation https://scikit-learn.org/stable/modules/clustering.htm

# import libraries 
import numpy as np

from sklearn.preprocessing import StandardScalar
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN

# for heirarchical clustering
from sklearn.cluster import AgglomerativeClustering 

# for scipy hierarchical clustering, linkage for computing, dendrogram for plotting
from scipy.cluster.hierarchy import dendrogram, linkage 

# X is the input data (features as numpy array)


##############################
## Density Based  Clustering #
##############################
# DBSCAN
# DBSCAN : https://scikit-learn.org/stable/modules/clustering.html#dbscan
# demo of DBSCAN: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

def cluster_db (X:np.array, eps:float = 0.3, min_samples:int=10, standard_scale:bool=False):
    if standard_scale:
        X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db
#  note: plot(X, hdb.labels_, hdb.probabilities_)

        
# HDBSCAN (should be preferred over DBSCAN)  
# demo of HDBSCAN https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#sphx-glr-auto-examples-cluster-plot-hdbscan-py
def cluster_hdb(X:np.array):
    hdb = HDBSCAN().fit(X)
    return hdb 
#  note: plot(X, hdb.labels_, hdb.probabilities_)

##############################
## Hierarchical Clustering ##
##############################
# 1) Agglomerative Clustering
##############################

def cluster_agglo(X:np.array, n_clusters:int=2, \
                   linkage: Literal['ward', 'complete', 'average', 'single']= 'ward', \
                     compute_distances: bool = False):
    
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, compute_distances=compute_distances).fit(X)
    return agg


##############################
## Hierarchical Clustering ##
##############################
# 2) Divisive clustering
##############################
