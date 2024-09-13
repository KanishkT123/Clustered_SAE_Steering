# Sci-Kit Learn Clusering Documentation https://scikit-learn.org/stable/modules/clustering.htm
# Sci-Kit Learn documentation on manifold learning functions: https://scikit-learn.org/stable/modules/manifold.html

# import libraries 
import numpy as np


from sklearn.preprocessing import StandardScalar
from sklearn import metrics
from sklearn.decomposition import PCA


# sklearn clustering methods
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, SpectralClustering, AgglomerativeClustering

# for scipy hierarchical clustering, linkage for computing, dendrogram for plotting
from scipy.cluster.hierarchy import dendrogram, linkage 

# umap
import umap

# plotting (only used for plotting dendrograms - set to False automatically, though this should be done separately)
import matplotlib.pyplot as plt

# decoder_matrix is the input data (features as numpy array)


def run_kmeans_on_matrix(decoder_matrix:np.array, K=5000,random_state=0):
  kmeans = KMeans(n_clusters=K, random_state=random_state)
  kmeans.fit(decoder_matrix)
  cluster_centers = kmeans.cluster_centers_
  labels = kmeans.labels_
  return cluster_centers, labels


def run_pca_on_matrix (decoder_matrix, n_components):
  pca = PCA(n_components=n_components)
  decoder_matrix_2d = pca.fit_transform(decoder_matrix)
  return decoder_matrix_2d

= 'ward', \
                     compute_distances: bool = False

def run_agg_clustering(decoder_matrix, linkage='ward', n_clusters=None, \
                       metric='eucledian',  distance_threshold=None, compute_distances=False):
  '''
  linkage can be any of: ['ward', 'complete', 'average', 'single']
  '''
  agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, \
                                            metric=metric, linkage=linkage, \
                                            compute_distances=compute_distances, \
                                            distance_threshold=distance_threshold)
  clusters = agg_clustering.fit_predict(decoder_matrix)
  return clusters


# hierarchical clustering 
def get_linkages (decoder_matrix, linkage='ward', plot_dendrogram = False):
   Z = linkage(decoder_matrix, 'ward')
   if plot_dendrogram:
      plt.figure(figsize=(10, 7))
      dendrogram(Z)
      plt.show()
   return Z
      
def run_spectral_clustering(decoder_matrix, n_clusters=10, affinity='nearest_neighbors', random_state=0):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)
    clusters = spectral.fit_predict(decoder_matrix)
    return clusters

def run_umap(decoder_matrix, n_components=2, random_state=0):
  umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
  embedding = umap_model.fit_transform(decoder_matrix)
  return embedding

##############################
## Density Based  Clustering #
##############################
# DBSCAN
# DBSCAN : https://scikit-learn.org/stable/modules/clustering.html#dbscan
# demo of DBSCAN: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

def run_dbscan (X:np.array, eps:float = 0.3, min_samples:int=10, standard_scale:bool=False):
    if standard_scale:
        X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db
#  note: plot(X, hdb.labels_, hdb.probabilities_)

# HDBSCAN (should be preferred over DBSCAN)  
# demo of HDBSCAN https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#sphx-glr-auto-examples-cluster-plot-hdbscan-py
def run_dbscan(X:np.array):
    hdb = HDBSCAN().fit(X)
    return hdb 