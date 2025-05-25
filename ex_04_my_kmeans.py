from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    
    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean", 
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """
    def __init__(self, k: int, max_iter: int = 100, distance_metric: DISTANCE_METRICS = "euclidean", init_method: INIT_METHOD = "kmeans++"):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.
        
        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).
        
        Returns:
            MyKMeans: Fitted estimator instance.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        
        # Code here
        dim = len(x.shape)

        if dim < 2 or dim > 3:
            raise ValueError("Input data must be a 2D or 3D array")

        # select initial centroids, either random or kmeans++
        if self.init_method == "kmeans++":
            self.centroids = self._initialize_centroids(x)
        elif self.init_method == "random":
            indices = np.random.sample(range(x.shape[0]), self.k)
            if dim == 2:
                self.centroids = x[indices, :]
            else:
                self.centroids = x[indices, :, :]
        else:
            raise ValueError("Initialization method must be 'kmeans++' or 'random'")

        # setup empty clusters
        clusters = np.empty(shape=(x.shape[0]))
        clusters[:] = np.nan
        
        # get initial distances
        distances = self._compute_distance(x, self.centroids)

        with tqdm(range(self.max_iter), desc="KMeans", unit="iter") as pbar:
            for _ in pbar:
                # assign each sample to cluster with centroid of lowest distance
                new_clusters = np.empty(shape=(x.shape[0]))
                new_clusters[:] = np.nan

                for i in range(distances.shape[0]):
                    c = np.argmin(distances[i])
                    new_clusters[i] = int(c)

                if np.array_equal(clusters, new_clusters):
                    # no more changes detected -> break out of for loop
                    break
                
                # calculate new centroids as means of all points within cluster
                clusters = np.copy(new_clusters)
                for c in range(self.k):
                    points_in_cluster = x[np.where(clusters == c)]
                    # assure cluster is not empty before calculating new centroid
                    if points_in_cluster.shape[0] > 0:
                        self.centroids[c] = np.mean(points_in_cluster, axis=0)
                    else:
                        # empty cluster encountered: select a random point as centroid for this emtpy cluster
                        self.centroids[c] = x[np.random.randint(0, x.shape[0])]
                
                # update progress bar with current iteration inertia (i.e. fit quality)
                distances = self._compute_distance(x, self.centroids)
                self.inertia_ = self._compute_inertia(distances=distances)
                pbar.set_postfix({"inertia": self.inertia_})

        return self
    

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.
        
        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)
        
        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the kmeans++ method.
        
        Args:
            x (np.ndarray): Training data.
            
        Returns:
            np.ndarray: Initial centroids.
        """
        dim = len(x.shape)

        if dim == 2:
            centroids = np.empty(shape=(self.k, x.shape[1]))
        else:
            centroids = np.empty(shape=(self.k, x.shape[1], x.shape[2]))
        
        centroids[:] = np.nan

        # choose first centroid randomly
        idx = np.random.randint(x.shape[0])
        if dim == 2:
            centroids[0] = x[idx, :]
        else:
            centroids[0] = x[idx, :, :]
        
        for c in range(self.k - 1):
            distances = self._compute_distance(x, centroids)
            # select next point as centroid with max distance to previous centroids
            if dim == 2:
                centroids[c+1] = x[np.argmax(distances), :]
            else:
                centroids[c+1] = x[np.argmax(distances), :, :]
        
        return np.array(centroids)

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance between samples and centroids.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, n_features) or (n_samples, time_steps, n_features).
            centroids (np.ndarray): Centroids of shape (k, n_features) or (k, time_steps, n_features).
            
        Returns:
            np.ndarray: Distances between each sample and each centroid, shape (n_samples, k).
        """
        if self.distance_metric == "dtw":
            return self._dtw(x, centroids)
        elif self.distance_metric != "euclidean" and self.distance_metric != "manhattan":
            raise ValueError("Invalid distance metric")

        distances = np.empty(shape=(x.shape[0], centroids.shape[0]))
        distances[:] = np.nan

        dim = len(x.shape)

        if dim == 2:
            for i in range(x.shape[0]):
                for k in range(centroids.shape[0]):
                    sum = 0
                    for j in range(x.shape[1]):
                        dist = x[i, j] - centroids[k, j]
                        if self.distance_metric == "euclidean":
                            sum += (dist)**2
                        elif self.distance_metric == "manhattan":
                            sum += abs(dist)
                    if self.distance_metric == "euclidean":
                        sum = np.sqrt(sum)
                    distances[i, k] = sum
        else:
            for i in range(x.shape[0]):
                for k in range(centroids.shape[0]):
                    sum = 0
                    for t in range(x.shape[1]):
                        for j in range(x.shape[2]):
                            dist = x[i, t, j] - centroids[k, t, j]
                            if self.distance_metric == "euclidean":
                                sum += (dist)**2
                            elif self.distance_metric == "manhattan":
                                sum += abs(dist)
                    if self.distance_metric == "euclidean":
                        sum = np.sqrt(sum)
                    distances[i, k] = sum

        return np.array(distances)

    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)
            
        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """
        distances = np.empty((x.shape[0], centroids.shape[0]))
        distances[:] = np.nan

        dim = len(x.shape)

        if dim == 2:
            for i in range(x.shape[0]):
                for k in range(centroids.shape[0]):
                    distances[i, k] = dtw.distance_fast(x[i], centroids[k], only_ub=False)
        else:
            for i in range(x.shape[0]):
                for k in range(centroids.shape[0]):
                    for j in range(x.shape[2]):
                        distances[i, k] = dtw.distance_fast(x[i, :, j], centroids[k, :, j], only_ub=False)

        return distances


    def _compute_inertia(self, distances: np.ndarray) -> float:
        inertia = 0.0

        for i in range(distances.shape[0]):
            idx = np.argmin(distances[i])
            inertia += (distances[i, idx])**2
        
        return inertia