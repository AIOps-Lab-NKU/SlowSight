from sklearn.metrics import silhouette_score
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re


def safe_standardize(data):
    """
    Safely standardizes the input data while avoiding NaN values caused by zero-variance features.

    Parameters:
    - data: A NumPy array containing the raw input data to be standardized.

    Returns:
    - scaled_data: The standardized data. Features with zero variance are left unchanged.

    Raises:
    - ValueError: If the input data contains no features (zero columns).
    """
    if data.shape[1] == 0:
        raise ValueError("Input data has no features to standardize.")

    scaler = StandardScaler()

    # Check variance of each feature
    variances = np.var(data, axis=0)
    zero_variance_mask = (variances == 0)

    # Skip standardization for features with zero variance
    if zero_variance_mask.any():
        print(
            f"Warning: The following features have zero variance and will not be standardized: {np.where(zero_variance_mask)[0]}")
        scaled_data = data.copy()
        non_zero_variance_mask = ~zero_variance_mask
        scaled_data[:, non_zero_variance_mask] = scaler.fit_transform(data[:, non_zero_variance_mask])
    else:
        scaled_data = scaler.fit_transform(data)

    return scaled_data


class KMeansClusterer:
    """
    A class for performing K-means clustering with automatic selection of optimal number of clusters (k).

    Attributes:
    - max_k (int): Maximum number of clusters to consider when selecting the best k.
    - _window_size (int): Size of a window used in some algorithms (not currently used in this code).
    - max_workers (int): Maximum number of threads/workers allowed for parallel processing.
    - pca (PCA): Instance of PCA for dimensionality reduction.
    """

    def __init__(self):
        # Initialize parameters for clustering
        self.max_k = 8
        self._window_size = 20
        self.max_workers = 4
        self.pca = None

    def find_best_k(self, pca_result):
        """
        Automatically determine the optimal number of clusters (k) using the Silhouette Score.

        Parameters:
        - pca_result: The transformed data after PCA dimensionality reduction.

        Returns:
        - best_k: Optimal number of clusters.
        """
        best_k = 2
        best_score = -1
        for k in range(2, self.max_k + 1):  # from 2 to max_k
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pca_result)
            if len(set(labels)) == 1:  # Skip if only one cluster is formed
                continue
            score = silhouette_score(pca_result, labels)
            if score > best_score:
                best_score = score
                best_k = k
        print(f"Selected optimal number of clusters k = {best_k} (Silhouette Score = {best_score:.4f})")
        return best_k

    def cluster_with_kmeans(self, pca_result, n_clusters):
        """
        Perform K-means clustering on PCA-reduced data.

        Parameters:
        - pca_result: Transformed 2D data after PCA.
        - n_clusters (int): Number of clusters to form.

        Returns:
        - cluster_labels: Labels assigned to each data point.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_result)
        return cluster_labels

    def process_disk(self, data, disk_metrics):
        """
        Process disk metrics data through standardization, PCA, and K-means clustering.

        Parameters:
        - data: Original dataset as a Pandas DataFrame.
        - disk_metrics: List of column names representing disk-related metrics.

        Returns:
        - pca_x: First principal component (x-coordinate).
        - pca_y: Second principal component (y-coordinate).
        - cluster_labels: Cluster assignment for each sample.
        """
        features = data[disk_metrics].values
        scaled = safe_standardize(features)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)

        # Automatically select the best number of clusters
        best_k = self.find_best_k(pca_result)

        # Perform K-means clustering with the selected number of clusters
        cluster_labels = self.cluster_with_kmeans(pca_result, best_k)

        return pca_result[:, 0], pca_result[:, 1], cluster_labels
