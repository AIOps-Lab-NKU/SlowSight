import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def safe_standardize(data):
    """
    Standardizes the input data while handling columns with zero variance.

    Parameters:
    - data: A NumPy array containing the data to be standardized.

    Returns:
    - scaled_data: The standardized data. Columns with zero variance are left unchanged.
    """
    if data.shape[1] == 0:
        raise ValueError("Input data has no features to standardize.")

    scaler = StandardScaler()
    variances = np.var(data, axis=0)
    zero_variance_mask = (variances == 0)

    if zero_variance_mask.any():
        print(
            f"Warning: The following features have zero variance and will not be standardized: {np.where(zero_variance_mask)[0]}")
        scaled_data = data.copy()
        non_zero_variance_mask = ~zero_variance_mask
        scaled_data[:, non_zero_variance_mask] = scaler.fit_transform(data[:, non_zero_variance_mask])
    else:
        scaled_data = scaler.fit_transform(data)

    return scaled_data


class DBSCANClusterer:
    """
    A class for performing DBSCAN clustering with predefined parameters.

    Attributes:
    - dbscan_eps (float): The maximum distance between two samples for one to be considered
                          as in the neighborhood of the other (default is 0.3).
    - dbscan_min_samples (int): The number of samples in a neighborhood for a point to be
                                considered as a core point (default is 20).
    """

    def __init__(self):
        # Initialize DBSCAN parameters
        self.dbscan_eps = 0.3
        self.dbscan_min_samples = 20

    def cluster_with_dbscan(self, data_2d):
        """
        Perform DBSCAN clustering on 2D data.

        Parameters:
        - data_2d: Input 2D data (NumPy array) to be clustered.

        Returns:
        - cluster_labels: Cluster labels for each data point. Noises are labeled as -1.
        """
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(data_2d)
        return cluster_labels
