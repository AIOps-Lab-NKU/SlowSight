import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode


def compute_statistics(features, col, col_data, col_mode_dict, threshold=1):
    # Basic statistical features
    mode_value = col_mode_dict[col]
    mean_value = np.mean(col_data)
    max_value = np.max(col_data)
    if (mean_value - mode_value < threshold and max_value - mode_value < threshold):
        features[f"{col}_mean"] = mode_value
        features[f"{col}_max"] = mode_value
    else:
        features[f"{col}_mean"] = mean_value
        features[f"{col}_max"] = max_value

    return features


def cluster_with_dbscan(feature_matrix, eps=0.5, min_samples=5, n_components=0.95):
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(feature_matrix)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(reduced_features)
    return labels, reduced_features

def match_with_dbscan(dbscan, feature_df):
    cluster_labels = dbscan.fit_predict(feature_df)
    return cluster_labels

def refine_clusters(reduced_features, labels, eps, alpha=0.1, beta=5.0):
    """
    Apply improved clustering and outlier definition.

    Parameters:
        reduced_features: Feature matrix after PCA dimensionality reduction
        labels: Original DBSCAN labels
        alpha: Threshold for base cluster (minimum proportion of samples in total)
        beta: Outlier distance multiplier

    Returns:
        Improved labels, with -1 indicating outliers
    """
    if len(reduced_features) == 0:
        return np.array([])

    n_samples = len(reduced_features)

    # Copy original labels
    improved_labels = labels.copy()

    # Return directly if all points are noise points
    if all(labels == -1):
        return improved_labels

    # Count and calculate ratio for each cluster
    unique_labels = np.unique(labels)
    cluster_counts = {label: np.sum(labels == label) for label in unique_labels}
    cluster_ratios = {label: count / n_samples for label, count in cluster_counts.items()}

    # Divide clusters into base and candidate clusters
    base_clusters = [label for label, ratio in cluster_ratios.items() if ratio > alpha]
    candidate_clusters = [label for label, ratio in cluster_ratios.items() if ratio <= alpha]

    if len(candidate_clusters) == 0:
        improved_labels = [0 for i in range(len(labels))]
        return improved_labels

    # Calculate center and radius for each base cluster
    cluster_centers = {}
    cluster_radii = {}

    for label in base_clusters:
        mask = labels == label
        cluster_points = reduced_features[mask]

        # Calculate center and maximum distance (radius)
        center = np.mean(cluster_points, axis=0)
        if len(cluster_points) > 1:
            distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            radius = np.max(distances)
        else:
            radius = eps  # If only one point, use DBSCAN's eps as the radius

        cluster_centers[label] = center
        cluster_radii[label] = radius

    # Process points in candidate clusters
    for label in candidate_clusters:
        mask = labels == label
        candidate_points_idx = np.where(mask)[0]

        for idx in candidate_points_idx:
            point = reduced_features[idx]

            # Determine whether it is an outlier by checking distance to all base cluster centers
            is_outlier = True

            for base_label in base_clusters:
                center = cluster_centers[base_label]
                radius = cluster_radii[base_label]

                # Calculate distance from point to center
                distance = np.sqrt(np.sum((point - center) ** 2))

                # Not an outlier if distance is within beta times the base cluster radius
                if distance <= beta * radius:
                    is_outlier = False
                    break

            # Mark as outlier if no matching base cluster found
            if is_outlier:
                improved_labels[idx] = -1

    # Re-evaluate original noise points (labeled -1)
    noise_mask = labels == -1
    noise_points_idx = np.where(noise_mask)[0]

    for idx in noise_points_idx:
        point = reduced_features[idx]

        # Check distance to all base cluster centers
        is_outlier = True

        for base_label in base_clusters:
            center = cluster_centers[base_label]
            radius = cluster_radii[base_label]

            # Calculate distance from point to center
            distance = np.sqrt(np.sum((point - center) ** 2))

            # Not an outlier if distance is within beta times the base cluster radius
            if distance <= beta * radius:
                is_outlier = False
                # Assign to nearest base cluster
                improved_labels[idx] = base_label
                break

    return improved_labels
