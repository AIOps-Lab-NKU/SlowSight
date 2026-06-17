import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance.dtw import warping_paths, best_path


def euclidean_numpy(u, v):
    """
    Compute Euclidean distance using NumPy.

    :param u: Vector 1
    :param v: Vector 2
    :return: Euclidean distance
    """
    u = np.asarray(u)
    v = np.asarray(v)
    return np.sqrt(np.sum((u - v) ** 2))


def calculate_dtw_distance(downsample_factor, sakoe_chiba_width, series1, series2):
    """
    Compute DTW distance with Sakoe-Chiba window.

    :param downsample_factor: Factor to downsample the time series
    :param sakoe_chiba_width: Width of the Sakoe-Chiba window
    :param series1: Time series 1
    :param series2: Time series 2
    :return: DTW distance
    """
    if downsample_factor > 1:
        series1 = series1[::downsample_factor]
        series2 = series2[::downsample_factor]

    # Compute DTW distance with Sakoe-Chiba window
    distance = dtw.distance(series1, series2, window=sakoe_chiba_width)
    return distance


def compare_single_disk(disks, t, window, metric, target_disk, downsample_factor, sakoe_chiba_width):
    """
    Compare a single disk's metric against the average of peer disks using DTW.

    :param disks: List of all disks in the group
    :param t: Current timestamp
    :param window: Current window data
    :param metric: Metric name to compare
    :param target_disk: Target disk to compare
    :param downsample_factor: Factor to downsample the time series
    :param sakoe_chiba_width: Width of the Sakoe-Chiba window for DTW
    :return: DTW distance result for this disk
    """
    target_series = window[f"{target_disk}_{metric}"].values

    # Calculate mean of peer disk metrics
    peer_columns = [f"{peer_disk}_{metric}" for peer_disk in disks if peer_disk != target_disk]
    peer_mean_series = window[peer_columns].mean(axis=1).values

    # Compute DTW distance between target and peer mean
    distance = calculate_dtw_distance(downsample_factor, sakoe_chiba_width, target_series, peer_mean_series)

    return t, f"{target_disk}_{metric}_dtw", distance
