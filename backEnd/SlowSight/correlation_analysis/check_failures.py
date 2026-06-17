import numpy as np
import json


def get_pattern(file_path='SlowSight/correlation_analysis/fail_slow.json', check_type='nic'):
    """
    Load metric patterns based on component type from a JSON file.

    Args:
        file_path (str): Path to the pattern definition JSON file.
        check_type (str): Component type to filter, e.g., 'nic' or 'disk'.

    Returns:
        dict: Dictionary mapping metric names to expected trend values:
            1 for increasing trend,
           -1 for decreasing trend,
            0 otherwise.
    """
    metric_pattern = {}
    with open(file_path, 'r') as f:
        patterns = json.load(f)
        for pattern in patterns:
            if pattern["component_type"] == check_type:
                # Assign trend value for main metric
                if pattern["failure_patterns"][0]["trend"] == 'increasing':
                    metric_pattern[pattern["metric_name"]] = 1
                elif pattern["failure_patterns"][0]["trend"] == 'decreasing':
                    metric_pattern[pattern["metric_name"]] = -1

                # Assign trends for related metrics
                for metric in pattern["failure_patterns"][0]["related_metrics"]:
                    if metric["expected_trend"] == 'increasing':
                        metric_pattern[metric["metric"]] = 1
                    elif metric["expected_trend"] == 'decreasing':
                        metric_pattern[metric["metric"]] = -1
    return metric_pattern


def divide(a, b):
    """
    Safe division function that returns zero when dividing by zero.

    Args:
        a (float): Numerator.
        b (float): Denominator.

    Returns:
        float: Result of a / b if b != 0, else 0.
    """
    if b != 0:
        return a / b
    else:
        return 0


def is_empty(data):
    """
    Check if data is empty or contains only zeros.

    Args:
        data (list or np.ndarray): Input data to be checked.

    Returns:
        bool: True if data is empty or all zeros; False otherwise.
    """
    if isinstance(data, list) and not data:
        return True

    if isinstance(data, np.ndarray) and not data.any():
        return True

    return False


def conv_smooth(data, box_pts):
    """
    Smooth data using convolution with uniform kernel (moving average).

    Args:
        data (np.ndarray): Time series data to smooth.
        box_pts (int): Length of smoothing window.

    Returns:
        np.ndarray: Smoothed data.
    """
    if is_empty(data):
        return data

    box = np.divide(np.ones(box_pts), box_pts)
    data_smooth = np.convolve(data, box, mode='valid')
    return data_smooth


def trend(data, win_len=None):
    """
    Detect overall trend in data: increasing, decreasing, or stable.

    Args:
        data (np.ndarray): Time series data.
        win_len (int or None): Window length for comparison. If None, use half of data length.

    Returns:
        int:
            1: Increasing trend detected.
           -1: Decreasing trend detected.
            0: No significant trend.
    """
    data = conv_smooth(data, box_pts=7)

    if not win_len:
        win_len = len(data) // 2

    if divide(np.mean(data[:win_len]), np.mean(data[-win_len:])) < 0.9:
        return 1

    elif divide(np.mean(data[:win_len]), np.mean(data[-win_len:])) > 1.1:
        return -1

    else:
        return 0
