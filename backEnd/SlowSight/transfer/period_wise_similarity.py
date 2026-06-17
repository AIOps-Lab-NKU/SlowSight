import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import json
from itertools import combinations
from scipy.stats import pearsonr


def show_node_data(file_path, metric, min_row, max_row):
    """
    Display and plot specific rows of a given column from the target file.

    Args:
        file_path (str): Path to the CSV file.
        metric (str): Name of the metric/column to extract.
        min_row (int): Start row index.
        max_row (int): End row index.

    Returns:
        None: Displays a plot of the selected data.
    """
    df = pd.read_csv(file_path)
    column_data = df.loc[min_row:max_row, metric]
    plt.plot(column_data)
    plt.xlabel('Index')
    plt.ylabel(metric)
    plt.title(f'Plot of {metric} from row {min_row} to {max_row}')
    plt.show()


def time_to_num(time_str):
    """
    Convert time string in 'HH:MM:SS' format to corresponding row index in the file.

    Args:
        time_str (str): Time string in 'HH:MM:SS' format.

    Returns:
        tuple: (hour, minute, second, row_index)
            hour (int): Hour part of input time.
            minute (int): Minute part of input time.
            second (int): Second part of input time.
            row_index (int): Corresponding row index in the file (with first row removed).
    """
    hour, minutes, seconds = [int(x) for x in time_str.split(':')]
    row_index = hour * 240 + minutes * 4 + seconds // 15
    return hour, minutes, seconds, row_index


def period_wise_similarity_in_file(file_path, metric_list, min_row, T_rows, T_times):
    """
    Calculate period-wise similarity by computing Pearson correlation between multiple periods of the same metric.

    Args:
        file_path (str): Path to the CSV file containing the data.
        metric_list (list): List of metrics/columns to analyze.
        min_row (int): Starting row index for data extraction.
        T_rows (int): Number of rows representing one period.
        T_times (int): Number of consecutive periods to consider.

    Returns:
        float: Average Pearson correlation coefficient across all period pairs.
    """
    # Read the relevant columns from the file
    df = pd.read_csv(file_path)
    try:
        selected_data = df[mertric_list]
    except KeyError:
        print(file_path)
        return -1

    temp_selected_T = selected_data.iloc[min_row:min_row + T_times * T_rows, :]

    # Normalize each column using Min-Max scaling
    for i in list(temp_selected_T.columns):
        Max = np.max(temp_selected_T[i])
        Min = np.min(temp_selected_T[i])
        temp_selected_T[i] = (temp_selected_T[i] - Min) / (Max - Min)
    temp_selected_T = temp_selected_T.fillna(0)

    # Store flattened matrix per period for similarity calculation
    T_data = []
    for i in range(T_times):
        start_row = i * T_rows
        matrix = temp_selected_T.iloc[start_row:start_row + T_rows, :].values.transpose().flatten()
        T_data.append(matrix)

    # Compute pairwise Pearson correlations
    pcor_sum = 0
    for pair in combinations(T_data, 2):
        pcor, _ = pearsonr(pair[0], pair[1])
        pcor_sum += pcor

    # Return average correlation
    avg_pcor = pcor_sum / 6  # Assuming 4 periods produce 6 unique pairs
    return avg_pcor


if __name__ == "__main__":
    """
    Observe the graph and corresponding file to select two points that represent one full period:

    2022/11/22  0:13:30  value is 1062  corresponds to row 56
    2022/11/22  1:03:15  value is 1363  corresponds to row 255

    Note: Since these two points define the start and end of a period,
          we increment the end row count by 1 for accurate slicing in later calculations.
    """

    # Start and end row indices for the first period
    _, _, _, min_row = time_to_num('0:13:30')
    _, _, _, max_row = time_to_num('1:03:15')
    max_row += 1  # Add 1 to include the last row

    # Number of rows representing one period
    T_rows = max_row - min_row

    # Number of periods to use for calculation
    T_times = 4

    pws_dict = {}
    used_mertric_file = "result_files/used_metric.json"
    used_mertric_list = json.load(open(used_mertric_file))

    folder_path = 'data'

    # Traverse each subfolder in the 'data' directory
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        print(folder_name)
        # Read CSV files
        for file_name in os.listdir(folder_full_path):
            file_full_path = os.path.join(folder_full_path, file_name)
            if file_name.endswith('.csv'):
                mertric_list = [folder_name + ':' + element for element in used_mertric_list]
                period_wise_similarity = period_wise_similarity_in_file(
                    file_full_path, mertric_list, min_row, T_rows, T_times
                )
                pws_dict[folder_name] = period_wise_similarity

    # Convert dictionary to DataFrame
    df_pws = pd.DataFrame.from_dict(pws_dict, orient='index', columns=['period_wise_similarity'])

    # Save to CSV with index name
    df_pws.index.name = 'entity'
    df_pws.to_csv('period_wise_similarity.csv')
