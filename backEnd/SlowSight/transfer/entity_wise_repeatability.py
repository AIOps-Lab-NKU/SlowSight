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
import networkx as nx
import warnings

warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage, fcluster

from SlowSight.transfer.period_wise_similarity import time_to_num


def entity_wise_repeatability_in_file(file_path, mertric_list, min_row, T_rows, T_times):
    """
    Calculate the median values of specified metrics across multiple time periods.

    Args:
        file_path (str): Path to the CSV file.
        mertric_list (list): List of metric names to select.
        min_row (int): Start row for data extraction.
        T_rows (int): Number of rows in one period.
        T_times (int): Number of periods to consider.

    Returns:
        np.ndarray: Flattened array containing median values per period.
    """
    df = pd.read_csv(file_path)
    try:
        selected_data = df[mertric_list]
    except KeyError:
        print(file_path)
        return np.array([])

    temp_selected_T = selected_data.iloc[min_row:min_row + T_times * T_rows, :]

    # Normalize each column using Min-Max scaling
    for i in list(temp_selected_T.columns):
        Max = np.max(temp_selected_T[i])
        Min = np.min(temp_selected_T[i])
        temp_selected_T[i] = (temp_selected_T[i] - Min) / (Max - Min)
    temp_selected_T = temp_selected_T.fillna(0)

    n = temp_selected_T.shape[1]  # number of columns
    median_matrix = np.zeros((T_rows, n))
    for col in range(n):
        column = temp_selected_T.iloc[:, col]
        for row in range(T_rows):
            median_value = column.iloc[row::T_rows].median()
            median_matrix[row, col] = median_value
    median_array = median_matrix.T.flatten()
    return median_array


def get_mean_pws(file_pws):
    """
    Calculate the average value from the second column of a CSV file,
    excluding negative values.

    Args:
        file_pws (str): Path to the CSV file.

    Returns:
        float: Mean value of the second column.
    """
    df = pd.read_csv(file_pws)
    df = df[df.iloc[:, 1] >= 0]  # remove values less than 0 in the second column
    mean_value = df.iloc[:, 1].mean()  # calculate the mean of the second column
    return mean_value


def binary_ewr_file(filepath, mean_pws):
    """
    Convert values in a CSV file to binary based on a threshold (mean_pws),
    and save the result.

    Args:
        filepath (str): Input CSV file path.
        mean_pws (float): Threshold used for binarization.
    """
    df = pd.read_csv(filepath, index_col=0)

    # Replace matrix values based on condition
    df = df.applymap(lambda x: 1 if x > mean_pws else 0)

    # Save to a new CSV file ('entity_wise_repeatability.csv')
    df.to_csv('entity_wise_repeatability.csv')


def check_file(filepath, targetfile):
    """
    Check for inconsistent triplets in a binary relation matrix where A-B and B-C are related,
    but A-C is not. Save these inconsistencies to a CSV file.

    Args:
        filepath (str): Path to the input CSV file.
        targetfile (str): Path to save the output results.

    Returns:
        list: List of inconsistent triplets found.
    """
    df = pd.read_csv(filepath, index_col=0)

    nodes = df.columns.tolist()
    false_set = []

    for i in range(len(nodes)):
        print(i)
        for j in range(i + 1, len(nodes)):
            if df.iloc[i, j]:
                for k in range(j + 1, len(nodes)):
                    if df.iloc[i, k] and not df.iloc[j, k]:
                        false_set.append({i, j, k})

    with open(targetfile, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write each item
        for item in false_set:
            writer.writerow([item] + [nodes[i] for i in item])

    return false_set


def classify_file(filepath):
    """
    Classify entities into groups where all elements in a group have mutual relations.

    Args:
        filepath (str): Path to the CSV file containing pairwise relations.

    Returns:
        dict: Mapping of entity names to class IDs.
    """
    entity_class_list = []
    entity_class_dict = {}

    df = pd.read_csv(filepath, index_col=0)
    class_id = 1
    nodes = df.columns.tolist()
    flag = [0] * len(nodes)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if df.iloc[i, j]:
                # Traverse each set; if i or j already exists in a set,
                # determine whether the other should be added.
                for oneset in entity_class_list:
                    if i in oneset:
                        allow_add = True
                        for element in oneset:
                            if df.iloc[j, element] + df.iloc[element, j] < 1:
                                allow_add = False
                        if allow_add:
                            oneset.add(j)
                        flag[j] = 1
                    elif j in oneset:
                        allow_add = True
                        for element in oneset:
                            if df.iloc[i, element] + df.iloc[element, i] < 1:
                                allow_add = False
                        if allow_add:
                            oneset.add(i)
                        flag[i] = 1
                    else:
                        entity_class_list.append({i, j})
        if not flag[i]:
            flag[i] = 1
            entity_class_list.append({i})

    for oneset in entity_class_list:
        for element in oneset:
            entity_class_dict[nodes[element]] = class_id
        class_id += 1

    return entity_class_dict


def draw_relation(filepath):
    """
    Draw a relationship graph based on a binary relation matrix.

    Args:
        filepath (str): Path to the CSV file containing relationships.
    """
    df = pd.read_csv(filepath, index_col=0)

    G = nx.DiGraph()

    for i, source_node in enumerate(df.columns):
        for j, target_node in enumerate(df.columns[i + 1:]):
            if df.iloc[j, i] == 1:
                G.add_edge(source_node.split('-')[1], target_node.split('-')[1])

    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue',
            edge_color='gray', arrows=False)
    plt.title("Relationship Graph")
    plt.show()


def index_to_value(file_path, refer_list):
    """
    Replace the first column in a CSV file with corresponding values from refer_list.

    Args:
        file_path (str): Path to the CSV file.
        refer_list (list): Reference list to map index to string.
    """
    df = pd.read_csv(file_path)

    csv_col1 = df.iloc[:, 0].tolist()

    result_list = [refer_list[i] if i < len(refer_list) else '' for i in csv_col1]

    df['Column1'] = result_list
    df.to_csv('new_data.csv', index=False)


def calculate_ewr(data_path, entities, used_mertric_list, min_row, T_rows, T_times):
    """
    Calculate Entity-Wise Repeatability values using Pearson correlation between different entities.

    Args:
        data_path (str): Path to the dataset CSV file.
        entities (list): List of unique entity names.
        used_mertric_list (list): List of metrics used.
        min_row (int): Starting row for data processing.
        T_rows (int): Number of rows per period.
        T_times (int): Number of periods.

    Returns:
        pd.DataFrame: DataFrame containing pairwise Pearson correlations between entities.
    """
    represent_curve = {}
    df = pd.read_csv(data_path)

    for entity in entities:
        mertric_list = [entity + '_' + element for element in used_mertric_list]
        median_array = entity_wise_repeatability_in_file(data_path, mertric_list, min_row, T_rows, T_times)
        if median_array.size:
            represent_curve[entity] = median_array

    keys = list(represent_curve.keys())
    matrix_size = len(keys)
    pearson_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        pearson_matrix[i, i] = 1.0
        for j in range(i + 1, matrix_size):
            key1 = keys[i]
            key2 = keys[j]
            array1 = represent_curve[key1]
            array2 = represent_curve[key2]
            if np.var(array1) == 0 or np.var(array2) == 0:
                pcor = 1.0
            else:
                pcor, _ = pearsonr(array1, array2)
            pearson_matrix[i, j] = pcor

    df_ewp = pd.DataFrame(pearson_matrix, index=keys, columns=keys)
    df_ewp.to_csv('entity_wise_repeatability_values.csv')

    return df_ewp


def draw_metrics(metrics_list, min_row, T_rows):
    """
    Plot normalized metric values over time for several metrics.

    Args:
        metrics_list (list): Metrics to plot.
        min_row (int): Starting row for plotting.
        T_rows (int): Number of rows to plot.
    """
    total_data = []
    for metric_name in metrics_list:
        folder_path = os.path.join('data', metric_name)
        for file_name in os.listdir(folder_path):
            file_full_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_full_path)
                used_mertric_file = "result_files/used_metric.json"
                used_mertric_list = json.load(open(used_mertric_file))
                mertric_list = [metric_name + ':' + element for element in used_mertric_list]
                selected_data = df[mertric_list]

                temp_selected_T = selected_data.iloc[min_row:min_row + T_rows, :]
                for i in list(temp_selected_T.columns):
                    Max = np.max(temp_selected_T[i])
                    Min = np.min(temp_selected_T[i])
                    temp_selected_T[i] = (temp_selected_T[i] - Min) / (Max - Min)
                temp_selected_T = temp_selected_T.fillna(0)

                metric_data = temp_selected_T.values.transpose().flatten()
                total_data.append(metric_data)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    for i, ax in enumerate(axes):
        data = total_data[i]
        ax.plot(data)
        ax.set_title(f"{metrics_list[i]}")
    plt.tight_layout()
    plt.show()


def cluster_entity(df_ewp, threshold=0.75):
    """
    Cluster entities based on their similarity scores.

    Args:
        df_ewp (pd.DataFrame): Pairwise similarity matrix.
        threshold (float): Correlation threshold for connecting two entities.

    Returns:
        dict: Clusters of entities grouped by similarity.
    """
    G = nx.Graph()
    G.add_nodes_from(df_ewp.columns)

    for i in range(len(df_ewp)):
        for j in range(i + 1, len(df_ewp)):
            if df_ewp.iloc[i, j] > threshold:
                G.add_edge(df_ewp.columns[i], df_ewp.columns[j])

    connected_components = list(nx.connected_components(G))
    cluster_dict = {i + 1: list(component) for i, component in enumerate(connected_components)}

    return cluster_dict


if __name__ == '__main__':
    _, _, _, min_row = time_to_num('16:00:00')
    _, _, _, max_row = time_to_num('17:00:00')
    max_row += 1
    T_rows = max_row - min_row  # Number of rows representing one period
    T_times = 4  # Number of periods used for calculation

    used_mertric_file = "../data/net/net_all/metric_candidates_all.json"
    used_mertric_list = json.load(open(used_mertric_file))

    used_metrics = []
    entities = []
    for metric in used_mertric_list:
        metric_pieces = metric.split("_")
        entity = metric_pieces[0] + '_' + metric_pieces[1]
        metric_name = metric_pieces[2]
        entities.append(entity)
        used_metrics.append(metric_name)

    used_metrics = list(set(used_metrics))
    entities = list(set(entities))
    print(used_metrics)
    print(entities)

    data_path = "../data/net/net_all/metric.csv"
    df_ewp = calculate_ewr(data_path, entities, used_metrics, min_row, T_rows, T_times)
    cluster_dict = cluster_entity(df_ewp, threshold=0.75)
    print(cluster_dict)
