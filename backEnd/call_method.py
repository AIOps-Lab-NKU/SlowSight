import json
import os
import sys

import pandas as pd
from Cluster.dbscan import DBSCANClusterer, safe_standardize
from Cluster.kmeans import KMeansClusterer
from sklearn.decomposition import PCA
from SlowSight.main import run
from SlowSight.utils.Logger import Logger
from SlowSight.utils.utils import ConfigHandler
import time

def call_slowSight(rawdataFile,result_list, config):
    """
        Call the hardware tool to process the raw data file.

        Parameters:
            - rawdataFile: string, the name of the raw data file.
            - file_name_list: Dictionary containing information about raw data.
            - result_list: Dictionary containing information about processed files and their results.

        Functionality:
            1. Sets up the input and output paths.
            2. Parses configuration parameters.
            3. Runs the hardware tool with the specified configuration.
            4. Saves the results to the output directory.
            5. Updates the result metadata in "result_info.json".
    """
    base_path = os.getcwd()
    rawdata_dir = os.path.join(base_path, f"{config['dataset_dir']}")  # FIXME
    method = 'SlowSight'
    output_path = os.path.join(base_path, config['result_dir'], rawdataFile, "SlowSight")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        return

    # Parameter analysis
    print(f"========utils:{rawdataFile}===========")
    cfg_para = {"dataset": rawdataFile, "rawdata_dir": rawdata_dir,
                "config_file": 'config.yml', "method": method}
    cfg = ConfigHandler(cfg_para).config

    # output
    pdf_path = cfg.result_dir
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
    sys.stdout = Logger(os.path.join(pdf_path, f"temp.log"))
    sys.stderr = Logger(os.path.join(pdf_path, f"temp.log"))
    paint_indicatorsPath = os.path.join(pdf_path, f"{rawdataFile}.pdf")
    rate_path = os.path.join(pdf_path, f"===={rawdataFile}-rate.csv")

    start_time = time.time()
    run(cfg, rawdataFile, paint_indicatorsPath, rate_path)
    end_time = time.time()
    print(f"run time: {int(end_time - start_time)} s.")

    dbscan = pd.read_csv(os.path.join(output_path, 'dbscan_instance_result.csv'))
    dbscan.to_csv(os.path.join(output_path, rawdataFile+'_'+f"{config['scatter_result']}.csv"), index=False)
    disk = pd.read_csv(os.path.join(output_path, "cause_up_metrics_mask.csv"))
    disk.to_csv(os.path.join(output_path, rawdataFile + '_' + f"{config['line_result']}.csv"), index=False)

    if rawdataFile in result_list:
        result_list[rawdataFile]['SlowSight'] = [config['line_result'], config['scatter_result']]
    else:
        result_list[rawdataFile] = {'SlowSight': [config['line_result'], config['scatter_result']]}
    with open(config['result_json'], "w") as f:
        json.dump(result_list, f, indent=4)


def call_dbscan(rawdataFile, file_name_list, result_list, config):
    """
    Calls the DBSCAN clustering algorithm on the specified dataset and saves the results.

    Parameters:
    rawdataFile (str): The name of the dataset file to be processed.
    - file_name_list: Dictionary containing information about raw data.
    - result_list: Dictionary containing information about processed files and their results.

    Returns:
    None
    """
    # Get the current working directory
    base_path = os.getcwd()
    # Define the input file path
    input_path = os.path.join(base_path, config['dataset_dir'], f"{rawdataFile}_{config['input_csv']}")
    # Define the output directory path
    output_path = os.path.join(base_path, config['result_dir'], rawdataFile, "DBSCAN")
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # If the output directory already exists, return immediately
        return
    # Get the information dictionary for the current dataset
    info_dict = file_name_list[rawdataFile]

    # Read the input data
    data = pd.read_csv(input_path, encoding='utf-8')
    # Preprocess the data by filling missing values
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    # Initialize the DBSCAN clustering algorithm
    clusterer = DBSCANClusterer()

    # Initialize the final result DataFrame
    final_result = pd.DataFrame()
    final_result['timestamp'] = data['timestamp']
    # Iterate through each item in the information dictionary
    for (key, value) in info_dict.items():
        for id in value['ids']:
            # Find all column names that match the current key and ID
            columns = [col for col in data.columns if f'{key}_{id}' in col]
            # Extract the feature data
            features = data[columns].values
            # Standardize the features
            scaled = safe_standardize(features)

            # Perform PCA dimensionality reduction
            pca = PCA(n_components=2)
            feature_pca = pca.fit_transform(scaled)
            # Perform DBSCAN clustering
            cluster_labels = clusterer.cluster_with_dbscan(feature_pca)

            # Create a temporary DataFrame to store the current key and ID results
            temp = pd.DataFrame()
            temp['timestamp'] = data['timestamp']
            temp[f'{key}_{id}_x'] = feature_pca[:, 0]
            temp[f'{key}_{id}_y'] = feature_pca[:, 1]
            temp[f'{key}_{id}_label'] = cluster_labels
            # Merge the temporary DataFrame with the final result DataFrame
            final_result = pd.merge(final_result, temp, on='timestamp', how='inner')

    # Save the final result to a CSV file
    final_result.to_csv(os.path.join(output_path + f"{rawdataFile}_{config['scatter_result']}.csv"), index=False)

    # Update the information dictionary with the DBSCAN result information
    if rawdataFile in result_list:
        result_list[rawdataFile]['DBSCAN'] = [config['scatter_result']]
    else:
        result_list[rawdataFile] = {'DBSCAN': [config['scatter_result']]}
    # Save the updated information dictionary to a JSON file
    with open(config['result_json'], "w") as f:
        json.dump(result_list, f, indent=4)


def call_kmeans(rawdataFile, file_name_list, result_list, config):
    """
    Perform K-means clustering on the provided raw data file.

    Reads the data from a specified file, performs clustering using the KMeansClusterer,
    and writes the results to a designated output path. Additionally, updates a dictionary
    tracking results by filename and writes this information to a JSON file.

    Parameters:
    - rawdataFile: Name of the file containing the raw data for clustering.
    - file_name_list: Dictionary containing information about raw data.
    - result_list: Dictionary containing information about processed files and their results.
    """
    # Get the current working directory as the base path
    base_path = os.getcwd()

    # Input path
    input_path = os.path.join(base_path, config['dataset_dir'], f"{rawdataFile}_{config['input_csv']}")
    # Output path
    output_path = os.path.join(base_path, config['result_dir'], rawdataFile, "Kmeans")

    # Check if the output path exists, if not, create it; if it does, exit the function
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        return

    # Retrieve the information dictionary for the current file
    info_dict = file_name_list[rawdataFile]

    # Read the input data
    data = pd.read_csv(input_path, encoding='utf-8')

    # Initialize the KMeansClusterer
    clusterer = KMeansClusterer()

    # Initialize the final result DataFrame
    final_result = pd.DataFrame()
    final_result['timestamp'] = data['timestamp']

    # Process each key-value pair in the info dictionary
    for (key, value) in info_dict.items():
        for id in value['ids']:
            # Identify columns of interest based on key and id
            columns = [col for col in data.columns if f'{key}_{id}' in col]

            # Use the clusterer to process the disk data and retrieve results
            x, y, labels = clusterer.process_disk(data, columns)

            # Create a temporary DataFrame to store the current set of results
            temp = pd.DataFrame()
            temp['timestamp'] = data['timestamp']
            temp[f'{key}_{id}_x'] = x
            temp[f'{key}_{id}_y'] = y
            temp[f'{key}_{id}_label'] = labels

            # Merge the temporary DataFrame with the final result DataFrame
            final_result = pd.merge(final_result, temp, on='timestamp', how='inner')

    # Write the final result to a CSV file
    final_result.to_csv(os.path.join(output_path + f"{rawdataFile}_{config['scatter_result']}.csv"), index=False)

    # Update the file name list with the clustering result information
    if rawdataFile in result_list:
        result_list[rawdataFile]['Kmeans'] = [config['scatter_result']]
    else:
        result_list[rawdataFile] = {'Kmeans': [config['scatter_result']]}

    # Write the updated file name list to a JSON file
    with open(config['result_json'], "w") as f:
        json.dump(result_list, f, indent=4)



if __name__ == '__main__':
    with open("file_info.json", "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    with open("result_info.json", "r") as f:
        try:
            result_list = json.load(f)
        except json.JSONDecodeError:
            result_list = {}
    with open("config.json", "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    call_slowSight('net_all', result_list, config)
