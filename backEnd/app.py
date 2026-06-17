import json
import shutil
import sys
import time
from datetime import datetime
import numpy as np

from flask import Flask, send_file
from flask_cors import CORS
from flask import request, jsonify,render_template
import pandas as pd
import os
from call_method import *

app = Flask(__name__)
if 'run' in sys.argv:
    sys.argv.remove('run')
cors = CORS(app)
app.config['path'] = {}
with open("config.json", "r") as f:
    try:
        app.config['path'] = json.load(f)
    except json.JSONDecodeError:
        app.config['path'] = {}

# ========================= data management==================================================


@app.route('/getBase', methods=['GET', 'POST'])
def get_base():
    """
       Retrieve the list of available datasets and corresponding method names based on the request type.

       Parameters:
          jsonify: A JSON response containing:
               - type: type of echarts

       Returns:
           jsonify: A JSON response containing:
               - dataset: Dictionary of file names loaded from "file_info.json".
               - method: List of method names based on the input type ('line' or 'scatter').

       Functionality:
           1. Parses the incoming JSON request to determine the type ('line' or 'scatter').
           2. Reads the dataset names from "file_info.json". If the file is empty or invalid, an empty dictionary is used.
           3. Assigns a list of method names based on the input type.
       """
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    input_data = request.get_json()
    with open(app.config['path']['dataset_json'], "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    method_names = []
    if input_data['type'] == 'line':
        method_names = app.config['path']['line_method']
    elif input_data['type'] == 'scatter':
        method_names = app.config['path']['scatter_method']
    return jsonify(dataset=file_name_list, method=method_names)


@app.route('/getResult')
def get_result():
    """
       Get result information

       This route is used to retrieve preprocessed result information, including dataset and method names.
       It attempts to load the dataset information from 'result_info.json'. If the file does not exist or is empty, it initializes as an empty dictionary.
    """
    base_path = os.getcwd()
    result_path = os.path.join(base_path, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(app.config['path']['result_json'], "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    method_names = []
    return jsonify(dataset=file_name_list, method=method_names)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    """
       Upload a data file and store it in the system.

       Parameters:
           dataType: int, indicating whether the file is a base dataset (0) or result data (other).
           name: string, the name of the file.
           data: dataframe, the content of the file.

       Returns:
           jsonify: A JSON object containing:
               - status: string, indicating success or warning.
               - info: string, providing additional information or error messages.
               - dataset: dict, the current state of stored datasets.

       Functionality:
           1. Validates the file name and structure.
           2. Parses the file content into a DataFrame.
           3. Updates the metadata in "file_info.json" or "result_info.json".
           4. Saves the file to the appropriate directory.
    """
    input_data = request.get_json()
    if input_data['dataType'] == 0:
        with open(app.config['path']['dataset_json'], "r") as f:
            try:
                file_name_list = json.load(f)
            except json.JSONDecodeError:
                file_name_list = {}
        if input_data['name'] not in list(file_name_list.keys()):
            # Validate file name format
            if len(input_data['name'].split('_metric')) != 2:
                return jsonify(status='warning', info="h.message.name_error", dataset=file_name_list)
            input_name= input_data['name'].split('_metric')[0]

            df = pd.DataFrame(input_data['data'])
            attributes = df.columns.tolist()[1:]
            # Build file records
            contains_dict = {}
            for attribute in attributes:
                parts = attribute.split('_')
                if len(parts) == 3:
                    if parts[0] in contains_dict:
                        if parts[1] not in contains_dict[parts[0]]['ids']:
                            contains_dict[parts[0]]['ids'].append(parts[1])
                        if parts[2] not in contains_dict[parts[0]]['metrics']:
                            contains_dict[parts[0]]['metrics'].append(parts[2])
                    else:
                        contains_dict[parts[0]] = {
                            'ids': [parts[1], ],
                            'metrics': [parts[2], ]
                        }
                else:
                    return jsonify(status='warning', info="h.message.metric_name_error", dataset=file_name_list)
            file_name_list[input_name] = contains_dict
            with open(f"{app.config['path']['dataset_dir']}/{input_name}_{app.config['path']['input_json']}", "w") as f:
                json.dump(attributes, f, indent=4)
            with open(app.config['path']['dataset_json'], "w") as f:
                json.dump(file_name_list, f, indent=4)
            df.to_csv(f"{app.config['path']['dataset_dir']}/{input_data['name']}",index=False, encoding="utf-8")
            return jsonify(status='success', info='h.message.save_success', dataset=file_name_list)
        else:
            return jsonify(status='warning', info='h.message.exist_file', dataset=file_name_list)
    else:
        with open(app.config['path']['result_json'], "r") as f:
            try:
                file_name_list = json.load(f)
            except json.JSONDecodeError:
                file_name_list = {}
        df = pd.DataFrame(input_data['data'])
        if app.config['path']['line_result'] in input_data['name']:
            item_name = input_data['name'].split(f"_{app.config['path']['line_result']}")[0]
            file_name = app.config['path']['line_result']
            try:
                with open(f"{app.config['path']['dataset_dir']}/{item_name}_{app.config['path']['input_json']}", "r") as f:
                    input_col = json.load(f)
            except FileNotFoundError:
                return jsonify(status='warning', info='h.message.no_rawdata', dataset=file_name_list)
            df_col = df.columns.tolist()[1:]
            if df_col != input_col:
                return jsonify(status='warning', info='h.message.upload_raw_not_match', dataset=file_name_list)
        elif app.config['path']['scatter_result'] in input_data['name']:
            item_name = input_data['name'].split(f"_{app.config['path']['scatter_result']}")[0]
            file_name = app.config['path']['scatter_result']
        else:
            return jsonify(status='warning', info='h.message.name_error_res', dataset=file_name_list)
        base_path = os.getcwd()
        output_path = os.path.join(base_path, {app.config['path']['result_dir']},{item_name},"none")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            if item_name in file_name_list:
                file_name_list[item_name]['none'] = [file_name]
            else:
                file_name_list[item_name] = {'none': [file_name]}
        else:
            file_name_list[item_name]['none'].append(file_name)
        df.to_csv(os.path.join(output_path, f'{item_name}_{file_name}.csv'), index=False, encoding="utf-8")
        with open(app.config['path']['result_json'], "w") as f:
            json.dump(file_name_list, f, indent=4 )
        return jsonify(status='success', info='h.message.save_success', dataset=file_name_list)


@app.route('/download', methods=['GET','POST'])
def download_file():
    """
       Function to handle file downloads.

       This function generates a file path based on the data type and name provided in the request,
       and sends the file to the client.
    """
    input_data = request.get_json()
    if input_data['dataType'] == 0:
        filename = f"{app.config['path']['dataset_dir']}/{input_data['name']}_{input_data['itemName']}.csv"
    else:
        filename = f"{app.config['path']['result_dir']}/{input_data['name']}/{input_data['method']}/{input_data['name']}_{input_data['itemName']}.csv"
    return send_file(filename, as_attachment=True)


@app.route('/delete', methods=['GET','POST'])
def delete_file():
    """
        Delete files or result sets based on the provided data type and list.
        Functionality:
            1. If dataType is 0, deletes files and related metadata from app.config['path']['dataset_json'].
            2. If dataType is not 0, deletes result directories and updates "result_info.json".
    """
    input_data = request.get_json()
    if input_data['dataType'] == 0:
        with open(app.config['path']['dataset_json'], 'r') as f:
            try:
                file_info = json.load(f)
            except json.JSONDecodeError:
                return jsonify(status='warning', info='h.message.no_target')
        for item in input_data['deleteList']:
            if item not in file_info:
                return jsonify(status='warning', info='h.message.no_target')
            del file_info[item]
            os.remove(f"{app.config['path']['dataset_dir']}/{item}_{app.config['path']['input_csv']}")
            os.remove(f"{app.config['path']['dataset_dir']}/{item}_{app.config['path']['input_json']}")
        with open(app.config['path']['dataset_json'], "w") as f:
            json.dump(file_info, f, indent=4)
    else:
        with open(app.config['path']['result_json'], 'r') as f:
            try:
                file_info = json.load(f)
            except json.JSONDecodeError:
                return jsonify(status='warning', info='h.message.no_target')
        for item, method in zip(input_data['deleteList'], input_data['methodList']):
            if item not in file_info:
                return jsonify(status='warning', info='h.message.no_target')
            if method not in file_info[item]:
                return jsonify(status='warning', info='h.message.no_target')
            folder_path = f"{app.config['path']['result_dir']}/{item}/{method}"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                del file_info[item][method]
            else:
                return jsonify(status='warning', info='h.message.no_target')
        with open(app.config['path']['result_json'], "w") as f:
            json.dump(file_info, f, indent=4)
    return jsonify(status='success', info='h.message.delete_success', dataset=file_info)

# =================================Scatter Labelling===================================


@app.route('/dbscanPredict', methods=['POST', 'GET'])
def dbscan_predict():
    """
        Use an algorithm to predict single metric multi-machine datasets and return the prediction results.

        Parameters:
            dataset: string, the name of the dataset.
            method: string, the algorithm to use.

        Returns:
            status: string, running status.
            option: dict, the prediction results.
    """
    base_path = os.getcwd()
    result_path = os.path.join(base_path, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = os.path.join(base_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    input_data = request.get_json()
    with open(app.config['path']['dataset_json'], "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    info_list = file_name_list[input_data['dataset']]

    with open(app.config['path']['result_json'], "r") as f:
        try:
            result_list = json.load(f)
        except json.JSONDecodeError:
            result_list = {}

    if input_data['method'] == 'SlowSight':
        call_slowSight(input_data['dataset'], result_list, app.config['path'])
    elif input_data['method'] == 'DBSCAN':
        call_dbscan(input_data['dataset'], file_name_list, result_list, app.config['path'])
    elif input_data['method'] == 'Kmeans':
        call_kmeans(input_data['dataset'], file_name_list, result_list, app.config['path'])
    try:
        data_result = pd.read_csv(f"{app.config['path']['result_dir']}/{input_data['dataset']}/{input_data['method']}/{input_data['dataset']}_{app.config['path']['scatter_result']}.csv", encoding='utf-8')
    except FileNotFoundError:
        return jsonify(status='warning', info='h.message.no_data')

    res_list = []
    for (key, value) in info_list.items():
        res_item = []
        name_list =[]
        for (index, item_id) in enumerate(value['ids']):
            name_list.append(key+'_'+item_id)
            target_columns = [key+'_'+item_id+'_x', key+'_'+item_id+'_y', key+'_'+item_id+'_label']
            target_columns.append('timestamp')
            data = data_result[target_columns]
            data['num'] = [index] * len(data)
            # data['num'] = data['num'].astype(int)
            res_item += data.values.tolist()
        res_list.append({'name': 'scatter result', 'data': res_item, 'nameList': name_list, 'selectCluster': None})
    data_result.to_csv(f"{app.config['path']['result_dir']}/{input_data['dataset']}/{input_data['method']}/{input_data['dataset']}_{app.config['path']['scatter_result']}.csv", index=False, encoding='utf-8')
    # print(res_list)

    return jsonify(status='success', option=res_list)


@app.route('/dbChangePredict', methods=['POST', 'GET'])
def db_change_predict():
    """
        Modify the labeling of existing results.

        Parameters:
            fileName: string, the file name.
            method: string, the algorithm to use
            targetName: string, the name of the target attribute.
            changeList: list, the range of changes.
            ifError: int, whether the change range is an anomaly.

        Returns:
            errorList: list, the list of errors.
    """
    input_data = request.get_json()
    df = pd.read_csv(f"{app.config['path']['result_dir']}/{input_data['fileName']}/{input_data['method']}/{input_data['fileName']}_{app.config['path']['scatter_result']}.csv", encoding='utf-8')
    for item in input_data['changeList']:
        target_row = df['timestamp'] == item[0]
        df.loc[target_row, item[1] + '_label'] = int(input_data['targetCluster'])
    df.to_csv(f"{app.config['path']['result_dir']}/{input_data['fileName']}/{input_data['method']}/{input_data['fileName']}_{app.config['path']['scatter_result']}.csv", index=False, encoding='utf-8')
    return jsonify(res_string='save successfully')



# =======================================Line Labelling================================
@app.route('/multi/predictTarget', methods=['POST', 'GET'])
def multi_predict_target():
    """
        Use an algorithm to predict single metric multi-machine datasets and return the prediction results.

        Parameters:
            dataset: string, the name of the dataset.
            method: string, the algorithm to use.

        Returns:
            option: dict, the prediction results.
    """
    base_path = os.getcwd()
    result_path = os.path.join(base_path, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = os.path.join(base_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    input_data = request.get_json()
    with open(app.config['path']['dataset_json'], "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    info_list = file_name_list[input_data['dataset']]

    with open(app.config['path']['result_json'], "r") as f:
        try:
            result_list = json.load(f)
        except json.JSONDecodeError:
            result_list = {}

    if input_data['method'] == 'SlowSight':
        call_slowSight(input_data['dataset'], result_list, app.config['path'])
    data_metrics = pd.read_csv(f"{app.config['path']['dataset_dir']}/{input_data['dataset']}_{app.config['path']['input_csv']}", encoding='utf-8')
    try:
        data_result = pd.read_csv(f"{app.config['path']['result_dir']}/{input_data['dataset']}/{input_data['method']}/{input_data['dataset']}_{app.config['path']['line_result']}.csv", encoding='utf-8')
    except FileNotFoundError:
        return jsonify(status='warning', info='h.message.no_data')
    data_result = data_result.drop_duplicates(subset='timestamp')
    data_result = data_result.set_index('timestamp').reindex(data_metrics['timestamp'], fill_value=0).reset_index()
    res_list = []
    for (key, value) in info_list.items():
        for metric in value['metrics']:
            target_columns = [key+'_'+item_id+'_'+metric for item_id in value['ids']]
            target_columns.insert(0,'timestamp')
            data = data_metrics[target_columns]
            label = data_result[target_columns]
            res_list.append(build_multi_return(metric, data, label, ''))
    data_result.to_csv(f"{app.config['path']['result_dir']}/{input_data['dataset']}/{input_data['method']}/{input_data['dataset']}_{app.config['path']['line_result']}.csv", index=False, encoding='utf-8')
    return jsonify(status='success', option=res_list)

@app.route('/multi/changePredict', methods=['POST', 'GET'])
def multi_change_predict():
    """
        Modify the labeling of existing results.

        Parameters:
            fileName: string, the file name.
            targetName: string, the name of the target machine.
            metricName: string, the name of  target metric.
            changeRange: list, the range of changes.
            ifError: int, whether the change range is an anomaly.
            method: string, the algorithm used.

        Returns:
            errorList: list, the list of errors.
    """
    input_data = request.get_json()
    df = pd.read_csv(f"{app.config['path']['result_dir']}/{input_data['fileName']}/{input_data['method']}/{input_data['fileName']}_{app.config['path']['line_result']}.csv", encoding='utf-8')
    for i in range(input_data['changeRange'][0], input_data['changeRange'][1]+1):
        df.loc[i, input_data['targetName']+'_'+input_data['metricName']] = input_data['ifError']
    error_list = list_error(df[input_data['targetName']+'_'+input_data['metricName']].to_list())
    df.to_csv(f"{app.config['path']['result_dir']}/{input_data['fileName']}/{input_data['method']}/{input_data['fileName']}_{app.config['path']['line_result']}.csv", index=False, encoding='utf-8')
    print(error_list)
    return jsonify(errorList=error_list)

@app.route('/multi/chooseMachine', methods=['POST', 'GET'])
def multi_choose_machine():
    """
       Select data from specified machines for display.

       Parameters:
           dataset: string, the file name.
           categoryName: string, the target category name.
           itemList: list, the selected machine names.
           method: string, the algorithm used.

       Returns:
           status: string, the save status.
           info: string, the prompt message.
    """
    input_data = request.get_json()
    data_metrics = pd.read_csv(f"{app.config['path']['dataset_dir']}/{input_data['dataset']}_{app.config['path']['input_csv']}", encoding='utf-8')
    data_result = pd.read_csv(f"{app.config['path']['result_dir']}/{input_data['dataset']}/{input_data['method']}/{input_data['dataset']}_{app.config['path']['line_result']}.csv",
                              encoding='utf-8')
    with open(app.config['path']['dataset_json'], "r") as f:
        try:
            file_name_list = json.load(f)
        except json.JSONDecodeError:
            file_name_list = {}
    info_list = file_name_list[input_data['dataset']]
    res_list = []
    for metric in info_list[input_data['categoryName']]['metrics']:
        target_columns = [input_data['categoryName'] + '_' + item_id + '_' + metric for item_id in input_data['itemList']]
        target_columns.insert(0, 'timestamp')
        # print(target_columns)
        data = data_metrics[target_columns]
        label = data_result[target_columns]
        # label = data_result['pred_adjusted']
        res_list.append(build_multi_return(metric, data, label, ''))
    return jsonify(option=res_list)

def build_multi_return(picName, data, label, target):
    """
        Construct the return data for multi-machine scenarios.

        Parameters:
            picName: string, the name of the picture.
            data: dataframe, the data to be processed.
            label: dataframe, the label data.
            target: string, the target identifier (currently not used in the function).

        Returns:
            res_dict: dict, the constructed return data containing:
                - picName: string, the name of the picture.
                - chooseTeam: string, the target identifier (currently an empty string).
                - time: list, the list of timestamps.
                - eachTeam: list, the list of processed data for each machine.
    """
    name = [item.replace(f'_{picName}', '') for item in list(data.columns)]
    time = list(data.iloc[:, 0])
    res = []
    for i in range(1, len(name)):
        eachData = data.iloc[:, i].to_list()
        eachError = list_error(label.iloc[:, i].to_list())
        res.append({
            'name': name[i],
            'data': eachData,
            'error': eachError
        })
    res_dict = {
        'picName': picName,
        'chooseTeam': target,
        'time': time,
        'eachTeam': res
    }
    return res_dict

def list_error(lst):
    """
       Identify error intervals in a list of labels.

       Parameters:
           lst: list, the list of labels (0 or 1).

       Returns:
           result: list, the list of error intervals, where each interval is represented as [start, end].
    """
    arr = np.array(lst)
    result = []
    starts = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0] + 1  # Continuous 1's start positions
    ends = np.where((arr[:-1] == 1) & (arr[1:] == 0))[0]  # Continuous 1's end positions
    if arr[0] == 1:
        starts = np.insert(starts, 0, 0)  # If the first element is 1, start from 0
    if arr[-1] == 1:
        ends = np.append(ends, len(arr) - 1)  # If the last element is 1, end at the last position
    for s, e in zip(starts, ends):
        result.append([int(s), int(e)])
    return result







if __name__ == '__main__':

    app.run(debug=True)
