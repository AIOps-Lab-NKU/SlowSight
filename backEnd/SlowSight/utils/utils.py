# -*- coding: utf-8 -*-
import os
import time
import random

import yaml
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from SlowSight.utils.spot import SPOT

import matplotlib.pyplot as plt


# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# Utility class to handle configuration files and runtime parameters
class ConfigHandler:
    """
    Class to handle configuration files and command-line arguments.

    Initializes config from a YAML file and allows runtime parameter overrides.
    """

    def __init__(self, run_time_para):
        """
        Load config from YAML file and merge with runtime parameters.
        """
        # load default config from the parent directory (backEnd)
        dir_ = os.path.dirname(os.path.abspath(__file__))
        dir_ = os.path.dirname(dir_)
        config_path = os.path.join(dir_, run_time_para['config_file'])
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config_dict = yaml.load(f, Loader=yaml.FullLoader)

        # update config according to executing parameters
        parser = argparse.ArgumentParser()
        for field, value in self._config_dict.items():
            parser.add_argument(f'--{field}', default=value)
        for field, value in run_time_para.items():
            parser.add_argument(f'--{field}', default=value)
        self._config = parser.parse_args()

        # complete config
        self._trans_format()
        self._complete_dirs()

    def _trans_format(self):
        """
        Convert invalid formats of config values to valid ones (e.g., string numbers to int/float).
        """
        config_dict = vars(self._config)
        for item, value in config_dict.items():
            if value == 'None':
                config_dict[item] = None
            elif isinstance(value, str) and is_number(value):
                if value.isdigit():
                    value = int(value)
                else:
                    value = float(value)
                config_dict[item] = value

    def _complete_dirs(self):
        """
        Complete directory paths in the config by appending dataset and method names.
        """
        if self._config.model_dir:
            self._config.model_dir = self._make_dir(self._config.model_dir)

        if self._config.result_dir:
            self._config.result_dir = self._make_dir(self._config.result_dir)

    def _make_dir(self, dir_):
        base = os.getcwd()
        print(base, dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        this_dir = os.path.join(dir_, self._config.dataset, self._config.method)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        return this_dir

    @property
    def config(self):
        return self._config


def get_data_dim(dataset):
    """
    Returns the number of dimensions/features based on the dataset name.
    """
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('service'):
        return 3
    elif str(dataset).startswith('explore'):
        return 24
    elif str(dataset) == 'new':
        return 19
    elif str(dataset) == 'CM20210815':
        return 937
    elif str(dataset) == 'CM20210815-14-277':
        return 277
    elif str(dataset) == 'CM20210815-7-277':
        return 277
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None, do_preprocess=True, train_start=0,
             test_start=0, prefix="processed", quntile=None, do_noise=False):
    """
    Load data from .pkl files and apply preprocessing.

    Args:
        dataset: Name of the dataset.
        max_train_size: Maximum size of training set.
        max_test_size: Maximum size of test set.
        do_preprocess: Whether to normalize the data.
        train_start: Start index of training data.
        test_start: Start index of test data.
        prefix: Path where data files are stored.
        quntile: Percentile for threshold-based normalization.
        do_noise: Whether to add Gaussian noise to zero columns.

    Returns:
        Processed train and test data with labels.
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('=' * 30)
    print(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f)
    f.close()
    train_data = train_data.reshape((-1, train_data.shape[1]))[train_start:train_end, :]
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f)
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    test_data = test_data.reshape((-1, test_data.shape[1]))[test_start:test_end, :]
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    test_data = fillData(test_data)
    train_data = fillData(train_data)

    if do_preprocess:
        train_data, test_data = preprocess(train_data, test_data, do_preprocess, quntile=quntile)

    if do_noise:
        mask_test = (test_data == 0).all(0)
        column_indices_test = np.where(mask_test)[0]
        mask_train = (train_data == 0).all(0)
        column_indices_train = np.where(mask_train)[0]
        if column_indices_train.shape[0] != 0:
            train_data = addNoise(train_data, column_indices_train)
        if column_indices_test.shape[0] != 0:
            test_data = addNoise(test_data, column_indices_test)

    if test_label is not None:
        print("test label shape: ", test_label.shape)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    return (train_data, None), (test_data, test_label)


def wgn(x, snr):
    """
    Add white Gaussian noise to signal x.
    """
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise


def gaussNoise(x, mu, sigma):
    """
    Add Gaussian noise to each element of the array.
    """
    for i in range(x.shape[0]):
        x[i] += random.gauss(mu, sigma)
    return x


def addNoise(data, index_column):
    """
    Add Gaussian noise to specific columns that contain all zeros.
    """
    for i in index_column:
        t = data[:, i]
        res = gaussNoise(t, 0, 0.05)
        data[:, i] = res
    return data


def fillData(np_data):
    """
    Fill missing values using linear interpolation.
    """
    df = pd.DataFrame(np_data)
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    res = np.array(df)
    return res


def preprocess(df_train, df_test, do_preprocess, quntile=None):
    """
    Normalize raw data using standardization or quantile-based scaling.

    Args:
        df_train: Training data.
        df_test: Test data.
        do_preprocess: Whether to perform normalization.
        quntile: Optional percentile used for thresholding.

    Returns:
        Normalized train and test data.
    """
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with interpolate')
        df_train = fillData(df_train)
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with interpolate')
        df_test = fillData(df_test)
    if do_preprocess:
        if quntile is None:
            scaler = StandardScaler().fit(df_train)
            df_train = scaler.transform(df_train)
            df_test = scaler.transform(df_test)
        else:
            df_min = np.percentile(np.sort(df_train, axis=0), 100 - quntile, axis=0)
            df_max = np.percentile(np.sort(df_train, axis=0), quntile, axis=0)

            scaler = df_max - df_min
            scaler[np.where(scaler == 0)[0]] = 1

            df_train = (df_train - df_min) / scaler
            df_test = (df_test - df_min) / scaler

    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    return df_train, df_test


def is_number(s):
    """
    Check if a string can be converted to a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def paint(datas, title=None, scores=None, names=None, on_dim=False, thresholds=None, store_path=None, x_names=None):
    """
    Plot time series data along with anomaly scores and optional thresholds.

    Args:
        datas: Input data with shape [dims, time_length].
        scores: Anomaly scores corresponding to the input data.
        names: Feature names to display.
        on_dim: Whether to plot scores per dimension.
        thresholds: Thresholds for anomaly detection.
        store_path: File path to save the figure.
        x_names: X-axis tick labels.
    """
    data = np.array(datas, dtype=float)
    scores = np.array(scores, dtype=float)

    total_row = data.shape[0]

    if on_dim:
        pair_num = 2
    else:
        pair_num = 1

    if scores is not None:
        if on_dim:
            total_row *= pair_num
            total_row += 1
        else:
            total_row += 1

    fig = plt.figure(figsize=(14, 7))
    for i in range(1, data.shape[0] + 1):
        ax = fig.add_subplot(total_row, 1, (i - 1) * pair_num + 1)
        ax.plot(data[i - 1], color='black')
        if names is None:
            ax.set_ylabel('dim ' + str(i - 1), rotation=0, labelpad=20)
        else:
            ax.set_ylabel(str(names[i - 1]), rotation=0, labelpad=20)

        if x_names is not None:
            a = [i for i in range(0, x_names.shape[0], x_names.shape[0] // 8)]
            label = [x_names[i] for i in a]
            ax.set_xticks(a)
            ax.set_xticklabels(label)

        if on_dim:
            ax = fig.add_subplot(total_row, 1, (i - 1) * pair_num + 2)
            ax.plot(scores[i - 1])
            if names is None:
                ax.set_ylabel('score ' + str(i - 1), rotation=0, labelpad=20)
            else:
                ax.set_ylabel('score ' + str(names[i - 1]), rotation=0, labelpad=20)
    # Plot overall score
    ax = fig.add_subplot(total_row, 1, total_row)
    if on_dim:
        scores = np.sum(scores, axis=0)
    scores = np.reshape(scores, -1)
    ax.plot(scores, color='blue')
    if names is None:
        ax.set_ylabel('dim ' + str(i - 1), rotation=0, labelpad=20)
    else:
        ax.set_ylabel(str(names[i - 1]), rotation=0, labelpad=20)

    if thresholds is not None:
        ax.plot(thresholds, color='red')
        anomaly_point = []
        for i in range(len(scores)):
            if scores[i] > thresholds[i]:
                anomaly_point.append(i)
        ax.scatter(anomaly_point, scores[anomaly_point], color='red', marker='.')

    if title:
        plt.title(title)
    if store_path is not None:
        plt.savefig(store_path, dpi=600)
    else:
        plt.show()


def pot_detect(init_score, score, windowsize, q=1e-3, level=0.98):
    """
    Run POT (Peak Over Threshold) method on given anomaly scores.

    Args:
        init_score: Data used to initialize threshold (usually train set scores).
        score: Data to detect anomalies on (usually test set scores).
        windowsize: Not used here.
        q: Detection risk level.
        level: Probability associated with the initial threshold.

    Returns:
        dict: POT result including alarms and thresholds.
    """
    # print("level:",level)
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, verbose=False)  # initialization step
    ret = s.run(dynamic=True)  # run

    return ret


def drawLoss(dataset_name, model_result_path, train_loss1, epoch, train_loss2=None):
    """
    Plot training loss curves.

    Args:
        dataset_name: Name of the dataset.
        model_result_path: Path to save the plot.
        train_loss1: First loss component.
        epoch: Number of training epochs.
        train_loss2: Optional second loss component.
    """
    X = range(epoch)
    fig = plt.figure(figsize=(14, 7))
    plt.plot(X, train_loss1, color="blue", label="loss1")
    if train_loss2:
        plt.plot(X, train_loss2, color="red", label="loss2")
    plt.legend(loc="upper left")
    plt.title(f'{dataset_name}')
    plt.savefig(f'{model_result_path}')
    plt.clf()


def paintReconstruct(processed_data, reconstruct_data, anomalies, thresholds_l, data, y_names, x_names,
                     all_scores=None, cause_indicator_index=None, fig_path=None):
    """
    Plot original and reconstructed data, optionally with anomaly scores and thresholds.

    Args:
        processed_data: Original data.
        reconstruct_data: Reconstructed data.
        anomalies: Detected anomaly indices.
        thresholds_l: Thresholds for anomaly detection.
        data: Dictionary containing metadata like train length.
        y_names: Y-axis feature names.
        x_names: X-axis time labels.
        all_scores: All anomaly scores.
        cause_indicator_index: Indices of root cause features.
        fig_path: Path to save the plot.
    """
    features_num = data['features_num']

    pair_num = 1
    anomaly_paint_flag = 0
    if all_scores is not None:
        anomaly_paint_flag = 1
    total_row = processed_data.shape[0] * pair_num + anomaly_paint_flag
    X_all = np.arange(0, processed_data.shape[1])
    X_train = np.arange(0, data['train_length'])
    X_test = np.arange(data['train_length'] + data['valid_length'], processed_data.shape[1])

    fig, ax = plt.subplots(total_row, 1, figsize=(16, 3 * total_row))
    a = [j for j in range(0, x_names.shape[0], x_names.shape[0] // 7)]
    x_label = [x_names[j] for j in a]

    if all_scores is not None:
        ax[0].plot(X_all, all_scores, label="anomaly_score")
        ax[0].plot(X_test, thresholds_l, '--', label="spot_threshold")
        ax[0].axvline(data['train_length'], c='#ff0097', ls='-.', lw=3)
        ax[0].axvline(data['train_length'] + data['valid_length'], c='#ff0097', ls='-.', lw=3)
        if len(anomalies) != 0:
            ax[0].scatter(anomalies, all_scores[anomalies], color='red', label='anomaly points')

        ax[0].text(0.35, 0.5,
                   f"number of anomalies:{len(anomalies)}\nnumber of indicators:{features_num}\nvalid_mse: {data['valid_mse']}  valid_mae: {data['valid_mae']}\ntest_mse: {data['test_mse']}  test_mae: {data['test_mae']}",
                   transform=ax[0].transAxes)
        ax[0].set_title('anomaly detection')
        ax[0].set_xticks(a)
        ax[0].set_xticklabels(x_label)
        ax[0].legend(loc='upper left', ncol=3)

    for i in range(processed_data.shape[0]):
        ax[(i) * pair_num + anomaly_paint_flag].plot(X_all, processed_data[i], label="original data")
        ax[(i) * pair_num + anomaly_paint_flag].plot(X_all, reconstruct_data[i], color='#ff8936',
                                                     label="reconstruct data")
        ax[i].axvline(data['train_length'], c='#ff0097', ls='-.', lw=3)
        ax[i].axvline(data['train_length'] + data['valid_length'], c='#ff0097', ls='-.', lw=3)
        if len(anomalies) != 0:
            ax[(i) * pair_num + anomaly_paint_flag].scatter(anomalies, processed_data[i, anomalies], color='red',
                                                            label='anomaly points')

        ax[(i) * pair_num + anomaly_paint_flag].set_title(str(i) + '-' + str(y_names[i]))

        ax[(i) * pair_num + anomaly_paint_flag].set_xticks(a)
        ax[(i) * pair_num + anomaly_paint_flag].set_xticklabels(x_label)

        ax[(i) * pair_num + anomaly_paint_flag].legend(loc='upper left', ncol=3)

    # show cause_indicator_index
    if cause_indicator_index is not None:
        for i in cause_indicator_index:
            ymin = min(np.min(reconstruct_data[i]), np.min(processed_data[i]))
            ymax = max(np.max(reconstruct_data[i]), np.max(processed_data[i]))
            fill_X = [0, 0, data['train_length'], data['train_length']]
            fill_Y = [ymin, ymax, ymax, ymin]
            ax[(i) * pair_num + anomaly_paint_flag].set_title(str(i) + '-' + str(y_names[i]), color='red')
            ax[(i) * pair_num + anomaly_paint_flag].fill(fill_X, fill_Y, color='#ffb3a7')

    fig.tight_layout()
    if fig_path is not None:
        print("save")
        plt.savefig(fig_path)
    else:
        plt.show()


def paintIndicators(all_indicators, required_indicator, processed_data, reconstruct_data, anomalies, thresholds_l,
                    data, x_names, all_scores=None, cause_indicator_index=None, fig_path=None):
    """
    Plot only selected indicators from all available indicators.
    """
    all_indicators = list(all_indicators)
    required_processed_data = []
    required_reconstruct_data = []
    for item in required_indicator:
        tmp_index = all_indicators.index(item)
        required_processed_data.append(processed_data[:, tmp_index])
        required_reconstruct_data.append(reconstruct_data[:, tmp_index])
    required_processed_data = np.array(required_processed_data)
    required_reconstruct_data = np.array(required_reconstruct_data)
    paintReconstruct(required_processed_data, required_reconstruct_data, anomalies,
                     thresholds_l, data, required_indicator, x_names,
                     all_scores, cause_indicator_index, fig_path)


def get_segment_dict(anomalies_cause_dict):
    anomalies_cause_dict.keys()
    result = []
    s = []  # empty stack
    for i in anomalies_cause_dict.keys():
        if len(s) == 0 or s[-1] + 1 == i:
            s.append(i)  # input
        else:
            result.append({ana:anomalies_cause_dict[ana] for ana in s})
            s = []  # clean
            s.append(i)  # output
    # The last round requires judgment
    result.append({ana:anomalies_cause_dict[ana] for ana in s})
    # Returns an exception
    return result


def get_segment_list(anomalies):
    """
    Group consecutive anomaly indices into lists.

    Args:
        anomalies: List of anomaly indices.

    Returns:
        List of lists, each representing a continuous anomaly segment.
    """
    if len(anomalies) == 0:
        return []
    result = []
    s = []
    for i in anomalies:
        if len(s) == 0 or s[-1] + 1 == i:
            s.append(i)
        else:
            result.append(s)
            s = []
            s.append(i)
    result.append(s)
    return result


def show_anomalies(ano_l):
    """
    Format anomaly indices into a compact string for visualization.

    Args:
        ano_l: List of anomaly indices.

    Returns:
        Formatted string showing detected anomalies.
    """
    t = ano_l[0]
    for i in range(1, len(ano_l)):
        if i % 6 == 0:
            t = t + '\n'
            t = t + ano_l[i]
        else:
            t = t + ','
            t = t + ano_l[i]
    return t