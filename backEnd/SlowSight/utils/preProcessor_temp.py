import numpy as np
import pandas as pd

import warnings

from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')


class preProcessor:
    @classmethod
    def preprocess_mix_max100(cls, np_train, np_valid, np_test, metric_length, bandwidth_l):
        """
        Preprocesses mixed data types: metrics are normalized using max-100 scaling,
        while logs are scaled using Min-Max normalization.

        Args:
            np_train (np.ndarray): Training data.
            np_valid (np.ndarray): Validation data.
            np_test (np.ndarray): Test data.
            metric_length (int): Number of metric columns in the data.
            bandwidth_l (list or np.ndarray): Bandwidth scale factors for specific metric columns.

        Returns:
            Processed train/valid/test data, along with scalers used.
        """
        np_train = np.asarray(np_train, dtype=np.float32)
        np_valid = np.asarray(np_valid, dtype=np.float32)
        np_test = np.asarray(np_test, dtype=np.float32)

        # Split into metric and log parts
        metric_train = np_train[:, :metric_length]
        metric_valid = np_valid[:, :metric_length]
        metric_test = np_test[:, :metric_length]

        log_train = np_train[:, metric_length:]
        log_valid = np_valid[:, metric_length:]
        log_test = np_test[:, metric_length:]

        # Apply Min-Max scaling on log features
        scale_log = MinMaxScaler()
        scale_log = scale_log.fit(log_train)
        log_train = scale_log.transform(log_train)
        log_valid = scale_log.transform(log_valid)
        log_test = scale_log.transform(log_test)

        # Apply Max-100 scaling on metric features
        cpu_mem_cols_len = metric_length - len(bandwidth_l)
        scale_metric = np.array([100] * cpu_mem_cols_len + bandwidth_l)
        metric_train = metric_train / scale_metric
        metric_valid = metric_valid / scale_metric
        metric_test = metric_test / scale_metric

        # Concatenate processed metric and log parts
        np_train = np.concatenate([metric_train, log_train], axis=1)
        np_valid = np.concatenate([metric_valid, log_valid], axis=1)
        np_test = np.concatenate([metric_test, log_test], axis=1)

        return np_train, np_valid, np_test, scale_metric, scale_log

    @classmethod
    def preprocess(cls, df_train, df_valid, df_test, pro_type, clip_alpha=None):
        """
        Applies various preprocessing methods to normalize or standardize the input data.

        Args:
            df_train (pd.DataFrame or np.ndarray): Training data.
            df_valid (pd.DataFrame or np.ndarray): Validation data.
            df_test (pd.DataFrame or np.ndarray): Test data.
            pro_type (str): Type of preprocessing method. Options include:
                'minmax', 'minmax_all', 'standard', 'max100', 'clip'.
            clip_alpha (float): Optional alpha value for clipping values during normalization.

        Returns:
            Processed train/valid/test data, scaler, and original data copies.
        """
        df_train = np.asarray(df_train, dtype=np.float32)
        df_valid = np.asarray(df_valid, dtype=np.float32)
        df_test = np.asarray(df_test, dtype=np.float32)

        ori_train = np.array(df_train)
        ori_valid = np.array(df_valid)
        ori_test = np.array(df_test)

        if pro_type == "minmax":
            # Min-Max scaling based on training data only
            scale = MinMaxScaler()
            scale = scale.fit(df_train)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)
            df_test = scale.transform(df_test)
        elif pro_type == "minmax_all":
            # Min-Max scaling on combined train+valid+test data
            df_all = np.concatenate((df_train, df_valid, df_test), axis=0)
            scale = MinMaxScaler()
            scale = scale.fit(df_all)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)
            df_test = scale.transform(df_test)
        elif pro_type == "standard":
            # Z-score standardization based on training data
            scale = StandardScaler().fit(df_train)
            df_train = scale.transform(df_train)
            df_valid = scale.transform(df_valid)
            df_test = scale.transform(df_test)
        elif pro_type == "max100":
            # Scale all values by 100 (simple normalization)
            scale = 100.0
            df_train = df_train / scale
            df_test = df_test / scale
        elif pro_type == "clip" and clip_alpha:
            # Clip extreme values based on mean ± alpha*std, then normalize
            alpha = clip_alpha

            def my_transform(value, mean=None, std=None):
                if mean is None:
                    mean = np.mean(value, axis=0)
                if std is None:
                    std = np.std(value, axis=0)
                    for j in range(len(std)):
                        if std[j] < 1e-4:
                            std[j] = 1
                for i in range(value.shape[0]):
                    clip_value = mean + alpha * std  # upper bound
                    temp = value[i] < clip_value
                    value[i] = temp * value[i] + (1 - temp) * clip_value

                    clip_value = mean - alpha * std  # lower bound
                    temp = value[i] > clip_value
                    value[i] = temp * value[i] + (1 - temp) * clip_value

                    value[i] = (value[i] - mean) / std  # normalization
                return value, mean, std

            df_train, mean_, std_ = my_transform(df_train)
            scale = {"mean": mean_, "std": std_}
            df_valid, _, _ = my_transform(df_valid, mean_, std_)
            df_test, _, _ = my_transform(df_test, mean_, std_)
        else:
            raise ValueError('Need to choose a valid preprocessing method')

        return df_train, df_valid, df_test, scale, ori_train, ori_valid, ori_test
