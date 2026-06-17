import os
import pickle
import time
import json
import copy

import pandas as pd
import numpy as np
from SlowSight.utils.preProcessor_temp import preProcessor


def smooth_data(df, window=3):
    for col in df.columns:
        if col == "timestamp":
            continue
        # Smooth the data of each column (except the 'timestamp' column) using a rolling window
        # Fill the resulting NaN values with backfill and then forward fill
        df[col] = df[col].rolling(window=window).mean().bfill().ffill().values


class dataLoader:
    def __init__(self, rawdata_file, config):
        self.rawdata_file = rawdata_file
        self.config = config

        self.__features = None
        self.__train_times = None
        self.__valid_times = None
        self.__test_times = None
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.ori_train = None
        self.ori_valid = None
        self.ori_test = None
        self.train_np = None
        self.valid_np = None
        self.test_np = None
        self.__scale = None
        # Read the data and perform initial processing
        self.__read_data()
        # Save the scaling information
        self.save_scale()

    def __read_data(self):
        # Read the raw metric data from a CSV file
        metric_raw = pd.read_csv(os.path.join(self.config.rawdata_dir, f"{self.rawdata_file}_metric.csv"))
        # Load the metric candidates from a JSON file
        with open(os.path.join(self.config.rawdata_dir, f"{self.rawdata_file}_metric_candidates.json"), "r") as f:
            metric_candidates = json.load(f)
        # Filter the metric candidates to only include columns present in the raw metric data
        metric_candidates = [col for col in metric_candidates if col in metric_raw.columns]

        # Smooth the metric data
        metric = metric_raw.copy()
        smooth_data(metric, window=self.config.smooth_window)
        channels_pd = metric[metric_candidates]
        time_stamp_str = metric["timestamp"]
        channels_pd['timestamp'] = pd.to_datetime(time_stamp_str).values

        # Split the data into training, validation, and test sets
        self.train_df, self.valid_df, self.test_df = self.__split_data(channels_pd)

        # Get the timestamps of the training, validation, and test sets
        self.__train_times, self.__valid_times, self.__test_times = self.train_df.timestamp.astype(
            str).tolist(), self.valid_df.timestamp.astype(str).tolist(), self.test_df.timestamp.astype(str).tolist()
        # Get all features of the whole dataset (excluding the timestamp column)
        self.__features = np.array(channels_pd.columns)[:-1]
        # Get the feature data of the training, validation, and test sets
        self.train_np, self.valid_np, self.test_np = self.train_df.values[:,
                                                                     :-1], self.valid_df.values[:, :-1], self.test_df.values[:, :-1]
        # Standardize the data features
        self.train_np, self.valid_np, self.test_np, self.__scale, self.ori_train, self.ori_valid, self.ori_test = preProcessor.preprocess(self.train_np,
                                                                                           self.valid_np,
                                                                                           self.test_np,
                                                                                           self.config.preprocess_type,
                                                                                           self.config.clip_alpha)

    def __split_data(self, rawdata_df):
        # Split the raw data into training set based on the training start and end times
        train_df = rawdata_df.loc[rawdata_df.timestamp < pd.to_datetime(self.config.train_end)]
        train_df = train_df.loc[rawdata_df.timestamp >= pd.to_datetime(self.config.train_start)]
        # Split the raw data into validation set based on the validation start and end times
        valid_df = rawdata_df.loc[rawdata_df.timestamp < pd.to_datetime(self.config.valid_end)]
        valid_df = valid_df.loc[rawdata_df.timestamp >= pd.to_datetime(self.config.valid_start)]
        # Split the raw data into test set based on the test start and end times
        test_df = rawdata_df.loc[rawdata_df.timestamp < pd.to_datetime(self.config.test_end)]
        test_df = test_df.loc[rawdata_df.timestamp >= pd.to_datetime(self.config.test_start)]

        return train_df, valid_df, test_df

    def save_scale(self):
        # Open a file to save the scaling information
        output_hal = open(os.path.join(
            self.config.result_dir, 'scaler.pkl'), 'wb')
        # Serialize the scale object and write it to the file
        tmp = pickle.dumps(self.__scale)
        output_hal.write(tmp)
        output_hal.close()

    def return_data(self):
        # Return the processed data including training, validation, test data and original data
        return self.train_np, self.valid_np, self.test_np, self.ori_train, self.ori_valid, self.ori_test

    def return_original_data(self):
        # Return the original data including training, validation, test data and features
        return self.train_df, self.valid_df, self.test_df, self.__features

    @property
    def scale(self):
        return self.__scale

    @property
    def features(self):
        return self.__features

    @property
    def train_times(self):
        return self.__train_times

    @property
    def valid_times(self):
        return self.__valid_times

    @property
    def test_times(self):
        return self.__test_times
