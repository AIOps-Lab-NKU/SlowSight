import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from SlowSight.utils.evaluation import evaluate_detection

class Reporter:
    def __init__(self, config, test_original_data, data_info, features, reconstruct_error, all_times, peer_results, peer_disk_labels, ground_truth_df=None, full_timestamps=None):
        """
        Initializes the Reporter instance with configuration and data required for anomaly analysis.

        Args:
            config: Configuration object containing paths and hyperparameters.
            test_original_data: Original test data including timestamps and metric values.
            data_info: Dictionary containing metadata about the dataset, such as train length.
            features: List of feature names (metrics) in the dataset.
            reconstruct_error: Reconstruction error from an autoencoder or similar model.
            all_times: Time stamps corresponding to each data point.
            peer_results: Dataframe containing results from peer comparison (e.g., DTW scores).
            peer_disk_labels: Anomaly labels for disks from peer comparison.
            full_timestamps: Optional. The full (train+valid+test) timestamps from the original D3
                             dataset (list / np.ndarray / pd.Series). When provided,
                             cause_up_metrics_mask.csv will be generated with len(full_timestamps)
                             rows and the same columns as test_original_data, matching the size
                             of the original D3 dataset.
        """

        self.config = config
        self.test_metric_data = test_original_data
        self.data_info = data_info
        self.all_features = features
        self.reconstruct_error = reconstruct_error
        self.all_times = all_times
        self.peer_results = peer_results
        self.peer_disk_labels = peer_disk_labels
        self.ground_truth_df = pd.DataFrame() if ground_truth_df is None else ground_truth_df
        self.merged_pred_df = None
        self.full_timestamps = None
        if full_timestamps is not None:
            if isinstance(full_timestamps, pd.Series):
                self.full_timestamps = full_timestamps.values
            elif isinstance(full_timestamps, list):
                self.full_timestamps = np.asarray(full_timestamps)
            else:
                self.full_timestamps = np.asarray(full_timestamps)

        self.merged_pred_df = None

    def merge_result(self):
        """
        Merges historical and peer prediction results based on timestamp using inner join,
        and combines predictions using logical OR operation. The merged result is saved to a CSV file.

        Note:
            - Fills missing values with 0 and converts to integer type.
            - Columns ending with '_1' are from history; '_2' are from peer comparisons.
        """
        history_df = pd.read_csv(os.path.join(self.config.result_dir, self.config.history_pred_labels))
        peer_df = pd.read_csv(os.path.join(self.config.result_dir, self.config.peer_pred_labels))
        self.merged_pred_df = pd.merge(
            history_df,
            peer_df,
            on="timestamp",
            suffixes=("_1", "_2"),
            how="inner"  # Use inner to keep shared timestamps
        )
        self.merged_pred_df["pred_1"] = self.merged_pred_df["pred_1"].fillna(0).astype(int)
        self.merged_pred_df["pred_2"] = self.merged_pred_df["pred_2"].fillna(0).astype(int)
        self.merged_pred_df["pred_adjusted_1"] = self.merged_pred_df["pred_adjusted_1"].fillna(0).astype(int)
        self.merged_pred_df["pred_adjusted_2"] = self.merged_pred_df["pred_adjusted_2"].fillna(0).astype(int)

        # Perform logical OR operations on pred and pred_adjusted columns
        self.merged_pred_df["pred"] = self.merged_pred_df["pred_1"] | self.merged_pred_df["pred_2"]
        self.merged_pred_df["pred_adjusted"] = self.merged_pred_df["pred_adjusted_1"] | self.merged_pred_df["pred_adjusted_2"]
        # ===== Whitelist recheck: for positions where pred_adjusted == 1,
        # verify the trend in a [-3, +5] window around whitelist metric columns =====
        WHITELIST_KEYWORDS = ['tcpsynretrans', 'tcptimeouts', 'retranssegs']
        RECHECK_BEFORE = 3
        RECHECK_AFTER = 5

        # 1) Select whitelist columns (latency/loss-related metrics) from the raw test data
        metric_cols = [c for c in self.test_metric_data.columns if c != 'timestamp']
        whitelist_cols = []
        for c in metric_cols:
            for kw in WHITELIST_KEYWORDS:
                if kw.lower() in str(c).lower():
                    whitelist_cols.append(c)
                    break

        # 2) Align merged_pred_df with test_metric_data by timestamp to build aligned
        # array of whitelist column values
        merged_times = self.merged_pred_df['timestamp'].astype(str).tolist()
        test_df_reidx = self.test_metric_data.copy()
        test_df_reidx['timestamp'] = test_df_reidx['timestamp'].astype(str)
        ordered_values = []
        for ts in merged_times:
            rows = test_df_reidx[test_df_reidx['timestamp'] == ts]
            if len(rows) > 0:
                ordered_values.append(rows[whitelist_cols].iloc[0].values.astype(float))
            else:
                ordered_values.append(np.full(len(whitelist_cols), np.nan))
        aligned_whitelist = np.array(ordered_values) if whitelist_cols else None
        def _window_increases(series, center_idx, before, after, MIN_RELATIVE_INCREASE=0.02, MIN_UP_RATIO=0.6):
            """
            Robust window-based upward-trend detection.
            Resistant to single-point jumps and boundary spikes.

            :param series: full time-series array of a single metric
            :param center_idx: center index of the anomaly
            :param before: number of points taken before the center
            :param after: number of points taken after the center
            :param MIN_RELATIVE_INCREASE: minimum relative increase threshold
            :param MIN_UP_RATIO: minimum ratio of monotonically increasing steps
            :return: True if a sustained increase is detected, False otherwise
            """
            n = len(series)
            start = max(0, center_idx - before)
            end = min(n, center_idx + after + 1)
            window_len = end - start
            if window_len < 5:
                return False

            # Extract the window and handle NaN values
            win = np.asarray(series[start:end], dtype=float)
            if np.any(np.isnan(win)):
                win = pd.Series(win).bfill().ffill().values

            # 1. Global linear slope (baseline trend)
            x = np.arange(window_len)
            slope = np.polyfit(x, win, 1)[0] if window_len >= 2 else 0.0

            # 2. Quantile comparison to avoid outlier influence (75th percentile instead of mean)
            mid = window_len // 2
            first = win[:mid]
            second = win[mid:]
            q3_first = np.percentile(first, 75)
            q3_second = np.percentile(second, 75)
            quant_diff = q3_second - q3_first

            # 3. Relative increase (based on the window median, more robust)
            win_median = np.median(np.abs(win))
            base = max(1e-8, win_median)
            rel = quant_diff / base

            # 4. Count ratio of increasing adjacent pairs, reject flat middle with boundary jumps
            up_count = 0
            for i in range(window_len - 1):
                if win[i + 1] >= win[i]:
                    up_count += 1
            up_ratio = up_count / (window_len - 1)

            # Require multiple conditions satisfied before reporting a real sustained worsening
            cond1 = slope >= 0                      # overall linear trend is upward
            cond2 = quant_diff > 0                  # second-half higher quantile excludes single-point spikes
            cond3 = rel > MIN_RELATIVE_INCREASE     # relative increase is meaningful
            cond4 = up_ratio >= MIN_UP_RATIO        # most adjacent pairs are increasing, rejects jump-only ends

            return cond1 and cond2 and cond3 and cond4
        n = len(self.merged_pred_df)
        # 4) Recheck only positions where pred_adjusted == 1 via whitelist columns in [-3, +5] window
        pred_raw = self.merged_pred_df["pred_adjusted"].values.astype(int).copy()
        pred_positions = int(np.sum(pred_raw == 1))

        # A prediction is kept only if at least MIN_RISING_COLS whitelist columns show an upward trend
        MIN_RISING_COLS = 3
        if aligned_whitelist is None or aligned_whitelist.shape[1] == 0:
            # No whitelist columns available for reference; fall back to no modification
            pred_rechecked = pred_raw
            print("[reporter] whitelist recheck: no whitelist columns, skipped")
            whitelist_details = []
        else:
            pred_rechecked = np.zeros(n, dtype=int)
            whitelist_details = []  # list of dicts, one entry per kept anomaly point
            # Key: iterate only over positions where pred == 1
            for i in np.where(pred_raw == 1)[0]:
                rising_count = 0
                rising_col_names = []
                for j in range(aligned_whitelist.shape[1]):
                    if _window_increases(aligned_whitelist[:, j], i, RECHECK_BEFORE, RECHECK_AFTER):
                        rising_count += 1
                        if whitelist_cols and j < len(whitelist_cols):
                            rising_col_names.append(whitelist_cols[j])
                        if rising_count >= MIN_RISING_COLS:
                            break
                if rising_count >= MIN_RISING_COLS:
                    pred_rechecked[i] = 1
                    # Persist attribute information for this kept anomaly point
                    ts_val = self.merged_pred_df["timestamp"].iloc[i]
                    whitelist_details.append({
                        "index": int(i),
                        "timestamp": str(ts_val),
                        "rising_count": int(rising_count),
                        "rising_columns": rising_col_names,
                        "window_before": RECHECK_BEFORE,
                        "window_after": RECHECK_AFTER,
                    })
            keep_count = int(np.sum(pred_rechecked == 1))
            print(
                f"[reporter] whitelist recheck: {keep_count}/{pred_positions} positions kept "
                f"after trend check (need >= {MIN_RISING_COLS} rising whitelist columns)"
            )
        self.merged_pred_df["pred_adjusted"] = pred_rechecked.astype(int)

        # Store per-point whitelist attributes alongside the main prediction output
        if whitelist_details:
            details_path = os.path.join(self.config.result_dir, "whitelist_kept_points.json")
            with open(details_path, "w") as f:
                json.dump(whitelist_details, f, indent=4)
        print("final_merge_result")
        pred_adjusted = evaluate_detection(self.merged_pred_df["ground_truth_1"].values, self.merged_pred_df["pred_adjusted"].values)
        print(self.merged_pred_df["ground_truth_1"].values, self.merged_pred_df["pred_adjusted"].values)
        print(os.path.join(self.config.result_dir, self.config.merged_pred_labels))
        self.merged_pred_df.to_csv(os.path.join(self.config.result_dir, self.config.merged_pred_labels), index=False)

    def get_series(self, data, start_time, end_time):
        """
        Filters the input DataFrame to only include rows within the specified time range.

        Args:
            data (pd.DataFrame): Input data with a 'timestamp' column.
            start_time (str or datetime): Start timestamp for filtering.
            end_time (str or datetime): End timestamp for filtering.

        Returns:
            pd.DataFrame: Filtered data between start_time and end_time.
        """
        time_stamp_str = data["timestamp"]
        data['timestamp'] = pd.to_datetime(time_stamp_str).values
        series_data = data.loc[data.timestamp <= pd.to_datetime(end_time)]
        series_data = series_data.loc[series_data.timestamp >= pd.to_datetime(start_time)]
        return series_data

    def get_dtw_rate(self, start_time, end_time):
        """
        Calculates average absolute anomaly score using DTW results for each metric.

        Args:
            start_time (str or datetime): Start timestamp for filtering the data.
            end_time (str or datetime): End timestamp for filtering the data.

        Returns:
            dict: A dictionary mapping each metric to its average anomaly score.
        """

        series_dtw = self.get_series(self.peer_results, start_time, end_time)
        metric_stats = {}
        dtw_rate = {}

        for col in series_dtw.columns:
            if col == "timestamp":
                continue
            metric_name = col.split("_")[2]
            target_column = col

            if metric_name in metric_stats:
                mean_metric_dtw = metric_stats[metric_name][0]
                std_metric_dtw = metric_stats[metric_name][1]
            else:
                metric_columns = [col for col in series_dtw.columns if metric_name in col]
                mean_metric_dtw = series_dtw[metric_columns].mean(axis=1)
                std_metric_dtw = series_dtw[metric_columns].std(axis=1)
                std_metric_dtw[np.where(std_metric_dtw == 0)[0]] = 1e-8
                metric_stats[metric_name] = [mean_metric_dtw, std_metric_dtw]

            target_values = series_dtw[col]
            anomaly_scores = (target_values - mean_metric_dtw) / std_metric_dtw
            # Calculate the average of absolute values
            mean_abs_anomaly_score = anomaly_scores.abs().mean()
            dtw_rate[target_column] = mean_abs_anomaly_score

        return dtw_rate

    def check_anomalous_type(self, start_time, end_time):
        """
        Checks if there are historical anomalies and peer anomalies during the specified time period.

        Args:
            start_time (str or datetime): Start timestamp for filtering the data.
            end_time (str or datetime): End timestamp for filtering the data.

        Returns:
            tuple:
                - history_anomaly (bool): True if any historical anomalies occurred.
                - peer_anomaly (dict): Mapping of disk IDs to whether they had anomalies.
        """
        series_pred = self.get_series(self.merged_pred_df, start_time, end_time)
        history_anomaly = False
        if series_pred['pred_adjusted_1'].sum() > 0:
            history_anomaly = True
        peer_anomaly = {}
        series_pred_disks = self.get_series(self.peer_disk_labels, start_time, end_time)
        for col in series_pred_disks.columns:
            if col == "timestamp":
                continue
            if series_pred_disks[col].sum() > 0:
                peer_anomaly[col] = True
            else:
                peer_anomaly[col] = False
        return history_anomaly, peer_anomaly

    def find_fault_segments(self):
        """
        Identifies continuous segments where anomalies (value == 1) occur in the merged prediction data.

        Returns:
            list of lists: Each sublist contains indices of consecutive anomaly points.
        """
        fault_segments = []
        adjusted_labels = self.merged_pred_df["pred_adjusted"].values
        start_idx = None
        for i, label in enumerate(adjusted_labels):
            if label == 1 and start_idx is None:
                # Start a consecutive 1
                start_idx = i
            elif label == 0 and start_idx is not None:
                fault_segments.append([k for k in range(start_idx, i)])
                start_idx = None
        # If the last paragraph is 1, add the last paragraph
        if start_idx is not None:
            fault_segments.append([k for k in range(start_idx, len(adjusted_labels))])
        return fault_segments

    def get_cause_weight(self, segment_l, threshold=None):
        """
        Calculates the contribution (weight) of each feature/metric to the anomaly in a given segment.

        Args:
            segment_l (list): List of indices representing a fault segment.
            threshold (float, optional): Threshold for selecting significant features.

        Returns:
            tuple:
                - alarm_index (np.array): Indices of top features exceeding the threshold.
                - array_kpi_degree_rate (np.array): Normalized importance of each feature.
                - alarm_index_full (np.array): All features sorted by importance.
                - overall_degree (float): Average anomaly severity across features.
        """
        train_reconstruct_avg = np.mean(self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std = np.std(self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std[np.where(train_reconstruct_std == 0)[0]] = 1e-8

        segment_require = segment_l[:10]
        kpi_normalised = abs(self.reconstruct_error[segment_require, :] - train_reconstruct_avg) / train_reconstruct_std
        param_coe = 3
        array_coe = np.array(
            [math.log(1 / (i / 10 + pow(math.e, -param_coe))) / param_coe for i in range(10)])  # A monotonically decreasing function
        array_kpi_degree = np.multiply(kpi_normalised.T,
                                       array_coe[:kpi_normalised.shape[0]])  # (features, 10) * (1, 10)
        array_kpi_degree = array_kpi_degree.sum(1)  # Degree of each feature
        array_kpi_degree_rate = array_kpi_degree / (array_kpi_degree.sum(0))  # Importance ratio of each feature
        overall_degree = array_kpi_degree.mean()  # Overall anomaly degree
        if threshold:
            alarm_index = np.where(array_kpi_degree_rate > threshold)[0]
        else:
            alarm_index_full = np.argsort(array_kpi_degree)[::-1]  # Root cause metrics sorted in descending order
            alarm_index = alarm_index_full[:2]  # Top 2 metrics for display
        return alarm_index, array_kpi_degree_rate, alarm_index_full, overall_degree

    def locate_cause(self, segment_list_l, threshold=None, rate_path=None, result_path=None):
        anomalies_cause_dict = {}
        anomalies_cause_full_dict = {}
        np_error_rate_all = []
        cause_TimeSeg_map = defaultdict(list)

        segment_labels, n_clusters, centroid_patterns = self.propagate_cluster_labels(segment_list_l)

        for segment_l in segment_list_l:
            if len(segment_l) != 0:
                segment_l_need = [i for i in segment_l]
                segment_cause_index, np_error_rate_segment, segment_cause_index_full, segment_degree = self.get_cause_weight(
                    segment_l_need, threshold)
                np_error_rate_all.append(np_error_rate_segment)
            else:
                segment_cause_index = []

            segment_key = " ".join([str(i) for i in segment_l])
            anomalies_cause_dict[segment_key] = segment_cause_index

            if len(segment_l) != 0:
                anomalies_cause_full_dict[segment_key] = {
                    "cause_index": segment_cause_index_full,
                    "error_rate": np_error_rate_segment,
                    "overall_error": segment_degree,
                    "cluster_label": segment_labels.get(segment_key, "unclustered"),
                    "n_clusters": int(n_clusters),
                }
        np_error_rate_all = np.array(np_error_rate_all).T
        if len(np_error_rate_all) != 0:
            df_error_rate_all = pd.DataFrame(np_error_rate_all, columns=[str(s_l[0]) for s_l in segment_list_l])
            if rate_path:
                df_error_rate_all.to_csv(rate_path)

        for k, v_l in anomalies_cause_dict.items():
            for v in v_l:
                cause_TimeSeg_map[v].append(k)

        cause_list = []
        for k, v in anomalies_cause_full_dict.items():
            k_list = k.split(" ")
            time_str_list = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in list(self.test_metric_data['timestamp'])]

            end_time_ = self.merged_pred_df['timestamp'][int(k_list[-1])]
            if isinstance(end_time_, pd.Timestamp):
                end_time_ = end_time_.strftime('%Y-%m-%d %H:%M:%S')
            end_idx = time_str_list.index(end_time_)
            start_time_ = self.merged_pred_df['timestamp'][int(k_list[0])]
            if isinstance(start_time_, pd.Timestamp):
                start_time_ = start_time_.strftime('%Y-%m-%d %H:%M:%S')
            start_idx = time_str_list.index(start_time_)

            anomaly_start = start_time_
            anomaly_end = end_time_
            features_name = [self.all_features[item] for item in v["cause_index"]]
            error_rate = [v["error_rate"][item] for item in v["cause_index"]]
            cur_cause_list = dict(zip(features_name, error_rate))

            if end_idx - start_idx <= self.config.classifier_seq_length // 2:
                if end_idx - self.config.classifier_seq_length + 1 < 0:
                    pattern_start_time = time_str_list[0]
                    pattern_end_time = time_str_list[0 + self.config.classifier_seq_length - 1]
                else:
                    pattern_start_time = time_str_list[end_idx - self.config.classifier_seq_length + 1]
                    pattern_end_time = end_time_
            else:
                if start_idx - self.config.classifier_seq_length // 2 + 1 < 0:
                    pattern_start_time = time_str_list[0]
                    pattern_end_time = time_str_list[0 + self.config.classifier_seq_length - 1]
                else:
                    pattern_start_time = time_str_list[start_idx - self.config.classifier_seq_length // 2 + 1]
                    pattern_end_time = time_str_list[
                        start_idx - self.config.classifier_seq_length // 2 + self.config.classifier_seq_length]
        # ===================== Generate a point-wise up-trend mask CSV 
        # Data source: whitelist_kept_points.json produced by the whitelist recheck in merge_result()
        whitelist_json_path = os.path.join(self.config.result_dir, "whitelist_kept_points.json")
        whitelist_kept = None
        if os.path.exists(whitelist_json_path):
            try:
                with open(whitelist_json_path, "r") as f:
                    whitelist_kept = json.load(f)
                print(f"[reporter] up-mask: loaded {len(whitelist_kept)} kept points from {whitelist_json_path}")
            except Exception as e:
                print(f"[reporter] up-mask: failed to parse {whitelist_json_path}: {e}")
                whitelist_kept = None
        else:
            print(f"[reporter] up-mask: {whitelist_json_path} not found, skipping mask generation")

        # Determine total row count and column set 
        use_full = (self.full_timestamps is not None and len(self.full_timestamps) > 0)
        src_cols_df = self.test_metric_data
        if use_full:
            n_rows = len(self.full_timestamps)
        else:
            n_rows = len(src_cols_df) if (src_cols_df is not None and not src_cols_df.empty) else 0

        if n_rows > 0 and src_cols_df is not None and not src_cols_df.empty and whitelist_kept:
            # Column order: first column is timestamp, the remaining metric columns follow
            # the original order of test_metric_data strictly
            metric_cols = [c for c in src_cols_df.columns if c != "timestamp"]
            all_cols = ["timestamp"] + metric_cols
            # Build a df of the same size; fill metric columns with zeros initially
            up_mask_df = pd.DataFrame(0, index=np.arange(n_rows), columns=all_cols)
            # Timestamp column (already the first): prefer full_timestamps ; otherwise fall back to test_metric_data
            if use_full:
                up_mask_df["timestamp"] = self.full_timestamps
            else:
                up_mask_df["timestamp"] = src_cols_df["timestamp"].values

            # Normalize up_mask_df's timestamp into a comparable Timestamp array
            raw_times = up_mask_df["timestamp"]
            try:
                src_time_list = pd.to_datetime(raw_times)
            except Exception:
                src_time_list = pd.Series(raw_times).astype(str)
                src_time_list = pd.to_datetime(src_time_list, errors="coerce")
            src_time_arr = src_time_list.values

            # Mark rows using whitelist_kept_points: for each kept anomaly point,
            # set its rising_columns to 1 at the matched timestamp row
            matched_points = 0
            for entry in whitelist_kept:
                ts_val = entry.get("timestamp")
                rising_cols = entry.get("rising_columns", [])
                if ts_val is None or not rising_cols:
                    continue
                try:
                    entry_dt = pd.to_datetime(ts_val)
                except Exception:
                    continue
                hit_mask = (src_time_arr >= entry_dt) & (src_time_arr <= entry_dt)
                hit_idx = np.where(hit_mask)[0]
                if hit_idx.size == 0:
                    continue
                # Restrict to metric columns that actually exist in the output
                valid_cols = [c for c in rising_cols if c in up_mask_df.columns]
                if not valid_cols:
                    continue
                for metric in valid_cols:
                    up_mask_df.loc[hit_idx, metric] = 1
                matched_points += 1

            out_dir = os.path.dirname(result_path) if result_path else os.getcwd()
            up_mask_csv = os.path.join(out_dir, "cause_up_metrics_mask.csv")
            up_mask_df.to_csv(up_mask_csv, index=False)
            marked_cols = [c for c in metric_cols if (up_mask_df[c].values == 1).any()]
            print(
                f"[reporter] cause_up_metrics_mask.csv saved"
            )
       
    def extract_segment_features(self, segment_l):
        """
        Extract feature vector for an anomaly segment.

        Args:
            segment_l (list): Indices within the anomaly segment.

        Returns:
            np.ndarray: Feature vector of shape (n_features,).
        """
        # Use statistical features of the reconstruction error as segment features
        segment_error = self.reconstruct_error[segment_l, :]
        
        features = []
        for i in range(segment_error.shape[1]):
            col_data = segment_error[:, i]
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                np.median(col_data)
            ])
        
        return np.array(features)
    
    def cluster_anomaly_segments(self, segment_list_l):
        """
        Group anomaly segments using hierarchical agglomerative clustering.

        Args:
            segment_list_l (list): List of anomaly segments.

        Returns:
            tuple: (cluster labels, cluster centroids, best number of clusters)
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.metrics import davies_bouldin_score

        # Extract features for all anomaly segments
        feature_vectors = []
        valid_segments = []

        for seg in segment_list_l:
            if len(seg) > 0:
                features = self.extract_segment_features(seg)
                feature_vectors.append(features)
                valid_segments.append(seg)

        if len(feature_vectors) < 2:
            return None, None, 1

        feature_vectors = np.array(feature_vectors)

        # Hierarchical clustering with Euclidean distance
        Z = linkage(feature_vectors, method='ward', metric='euclidean')

        # Select the optimal number of clusters via the Davies-Bouldin index
        max_clusters = min(10, len(feature_vectors))
        best_db_score = float('inf')
        best_k = 2
        best_labels = None
        
        for k in range(2, max_clusters + 1):
            labels = fcluster(Z, k, criterion='maxclust')
            db_score = davies_bouldin_score(feature_vectors, labels)
            
            if db_score < best_db_score:
                best_db_score = db_score
                best_k = k
                best_labels = labels
        
        # 获取聚类中心
        centroids = {}
        for cluster_id in np.unique(best_labels):
            mask = best_labels == cluster_id
            cluster_points = feature_vectors[mask]
            centroid = np.mean(cluster_points, axis=0)
            centroids[cluster_id] = centroid
        
        return best_labels, centroids, best_k, valid_segments
    
    def propagate_cluster_labels(self, segment_list_l):
        labels, centroids, best_k, valid_segments = self.cluster_anomaly_segments(segment_list_l)
        
        if labels is None:
            return {}, 1, {}
        
        # Assign a pattern label to each cluster centroid
        centroid_patterns = {}
        for cluster_id, centroid in centroids.items():
            centroid_patterns[cluster_id] = f"pattern_{cluster_id}"
        
        # Propagate labels to all members
        segment_labels = {}
        for i, seg in enumerate(valid_segments):
            segment_key = " ".join([str(idx) for idx in seg])
            segment_labels[segment_key] = centroid_patterns[labels[i]]
        
        return segment_labels, int(best_k), centroid_patterns