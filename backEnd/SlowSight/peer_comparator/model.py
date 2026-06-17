import os
import copy
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode
from concurrent.futures import ThreadPoolExecutor
from SlowSight.peer_comparator.dtw import compare_single_disk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from SlowSight.utils.evaluation import evaluate_detection
from SlowSight.peer_comparator.dbscan import cluster_with_dbscan, compute_statistics, match_with_dbscan, refine_clusters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def smooth_data(df, window=3):
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().ffill().values


class PeerComparator:
    def __init__(self, ground_truth_df, train_data, valid_data, test_data, metric_candidates, result_path, smooth_window=18,
                 window_size=38, batch_size=50, func='dtw', **kwargs):
        self.smooth_window = smooth_window
        self._window_size = window_size
        self._batch_size = batch_size
        self.ground_truth_df = ground_truth_df
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.test_times = np.array(self.test_data.timestamp.astype(str).tolist())
        self.train_process_data = copy.deepcopy(train_data)
        self.test_process_data = copy.deepcopy(test_data)
        self.metric_candidates = metric_candidates
        self.disks = []
        self.instances = []
        self.metrics = []
        self.node_columns = {}
        self.pred_labels = None
        self.host_labels = pd.DataFrame({"timestamp": self.test_times, "ground_truth": np.zeros(len(self.test_times))})
        self.key_features = ['latency']
        self.max_workers = kwargs.get('max_workers', 8)
        self.downsample_factor = kwargs.get('downsample_factor', 1)
        self.sakoe_chiba_width = kwargs.get('sakoe_chiba_width', 10)
        self.dbscan_eps = kwargs.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = kwargs.get('dbscan_min_samples', 5)
        self.n_components = kwargs.get('n_components', 2)

        self.__train_times = None
        self.__valid_times = None
        self.__test_times = None
        self.train_results = None
        self.valid_results = None
        self.test_results = None

        self.normal_clusters = None
        self.dbscan = None
        self.pca = None
        self.col_mode_dict = None

        self.result_path = result_path

        self.__get_all_features()
        if func == 'dtw':
            # self.preprocess()
            self.train_results = self.compare_dtw_with_peers(self.train_data, 0)
            self.valid_results = self.compare_dtw_with_peers(self.valid_data, len(self.train_data))
            self.test_results = self.compare_dtw_with_peers(self.test_data,
                                                            len(self.train_data) + len(self.valid_data))
        elif func == 'cluster':
            self.col_mode_dict = self.get_mode_value(self.train_data)
            # self.train_results = self.compare_dbscan_with_peers(self.train_data, flag='train')
            # self.valid_results = self.compare_dbscan_with_peers(self.valid_data, flag='valid')
            self.test_results = self.compare_dbscan_with_peers(self.test_data, window_size=10)
            # self.plot_dbscan_all(train_cluster_labels, train_feature_pca, test_cluster_labels, test_feature_pca, 'peer_dbscan_result_all.png')

    def __smooth_data(self):
        smooth_data(self.train_data, window=self.smooth_window)
        smooth_data(self.valid_data, window=self.smooth_window)
        smooth_data(self.test_data, window=self.smooth_window)

    def __get_all_features(self):
        for metric_candidate in self.metric_candidates:
            node_pieces = metric_candidate.split('_')
            instance = node_pieces[0] + '_' + node_pieces[1]
            metric = node_pieces[2]
            if instance not in self.instances:
                self.instances.append(instance)
            if metric not in self.metrics:
                self.metrics.append(metric)
            if node_pieces[1] not in self.node_columns:
                self.node_columns[instance] = [metric_candidate]
            else:
                self.node_columns[instance].append(metric_candidate)

    def return_results(self):
        return self.train_results, self.valid_results, self.test_results

    def preprocess(self):
        self.__train_times, self.__test_times = self.train_process_data.timestamp.astype(
            str).tolist(), self.test_process_data.timestamp.astype(str).tolist()
        # Entire dataset feature matrix
        train_np, test_np = self.train_process_data.values[:, :-1], self.test_process_data.values[:, :-1]
        df_train = np.asarray(train_np, dtype=np.float32)
        df_test = np.asarray(test_np, dtype=np.float32)
        scale = MinMaxScaler().fit(df_train)
        df_train = scale.transform(df_train)
        df_test = scale.transform(df_test)

    def get_mode_value(self, values):
        col_mode_dict = {}
        for node, columns in self.node_columns.items():
            node_data_all = values[columns]

            for col in columns:
                col_data_all = node_data_all[col].values
                col_mode = mode(col_data_all)
                mode_value = col_mode.mode[0]
                col_mode_dict[col] = mode_value
        return col_mode_dict

    def compare_dtw_with_peers(self, values, idx):
        """
        Compare each disk's metrics with the average of peer disks using DTW distance.
        """
        result_df = pd.DataFrame({"timestamp": values["timestamp"]})

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for t in range(len(values)):
                # Select window
                if t < self._window_size:
                    window = values.iloc[:t + 1]
                else:
                    window = values.iloc[t - self._window_size + 1:t + 1]

                for metric in self.metrics:
                    for target_disk in self.disks:
                        futures.append(
                            executor.submit(compare_single_disk, self.disks, t, window, metric, target_disk,
                                            self.downsample_factor, self.sakoe_chiba_width)
                        )

            # Collect results from threads
            for future in futures:
                t, column_name, distance = future.result()
                result_df.loc[t + idx, column_name] = distance

        return result_df

    def compare_dbscan_with_peers(self, values, window_size, eps=0.5, min_samples=5, n_components=0.95):
        result_df = pd.DataFrame({"timestamp": values["timestamp"]})
        feature_df = pd.DataFrame({"timestamp": values["timestamp"]})
        timestamps = values['timestamp'].unique()

        for t_idx in tqdm(range(0, window_size)):
            current_time = timestamps[t_idx]
            for node, columns in self.node_columns.items():
                result_df.loc[result_df["timestamp"] == current_time, f"{node}_cluster"] = 0

        stat_df = {}

        # Iterate over each timestamp
        for t_idx in tqdm(range(window_size, len(timestamps))):
            current_time = timestamps[t_idx]

            if 'timestamp' not in stat_df:
                stat_df['timestamp'] = [current_time]
            else:
                stat_df['timestamp'].append(current_time)

            # Get window data
            window_data = values[values['timestamp'].between(timestamps[t_idx - window_size], current_time)]

            # Compute statistical features for each node
            node_features = {}
            for node, columns in self.node_columns.items():
                node_data = window_data[columns]

                normalized_node_data = {}
                for col in columns:
                    col_data = node_data[col].values
                    if np.std(col_data) > 1e-10:  # Avoid division by near-zero standard deviation
                        normalized_node_data[col] = (col_data - np.mean(col_data)) / np.std(col_data)
                    else:
                        normalized_node_data[col] = col_data

                # Compute statistics
                features = {}
                for col in columns:
                    col_data = node_data[col].values
                    # col_data = normalized_node_data[col]
                    features = compute_statistics(features, col, col_data, self.col_mode_dict)

                for key in features:
                    if key not in stat_df:
                        stat_df[key] = [features[key]]
                    else:
                        stat_df[key].append(features[key])
                node_features[node] = features

            # Build feature matrix
            feature_matrix = []
            nodes = []
            for node, features in node_features.items():
                feature_vector = list(features.values())
                feature_matrix.append(feature_vector)
                nodes.append(node)

            # Standardize feature matrix to unify feature scales
            if len(feature_matrix) > 0:
                feature_matrix_array = np.array(feature_matrix)
                scaler = StandardScaler()
                feature_matrix_scaled = scaler.fit_transform(feature_matrix_array)

                # Fine-tune DBSCAN parameters for better tolerance of small fluctuations
                labels, reduced_features = cluster_with_dbscan(feature_matrix_scaled, eps=0.7, min_samples=3,
                                                               n_components=2)

                # Use more intelligent outlier detection logic
                improved_labels = refine_clusters(reduced_features, labels, eps)

                if -1 in improved_labels:
                    self.visualize_clustering_at_timestamp(reduced_features, labels, nodes, current_time, eps)

                # Record results
                for i, node in enumerate(nodes):
                    feature_df.loc[feature_df["timestamp"] == current_time, f"{node}_x"] = reduced_features[i][0]
                    feature_df.loc[feature_df["timestamp"] == current_time, f"{node}_y"] = reduced_features[i][1]
                    if improved_labels[i] == -1:
                        result_df.loc[result_df["timestamp"] == current_time, f"{node}_cluster"] = -1
                        feature_df.loc[feature_df["timestamp"] == current_time, f"{node}_label"] = -1
                    else:
                        result_df.loc[result_df["timestamp"] == current_time, f"{node}_cluster"] = 0
                        feature_df.loc[feature_df["timestamp"] == current_time, f"{node}_label"] = 0

        stat_df = pd.DataFrame(stat_df)
        stat_df.to_csv(os.path.join(self.result_path, 'feature_result.csv'), index=False)
        feature_df = feature_df.fillna(0)
        feature_df.to_csv(os.path.join(self.result_path, 'dbscan_instance_result.csv'), index=False)
        return result_df

    def get_ground_truth_times(self):
        def __time_plus(timestr, var_time):
            timestamp = time.mktime(time.strptime(timestr, "%Y-%m-%d %H:%M:%S"))
            timestamp += int(var_time.replace("s", ""))
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            return timestr

        self.ground_truth_df = self.ground_truth_df.reset_index(drop=True)
        ground_truth_anomaly_times = []
        for i in range(self.ground_truth_df.shape[0]):
            start_time = self.ground_truth_df.loc[i, "start_time(utc)"]
            duration = self.ground_truth_df.loc[i, "duration"]
            temp = self.test_times[
                (self.test_times >= start_time) & (self.test_times < __time_plus(start_time, duration))].tolist()
            ground_truth_anomaly_times += temp
        return ground_truth_anomaly_times

    def detect_anomalies_dtw(self, k=3):
        """
        Determine whether the DTW distance of a disk's metric is greater than the global mean + k*std across all disks for that metric.
        :param dtw_results: DataFrame containing DTW distances for each disk and metric at each window
        :param disks: list of disk names (e.g., ["disk1", "disk2", "disk3"])
        :param metrics: list of metric names to monitor (e.g., ["latency", "throughput"])
        :param k: float, number of standard deviations used as threshold for anomaly detection
        :return: DataFrame with an additional column indicating anomalies
        """
        dtw_results = pd.DataFrame({"timestamp": self.test_results["timestamp"]})
        for metric in self.metrics:
            # Extract column names for this metric across all disks
            metric_columns = [f"{disk}_{metric}_dtw" for disk in self.disks]

            # Calculate global mean and standard deviation
            global_mean = self.test_results[metric_columns].mean(axis=1)
            global_std = self.test_results[metric_columns].std(axis=1)

            # Iterate over each disk and determine anomalies
            for disk in self.disks:
                column_name = f"{disk}_{metric}_dtw"
                if column_name in self.test_results.columns:
                    # Dynamically calculate threshold
                    threshold = global_mean + k * global_std

                    # Determine if it's an anomaly
                    dtw_results[f"{column_name}_anomaly"] = self.test_results[column_name] > threshold

        self.detect_key_anomalies_dtw(dtw_results)
        anomalous_disks = self.find_anomalous_disks()

    def detect_key_anomalies_dtw(self, dtw_results):
        self.pred_labels = pd.DataFrame({"timestamp": self.test_results["timestamp"]})
        # Iterate over each disk
        for disk in self.disks:
            # Find anomaly columns for key metrics of this disk
            anomaly_columns = [f"{disk}_{metric}_dtw_anomaly" for metric in self.key_features if
                               f"{disk}_{metric}_dtw_anomaly" in dtw_results.columns]

            # Skip disk if no key metric anomalies are found
            if not anomaly_columns:
                continue

            # Logical OR on anomaly columns (any true means anomaly)
            self.pred_labels[disk] = dtw_results[anomaly_columns].any(axis=1).astype(int)  # True -> 1, False -> 0

        ground_truth_anomaly_times = self.get_ground_truth_times()
        ground_truth_index = self.host_labels[
            self.host_labels["timestamp"].isin(ground_truth_anomaly_times)].index
        self.host_labels.loc[ground_truth_index, "ground_truth"] = 1
        self.host_labels['pred'] = list(
            self.pred_labels.drop(columns=["timestamp"]).any(axis=1).astype(int))
        adjusted_labels = evaluate_detection(self.host_labels["ground_truth"].values,
                                             self.host_labels['pred'].values)
        self.host_labels["pred_adjusted"] = adjusted_labels
        self.host_labels.to_csv(os.path.join(self.result_path, 'peer_result.csv'), index=None)
        self.pred_labels.to_csv(os.path.join(self.result_path, 'peer_pred_disks.csv'), index=None)
        self.find_fault_segments()

    def find_anomalous_disks(self):
        anomalous_disks = {}

        for disk in self.disks:
            if disk in self.pred_labels.columns:
                # Find all rows where value is 1
                timestamps = self.pred_labels.loc[self.pred_labels[disk] == 1, "timestamp"].tolist()
                if timestamps:
                    anomalous_disks[disk] = timestamps

        return anomalous_disks

    def detect_anomalies_dbscan(self):
        self.pred_labels = pd.DataFrame({"timestamp": self.test_results["timestamp"]})
        for instance in self.instances:
            cluster_col = f"{instance}_cluster"
            self.pred_labels[instance] = self.test_results[cluster_col].apply(lambda x: 0 if x == 0 else 1)

        ground_truth_anomaly_times = self.get_ground_truth_times()
        ground_truth_index = self.host_labels[self.host_labels["timestamp"].isin(ground_truth_anomaly_times)].index
        self.host_labels.loc[ground_truth_index, "ground_truth"] = 1
        self.host_labels['pred'] = list(self.pred_labels.drop(columns=["timestamp"]).any(axis=1).astype(int))
        print("detect_anomalies_dbscan")
        adjusted_labels = evaluate_detection(self.host_labels["ground_truth"].values, self.host_labels['pred'].values)
        self.host_labels["pred_adjusted"] = adjusted_labels
        self.host_labels.to_csv(os.path.join(self.result_path, 'peer_result_dbscan.csv'), index=None)
        self.pred_labels.to_csv(os.path.join(self.result_path, 'peer_pred_instances_dbscan.csv'), index=None)
        self.find_fault_segments()

    def find_fault_segments(self):
        """
        Find continuous segments of 1s, record start/end times and faulty disks, and save to JSON.

        :param adjusted_labels: np.ndarray, host adjusted labels (1 for anomaly, 0 for normal)
        :param timestamps: pd.Series, timestamps corresponding to labels
        :param pred_labels: pd.DataFrame, per-disk anomaly status (1 for anomaly, 0 for normal)
        :param disks: list, disk names (e.g., ["disk1", "disk2", "disk3"])
        :param output_file: str, output JSON filename
        :return: None
        """
        # Initialize result list
        fault_segments = []

        adjusted_labels = self.host_labels["pred_adjusted"].values
        timestamps = self.host_labels["timestamp"].values

        # Loop through adjusted_labels to find continuous 1s
        start_idx = None
        for i, label in enumerate(adjusted_labels):
            if label == 1 and start_idx is None:
                # Start of a new segment
                start_idx = i
            elif label == 0 and start_idx is not None:
                # End of a segment
                start_time = timestamps[start_idx]
                end_time = timestamps[i - 1]

                # Identify faulty disks during this period
                fault_disks = []
                for instance in self.instances:
                    if self.pred_labels[instance].iloc[start_idx:i].sum() > 0:
                        fault_disks.append(instance)

                # Record segment info
                fault_segments.append({
                    "TimeStamp": start_time,
                    "TimeStamp_end": end_time,
                    "fault_disks": fault_disks
                })

                # Reset start_idx
                start_idx = None

        # Handle last segment if still active
        if start_idx is not None:
            start_time = timestamps[start_idx]
            end_time = timestamps[len(adjusted_labels) - 1]
            fault_disks = []
            for instance in self.instances:
                if self.pred_labels[instance].iloc[start_idx:].sum() > 0:
                    fault_disks.append(instance)
            fault_segments.append({
                "TimeStamp": start_time,
                "TimeStamp_end": end_time,
                "fault_disks": fault_disks
            })

        # Save to JSON
        with open(os.path.join(self.result_path, 'peer_result.json'), "w") as f:
            json.dump(fault_segments, f, indent=4)

    def plot_dbscan(self, cluster_labels, feature_pca, fig_path):
        # Visualize clustering result
        plt.figure(figsize=(8, 6))

        unique_labels = sorted(set(cluster_labels))
        n_clusters = len(unique_labels)
        colors = plt.cm.jet(np.linspace(0, 1, n_clusters))

        for label, color in zip(unique_labels, colors):
            mask = (cluster_labels == label)
            plt.scatter(feature_pca[mask, 0], feature_pca[mask, 1], c=[color], label=f'Cluster {label}')

        plt.title("DBSCAN Clustering Result")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.result_path, fig_path))
        plt.close()

    def plot_dbscan_all(self, train_cluster_labels, train_feature_pca, test_cluster_labels, test_feature_pca, fig_path):
        # Get all unique cluster labels
        all_labels = set(train_cluster_labels) | set(test_cluster_labels)
        all_labels.discard(-1)  # Remove noise label

        # Assign a color to each cluster
        colors = plt.cm.get_cmap('tab20', len(all_labels))

        plt.figure(figsize=(10, 8))

        # Plot training points
        for label in all_labels:
            train_points = train_feature_pca[train_cluster_labels == label]
            plt.scatter(train_points[:, 0], train_points[:, 1], s=50, c=[colors(label)], label=f'Train Cluster {label}')

        # Plot test points
        for label in all_labels:
            test_points = test_feature_pca[test_cluster_labels == label]
            plt.scatter(test_points[:, 0], test_points[:, 1], s=50, c=[colors(label)], marker='x',
                        label=f'Test Cluster {label}')

        # Plot noise points
        noise_train_points = train_feature_pca[train_cluster_labels == -1]
        noise_test_points = test_feature_pca[test_cluster_labels == -1]
        plt.scatter(noise_train_points[:, 0], noise_train_points[:, 1], s=50, c='black', marker='o',
                    label='Train Noise')
        plt.scatter(noise_test_points[:, 0], noise_test_points[:, 1], s=50, c='black', marker='x', label='Test Noise')

        plt.title('DBSCAN Clustering - Train vs Test')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.savefig(os.path.join(self.result_path, fig_path))
        plt.show()

    def visualize_clustering_at_timestamp(self, reduced_features, labels, nodes, timestamp, eps):
        """
        Visualize clustering results at a specific timestamp

        Parameters:
        reduced_features: Feature matrix after PCA dimensionality reduction
        labels: DBSCAN cluster labels
        nodes: List of node names
        timestamp: Timestamp
        eps: DBSCAN eps parameter
        """
        plt.figure(figsize=(12, 10))

        n_samples = reduced_features.shape[0]

        if reduced_features.shape[1] == 1:
            features_2d = np.column_stack((reduced_features, np.zeros_like(reduced_features)))
            x_label, y_label = 'PC1', 'Arbitrary Dimension'
        elif reduced_features.shape[1] == 2:
            features_2d = reduced_features
            x_label, y_label = 'PC1', 'PC2'
        else:
            if n_samples > 5:
                perplexity = min(30, n_samples - 1)
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(reduced_features)
                    x_label, y_label = 'TSNE-1', 'TSNE-2'
                except Exception as e:
                    print(f"t-SNE reduction failed: {e}")
                    if reduced_features.shape[1] >= 2:
                        features_2d = reduced_features[:, :2]
                        x_label, y_label = 'PC1', 'PC2'
                    else:
                        features_2d = np.column_stack((reduced_features, np.zeros_like(reduced_features)))
                        x_label, y_label = 'PC1', 'Arbitrary Dimension'
            else:
                if reduced_features.shape[1] >= 2:
                    features_2d = reduced_features[:, :2]
                    x_label, y_label = 'PC1', 'PC2'
                else:
                    features_2d = np.column_stack((reduced_features, np.zeros_like(reduced_features)))
                    x_label, y_label = 'PC1', 'Arbitrary Dimension'

        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # Anomalies
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c='black', s=100, marker='X', label='Anomalies')
                for j, node in enumerate(nodes):
                    if labels[j] == -1:
                        plt.annotate(node, (features_2d[j, 0], features_2d[j, 1]),
                                     fontsize=9, ha='right')
            else:  # Normal clusters
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=[colors[i]], s=50, marker='o',
                            label=f'Cluster {label}')

        if reduced_features.shape[1] <= 2:
            for i in range(len(features_2d)):
                if labels[i] != -1:
                    circle = plt.Circle((features_2d[i, 0], features_2d[i, 1]),
                                        eps, color='gray', fill=False, alpha=0.2)
                    plt.gca().add_patch(circle)

        plt.title(f'Clustering Results at {timestamp}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        anomaly_nodes = [nodes[i] for i, label in enumerate(labels) if label == -1]
        anomaly_text = "Anomalous Nodes: " + ", ".join(anomaly_nodes) if anomaly_nodes else "No anomalies detected"
        plt.figtext(0.5, 0.01, anomaly_text, ha='center', fontsize=12, bbox={"facecolor": "lightpink", "alpha": 0.5})

        plt.tight_layout()
        plt.show()

        return anomaly_nodes
    def workload_aware_sample_components(self, sample_ratio=0.3, n_groups=None):
        """
        Group components by their workload characteristics (statistics of historical
        DTW distances), randomly sample a percentage of components from each group,
        and return the whitelist of sampled components.

        Returns:
            set: Sampled instance IDs, e.g. {"node_0_nic", "node_1_nic"}
        """
        from sklearn.cluster import KMeans

        component_features = {}
        for instance in self.instances:
            dtw_cols = [col for col in self.test_results.columns if instance in col and col != "timestamp"]
            if dtw_cols:
                component_features[instance] = self.test_results[dtw_cols].mean().values

        if not component_features:
            return set(self.instances)

        component_ids = list(component_features.keys())
        features = np.array([component_features[cid] for cid in component_ids])

        if n_groups is None:
            n_groups = max(2, min(5, len(component_ids)))

        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(features)

        groups = {}
        for i, cid in enumerate(component_ids):
            gid = int(group_labels[i])
            groups.setdefault(gid, []).append(cid)

        sampled = set()
        for gid, members in groups.items():
            n_sample = max(1, int(len(members) * sample_ratio))
            indices = np.random.choice(len(members), size=n_sample, replace=False)
            for idx in indices:
                sampled.add(members[idx])

        return sampled