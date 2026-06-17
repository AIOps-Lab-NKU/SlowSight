from matplotlib import pyplot as plt, patches

from SlowSight.utils.utils import paintIndicators, paintReconstruct, get_segment_dict, get_segment_list, show_anomalies
from SlowSight.utils.spot import SPOT, biSPOT
from SlowSight.utils.evaluation import evaluate_detection
import pandas as pd
import math
import numpy as np
from sklearn import metrics
from collections import defaultdict
import time
import json

from SlowSight.correlation_analysis.check_failures import get_pattern, trend


class postProcessor:
    def __init__(self, original_data, processed_data, reconstruct_data, generate_data, data_info, all_features, all_times, scaler):
        '''
        All input data is the full dataset; no distinction between training and test sets.
        :param processed_data:
        :param reconstruct_data:
        '''
        self.original_data = original_data
        self.model_processed_data = processed_data
        self.model_reconstruct_data = reconstruct_data
        self.model_generate_data = generate_data
        self.data_info = data_info
        self.all_features = all_features
        self.all_times = all_times
        self.scaler = scaler

        self.cause_info = []
        self.cause_index = {}
        self.cause_TimeSeg_map = defaultdict(list)
        self.scores = None
        self.reconstruct_error = None
        self.spot_detect_res = None
        self.upper_bounding = None

        self.metric_patterns = get_pattern()

        if isinstance(scaler, dict):
            self.true_processed_data = self.model_processed_data * scaler["std"] + scaler["mean"]
            self.true_reconstruct_data = self.model_reconstruct_data * scaler["std"] + scaler["mean"]
        else:  # scaler is a sklearn.preprocessing._data.xxScaler
            self.true_processed_data = self.scaler.inverse_transform(self.model_processed_data)
            self.true_reconstruct_data = self.scaler.inverse_transform(self.model_reconstruct_data)

    def compute_score(self, score_type, alpha):
        if score_type == 'common':
            # self.reconstruct_error = np.square(self.model_processed_data - self.model_reconstruct_data)
            self.reconstruct_error = alpha * np.square(self.model_processed_data - self.model_generate_data) + \
                                     (1 - alpha) * np.square(self.model_processed_data - self.model_reconstruct_data)
        else:
            raise NotImplementedError()
        scores = np.sum(self.reconstruct_error, axis=1)  # sum of reconstruction errors across all features
        # differential of anomaly scores
        scores_diff = np.diff(scores)
        self.scores = np.insert(scores_diff, 0, scores_diff[0])
        return self.reconstruct_error

    def spot_fit(self, q=1e-3, level=0.98, threshold=None):
        train_data = np.copy(self.scores[:self.data_info['train_length']])

        s = SPOT(q)  # SPOT object
        s.fit(train_data,  # training data
              self.scores[self.data_info['train_length'] + self.data_info['valid_length']:])  # test data
        s.initialize(level=level, verbose=True)  # initialization step
        self.spot_detect_res = s.run()  # run
        self.spot_detect_res["upper_thresholds"] = self.spot_detect_res["thresholds"]

        return self.spot_detect_res

    def spot_fit_single(self, key_metric, q=1e-3, level=0.98, threshold=None):
        rec_error = self.reconstruct_error.copy()
        index = self.all_features.index(key_metric)
        score = rec_error[:, index]
        score_diff = np.diff(score)
        metric_score = np.insert(score_diff, 0, score_diff[0])

        train_data = np.copy(metric_score[:self.data_info['train_length']])

        s = SPOT(q)  # SPOT object
        s.fit(train_data,  # training data
              metric_score[self.data_info['train_length'] + self.data_info['valid_length']:])  # test data
        s.initialize(level=level, verbose=True)  # initialization step
        spot_detect_res = s.run()  # run
        spot_detect_res["upper_thresholds"] = spot_detect_res["thresholds"]
        anomalies_l = spot_detect_res['alarms']
        anomalies = np.array(anomalies_l) + self.data_info['train_length'] + self.data_info['valid_length']

        return anomalies

    def check_trend(self, segment_l):
        metric_flag = {}
        for metric in self.metric_patterns.keys():
            for feature in self.all_features:
                if metric in feature:
                    index = self.all_features.index(feature)
                    data = self.model_processed_data[segment_l[0] - 12: segment_l[0] + 10, index]
                    metric_t = trend(data)
                    if metric_t != self.metric_patterns[metric]:
                        metric_flag[metric] = False
                    else:
                        metric_flag[metric] = True
        if all(metric_flag.values()):
            return True
        else:
            return False

    def process_alarm(self, segment_list_l):
        check_segment_list = []
        for segment_l in segment_list_l:
            if len(segment_l) != 0:
                flag = self.check_trend(segment_l)
                if flag:
                    check_segment_list.append(segment_l)
        return check_segment_list

    def locate_cause(self, threshold=None, rate_path=None, result_path=None):
        anomalies_l = self.spot_detect_res['alarms']
        anomalies = np.array(anomalies_l) + self.data_info['train_length'] + self.data_info['valid_length']
        segment_list_l = get_segment_list(anomalies)
        check_segment_list_l = self.process_alarm(segment_list_l)
        anomalies_spot = np.array(self.spot_detect_res['alarms']) + self.data_info['train_length'] + self.data_info[
            'valid_length']
        anomalies_cause_dict = {}  # key represents this anomaly segment, value represents top root causes
        anomalies_cause_full_dict = {}  # key represents this anomaly segment, value includes cause metrics and their scores

        np_error_rate_all = []

        for segment_l in check_segment_list_l:
            if len(segment_l) != 0:
                segment_l_need = [i for i in segment_l if i in anomalies_spot]
                # segment_cause_index = self.get_cause_frequency(segment_l_need)
                segment_cause_index, np_error_rate_segment, segment_cause_index_full, segment_degree = self.get_cause_weight(
                    segment_l_need, threshold)  # top 2 root cause metrics for this segment
                np_error_rate_all.append(np_error_rate_segment)
                # segment_cause_index = self.get_cause_threshold(segment_l_need)
            else:
                segment_cause_index = []

            anomalies_cause_dict[" ".join([str(i) for i in segment_l])] = segment_cause_index
            if len(segment_l) != 0:
                anomalies_cause_full_dict[" ".join([str(i) for i in segment_l])] = {
                    "cause_index": segment_cause_index_full,
                    "error_rate": np_error_rate_segment,
                    "overall_error": segment_degree}

        np_error_rate_all = np.array(np_error_rate_all).T
        if len(np_error_rate_all) != 0:
            df_error_rate_all = pd.DataFrame(np_error_rate_all, columns=[str(s_l[0]) for s_l in check_segment_list_l])
            if rate_path:
                df_error_rate_all.to_csv(rate_path)

        # cause_TimeSeg_map records which segments each metric is involved in
        # key: metric index, value: list of segments (as strings separated by spaces)
        for k, v_l in anomalies_cause_dict.items():
            for v in v_l:
                self.cause_TimeSeg_map[v].append(k)

        # Prepare output JSON format with cause metrics and scores
        cause_list = []
        for k, v in anomalies_cause_full_dict.items():
            k_list = k.split(" ")
            anomaly_start = self.all_times[int(k_list[0])]
            anomaly_end = self.all_times[int(k_list[-1])]
            features_name = [self.all_features[item] for item in v["cause_index"]]
            error_rate = [v["error_rate"][item] for item in v["cause_index"]]
            cur_cause_list = dict(zip(features_name, error_rate))

            def construct_metric_info(metric_name, metric_score, metric_labels=None):
                """
                Build output information for an anomalous metric
                """
                if not metric_labels:
                    metric_labels = {
                        "instance": "xxx",
                        "job": "xxx",
                        "machine_id": "xxx",
                        "origin": "xxx"
                    }
                metric_info = {
                    "metric": metric_name,
                    "labels": metric_labels,
                    "score": metric_score,
                    "description": f"{metric_name} anomaly"
                }
                return metric_info

            cur_result_json = {}
            cur_result_json["TimeStamp"] = anomaly_start
            cur_result_json["TimeStamp_end"] = anomaly_end
            cur_result_json["Attributes"] = {
                "entity_id": "<machine_id>_<entity_name>_<key1>_<key2>_..",
                "event_id": "<timestamp>_<entity_id>",
                "event_type": "app or os",
                "event_source": "historical_detector",
                "anomaly_score": v["overall_error"]
            }

            top1_metric = construct_metric_info(features_name[0], error_rate[0])  # highest scoring metric
            cur_result_json["Resource"] = top1_metric
            cur_cause_list = [construct_metric_info(metric_name, metric_score) for metric_name, metric_score in cur_cause_list.items()]
            cur_result_json["Resource"]["cause_metrics"] = cur_cause_list

            cause_list.append(cur_result_json)

        # Save results to JSON file
        if result_path:
            with open(result_path, "w") as f:
                json.dump(cause_list, f, indent=4, ensure_ascii=False)

        return anomalies_cause_full_dict


    # Weight calculation for root cause localization
    def get_cause_weight(self, segment_l, threshold=None):
        train_reconstruct_avg = np.mean(self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std = np.std(self.reconstruct_error[:self.data_info['train_length']], axis=0)
        train_reconstruct_std[np.where(train_reconstruct_std == 0)[0]] = 1e-8

        segment_require = segment_l[:10]
        # kpi_normalised = abs(self.reconstruct_error[segment_require, :] - train_reconstruct_avg)
        kpi_normalised = abs(self.reconstruct_error[segment_require, :] - train_reconstruct_avg) / train_reconstruct_std
        param_coe = 3
        array_coe = np.array([math.log(1 / (i / 10 + pow(math.e, -param_coe))) / param_coe for i in range(10)])  # monotonically decreasing function
        array_kpi_degree = np.multiply(kpi_normalised.T, array_coe[:kpi_normalised.shape[0]])  # (features, 10) * (1, 10)
        array_kpi_degree = array_kpi_degree.sum(1)  # degree per feature
        array_kpi_degree_rate = array_kpi_degree / (array_kpi_degree.sum(0))  # proportion per feature
        overall_degree = array_kpi_degree.mean()  # overall anomaly severity

        if threshold:
            alarm_index = np.where(array_kpi_degree_rate > threshold)[0]
        else:
            alarm_index_full = np.argsort(array_kpi_degree)[::-1]  # descending order
            alarm_index = alarm_index_full[:2]  # top 2 features

        return alarm_index, array_kpi_degree_rate, alarm_index_full, overall_degree

    def paint_indicators(self, figPath=None):
        # Prepare data
        valid_all_model_processed_data = self.model_processed_data[
                                         self.data_info['train_length']:self.data_info['train_length'] + self.data_info[
                                             'valid_length'], :]
        valid_all_model_G_D = self.model_reconstruct_data[
                              self.data_info['train_length']:self.data_info['train_length'] + self.data_info[
                                  'valid_length'], :]
        test_all_model_processed_data = self.model_processed_data[
                                        self.data_info['train_length'] + self.data_info['valid_length']:, :]
        test_all_model_G_D = self.model_reconstruct_data[
                             self.data_info['train_length'] + self.data_info['valid_length']:, :]

        data_para = {'valid_mae': metrics.mean_absolute_error(valid_all_model_processed_data, valid_all_model_G_D),
                     'valid_mse': metrics.mean_squared_error(valid_all_model_processed_data, valid_all_model_G_D),
                     'test_mae': metrics.mean_absolute_error(test_all_model_processed_data, test_all_model_G_D),
                     'test_mse': metrics.mean_squared_error(test_all_model_processed_data, test_all_model_G_D),
                     'features_num': len(self.all_features), 'train_length': self.data_info['train_length'],
                     'valid_length': self.data_info['valid_length'],
                     "upper_thresholds": self.spot_detect_res['upper_thresholds'],
                     'spot_anomalies_segment': get_segment_list(
                         np.array(self.spot_detect_res['alarms']) + (
                                 self.data_info['train_length'] + self.data_info['valid_length']))
                     }
        self.paintReconstruct(data_para, figPath)
        self.paintOriginal(data_para, figPath)

    def paintReconstruct(self, data, fig_path=None):
        pair_num = 1
        scores_paint_flag = 1

        normal_indicate_row = scores_paint_flag
        total_row = self.model_processed_data.T.shape[0] * pair_num + normal_indicate_row
        X_all = np.arange(0, self.model_processed_data.T.shape[1])
        X_train = np.arange(0, data['train_length'])
        X_test = np.arange(data['train_length'] + data['valid_length'], self.model_processed_data.T.shape[1])
        fig, ax = plt.subplots(total_row, 1, figsize=(16, 3 * total_row))
        a = [j for j in range(0, self.all_times.shape[0], self.all_times.shape[0] // 7)]
        x_label = [self.all_times[j] for j in a]
        anomalies = np.array(self.spot_detect_res['alarms']) + + (
                self.data_info['train_length'] + self.data_info['valid_length'])

        # Paint anomaly detection
        if scores_paint_flag == 1:
            ax[0].plot(X_all, self.scores, label="anomaly_score")
            ax[0].plot(X_test, data['upper_thresholds'], '--', label="upper_threshold")

            ax[0].axvline(data['train_length'], c='#ff0097', ls='-.', lw=2)
            ax[0].axvline(data['train_length'] + data['valid_length'], c='#ff0097', ls='-.', lw=2)
            if len(data['spot_anomalies_segment']) != 0:
                for segments in data['spot_anomalies_segment']:
                    ax[0].add_patch(patches.Rectangle((segments[0], ax[0].get_ylim()[0]),
                                                      segments[-1] - segments[0],
                                                      ax[0].get_ylim()[1] -
                                                      ax[0].get_ylim()[0],
                                                      transform=ax[0].transData, alpha=0.3,
                                                      color="#ff0097"))
                spot_print = list(
                    map(lambda e_l: str(e_l[0]) if len(e_l) == 1 or 0 else f"{str(e_l[0])}~{str(e_l[-1])}",
                        data['spot_anomalies_segment']))
                spot_print = show_anomalies(spot_print)

                ax[0].text(0.05, 0.2,
                           f"number of spot anomalies:{len(data['spot_anomalies_segment'])}\nThey are:{spot_print}\n",
                           transform=ax[0].transAxes)
            else:
                ax[0].text(0.35, 0.7,
                           f"number of spot anomalies:{len(data['spot_anomalies_segment'])}\n",
                           transform=ax[0].transAxes)

            ax[0].text(0.35, 0.4,
                       f"number of indicators:{data['features_num']}\ntest_mse: {data['test_mse']}  test_mae: {data['test_mae']}",
                       transform=ax[0].transAxes)
            ax[0].set_title('anomaly detection')
            ax[0].set_xticks(a)
            ax[0].set_xticklabels(x_label)
            ax[0].legend(loc='upper left', ncol=3)

        for i in range(self.model_processed_data.T.shape[0]):
            ax[i * pair_num + normal_indicate_row].plot(X_all, self.model_processed_data.T[i], label="original data")
            ax[i * pair_num + normal_indicate_row].plot(X_all, self.model_reconstruct_data.T[i], color='#ff8936',
                                                        label="reconstruct data")

            ax[i * pair_num + normal_indicate_row].axvline(data['train_length'], c='#ff0097', ls='-.', lw=2)
            ax[i * pair_num + normal_indicate_row].axvline(data['train_length'] + data['valid_length'], c='#ff0097',
                                                           ls='-.', lw=2)

            ax[i * pair_num + normal_indicate_row].set_title(str(i) + '-' + str(self.all_features[i]))
            ax[i * pair_num + normal_indicate_row].set_xticks(a)
            ax[i * pair_num + normal_indicate_row].set_xticklabels(x_label)
            ax[i * pair_num + normal_indicate_row].legend(loc='upper left', ncol=3)

        # Show cause_indicator_index
        if len(self.cause_TimeSeg_map) != 0:
            for k, str_l in self.cause_TimeSeg_map.items():  # k is feature index, v is anomaly segments
                fill_X = [0, 0, data['train_length'], data['train_length']]
                fill_Y = [ax[k * pair_num + normal_indicate_row].get_ylim()[0],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[1],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[1],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[0]]

                segs_l = [seg.split(" ") for seg in str_l]
                for segments in segs_l:
                    segments = [int(i) for i in segments]
                    ax[k * pair_num + normal_indicate_row].add_patch(
                        patches.Rectangle((segments[0], ax[k * pair_num + normal_indicate_row].get_ylim()[0]),
                                          segments[-1] - segments[0],
                                          ax[k * pair_num + normal_indicate_row].get_ylim()[1] -
                                          ax[k * pair_num + normal_indicate_row].get_ylim()[0],
                                          transform=ax[k * pair_num + normal_indicate_row].transData, alpha=0.3,
                                          color="red"))

                segs_print = list(
                    map(lambda e_l: str(e_l[0]) if len(e_l) == 1 or 0 else f"{str(e_l[0])}~{str(e_l[-1])}", segs_l))
                segs_print = show_anomalies(segs_print)
                ax[k * pair_num + scores_paint_flag].text(0.35, 0.8,
                                                          f"number of anomalies of segments:{len(segs_l)}\nThey are {segs_print}",
                                                          transform=ax[k * pair_num + normal_indicate_row].transAxes)

                ax[k * pair_num + normal_indicate_row].set_title(str(k) + '-' + str(self.all_features[k]), color='red')
                ax[k * pair_num + normal_indicate_row].fill(fill_X, fill_Y, color='#ffb3a7')

        fig.tight_layout()
        if fig_path is not None:
            print(f"figure saved in {fig_path}")
            plt.savefig(fig_path)
        else:
            plt.show()

    def paintOriginal(self, data, fig_path=None):
        fig_path = fig_path[:-4] + "_original.pdf"

        pair_num = 1
        scores_paint_flag = 1

        normal_indicate_row = scores_paint_flag
        total_row = self.original_data.T.shape[0] * pair_num + normal_indicate_row
        X_all = np.arange(0, self.original_data.T.shape[1])
        X_train = np.arange(0, data['train_length'])
        X_test = np.arange(data['train_length'] + data['valid_length'], self.original_data.T.shape[1])
        fig, ax = plt.subplots(total_row, 1, figsize=(16, 3 * total_row))
        a = [j for j in range(0, self.all_times.shape[0], self.all_times.shape[0] // 7)]
        x_label = [self.all_times[j] for j in a]
        anomalies = np.array(self.spot_detect_res['alarms']) + + (
                self.data_info['train_length'] + self.data_info['valid_length'])
        # paint anomaly detection
        if scores_paint_flag == 1:
            ax[0].plot(X_all, self.scores, label="anomaly_score")
            ax[0].plot(X_test, data['upper_thresholds'], '--', label="upper_threshold")

            ax[0].axvline(data['train_length'], c='#ff0097', ls='-.', lw=2)
            ax[0].axvline(data['train_length'] + data['valid_length'], c='#ff0097', ls='-.', lw=2)
            if len(data['spot_anomalies_segment']) != 0:
                for segments in data['spot_anomalies_segment']:
                    ax[0].add_patch(patches.Rectangle((segments[0], ax[0].get_ylim()[0]),
                                                      segments[-1] - segments[0],
                                                      ax[0].get_ylim()[1] -
                                                      ax[0].get_ylim()[0],
                                                      transform=ax[0].transData, alpha=0.3,
                                                      color="#ff0097"))
                spot_print = list(
                    map(lambda e_l: str(e_l[0]) if len(e_l) == 1 or 0 else f"{str(e_l[0])}~{str(e_l[-1])}",
                        data['spot_anomalies_segment']))
                spot_print = show_anomalies(spot_print)
                ax[0].text(0.05, 0.2,
                           f"number of spot anomalies:{len(data['spot_anomalies_segment'])}\nThey are:{spot_print}\n",
                           transform=ax[0].transAxes)
            else:
                ax[0].text(0.35, 0.7,
                           f"number of spot anomalies:{len(data['spot_anomalies_segment'])}\n",
                           transform=ax[0].transAxes)

            ax[0].text(0.35, 0.4,
                       f"number of indicators:{data['features_num']}\ntest_mse: {data['test_mse']}  test_mae: {data['test_mae']}",
                       transform=ax[0].transAxes)
            ax[0].set_title('anomaly detection')
            ax[0].set_xticks(a)
            ax[0].set_xticklabels(x_label)
            ax[0].legend(loc='upper left', ncol=3)

        for i in range(self.original_data.T.shape[0]):
            ax[i * pair_num + normal_indicate_row].plot(X_all, self.original_data.T[i], label="original data")
            # ax[i * pair_num + normal_indicate_row].plot(X_all, self.true_reconstruct_data.T[i], color='#ff8936',
            #                                             label="reconstruct data")

            ax[i * pair_num + normal_indicate_row].axvline(data['train_length'], c='#ff0097', ls='-.', lw=2)
            ax[i * pair_num + normal_indicate_row].axvline(data['train_length'] + data['valid_length'], c='#ff0097',
                                                           ls='-.', lw=2)

            ax[i * pair_num + normal_indicate_row].set_title(str(i) + '-' + str(self.all_features[i]))
            ax[i * pair_num + normal_indicate_row].set_xticks(a)
            ax[i * pair_num + normal_indicate_row].set_xticklabels(x_label)
            ax[i * pair_num + normal_indicate_row].legend(loc='upper left', ncol=3)

        # Show cause_indicator_index
        if len(self.cause_TimeSeg_map) != 0:
            for k, str_l in self.cause_TimeSeg_map.items():
                fill_X = [0, 0, data['train_length'], data['train_length']]
                fill_Y = [ax[k * pair_num + normal_indicate_row].get_ylim()[0],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[1],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[1],
                          ax[k * pair_num + normal_indicate_row].get_ylim()[0]]

                segs_l = [seg.split(" ") for seg in str_l]
                for segments in segs_l:
                    segments = [int(i) for i in segments]
                    ax[k * pair_num + normal_indicate_row].add_patch(
                        patches.Rectangle((segments[0], ax[k * pair_num + normal_indicate_row].get_ylim()[0]),
                                          segments[-1] - segments[0],
                                          ax[k * pair_num + normal_indicate_row].get_ylim()[1] -
                                          ax[k * pair_num + normal_indicate_row].get_ylim()[0],
                                          transform=ax[k * pair_num + normal_indicate_row].transData, alpha=0.3,
                                          color="red"))

                segs_print = list(
                    map(lambda e_l: str(e_l[0]) if len(e_l) == 1 or 0 else f"{str(e_l[0])}~{str(e_l[-1])}", segs_l))
                segs_print = show_anomalies(segs_print)
                ax[k * pair_num + scores_paint_flag].text(0.35, 0.8,
                                                          f"number of anomalies of segments:{len(segs_l)}\nThey are {segs_print}",
                                                          transform=ax[k * pair_num + normal_indicate_row].transAxes)

                ax[k * pair_num + normal_indicate_row].set_title(str(k) + '-' + str(self.all_features[k]), color='red')
                ax[k * pair_num + normal_indicate_row].fill(fill_X, fill_Y, color='#ffb3a7')

        fig.tight_layout()
        if fig_path is not None:
            print(f"figure saved in {fig_path}")
            plt.savefig(fig_path)
        else:
            plt.show()

    # Read failure start time and duration, compute corresponding timestamps
    def get_ground_truth_times(self, ground_truth_df, test_times):
        def __time_plus(timestr, var_time):
            timestamp = time.mktime(time.strptime(timestr, "%Y-%m-%d %H:%M:%S"))
            timestamp += int(var_time.replace("s", ""))  # only supports seconds
            timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            return timestr

        ground_truth_df = ground_truth_df.reset_index(drop=True)
        ground_truth_anomaly_times = []
        for i in range(ground_truth_df.shape[0]):
            start_time = ground_truth_df.loc[i, "start_time(utc)"]
            duration = ground_truth_df.loc[i, "duration"]
            temp = test_times[(test_times >= start_time) & (test_times < __time_plus(start_time, duration))].tolist()
            ground_truth_anomaly_times += temp
        return ground_truth_anomaly_times

    # Evaluate anomaly detection results
    def get_anomalies(self, ground_truth_df, anomalies_cause_full_dict, result_path, disks_result_csv_path):
        anomalies = np.array(self.spot_detect_res['alarms']) + (
                    self.data_info['train_length'] + self.data_info['valid_length'])
        pred_anomaly_times = []
        for i in anomalies:
            pred_anomaly_times.append(self.all_times[i])

        if ground_truth_df.empty:
            for i in anomalies:
                print(self.all_times[i])
            return

        test_times = self.all_times[(self.data_info['train_length'] + self.data_info['valid_length']):]
        result_df = pd.DataFrame({"timestamp": test_times, "ground_truth": np.zeros(test_times.size),
                                  "pred": np.zeros(test_times.size)})
        ground_truth_anomaly_times = self.get_ground_truth_times(ground_truth_df, test_times)
        ground_truth_index = result_df[result_df["timestamp"].isin(ground_truth_anomaly_times)].index
        result_df.loc[ground_truth_index, "ground_truth"] = 1
        pred_index = result_df[result_df["timestamp"].isin(pred_anomaly_times)].index
        result_df.loc[pred_index, "pred"] = 1
        print("get_anomalies")
        pred_adjusted = evaluate_detection(result_df["ground_truth"].values, result_df["pred"].values)
        result_df["pred_adjusted"] = pred_adjusted
        result_df.to_csv(result_path, index=None)

        self.get_disks_anomalies(anomalies_cause_full_dict, disks_result_csv_path)

    def get_disks_anomalies(self, anomalies_cause_full_dict, disks_result_path):
        disks_result_df = pd.DataFrame({"timestamp": self.all_times})
        for feature in self.all_features:
            disks_result_df[feature] = np.zeros(self.all_times.size)

        times = list(self.all_times)
        for k, v in anomalies_cause_full_dict.items():
            k_list = k.split(" ")
            start_idx = k_list[0]
            end_idx = k_list[-1]

            features_name = [self.all_features[item] for item in v["cause_index"]]
            error_rate = [v["error_rate"][item] for item in v["cause_index"]]
            cur_cause_list = dict(zip(features_name, error_rate))
            sum_scores = 0.0
            num_metrics = 0
            for metric_name, metric_score in cur_cause_list.items():
                sum_scores += metric_score
                num_metrics += 1
            mean_score = sum_scores / num_metrics
            for metric_name, metric_score in cur_cause_list.items():
                if metric_score > mean_score:
                    disks_result_df.loc[start_idx:end_idx, metric_name] = 1

        disks_result_df.to_csv(disks_result_path, index=None)

    def get_scores(self):
        return self.scores, self.reconstruct_error

    def get_data(self):
        return self.true_processed_data, self.true_reconstruct_data