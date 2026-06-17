import numpy as np


def sigmoid(z, alpha, offset=0.5):
    y = 1 / (1 + np.exp(-alpha * (z - offset)))
    return y


def ReLU(x, alpha, offset, liner_criteria=None):
    if liner_criteria is None:
        liner_criteria = 0.5
    if x > liner_criteria:
        return sigmoid(x, alpha=alpha, offset=offset)
    elif x <= 0:
        return 0
    else:
        return x


#         health_score = [health.run(x, err, thre) for x, err, thre in zip(post.true, post.score, post.spot_res['upper_thresholds'])]
#         health_score = np.array(health_score)

class Health_single:
    def __init__(self, train_std, criterion=None):
        # self.threshold = threshold  # Model-determined anomaly threshold
        self.train_std = train_std  # Standard deviation of data
        self.criterion = criterion  # Expert-defined red line
        self.anomaly_window = {}  # Used to store sustained anomaly scores when continuous anomalies occur
        self.sustained_healthy_number = 0  # Records the number of continuously healthy points after a sustained anomaly window has been set
        self.window_length_limit_low = 3
        self.window_length_limit_up = 10
        self.recover_limit = 3

        self.dev_weight = 0.5
        self.dur_weight = 0.5

        self.update_flag = False
        self.alarms = []
        self.health_info = {}

    def get_weight(self, x, error, threshold):
        ratio = 1.0
        if self.criterion is not None:
            criterion = max(self.criterion, threshold)
            ratio = x / criterion
        ratio = min(ratio, 1.0)
        ratio = ratio * 100
        if ratio % 5 == 0:
            self.dev_weight = (ratio // 5) * 5
        else:
            self.dev_weight = ((ratio // 5) + 1) * 5
        self.dev_weight = self.dev_weight / 100
        self.dur_weight = 1.0 - self.dev_weight
        return ratio / 100

    # Deviation score
    def get_deviation_score(self, x, error, threshold):
        if error >= threshold:
            ratio = 1.0
            if self.criterion is not None:
                criterion = max(self.criterion, threshold)
                ratio = x / criterion
            ratio = min(ratio, 1.0)
            score = ReLU(ratio, 14, 0.5)  # Transformation applied to deviation score
        else:
            score = 0
        return score

    # Internal class variables used: self.anomaly_window; self.window_length_limit_low; self.sustained_healthy_number; self.update_flag = True
    def get_sustained_score(self, index, x, error, threshold):
        if error > threshold:
            if x < 1e3:
                self.anomaly_window[index] = [0, self.dur_weight]
            else:
                self.anomaly_window[index] = [1, self.dur_weight]
            self.anomaly_window[index] = [1, self.dur_weight]
            self.broaden_window()
            if len(self.anomaly_window) < self.window_length_limit_low:  # If the window is still small
                score = 0
            else:  # When this point is the `self.window_length_limit`th point
                score = sum([v[0] for k, v in self.anomaly_window.items()]) / len(self.anomaly_window)
            self.sustained_healthy_number = 0
        else:  # Encounter a normal point
            self.sustained_healthy_number = self.sustained_healthy_number + 1
            self.anomaly_window[index] = [0, self.dur_weight]
            self.broaden_window()
            if self.sustained_healthy_number < self.recover_limit:
                if len(self.anomaly_window) < self.window_length_limit_low:  # If the window is still small
                    score = 0
                else:  # When this point is the third consecutive zero
                    score = sum([v[0] for k, v in self.anomaly_window.items()]) / len(self.anomaly_window)
            else:  # This point is the third zero; need to recover
                self.window_length_limit_low = 3
                self.anomaly_window = {}
                self.update_flag = False  # Window cleared, need to update when subsequent anomalies occur
                score = 0

        if score < 0.6:
            score = 0
        elif self.dur_weight <= 0.6:
            score = sigmoid(score, 10, 0.6)
        else:
            score = ReLU(score, 14, self.dur_weight * 0.8, liner_criteria=self.dur_weight * 0.8)

        return score

    def broaden_window(self):
        # Dynamically adjust window size
        if len(self.anomaly_window) == self.window_length_limit_low:
            if np.mean([v[1] for k, v in self.anomaly_window.items()]) > 0.6:  # The average weight of continuity is high, expand the window
                self.window_length_limit_low = self.window_length_limit_up

    def fit(self, index, x, error, threshold):  # x: raw data; error: anomaly score; threshold: SPOT threshold
        ratio = self.get_weight(x, error, threshold)
        deviation_score = self.get_deviation_score(x, error, threshold)
        sustained_score = self.get_sustained_score(index, x, error, threshold)
        health_score = 1 - (deviation_score * self.dev_weight + sustained_score * self.dur_weight)
        self.health_info[index] = [health_score, deviation_score, sustained_score, self.dur_weight, ratio]
        self.update(health_score, sustained_score)

    def predict(self, threshold):
        for index, score_l in self.health_info.items():
            if score_l[0] < threshold:
                self.alarms.append(index)
        return self.alarms

    def update(self, health_score, new_sustained_score):
        if health_score < 0.2:  # Indicates an anomaly should be reported
            if not self.update_flag:  # Previous points have already been modified
                for Id in self.anomaly_window.keys():
                    origin_health_score_info = self.health_info[Id]
                    # origin_health_score, origin_dev_score, origin_dur_score = self.health_info[Id]
                    new_health_score = 1 - (origin_health_score_info[1] * self.dev_weight + new_sustained_score * self.dur_weight)
                    origin_health_score_info[0] = new_health_score
                    origin_health_score_info[2] = new_sustained_score
                    self.health_info[Id] = origin_health_score_info
                self.update_flag = True
        else:  # Do nothing
            pass

    @property
    def health_score_info(self):
        res = []
        for v in self.health_info.values():
            res.append(v)
        return res


class Health_multi:
    def __init__(self, train_std, criterion_np=None):
        # self.threshold = threshold  # Model-determined anomaly threshold
        self.train_std = train_std  # Standard deviation of data
        self.criterion_np = criterion_np  # Expert-defined red line
        self.window_length_limit = 5
        self.anomaly_window = {}  # Used to store sustained anomaly scores when continuous anomalies occur
        self.sustained_healthy_number = 0  # Records the number of continuously healthy points after a sustained anomaly
        self.anomaly_happen = False

        self.dev_weight = 0.5
        self.dur_weight = 0.5

        self.update_flag = False
        self.alarms = []
        self.health_info = {}

    def get_weight(self, x_np, error, threshold):
        if self.criterion_np is not None:
            if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                self.dev_weight = 1.0
                self.dur_weight = 0.0
            else:
                self.dev_weight = 0.0
                self.dur_weight = 1.0
        else:
            self.dev_weight = 0.0
            self.dur_weight = 1.0

    # Deviation score
    def get_deviation_score(self, x_np, error, threshold):
        if error >= threshold:
            ratio = 1.0
            if self.criterion_np is not None:
                if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                    return 1.0
                else:
                    # self.criterion_np[np.where(self.criterion_np < threshold)[0]] = threshold
                    ratio = np.max(np.abs(x_np - threshold) / (self.criterion_np - threshold + 1e-4))
            ratio = min(ratio, 1.0)
            score = (error - threshold) / self.train_std
            score = ReLU(ratio, 14, 0.5) * sigmoid(score, 5 / 6)
        else:
            score = 0
        return score

    # Internal class variables used: self.anomaly_window; self.window_length_limit; self.sustained_healthy_number; self.update_flag = True; self.anomaly_happen
    def get_sustained_score(self, index, error, threshold):
        # Judged by SPOT
        if error > threshold:  # Current point is abnormal
            self.anomaly_window[index] = 1
            self.sustained_healthy_number = 0
            if len(self.anomaly_window) <= self.window_length_limit:  # First 5 points
                if np.sum([v for k, v in self.anomaly_window.items()]) < 3:
                    score = 0
                else:
                    score = 1
                    self.anomaly_happen = True
            else:  # Once the anomaly window exceeds 5 points and hasn't been cleared, subsequent points are considered anomalies
                score = 1
                self.anomaly_happen = True

        else:  # Current point is normal
            self.sustained_healthy_number = self.sustained_healthy_number + 1
            if self.anomaly_happen:  # There was an anomaly before
                if self.sustained_healthy_number < 3:
                    score = 1
                    self.anomaly_window[index] = 0
                else:  # Three consecutive healthy points
                    score = 0
                    self.anomaly_window = {}
                    self.anomaly_happen = False
                    self.update_flag = False
            else:  # No previous anomaly
                score = 0
                if self.sustained_healthy_number < 3:
                    self.anomaly_window[index] = 0
                else:
                    self.anomaly_window = {}
                    self.update_flag = False
                    self.anomaly_happen = False

        return score

    def fit(self, index, x_np, error, threshold):  # index: current point; x: raw data; error: anomaly score; threshold: SPOT threshold
        deviation_score = self.get_deviation_score(x_np, error, threshold)
        sustained_score = self.get_sustained_score(index, error, threshold)
        self.get_weight(x_np, error, threshold)
        health_score = 1 - (deviation_score * self.dev_weight + sustained_score * self.dur_weight)
        self.health_info[index] = [health_score, deviation_score, sustained_score, self.dur_weight]
        self.update(health_score, sustained_score)

    def predict(self, threshold):
        for index, score_l in self.health_info.items():
            if score_l[0] < threshold:
                self.alarms.append(index)
        return self.alarms

    def update(self, health_score, new_sustained_score):
        if health_score < 0.2:  # Indicates an anomaly should be reported
            if not self.update_flag:  # Previous points have already been modified
                for Id in self.anomaly_window.keys():
                    origin_health_score_info = self.health_info[Id]
                    # origin_health_score, origin_dev_score, origin_dur_score = self.health_info[Id]
                    new_health_score = 1 - (origin_health_score_info[1] * self.dev_weight + new_sustained_score * self.dur_weight)
                    origin_health_score_info[0] = new_health_score
                    origin_health_score_info[2] = new_sustained_score
                    self.health_info[Id] = origin_health_score_info
                self.update_flag = True
        else:  # Do nothing
            pass

    @property
    def health_score_info(self):
        res = []
        for v in self.health_info.values():
            res.append(v)
        return res
