# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple
from sklearn.metrics import precision_recall_curve


def adjust_labels(pred_labels: np.ndarray, raw_labels: np.ndarray,
                  delay: int = None, inplace: bool = False, advance=1) -> np.ndarray:
    """
    This function corrects labels Donut outputs.

    :param pred_labels: labels output by Donut
    :param raw_labels: ground truth
    :param delay: maximum allowed delay
    :param inplace: whether to modify the origin array or create a copy of it
    :return: labels after correction
    """

    # in case that the lengths of pred_labels and true_labels are not the same
    if np.shape(pred_labels) != np.shape(raw_labels):
        raise ValueError("Two series have the different shape!")

    # if no delay argument passed in, the delay will be set as the length of raw_labels,
    #  which means any point of an anomaly segment detected will be regarded the detection of the whole segment.
    if delay is None:
        delay = len(raw_labels)
    # specify the start of continuous zeros and ones
    splits = np.where(raw_labels[1:] != raw_labels[:-1])[0] + 1
    # a flag to show whether current part is continuous ones
    is_anomaly = (raw_labels[0] == 1)
    adjusted_labels = np.copy(pred_labels) if not inplace else pred_labels
    # Evaluation results do not consider individual reported outliers
    for index in range(len(adjusted_labels)):
        if index > 0 and index < len(adjusted_labels) - 1 and adjusted_labels[index] == 1:
            if adjusted_labels[index-1] == 0 and adjusted_labels[index+1] == 0:
                adjusted_labels[index] = 0

    # start position of every segment
    pos = 0
    for part in splits:
        # 'part' is the start position of next segment, as well as the end of last segment
        if is_anomaly:
            # split ground truth by maximum allowed delay
            ptr = min(pos + delay + 1, part)
            adjusted_labels[pos - advance: part] = 1 if np.sum(pred_labels[pos - advance: ptr]) > 0 else 0
        is_anomaly = not is_anomaly
        pos = part
    part = len(raw_labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_labels[pos - advance: part] = 1 if np.sum(pred_labels[pos - advance: ptr]) > 0 else 0

    return adjusted_labels

# Get consecutive exception segments
def get_anomaly_segment(labels):
    label_segments = []
    mark = False
    temp = None
    for idx, value in enumerate(labels):
        if value == 1:
            if mark == False:
                mark = True
                temp = set([idx])
            else:
                temp.add(idx)
        elif value == 0 and mark == True:
            label_segments.append(temp)
            mark = False
            temp = None
    if temp != None:
        label_segments.append(temp)
    return label_segments

#"""
def evaluate(pred_labels: np.ndarray, raw_labels: np.ndarray, delay: int = None) -> Tuple[int, int, int, int]:
    '''by segment
    This function is used for calculating TP, FP, FN. The input for this function is
    supposed to be 1-D vector which contains only 0 and 1 values.

    :param pred_labels: the predicted labels Donut outputs
    :param raw_labels: labels that are manually set in datasets
    :param delay: maximum allowed delay
    :return: a tuple, which is (TP, FP, FN, adjusted_labels)
    '''
    # in case that the lengths of pred_labels and true_labels are not the same
    if np.shape(pred_labels) != np.shape(raw_labels):
        raise ValueError("Two series have the different shape!")

    # adjust pred_labels
    adjusted_labels = adjust_labels(pred_labels, raw_labels, delay=delay)

    # segment statistic
    raw_labels_segments = get_anomaly_segment(raw_labels)
    adjusted_labels_segments = get_anomaly_segment(adjusted_labels)
    TP = 0
    for label_item in raw_labels_segments:
        for pred_item in adjusted_labels_segments:
            if len(pred_item & label_item) != 0:
                TP += 1
                break
    FP = len(adjusted_labels_segments) - TP
    FN = len(raw_labels_segments) - TP

    return int(TP), int(FP), int(FN), adjusted_labels
#"""

# source code from original Donut project
def adjust_scores(scores, labels, delay: int = None, inplace: bool = False) -> np.ndarray:
    assert np.shape(scores) == np.shape(labels)
    if delay is None:
        delay = len(scores)
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    is_anomaly = labels[0] == 1
    adjusted_scores = np.copy(scores) if not inplace else scores
    pos = 0
    for part in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, part)
            adjusted_scores[pos: ptr] = np.max(adjusted_scores[pos: ptr])
            adjusted_scores[ptr: part] = np.maximum(adjusted_scores[ptr: part], adjusted_scores[pos])
        is_anomaly = not is_anomaly
        pos = part
    part = len(labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_scores[pos: part] = np.max(adjusted_scores[pos: ptr])
    return adjusted_scores

def best_f1score(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)

    best_threshold = thresholds[np.argmax(f1score)]
    best_precision = precision[np.argmax(f1score)]
    best_recall = recall[np.argmax(f1score)]

    return best_threshold, best_precision, best_recall, np.max(f1score)

def evaluate_detection(ground_truth, pred, delay=None):
    TP, FP, FN, adjusted_labels = evaluate(pred_labels=pred, raw_labels=ground_truth, delay=delay) #All five points of delayed hits count as correct
    if TP + FP == 0:
        P = 0
    else:
        P = 100 * TP / (TP + FP)
    if TP + FN == 0:
        R = 0
    else:
        R = 100 * TP / (TP + FN)
    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    print(
        'true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
        .format(TP, FP, FN, P, R, F1))
    return adjusted_labels


# delete this part of test code when plugged into the model
if __name__ == '__main__':
    ground_truth = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    pred =         np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    evaluate_detection(ground_truth, pred)
