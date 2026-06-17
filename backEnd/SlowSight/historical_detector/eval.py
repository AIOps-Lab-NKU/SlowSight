import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from SlowSight.historical_detector.utils.tools import adjustment

from SlowSight.utils.evaluation import evaluate_detection
from SlowSight.utils.spot import SPOT


def itransformer_detection(test_labels, test_energy, train_energy, anomaly_ratio=0.1):
    """
    Detect anomalies using energy score thresholding based on percentile.

    Parameters:
        test_labels (np.ndarray): Ground truth labels for test data.
        test_energy (np.ndarray): Energy scores for test data.
        train_energy (np.ndarray): Energy scores for training data used as reference.
        anomaly_ratio (float): Expected ratio of anomalies in the data for threshold estimation.

    Returns:
        None
    """
    # Combine train and test energy to compute a global threshold
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
    print("Threshold :", threshold)

    # Apply threshold to generate binary predictions
    pred = (test_energy > threshold).astype(int)

    # Convert ground truth labels to NumPy array
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)

    original_pred = np.array(pred)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # Adjust predictions to align with ground truth
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    gt = np.squeeze(gt)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    # Calculate evaluation metrics
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

    # Save results to file
    f = open("result_anomaly_detection.txt", 'a')
    f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))
    f.write('\n')
    f.write('\n')
    f.close()
    return


def evaluate(train_energy, test_energy, gt, pred, original_pred, test_labels, folder_path='detection_results'):
    """
    Evaluate detection performance and save results.

    Parameters:
        train_energy (np.ndarray): Energy scores from training data.
        test_energy (np.ndarray): Energy scores from test data.
        gt (np.ndarray): Ground truth labels.
        pred (np.ndarray): Adjusted predicted labels.
        original_pred (np.ndarray): Original predicted labels before adjustment.
        test_labels (np.ndarray): Original test labels.
        folder_path (str): Directory path to save result CSV files.

    Returns:
        None
    """
    # Run standard evaluation method
    print("historical_detector/eval.py-evaluate1")
    evaluate_detection(gt, pred)

    # Save results into CSV file
    result_df = pd.DataFrame({"ground_truth": gt, "pred": original_pred, "pred_adjusted": pred})
    result_df.to_csv(os.path.join(folder_path, "result.csv"), index=False)

    # spot
    spot_pred, spot_detect_res = spot_fit(gt, train_energy, test_energy, q=1e-4, level=0.98)

    # Convert to NumPy arrays
    spot_original_pred = np.array(spot_pred)
    spot_gt = test_labels.astype(int)

    # Align predictions with ground truth
    spot_gt, spot_pred = adjustment(spot_gt, spot_pred)

    spot_pred = np.array(spot_pred)
    spot_gt = np.array(spot_gt)
    spot_gt = np.squeeze(spot_gt)

    # Compute metrics for SPOT-based detection
    spot_accuracy = accuracy_score(spot_gt, spot_pred)
    spot_precision, spot_recall, spot_f_score, spot_support = precision_recall_fscore_support(spot_gt, spot_pred,
                                                                                            average='binary')
    print(
        "SPOT Accuracy : {:0.4f}, SPOT Precision : {:0.4f}, SPOT Recall : {:0.4f}, SPOT F-score : {:0.4f} ".format(
            spot_accuracy, spot_precision,
            spot_recall, spot_f_score))

    # Run standard evaluation for SPOT predictions
    print("historical_detector/eval.py-evaluate2")
    evaluate_detection(spot_gt, spot_pred)

    # Save SPOT results to CSV
    spot_result_df = pd.DataFrame({"ground_truth": spot_gt, "pred": spot_original_pred, "pred_adjusted": spot_pred})
    spot_result_df.to_csv(os.path.join(folder_path, "spot_result.csv"), index=False)

    # Append SPOT metrics to result file
    f = open("result_anomaly_detection.txt", 'a')
    f.write(
        "SPOT Accuracy : {:0.4f}, SPOT Precision : {:0.4f}, SPOT Recall : {:0.4f}, SPOT F-score : {:0.4f} ".format(
            spot_accuracy, spot_precision,
            spot_recall, spot_f_score))
    f.write('\n')
    f.write('\n')
    f.close()
    return


def spot_fit(gt, train_energy, test_energy, q=1e-3, level=0.98):
    """
    Use SPOT algorithm to detect anomalies based on energy scores.

    Parameters:
        gt (np.ndarray): Ground truth labels for test data.
        train_energy (np.ndarray): Energy scores from training data.
        test_energy (np.ndarray): Energy scores from test data.
        q (float): Risk parameter used for threshold estimation.
        level (float): Confidence level for extreme value theory.

    Returns:
        spot_pred (np.ndarray): Predicted anomaly labels using SPOT.
        spot_detect_res (dict): Dictionary containing thresholds and detected alarms.
    """
    train_data = np.copy(train_energy)

    s = SPOT(q)  # Create SPOT object
    s.fit(train_data, test_energy)  # Fit SPOT model
    s.initialize(level=level, verbose=True)  # Initialize threshold
    spot_detect_res = s.run()  # Run SPOT to detect anomalies
    spot_detect_res["upper_thresholds"] = spot_detect_res["thresholds"]  # Store upper thresholds

    # Generate prediction vector with 1 for anomalies
    anomalies = np.array(spot_detect_res['alarms'])
    spot_pred = np.zeros(gt.shape[0])
    spot_pred[anomalies] = 1

    return spot_pred, spot_detect_res
