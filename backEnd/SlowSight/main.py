import os
import sys
import argparse

import numpy as np
import pandas as pd

from SlowSight.utils.Logger import Logger
from SlowSight.utils.dataLoader_temp import dataLoader
from SlowSight.utils.postProcessor_temp import postProcessor


import json
from SlowSight.utils.utils import *
from SlowSight.historical_detector.model import TransGAN

from SlowSight.peer_comparator.model import PeerComparator
from SlowSight.report import Reporter


def run(config, rawdata_file, paint_indicators_path, ratePath):
    """
    Main workflow for anomaly detection and root cause analysis

    Args:
        config: Configuration object containing parameters
        rawdata_file: Name of the raw data file
        paint_indicators_path: Path to save visualization results
        ratePath: Path to save anomaly rate data
    """
    # Load ground truth data if available
    ground_truth_path = os.path.join(config.rawdata_dir, "ground_truth.csv")
    ground_truth_df = pd.read_csv(ground_truth_path) if os.path.exists(ground_truth_path) else pd.DataFrame()

    # Initialize data loader and load datasets
    data_loader = dataLoader(rawdata_file, config)
    train_df, valid_df, test_df, metric_candidates = data_loader.return_original_data()

    # Perform peer comparison-based anomaly detection
    peer_comparator = PeerComparator(
        ground_truth_df, train_df, valid_df, test_df, metric_candidates,
        config.result_dir, smooth_window=config.smooth_window,
        window_size=config.window_size, batch_size=config.batch_size,
        func=config.peer_eval_func
    )
    peer_comparator.detect_anomalies_dbscan()

    # Save peer comparison results
    peer_train_results, peer_valid_results, peer_test_results = peer_comparator.return_results()
    peer_disk_labels = peer_comparator.pred_labels
    peer_results = pd.concat([peer_train_results, peer_valid_results, peer_test_results])
    peer_results.to_csv(os.path.join(config.result_dir, config.dbscan_result_name), index=False)

    # Load processed data for model training
    train_processed, valid_processed, test_processed, train_original, valid_original, test_original = data_loader.return_data()
    print("Dataset shapes:", train_processed.shape, valid_processed.shape, test_processed.shape)

    # Initialize TransGAN model for anomaly detection
    model = TransGAN(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        total_metrics=config.total_metrics,
        num_entities=config.num_entities,
        d_model=config.d_model,
        d_entity=config.d_entity,
        factor=config.factor,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        activation=config.activation,
        e_layers=config.e_layers,
        dropout=config.dropout,
        train_epochs=config.train_epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
        weight_decay=config.weight_decay,
        max_len=config.max_len,
        batch_size=config.data_batch_size
    )

    # Process data through the model
    model.processed_data(train_processed, valid_processed, test_processed)
    print('Data shapes:', train_processed.shape, valid_processed.shape, test_processed.shape)

    # # Train the model
    # print('---- Training model ----')
    # model_metrics = model.fit()
    # model.save(os.path.join(config.model_dir, 'encoder'), os.path.join(config.model_dir, 'decoder_G'),
    #            os.path.join(config.model_dir, 'decoder_D'))
    # print(f'model saved in {config.model_dir}')
    # drawLoss(dataset_name=rawdata_file[:-4], model_result_path=os.path.join(config.result_dir, 'train_loss.png'),
    #          train_loss1=model_metrics['train_loss1'], train_loss2=model_metrics['train_loss2'],
    #          epoch=model_metrics['epoch'])
    # drawLoss(dataset_name=rawdata_file[:-4], model_result_path=os.path.join(config.result_dir, 'valid_loss.png'),
    #          train_loss1=model_metrics['valid_loss1'], train_loss2=model_metrics['valid_loss2'],
    #          epoch=model_metrics['epoch'])
    
    # ============= restore model ===============
    print('----restore model----')
    model.restore(os.path.join(config.model_dir, 'encoder'), os.path.join(config.model_dir, 'decoder_G'),
                  os.path.join(config.model_dir, 'decoder_D'))
    print(f'----model restored from {config.model_dir}----')

    # ============= predict ===============
    print('----predict----')
    model_data = model.predict()
    train_predict_G = model_data['train_predict_G']
    train_predict_G_D = model_data['train_predict_G_D']
    valid_predict_G = model_data['valid_predict_G']
    valid_predict_G_D = model_data['valid_predict_G_D']
    test_predict_G = model_data['test_predict_G']
    test_predict_G_D = model_data['test_predict_G_D']

    # Prepare data for post-processing
    data_info = {
        'train_length': train_predict_G_D.shape[0],
        'valid_length': valid_predict_G_D.shape[0],
        'dataset': rawdata_file
    }

    # Concatenate processed data and predictions
    x_all_processed = np.concatenate((
        train_processed[-len(train_predict_G_D):],
        valid_processed[-len(valid_predict_G_D):],
        test_processed[-len(test_predict_G_D):]
    ), axis=0)

    all_G = np.concatenate((train_predict_G, valid_predict_G, test_predict_G), axis=0)
    all_G_D = np.concatenate((train_predict_G_D, valid_predict_G_D, test_predict_G_D), axis=0)

    # Combine time stamps and original data
    all_times = np.concatenate((
        data_loader.train_times[-len(train_predict_G_D):],
        data_loader.valid_times[-len(valid_predict_G_D):],
        data_loader.test_times[-len(test_predict_G_D):]
    ), axis=0)

    x_all_original = np.concatenate((
        train_original[-len(train_predict_G_D):],
        valid_original[-len(valid_predict_G_D):],
        test_original[-len(test_predict_G_D):]
    ), axis=0)

    # Save processed data and predictions
    np.save(os.path.join(config.result_dir, config.all_processed_data), x_all_processed)
    np.save(os.path.join(config.result_dir, config.all_G_D), all_G_D)
    np.save(os.path.join(config.result_dir, config.all_times), all_times)
    np.save(os.path.join(config.result_dir, config.features), data_loader.features)

    with open(os.path.join(config.result_dir, 'data_info.json'), 'w') as f:
        json.dump(data_info, f)
    print("Data info saved:", data_info)

    # ============= postProcessor ===============
    disks_result_csv_path = os.path.join(config.result_dir, "disks_result.csv")
    result_csv_path = os.path.join(config.result_dir, "result.csv")
    result_json_path = os.path.join(config.result_dir, "result.json")
    post = postProcessor(processed_data=x_all_processed, reconstruct_data=all_G_D, generate_data=all_G,
                         original_data=x_all_original,
                         data_info=data_info,
                         all_features=data_loader.features.tolist(),
                         all_times=all_times,
                         scaler=data_loader.scale)
    reconstruct_error = post.compute_score(score_type="common", alpha=config.alpha)
    np.save(os.path.join(config.result_dir, config.reconstruct_error), reconstruct_error)

    # Fit SPOT algorithm for thresholding
    post.spot_fit(q=config.q, level=config.level)

    # Locate root causes of anomalies and generate reports
    anomalies_cause_full_dict = post.locate_cause(rate_path=ratePath, result_path=result_json_path)
    post.paint_indicators(figPath=paint_indicators_path)
    post.get_anomalies(ground_truth_df, anomalies_cause_full_dict, result_csv_path, disks_result_csv_path)

    # Configure and generate final report
    report_config_dict = {
        'rawdata_dir': config.rawdata_dir,
        'result_dir': config.result_dir,
        'window_size': config.window_size,
        'history_pred_labels': config.history_labels_name,
        'peer_pred_labels': config.peer_labels_name,
        'merged_pred_labels': config.merged_labels_name,
        'classifier_type': config.shape_type,
        'classifier_seq_length': config.shape_seq_length,
    }

    # Parse report configuration
    parser = argparse.ArgumentParser()
    for field, value in report_config_dict.items():
        parser.add_argument(f'--{field}', default=value)
    report_config = parser.parse_args()

    # Load classifier model for anomaly pattern analysis
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(cur_dir)
    # Concatenate all timestamps (train/valid/test) and pass to Reporter
    # so the generated up-trend mask CSV matches the full D3 data size
    full_ts = np.concatenate([
        pd.to_datetime(test_df["timestamp"]).values,
        pd.to_datetime(train_df["timestamp"]).values,
        pd.to_datetime(valid_df["timestamp"]).values,
    ], axis=0)
    reporter = Reporter(report_config, test_df, data_info,
                        data_loader.features, reconstruct_error, all_times, peer_results, peer_disk_labels,
                        full_timestamps=full_ts)
    reporter.merge_result()
    fault_segments = reporter.find_fault_segments()
    sampled_components = peer_comparator.workload_aware_sample_components(sample_ratio=0.3)

    # Cluster the sampled anomaly segments and print a summary
    segment_labels, n_clusters, centroid_patterns = reporter.propagate_cluster_labels(fault_segments)
    print(f"[HAC] {len(fault_segments)}/{len(fault_segments)} segments sampled -> {n_clusters} clusters")

    # Continue with root cause localization; locate_cause internally re-clusters and writes JSON
    reporter.locate_cause(fault_segments, result_path=os.path.join(config.result_dir, config.merged_results_name))


if __name__ == '__main__':
    # Configuration parameters
    rawdataFile = "net_all"  # Input data file
    rawdata_dir = f"data/net/{rawdataFile}"  # Data directory
    method = 'test'  # Execution mode

    # Load configuration settings
    print(f"======== Processing: {rawdataFile} =========")
    cfg_para = {
        "dataset": rawdataFile,
        "rawdata_dir": rawdata_dir,
        "config_file": 'config.yml',
        "method": method
    }
    cfg = ConfigHandler(cfg_para).config

    # Setup output directories and logging
    pdf_path = cfg.result_dir
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)

    # Redirect console output to log file
    sys.stdout = Logger(os.path.join(pdf_path, f"temp.log"))
    sys.stderr = Logger(os.path.join(pdf_path, f"temp.log"))

    # Define output paths
    paint_indicatorsPath = os.path.join(pdf_path, f"{rawdataFile}.pdf")
    rate_path = os.path.join(pdf_path, f"===={rawdataFile}-rate.csv")

    # Execute main workflow
    start_time = time.time()
    run(cfg, rawdataFile, paint_indicatorsPath, rate_path)
    end_time = time.time()
    print(f"Execution time: {int(end_time - start_time)} seconds")