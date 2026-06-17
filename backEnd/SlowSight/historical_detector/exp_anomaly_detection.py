from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from SlowSight.historical_detector.utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from SlowSight.utils.spot import SPOT
from SlowSight.utils.evaluation import evaluate_detection
from SlowSight.utils.lr_schedulers import WarmupPolyLR
from SlowSight.utils.postProcessor_temp import paint_indicators, get_anomalies, locate_cause, paint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

# Set tensor sharing strategy to avoid multiprocessing errors
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        """
        Build the anomaly detection model.

        Returns:
            model (nn.Module): The initialized model.
        """
        model = self.model_dict[self.args.model].Model(self.args).float()

        # Multi-GPU support
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Load dataset and data loader based on flag (train/val/test).

        Parameters:
            flag (str): Indicates which dataset to load ('train', 'val', or 'test').

        Returns:
            data_set: Loaded dataset.
            data_loader: DataLoader for the dataset.
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Select optimizer for training.

        Returns:
            model_optim (optim.Optimizer): Optimizer instance.
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Select loss function for training.

        Returns:
            criterion (nn.Module): Loss function instance.
        """
        criterion = nn.MSELoss()
        return criterion

    @staticmethod
    def get_model_gradients(model):
        """
        Get gradients of all parameters in the model.

        Parameters:
            model (nn.Module): PyTorch model.

        Returns:
            grads (OrderedDict): Dictionary of parameter names and their gradients.
        """
        grads = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
        return grads

    @staticmethod
    def update_grads(model, grads):
        """
        Add existing gradients with new gradients.

        Parameters:
            model (nn.Module): Model whose gradients are being updated.
            grads (dict): Dictionary of gradients to be added.

        Returns:
            None
        """
        for name, param in model.named_parameters():
            if name in grads:
                param.grad = param.grad + grads[name]

    def train(self, setting):
        """
        Train the anomaly detection model.

        Parameters:
            setting (str): Configuration string used for saving checkpoints.

        Returns:
            model (nn.Module): Trained model.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler_all = WarmupPolyLR(model_optim, self.args.train_epochs, warmup_iters=5)

        train_loss1 = []
        train_loss2 = []
        valid_loss1 = []
        valid_loss2 = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.model.train()
            epoch_time = time.time()

            train_losses1 = []
            train_losses2 = []

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                _, L, N = batch_x.shape

                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)

                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

                loss_G = (1 / (epoch+1)) * criterion(w_G, batch_x) + (1 - 1 / (epoch+1)) * criterion(w_G_D, batch_x)
                loss_D = (1 / (epoch+1)) * criterion(w_D, batch_x) - (1 - 1 / (epoch+1)) * criterion(w_G_D, batch_x)

                model_optim.zero_grad()

                loss_G.backward(retain_graph=True)
                g_encoder_grads = self.get_model_gradients(self.model.encoder)
                loss_D.backward()
                self.update_grads(self.model.encoder, g_encoder_grads)

                model_optim.step()

                train_losses1.append(loss_G.detach().numpy())
                train_losses2.append(loss_D.detach().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss1: {2:.7f} | loss2: {3:.7f}".format(i + 1, epoch + 1, loss_G.item(), loss_D.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            scheduler_all.step()

            train_loss1_mean = np.mean(train_losses1)
            train_loss2_mean = np.mean(train_losses2)
            train_loss1.append(train_loss1_mean)
            train_loss2.append(train_loss2_mean)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            val_losses1 = []
            val_losses2 = []

            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

                val_loss1 = 1 / (epoch+1) * criterion(batch_x, w_G) + (1 - 1 / (epoch+1)) * criterion(batch_x, w_G_D)
                val_loss2 = 1 / (epoch+1) * criterion(batch_x, w_D) - (1 - 1 / (epoch+1)) * criterion(batch_x, w_G_D)
                val_losses1.append(val_loss1.detach().numpy())
                val_losses2.append(val_loss2.detach().numpy())
                print("Epoch: {0} | Vali Loss1: {1:.7f} Vali Loss2: {2:.7f}".format(epoch + 1, val_loss1.item(), val_loss2.item()))

            val1_loss = np.mean(val_losses1)
            val2_loss = np.mean(val_losses2)
            valid_loss1.append(val1_loss)
            valid_loss2.append(val2_loss)

            early_stopping(np.average(valid_loss1), self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        Evaluate the trained model on the test set and detect anomalies.

        Parameters:
            setting (str): Identifier for the current experiment.
            test (int): Whether to load a pre-trained model (0: no, 1: yes).

        Returns:
            None
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        train_reconstruct_error = []
        train_model_processed_data = []
        train_model_reconstruct_data = []

        # Step 1: Statistic calculation on the training set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)
                # print(batch_x.shape) # torch.Size([128, 100, 83])
                # print(batch_y.shape) # torch.Size([128, 100, 1])
                # print(z.shape) # torch.Size([128, 83, 128])
                # print(w_G.shape) # torch.Size([128, 100, 83])
                # print(w_D.shape) # torch.Size([128, 100, 83])
                # print(w_G_D.shape) # torch.Size([128, 100, 83])
                reconstruct_error = 0.1 * np.square(batch_x[:, -1, :].detach().numpy() - w_G[:, -1, :].detach().numpy()) + \
                                   (1 - 0.1) * np.square(batch_x[:, -1, :].detach().numpy() - w_G_D[:, -1, :].detach().numpy())

                score = np.sum(reconstruct_error, axis=-1)
                attens_energy.extend(score)

                train_model_processed_data.append(batch_x[:, -1, :].detach().numpy())
                train_model_reconstruct_data.append(w_G_D[:, -1, :].detach().numpy())

                train_reconstruct_error.extend(reconstruct_error)

            train_model_processed_data = np.concatenate(train_model_processed_data, axis=0)
            train_model_reconstruct_data = np.concatenate(train_model_reconstruct_data, axis=0)

        train_energy = np.array(attens_energy)

        test_reconstruct_error = []
        test_model_processed_data = []
        test_model_reconstruct_data = []

        # Step 2: Threshold estimation using combined energy from train and test
        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
            w_G = self.model.decoder_G(z).permute(0, 2, 1)
            w_D = self.model.decoder_D(z).permute(0, 2, 1)
            w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

            # criterion
            reconstruct_error = 0.1 * np.square(batch_x[:, -1, :].detach().numpy() - w_G[:, -1, :].detach().numpy()) + \
                                (1 - 0.1) * np.square(batch_x[:, -1, :].detach().numpy() - w_G_D[:, -1, :].detach().numpy())

            score = np.sum(reconstruct_error, axis=-1)
            attens_energy.extend(score)
            test_labels.extend(batch_y[:, -1].detach().numpy())

            test_model_processed_data.append(batch_x[:, -1, :].detach().numpy())
            test_model_reconstruct_data.append(w_G_D[:, -1, :].detach().numpy())

            test_reconstruct_error.extend(reconstruct_error)

        test_model_processed_data = np.concatenate(test_model_processed_data, axis=0)
        test_model_reconstruct_data = np.concatenate(test_model_reconstruct_data, axis=0)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # Step 3: Anomaly prediction using threshold
        pred = (test_energy > threshold).astype(int)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        original_pred = np.array(pred)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # Step 4: Adjust predictions to align with ground truth
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        gt = np.squeeze(gt)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        # Save results
        evaluate_detection(gt, pred)
        result_df = pd.DataFrame({"ground_truth": gt, "pred": original_pred, "pred_adjusted": pred})
        result_df.to_csv(os.path.join(folder_path, "result.csv"), index=False)

        # Use SPOT algorithm for adaptive thresholding
        spot_pred, spot_detect_res = self.spot_fit(gt, train_energy, test_energy, q=1e-4, level=0.98)
        spot_original_pred = np.array(spot_pred)
        spot_gt = test_labels.astype(int)
        spot_gt, spot_pred = adjustment(spot_gt, spot_pred)
        spot_pred = np.array(spot_pred)
        spot_gt = np.array(spot_gt)
        spot_gt = np.squeeze(spot_gt)

        spot_accuracy = accuracy_score(spot_gt, spot_pred)
        spot_precision, spot_recall, spot_f_score, spot_support = precision_recall_fscore_support(spot_gt, spot_pred, average='binary')
        print("SPOT Accuracy : {:0.4f}, SPOT Precision : {:0.4f}, SPOT Recall : {:0.4f}, SPOT F-score : {:0.4f} ".format(
            spot_accuracy, spot_precision,
            spot_recall, spot_f_score))

        evaluate_detection(spot_gt, spot_pred)
        spot_result_df = pd.DataFrame({"ground_truth": spot_gt, "pred": spot_original_pred, "pred_adjusted": spot_pred})
        spot_result_df.to_csv(os.path.join(folder_path, "spot_result.csv"), index=False)

        # Plot reconstruction and save final results
        pdf_path = "./results"
        rate_path = os.path.join(pdf_path, f"rate.csv")
        result_csv_path = os.path.join(pdf_path, "result.csv")
        result_json_path = os.path.join(pdf_path, f"result.json")
        paint_indicators_path = os.path.join(pdf_path, f"result.pdf")
        ground_truth_path = os.path.join("./dataset/PSM/", "ground_truth.csv")
        ground_truth_df = pd.read_csv(ground_truth_path) if os.path.exists(ground_truth_path) else pd.DataFrame()

        all_features = ['gala_gopher_sli_tps@docker@sysbench','gala_gopher_nic_txspeed_KB@machine','gala_gopher_nic_rxspeed_KB@machine','gala_gopher_block_access_pagecache@machine@vda','gala_gopher_cpu_net_rx@machine','gala_gopher_cpu_sched@machine','gala_gopher_disk_util@machine@vda','gala_gopher_disk_util@machine@vda4','gala_gopher_block_count_latency_device@machine@vda','gala_gopher_block_count_latency_driver@machine@vda','gala_gopher_block_count_latency_req@machine@vda','gala_gopher_disk_wareq@machine@vda4','gala_gopher_disk_wareq@machine@vda','gala_gopher_disk_wspeed_kB@machine@vda','gala_gopher_block_write_bytes@machine@vda','gala_gopher_disk_wspeed@machine@dm-0','gala_gopher_disk_wspeed@machine@vda3','gala_gopher_disk_wspeed_kB@machine@dm-0','gala_gopher_disk_wspeed_kB@machine@vda3','gala_gopher_block_mark_buffer_dirty@machine','gala_gopher_disk_aqu@machine@vda3','gala_gopher_block_mark_buffer_dirty@machine@vda','gala_gopher_nic_tc_sent_drop@machine','gala_gopher_disk_wareq@machine@vda3','gala_gopher_mem_free_kB@machine','gala_gopher_proc_pm_size@machine@python3','gala_gopher_tcp_link_srtt@docker@worker','gala_gopher_block_latency_req_max@machine@vda','gala_gopher_block_latency_req_jitter@machine@vda','gala_gopher_cpu_rcu@machine','gala_gopher_disk_aqu@machine@vda','gala_gopher_disk_w_await@machine@vda','gala_gopher_proc_shared_clean_size@machine@python3','gala_gopher_mem_swap_util@machine','gala_gopher_mem_swap_free_kB@machine','gala_gopher_disk_w_await@machine@vda4','gala_gopher_disk_aqu@machine@vda4','gala_gopher_tcp_link_rcv_rtt@docker@WALreceiver','gala_gopher_cpu_timer@machine','gala_gopher_tcp_link_srtt@docker@WalSender','gala_gopher_mem_available_kB@machine','gala_gopher_block_mark_page_dirty@machine@vda','gala_gopher_tcp_link_rcv_rtt@docker@worker','gala_gopher_cpu_total_used_per@machine','gala_gopher_nic_tc_backlog@machine','gala_gopher_disk_wspeed_kB@machine@vda4','gala_gopher_disk_wspeed@machine@vda4','gala_gopher_sli_tps@docker@omm','gala_gopher_net_tcp_curr_estab@machine','gala_gopher_sli_tps@docker@postgres','gala_gopher_disk_rspeed@machine@vda4','gala_gopher_disk_rspeed@machine@vda','gala_gopher_net_tcp_retrans_segs@machine','gala_gopher_mem_util@machine','gala_gopher_proc_private_clean_size@machine@python3','gala_gopher_mem_buffers_kB@machine','gala_gopher_disk_rspeed_kB@machine@vda4','gala_gopher_disk_rspeed_kB@machine@vda','gala_gopher_disk_r_await@machine@vda','gala_gopher_disk_wspeed@machine@vda','gala_gopher_tcp_link_srtt@docker@python3','gala_gopher_block_read_bytes@machine@vda','gala_gopher_disk_wareq@machine@dm-1','gala_gopher_mem_inactive_kB@machine','gala_gopher_block_write_bytes@machine','gala_gopher_disk_r_await@machine@vda4','gala_gopher_tcp_link_sk_drops@docker@WalSender','gala_gopher_disk_rareq@machine@vda','gala_gopher_mem_cache_kB@machine','gala_gopher_disk_rareq@machine@vda4','gala_gopher_tcp_link_notack_bytes@docker@WalSender','gala_gopher_net_tcp_out_segs@machine','gala_gopher_disk_rareq@machine@dm-1','gala_gopher_mem_active_kB@machine','gala_gopher_disk_rareq@machine@vda3','gala_gopher_tcp_link_srtt@docker@WALreceiver','gala_gopher_cpu_iowait_msec@machine','gala_gopher_cpu_irq_msec@machine','gala_gopher_cpu_nice_msec@machine','gala_gopher_cpu_softirq_msec@machine','gala_gopher_cpu_system_msec@machine','gala_gopher_cpu_user_msec@machine','gala_gopher_tcp_link_retran_packets@docker@worker']

        model_processed_data = np.concatenate([train_model_processed_data, test_model_processed_data], axis=0)
        model_reconstruct_data = np.concatenate([train_model_reconstruct_data, test_model_reconstruct_data], axis=0)
        scores = np.array(combined_energy)
        all_reconstruct_error = np.concatenate([train_reconstruct_error, test_reconstruct_error], axis=0)

        import pickle
        with open(os.path.join('./results', 'scaler.pkl'), 'rb') as file:
            scaler = pickle.load(file)
        if isinstance(scaler, dict):
            true_processed_data = model_processed_data * scaler["std"] + scaler["mean"]
            true_reconstruct_data = model_reconstruct_data * scaler["std"] + scaler["mean"]
        else:
            true_processed_data = scaler.inverse_transform(model_processed_data)
            true_reconstruct_data = scaler.inverse_transform(model_reconstruct_data)
        print('true_processed_data: ', true_processed_data.shape)
        print('true_reconstruct_data: ', true_reconstruct_data.shape)

        cause_TimeSeg_map = defaultdict(list)
        all_times = np.array([float(i) for i in range(10080)])

        # Visualize indicators and anomalies
        paint(spot_detect_res, all_times, scores, all_features, true_processed_data, true_reconstruct_data, model_processed_data, model_reconstruct_data, fig_path=paint_indicators_path)

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.write("SPOT Accuracy : {:0.4f}, SPOT Precision : {:0.4f}, SPOT Recall : {:0.4f}, SPOT F-score : {:0.4f} ".format(
            spot_accuracy, spot_precision,
            spot_recall, spot_f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def spot_fit(self, gt, train_energy, test_energy, q=1e-3, level=0.98):
        """
        Fit SPOT algorithm to find optimal thresholds and detect anomalies.

        Parameters:
            gt (np.ndarray): Ground truth labels.
            train_energy (np.ndarray): Energy scores from training data.
            test_energy (np.ndarray): Energy scores from test data.
            q (float): Risk parameter for SPOT.
            level (float): Confidence level for extreme value theory.

        Returns:
            spot_pred (np.ndarray): Predicted anomaly labels using SPOT.
            spot_detect_res (dict): Detection results including alarms and thresholds.
        """
        train_data = np.copy(train_energy)

        s = SPOT(q)  # Create SPOT object
        s.fit(train_data, test_energy)  # Fit SPOT model
        s.initialize(level=level, verbose=True)  # Initialize thresholds
        spot_detect_res = s.run()  # Run SPOT to detect anomalies
        spot_detect_res["upper_thresholds"] = spot_detect_res["thresholds"]

        anomalies = np.array(spot_detect_res['alarms'])
        spot_pred = np.zeros(gt.shape[0])
        spot_pred[anomalies] = 1

        return spot_pred, spot_detect_res
