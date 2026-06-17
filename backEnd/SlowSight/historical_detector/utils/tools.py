import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj='type1'):
    """
    Adjust the learning rate during training according to a predefined schedule.

    Args:
        optimizer: The optimizer whose learning rate will be adjusted.
        epoch: Current training epoch.
        learning_rate: Initial learning rate.
        lradj: Learning rate adjustment strategy. Options:
            - 'type1': Decay by half every 1 epoch.
            - 'type2': Manual step decay based on specific epochs.
            - 'cosine': Cosine decay over training epochs.

    Returns:
        None
    """
    # Define learning rate schedules
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            patience (int): Number of epochs to wait after last improvement.
            verbose (bool): If True, prints a message when a new best model is saved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Call method to evaluate whether to stop training.

        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to save if validation improves.
            path (str): Path to save checkpoint.

        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): PyTorch model.
            path (str): Directory to save the model.

        Returns:
            None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Args:
        mean (float): Mean of the training data.
        std (float): Standard deviation of the training data.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transform input data using standardization.

        Args:
            data (np.ndarray or torch.Tensor): Input tensor or array.

        Returns:
            Transformed data with zero mean and unit variance.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse the standardization transformation.

        Args:
            data (np.ndarray or torch.Tensor): Scaled data.

        Returns:
            Original unscaled data.
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualize true values and predicted values.

    Args:
        true (array-like): Ground truth values.
        preds (array-like or None): Predicted values.
        name (str): File path to save the plot.

    Returns:
        None
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    """
    Adjust prediction labels based on ground truth to ensure anomaly consistency.

    This function propagates predictions to align with ground truth anomalies,
    making sure once a true anomaly is detected, all related predictions are marked as anomaly too.

    Args:
        gt (list or np.array): Ground truth binary labels (0 for normal, 1 for anomaly).
        pred (list or np.array): Predicted binary labels before adjustment.

    Returns:
        tuple: Adjusted ground truth and predicted arrays.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Back-propagate anomaly from current point
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Forward-propagate anomaly
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculate accuracy between predicted and true labels.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_pred == y_true)
