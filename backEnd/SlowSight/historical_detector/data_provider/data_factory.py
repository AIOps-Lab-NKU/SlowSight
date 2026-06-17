# data_factory.py
# This file defines a unified data provider interface for different datasets used in the iTransGAN project.

from SlowSight.historical_detector.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, DataSegLoader
from torch.utils.data import DataLoader

# Mapping from dataset names to corresponding dataset classes
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(flag, train_data, valid_data, test_data, batch_size, seq_len, num_workers=10):
    """
    Provide data loader and dataset based on flag (train/val/test).

    Parameters:
        flag (str): Indicates which type of data is needed ('train', 'val', or 'test').
        train_data: Training data used for fitting.
        valid_data: Validation data used for model selection.
        test_data: Test data used for final evaluation.
        batch_size (int): Batch size for data loader.
        seq_len (int): Length of input sequence.
        num_workers (int): Number of workers for pytorch DataLoader. Default is 10.

    Returns:
        data_set: The dataset object.
        data_loader: PyTorch DataLoader for the dataset.
    """
    Data = DataSegLoader

    # Determine whether to shuffle the data
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = batch_size

    # Whether to drop the last incomplete batch
    drop_last = False

    # Build the dataset using DataSegLoader
    data_set = Data(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        win_size=seq_len,
        flag=flag,
    )

    print('data_provider', flag, len(data_set))

    # Create DataLoader with appropriate settings
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
