import math
import numpy as np


class SlidingWindowDataset():

    def __init__(self, values, window_size):
        self._values = values  # Training data, 2D ndarray
        self._window_size = window_size
        # 2D array where each element is a list of window indices [number of windows, window size]
        self._strided_values = self._to_windows(self._values)

    def _to_windows(self, values):
        sliding_windows = []  # Elements are lists, each representing window indices
        for i in range(values.shape[0] - self._window_size + 1):
            sliding_windows.append(self._values[i:i + self._window_size])
        return np.array(sliding_windows)  # [Number of windows, Window size]

    def __getitem__(self, index):  # Returns the specified window
        return np.copy(self._strided_values[index]).astype(np.float32)

    def __len__(self):  # Number of windows
        return np.size(self._strided_values, 0)


class SlidingWindowDataLoader():

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self._dataset = dataset  # Dataset of sliding windows
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

        if self._shuffle:
            # [Number of windows]
            self._idxs = np.random.permutation(self._dataset.shape[0])
        else:
            self._idxs = np.arange(self._dataset.shape[0])

        if self._drop_last:
            self._total = len(self._dataset) // self._batch_size
        else:
            self._total = (len(self._dataset) + self._batch_size - 1) // self._batch_size

    def get_item(self, idx):
        if (idx + 1) * self._batch_size > self._dataset.shape[0]:
            # 1D vector
            batch_idx = self._idxs[idx * self._batch_size:]
            batch = self._dataset[batch_idx]
        else:
            batch_idx = self._idxs[idx * self._batch_size:(idx + 1) * self._batch_size]
            batch = self._dataset[batch_idx]

        return batch


if __name__ == '__main__':
    # Create example data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    window_size = 4

    # Create dataset
    data_sliding_window = SlidingWindowDataLoader(
        SlidingWindowDataset(data, window_size)._strided_values,
        batch_size=2
    )
    for step in range(data_sliding_window._total):
        print(data_sliding_window.get_item(step))
        print('---')
