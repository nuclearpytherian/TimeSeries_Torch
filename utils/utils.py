
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

def data_to_series(X, y, TIME_STEP):
    N = len(X)
    output_X = []
    output_y = []
    for i in range(TIME_STEP, N):
        t = []
        Xi = X[i-TIME_STEP:i, :]
        for j in range(TIME_STEP):
            t.append(Xi[[j], :])
        output_X.append(t)
        output_y.append(y[i])
    return np.squeeze(np.array(output_X)), np.array(output_y)

def Conv1d_output_dim(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def random_pandasDF(N, N_FEATURES, NUM_CLASSES):
    data = pd.DataFrame(np.random.normal(20, 5, (N, N_FEATURES)))
    data['y'] = np.random.choice(np.array(range(NUM_CLASSES)), N)
    data.index = pd.date_range(start="1/1/2010", end="1/1/2015", periods=N)
    return data


class PandasTimeClassifierDataset(Dataset):

    def __init__(self, data_df, label_col, TIME_STEP, date_format="%Y/%m/%d %H:%M:%S"):
        super().__init__()
        self.x = data_df.drop(label_col, axis=1).values
        self.y = data_df[label_col].values
        self.time = pd.to_datetime(data_df.index, format=date_format)
        self.label_col = label_col
        self.TIME_STEP = TIME_STEP

    def __getitem__(self, index):
        x_data, y_data = self.toTimeSeries()
        x_data, y_data = torch.from_numpy(x_data).float(), torch.from_numpy(y_data).long()
        return x_data[index], y_data[index]

    def __len__(self):
        return len(self.y) - self.TIME_STEP

    def toTimeSeries(self):
        N = len(self.y)
        output_X = []
        output_y = []
        for i in range(self.TIME_STEP, N):
            t = []
            Xi = self.x[i - self.TIME_STEP:i, :]
            for j in range(self.TIME_STEP):
                t.append(Xi[[j], :])
            output_X.append(t)
            output_y.append(self.y[i])
        return np.squeeze(np.array(output_X)), np.array(output_y)
