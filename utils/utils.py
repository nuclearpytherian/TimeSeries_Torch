
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.metrics import confusion_matrix


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


def random_pandasDFClassifier(batch_size, N_FEATURES, NUM_CLASSES):
    for i in range(NUM_CLASSES):
        m = np.random.randint(5,100, 1)
        data = pd.DataFrame(np.random.normal(m, 2, (batch_size*2, N_FEATURES)))
        data['y'] = [i for _ in range(batch_size*2)]
        tr_data = data[:batch_size]
        val_data = data[batch_size:]
        if i == 0:
            tr_df = tr_data
            val_df = val_data
        else:
            tr_df = pd.concat([tr_df, tr_data], axis=0)
            val_df = pd.concat([val_df, val_data], axis=0)
    tr_df.index = pd.date_range(start="1/1/2010", end="1/1/2015", periods=len(tr_df))
    val_df.index = pd.date_range(start="1/1/2010", end="1/1/2015", periods=len(val_df))
    return tr_df, val_df


def random_pandasDFRegressor(batch_size, N_FEATURES):
    cos = np.cos(list(range(batch_size * 2))) + np.array(list(range(batch_size * 2))) * 0.01
    for i in range(N_FEATURES):
        m = np.random.randint(5, 100, 1)
        x = cos * m + np.random.normal(m/3,m/3,batch_size*2)
        data = pd.DataFrame(x)
        if i == 0:
            df_cat = data
        else:
            df_cat = pd.concat([df_cat, data], axis=1)
    df_cat['y'] = cos
    df_cat.index = pd.date_range(start="1/1/2010", end="1/1/2015", periods=len(df_cat))
    tr_data = df_cat[:batch_size]
    val_data = df_cat[batch_size:]
    return tr_data, val_data


class PandasTimeDataset(Dataset):

    def __init__(self, data_df, label_col, TIME_STEP, mode:'classifier or regressor', date_format="%Y/%m/%d %H:%M:%S"):
        super().__init__()
        self.x = data_df.drop(label_col, axis=1).values
        self.y = data_df[label_col].values
        self.time = pd.to_datetime(data_df.index, format=date_format)
        self.label_col = label_col
        self.TIME_STEP = TIME_STEP
        self.mode = mode

    def __getitem__(self, index):
        x_data, y_data = self.toTimeSeries()
        if self.mode == 'classifier':
            x_data, y_data = torch.from_numpy(x_data).float(), torch.from_numpy(y_data).long()
        elif self.mode == 'regressor':
            x_data, y_data = torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float()
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



class Predictor:
    def __init__(self, trained_model, dataloader):
        self.model = trained_model
        self.dataloader = dataloader

    def __call__(self, dataloader):
        preds = []
        for i, (x, y) in enumerate(dataloader):
            pred = self.model_predict(self.model, x)
            preds.append(int(pred))
            if i == len(dataloader)-1:
                break
        return preds

    def model_predict(self, model, input):
        if input.ndim == 1:
            input = input.unsqueeze(0)
        pred = model(input).data.max(1, keepdim=True)[1].squeeze()
        return pred

    def get_preds(self):
        preds = []
        labels = []
        for i, (x, y) in enumerate(self.dataloader):
            pred = self.model_predict(self.model, x)
            preds.append(int(pred))
            labels.append(int(y))
            if i == len(self.dataloader) - 1:
                break
        return preds, labels

    def get_accuracy(self):
        preds, labels = self.get_preds()
        return np.sum([1 for x, y in zip(preds, labels) if int(x) == int(y)]) / len(preds)

    def confusion_matrix(self):
        preds, labels = self.get_preds()
        df = pd.DataFrame(confusion_matrix(preds, labels))
        return df